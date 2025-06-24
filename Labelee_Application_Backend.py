# Labelee Application Backend
# Attribution: Labelee model developed by @1mman0
# This application utilizes the Labelee model for multimodal vision-language tasks

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
    print("flask-cors not installed. Install it using 'pip install flask-cors' for cross-origin support.")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import math
import os
import re
from collections import Counter
import time
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging for performance metrics
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/labelee_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Labelee Model Classes (developed by @1mman0)
class SpatialAttentionModule(nn.Module):
    """Novel spatial attention mechanism for image features"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv_query = nn.Conv2d(channels, channels // 4, 1)
        self.conv_key = nn.Conv2d(channels, channels // 4, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        for m in [self.conv_query, self.conv_key, self.conv_value]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.conv_query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch_size, -1, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)
        attention = torch.bmm(query, key) / math.sqrt(channels // 4)
        attention = torch.clamp(attention, -5, 5)
        attention = F.softmax(attention, dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, channels, height, width)
        return self.gamma * out + x

class HierarchicalImageEncoder(nn.Module):
    """Hierarchical image encoder that captures multi-scale features"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale_1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            SpatialAttentionModule(64)
        )
        self.scale_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SpatialAttentionModule(128)
        )
        self.scale_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SpatialAttentionModule(256)
        )
        self.scale_4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SpatialAttentionModule(512)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64 + 128 + 256 + 512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, feature_dim, 1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveMaxPool2d(4)
        local_feat_size = feature_dim * 16
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim + local_feat_size, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        x = torch.clamp(x, -3.0, 3.0)
        feat_1 = self.scale_1(x)
        feat_2 = self.scale_2(feat_1)
        feat_3 = self.scale_3(feat_2)
        feat_4 = self.scale_4(feat_3)
        target_size = feat_2.shape[2:]
        feat_1_up = F.interpolate(feat_1, size=target_size, mode='bilinear', align_corners=False)
        feat_3_up = F.interpolate(feat_3, size=target_size, mode='bilinear', align_corners=False)
        feat_4_up = F.interpolate(feat_4, size=target_size, mode='bilinear', align_corners=False)
        fused_features = torch.cat([feat_1_up, feat_2, feat_3_up, feat_4_up], dim=1)
        fused = self.fusion_conv(fused_features)
        global_feat = self.global_pool(fused).flatten(1)
        local_feat = self.local_pool(fused).flatten(1)
        combined = torch.cat([global_feat, local_feat], dim=1)
        refined_features = self.refinement(combined)
        return torch.clamp(refined_features, -10, 10)

class ContextualWordEmbedding(nn.Module):
    """Contextual word embedding with dynamic attention"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.context_transform = nn.Linear(embed_dim, embed_dim)
        self.dynamic_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.normal_(self.word_embedding.weight, std=0.02)
        nn.init.constant_(self.word_embedding.weight[0], 0)
        
    def forward(self, input_ids):
        word_emb = self.word_embedding(input_ids)
        context = self.context_transform(word_emb)
        gate = torch.sigmoid(self.dynamic_gate(word_emb))
        return word_emb + gate * context

class SemanticTextEncoder(nn.Module):
    """Novel text encoder with semantic understanding layers"""
    
    def __init__(self, vocab_size: int = 10000, feature_dim: int = 768, max_length: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.word_encoder = ContextualWordEmbedding(vocab_size, feature_dim)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, feature_dim) * 0.02)
        self.semantic_layers = nn.ModuleList([
            self._make_semantic_layer(feature_dim) for _ in range(4)
        ])
        self.word_aggregator = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.phrase_aggregator = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.sentence_aggregator = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.final_projection = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _make_semantic_layer(self, dim):
        return nn.ModuleDict({
            'self_attention': nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.1),
            'feed_forward': nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim)
            ),
            'norm1': nn.LayerNorm(dim),
            'norm2': nn.LayerNorm(dim)
        })
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]
            seq_len = self.max_length
        x = self.word_encoder(input_ids)
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        if attention_mask is None:
            attention_mask = (input_ids != 0)
        key_padding_mask = ~attention_mask.bool()
        for layer in self.semantic_layers:
            try:
                attn_out, _ = layer['self_attention'](x, x, x, key_padding_mask=key_padding_mask)
                attn_out = torch.clamp(attn_out, -5, 5)
                x = layer['norm1'](x + attn_out)
                ff_out = layer['feed_forward'](x)
                x = layer['norm2'](x + ff_out)
            except RuntimeError as e:
                logging.error(f"Semantic layer error: {e}")
                x = layer['norm1'](x)
        attention_mask_float = attention_mask.float()
        seq_lengths = attention_mask.sum(-1, keepdim=True).clamp(min=1)
        try:
            word_repr, _ = self.word_aggregator(x, x, x, key_padding_mask=key_padding_mask)
            word_pooled = (word_repr * attention_mask_float.unsqueeze(-1)).sum(1) / seq_lengths
        except:
            word_pooled = (x * attention_mask_float.unsqueeze(-1)).sum(1) / seq_lengths
        try:
            phrase_pooled = word_pooled
            window_size = min(3, seq_len)
            if seq_len >= window_size:
                phrase_features = []
                for i in range(0, seq_len - window_size + 1, window_size):
                    end_idx = min(i + window_size, seq_len)
                    window_mask = attention_mask[:, i:end_idx]
                    if window_mask.sum() > 0:
                        window_feat = x[:, i:end_idx]
                        window_lengths = window_mask.sum(-1, keepdim=True).clamp(min=1)
                        phrase_out = (window_feat * window_mask.float().unsqueeze(-1)).sum(1) / window_lengths
                        phrase_features.append(phrase_out)
                if phrase_features:
                    phrase_pooled = torch.stack(phrase_features, dim=1).mean(1)
        except:
            phrase_pooled = word_pooled
        try:
            sentence_repr, _ = self.sentence_aggregator(x, x, x, key_padding_mask=key_padding_mask)
            sentence_pooled = (sentence_repr * attention_mask_float.unsqueeze(-1)).sum(1) / seq_lengths
        except:
            sentence_pooled = word_pooled
        combined_repr = torch.cat([word_pooled, phrase_pooled, sentence_pooled], dim=1)
        final_features = self.final_projection(combined_repr)
        return torch.clamp(final_features, -10, 10)

class CrossModalFusionNetwork(nn.Module):
    """Novel cross-modal fusion using interactive attention"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        self.vision_to_text = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.text_to_vision = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.interactive_fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.vision_refine = nn.Linear(feature_dim, feature_dim)
        self.text_refine = nn.Linear(feature_dim, feature_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, vision_features, text_features):
        vision_features = torch.clamp(vision_features, -10, 10)
        text_features = torch.clamp(text_features, -10, 10)
        vision_expanded = vision_features.unsqueeze(1)
        text_expanded = text_features.unsqueeze(1)
        try:
            v2t_attended, _ = self.vision_to_text(vision_expanded, text_expanded, text_expanded)
            v2t_attended = torch.clamp(v2t_attended, -5, 5)
        except:
            v2t_attended = vision_expanded
        try:
            t2v_attended, _ = self.text_to_vision(text_expanded, vision_expanded, vision_expanded)
            t2v_attended = torch.clamp(t2v_attended, -5, 5)
        except:
            t2v_attended = text_expanded
        v2t_attended = v2t_attended.squeeze(1)
        t2v_attended = t2v_attended.squeeze(1)
        fused_features = torch.cat([vision_features, text_features, v2t_attended, t2v_attended], dim=1)
        interactive_output = self.interactive_fusion(fused_features)
        refined_vision = self.vision_refine(vision_features + interactive_output)
        refined_text = self.text_refine(text_features + interactive_output)
        return refined_vision, refined_text, interactive_output

class Labelee(nn.Module):
    """Main model with hierarchical understanding and cross-modal fusion
    Developed by @1mman0"""
    
    def __init__(self, vocab_size: int = 10000, feature_dim: int = 768, num_classes: int = 1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_encoder = HierarchicalImageEncoder(feature_dim)
        self.text_encoder = SemanticTextEncoder(vocab_size, feature_dim)
        self.fusion_network = CrossModalFusionNetwork(feature_dim)
        self.similarity_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        self.vision_reconstruction = nn.Linear(feature_dim, feature_dim)
        self.text_reconstruction = nn.Linear(feature_dim, feature_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, images, input_ids, attention_mask=None, task='similarity'):
        try:
            vision_features = self.image_encoder(images)
        except Exception as e:
            logging.error(f"Image encoder error: {e}")
            batch_size = images.shape[0]
            vision_features = torch.randn(batch_size, self.feature_dim, device=images.device) * 0.1
        try:
            text_features = self.text_encoder(input_ids, attention_mask)
        except Exception as e:
            logging.error(f"Text encoder error: {e}")
            batch_size = input_ids.shape[0]
            text_features = torch.randn(batch_size, self.feature_dim, device=input_ids.device) * 0.1
        try:
            fused_vision, fused_text, interactive_features = self.fusion_network(vision_features, text_features)
        except Exception as e:
            logging.error(f"Fusion network error: {e}")
            fused_vision = vision_features
            fused_text = text_features
            interactive_features = (vision_features + text_features) / 2
        if task == 'similarity':
            similarity_scores = self.similarity_head(interactive_features)
            similarity_scores = torch.sigmoid(similarity_scores)
            return similarity_scores
        elif task == 'classification':
            class_logits = self.classification_head(interactive_features)
            return class_logits
        elif task == 'retrieval':
            return fused_vision, fused_text
        elif task == 'reconstruction':
            vision_recon = self.vision_reconstruction(fused_vision)
            text_recon = self.text_reconstruction(fused_text)
            return vision_recon, text_recon, vision_features, text_features
        else:
            return fused_vision, fused_text, interactive_features

class MultiTaskLoss(nn.Module):
    """Multi-task loss combining various objectives"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def similarity_loss(self, scores, labels):
        scores = scores.squeeze()
        labels = labels.float().squeeze()
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return F.binary_cross_entropy(scores, labels, reduction='mean')
    
    def reconstruction_loss(self, recon_v, recon_t, orig_v, orig_t):
        loss_v = F.mse_loss(recon_v, orig_v, reduction='mean')
        loss_t = F.mse_loss(recon_t, orig_t, reduction='mean')
        return loss_v + loss_t
    
    def contrastive_loss(self, vision_features, text_features, temperature=0.1):
        vision_norm = F.normalize(vision_features, p=2, dim=1, eps=1e-8)
        text_norm = F.normalize(text_features, p=2, dim=1, eps=1e-8)
        sim_matrix = torch.matmul(vision_norm, text_norm.T) / temperature
        sim_matrix = torch.clamp(sim_matrix, -10, 10)
        batch_size = vision_features.shape[0]
        labels = torch.arange(batch_size, device=vision_features.device)
        loss_v2t = F.cross_entropy(sim_matrix, labels, reduction='mean')
        loss_t2v = F.cross_entropy(sim_matrix.T, labels, reduction='mean')
        return (loss_v2t + loss_t2v) / 2
    
    def forward(self, model_outputs, labels, task='similarity'):
        total_loss = 0
        if task == 'similarity':
            scores = model_outputs
            sim_loss = self.similarity_loss(scores, labels)
            total_loss += self.alpha * sim_loss
        elif task == 'reconstruction':
            recon_v, recon_t, orig_v, orig_t = model_outputs
            recon_loss = self.reconstruction_loss(recon_v, recon_t, orig_v, orig_t)
            total_loss += self.beta * recon_loss
        elif task == 'retrieval':
            vision_feat, text_feat = model_outputs
            contrastive_loss = self.contrastive_loss(vision_feat, text_feat)
            total_loss += self.gamma * contrastive_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        return total_loss

class EnhancedTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, 
            "<MASK>": 4, "<NUM>": 5, "<PUNCT>": 6
        }
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_built = False
        
    def build_vocab(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            text = text.lower()
            text = re.sub(r'\d+', '<NUM>', text)
            text = re.sub(r'[^\w\s]', '<PUNCT>', text)
            words = text.split()
            word_counts.update(words)
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        idx = len(self.special_tokens)
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        self.vocab_built = True
        
    def encode(self, text: str, max_length: int = 128):
        if not self.vocab_built:
            raise ValueError("Build vocabulary first!")
        text = text.lower()
        text = re.sub(r'\d+', '<NUM>', text)
        text = re.sub(r'[^\w\s]', '<PUNCT>', text)
        words = ["<CLS>"] + text.split() + ["<SEP>"]
        input_ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words[:max_length]]
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.word2idx["<PAD>"])
            attention_mask.append(0)
        return input_ids, attention_mask

# Placeholder dataset class (replace with your actual implementation)
class VisionLanguageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        # Mock data for testing
        self.data = [
            {"image_path": "dummy1.jpg", "text": "A dog running in the park", "label": 0.9},
            {"image_path": "dummy2.jpg", "text": "A cat sleeping on a couch", "label": 0.8},
            {"image_path": "dummy3.jpg", "text": "A car driving on a road", "label": 0.7}
        ]
        self.tokenizer = EnhancedTokenizer(vocab_size=10000)
        self.tokenizer.build_vocab([item["text"] for item in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Mock image (replace with actual image loading)
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        input_ids, attention_mask = self.tokenizer.encode(item["text"])
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        label = torch.tensor(item["label"], dtype=torch.float)
        return image, input_ids, attention_mask, label

app = Flask(__name__)
if CORS:
    CORS(app)  # Enable CORS only if flask-cors is available
else:
    logging.warning("CORS not enabled. Frontend and backend must be on the same origin.")

@app.route("/api/process", methods=["POST"])
def process_data():
    """Process image and/or text with Labelee model for inference"""
    start_time = time.time()
    try:
        image = request.files.get("image")
        text = request.form.get("text")
        task = request.form.get("task", "similarity")
        user_id = request.form.get("user_id", "demo_user")

        if not image and not text:
            return jsonify({"error": "Image or text input required"}), 400

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Labelee(vocab_size=10000, feature_dim=768, num_classes=1000).to(device)
        checkpoint_path = f"checkpoints/{user_id}/model.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Process image
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_data = None
        if image:
            img = Image.open(image).convert('RGB')
            image_data = image_transform(img).unsqueeze(0).to(device)

        # Process text
        tokenizer = EnhancedTokenizer(vocab_size=10000)
        input_ids, attention_mask = [], []
        if text:
            tokenizer.build_vocab([text])
            input_ids, attention_mask = tokenizer.encode(text)
            input_ids = torch.tensor([input_ids]).to(device)
            attention_mask = torch.tensor([attention_mask]).to(device)

        # Run model
        with torch.no_grad():
            outputs = model(image_data, input_ids, attention_mask, task=task)

        # Format results
        results = []
        if task == "similarity":
            score = outputs.item() if outputs.numel() == 1 else outputs[0].item()
            results.append({
                "title": "Similarity Score",
                "value": f"{score:.2f}",
                "description": f"Image-text similarity score: {score:.2f}",
                "confidence": score
            })
            logging.info(f"Similarity task: score={score:.2f}, user_id={user_id}")
        elif task == "classification":
            logits = outputs.softmax(dim=-1)
            class_idx = logits.argmax().item()
            confidence = logits.max().item()
            class_names = [f"Class_{i}" for i in range(1000)]  # Replace with actual classes
            class_name = class_names[class_idx]
            results.append({
                "title": "Classification Result",
                "value": class_name,
                "description": f"Classified as {class_name} (confidence: {confidence:.2f})",
                "confidence": confidence
            })
            logging.info(f"Classification task: class={class_name}, confidence={confidence:.2f}, user_id={user_id}")
        elif task == "retrieval":
            vision_features, text_features = outputs
            vision_norm = vision_features.norm().item()
            text_norm = text_features.norm().item()
            results.append({
                "title": "Vision Feature Norm",
                "value": f"{vision_norm:.2f}",
                "description": f"L2 norm of vision features: {vision_norm:.2f}",
                "confidence": 0.5
            })
            results.append({
                "title": "Text Feature Norm",
                "value": f"{text_norm:.2f}",
                "description": f"L2 norm of text features: {text_norm:.2f}",
                "confidence": 0.5
            })
            logging.info(f"Retrieval task: vision_norm={vision_norm:.2f}, text_norm={text_norm:.2f}, user_id={user_id}")
        elif task == "reconstruction":
            vision_recon, text_recon, vision_orig, text_orig = outputs
            vision_loss = F.mse_loss(vision_recon, vision_orig).item()
            text_loss = F.mse_loss(text_recon, text_orig).item()
            results.append({
                "title": "Vision Reconstruction Loss",
                "value": f"{vision_loss:.4f}",
                "description": f"Vision MSE loss: {vision_loss:.4f}",
                "confidence": 0.5
            })
            results.append({
                "title": "Text Reconstruction Loss",
                "value": f"{text_loss:.4f}",
                "description": f"Text MSE loss: {text_loss:.4f}",
                "confidence": 0.5
            })
            logging.info(f"Reconstruction task: vision_loss={vision_loss:.4f}, text_loss={text_loss:.4f}, user_id={user_id}")

        inference_time = time.time() - start_time
        logging.info(f"Inference completed in {inference_time:.2f}s for task={task}, user_id={user_id}")
        return jsonify(results)
    except Exception as e:
        logging.error(f"Process error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def train_model():
    """Train Labelee model on uploaded dataset"""
    start_time = time.time()
    try:
        dataset = request.files.get("dataset")
        user_id = request.form.get("user_id", "demo_user")
        task = request.form.get("task", "similarity")

        if not dataset or not user_id or not task:
            return jsonify({"error": "Missing dataset, user_id, or task"}), 400

        # Save dataset
        dataset_path = f"data/{user_id}/dataset"
        os.makedirs(dataset_path, exist_ok=True)
        dataset_file_path = os.path.join(dataset_path, dataset.filename)
        dataset.save(dataset_file_path)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Labelee(vocab_size=10000, feature_dim=768, num_classes=1000).to(device)
        checkpoint_path = f"checkpoints/{user_id}/model.pth"
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Load dataset
        dataset = VisionLanguageDataset(dataset_path, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        dataloader = DataLoader(dataset, batch_size=16, num_workers=2, pin_memory=True)

        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = MultiTaskLoss(alpha=1.0, beta=0.5, gamma=0.3)
        loss_curve = {'epochs': [], 'training': []}
        for epoch in range(5):
            model.train()
            epoch_loss = 0
            for batch in dataloader:
                images, input_ids, attention_mask, labels = batch
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(images, input_ids, attention_mask, task=task)
                loss = criterion(outputs, labels, task=task)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            avg_loss = epoch_loss / len(dataloader)
            loss_curve['epochs'].append(epoch + 1)
            loss_curve['training'].append(avg_loss)
            logging.info(f"Epoch {epoch+1}/{5}, Task: {task}, Avg Loss: {avg_loss:.4f}, user_id={user_id}")

        # Save checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f}s for task={task}, user_id={user_id}")
        return jsonify({
            "status": "completed",
            "checkpoint": checkpoint_path,
            "final_loss": loss_curve['training'][-1],
            "epochs": 5,
            "loss_curve": loss_curve
        })
    except Exception as e:
        logging.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    app.run(host="0.0.0.0", port=8000, debug=True)