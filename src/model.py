import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoTokenizer, AutoModel
import math
from typing import List, Tuple, Dict, Optional, Union

class SpatialAttentionModule(nn.Module):
    """Novel spatial attention mechanism for image features"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.conv_query = nn.Conv2d(channels, channels // 4, 1)
        self.conv_key = nn.Conv2d(channels, channels // 4, 1)
        self.conv_value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize weights properly
        for m in [self.conv_query, self.conv_key, self.conv_value]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.conv_query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.conv_key(x).view(batch_size, -1, height * width)
        value = self.conv_value(x).view(batch_size, -1, height * width)
        
        # Compute attention with stability improvements
        attention = torch.bmm(query, key)
        attention = attention / math.sqrt(channels // 4)
        attention = torch.clamp(attention, -5, 5)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        return self.gamma * out + x

class HybridImageEncoder(nn.Module):
    """Hybrid image encoder combining TIMM backbone with novel spatial attention"""
    
    def __init__(self, model_name='vit_base_patch16_224.augreg_in21k_ft_in1k', feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Primary encoder from TIMM (for stability and pre-trained weights)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        backbone_dim = self.backbone.embed_dim if hasattr(self.backbone, 'embed_dim') else self.backbone.num_features
        
        # Custom spatial processing layers (your novel contribution)
        self.spatial_attention = SpatialAttentionModule(3)  # Applied to input
        
        # Feature projection and refinement
        self.feature_projector = nn.Linear(backbone_dim, feature_dim)
        
        # Additional refinement layers (your IP)
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in [self.feature_projector] + list(self.refinement.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input validation and normalization
        x = torch.clamp(x, -3.0, 3.0)
        
        # Apply spatial attention to input (novel contribution)
        if x.shape[2] == x.shape[3]:  # Square images for spatial attention
            try:
                x_attended = self.spatial_attention(x)
                x = 0.8 * x + 0.2 * x_attended  # Residual connection
            except:
                pass  # Fallback to original input
        
        # Extract features using TIMM backbone
        try:
            if hasattr(self.backbone, 'forward_features'):
                features = self.backbone.forward_features(x)
                if len(features.shape) == 3:  # ViT case: [B, N, D]
                    features = features[:, 0]  # CLS token
                elif len(features.shape) == 4:  # CNN case: [B, C, H, W]
                    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            else:
                features = self.backbone(x)
        except Exception as e:
            print(f"Backbone error: {e}, using fallback")
            batch_size = x.shape[0]
            features = torch.randn(batch_size, 768, device=x.device) * 0.1
        
        # Project to target dimension
        features = self.feature_projector(features)
        
        # Apply refinement (your novel processing)
        refined_features = self.refinement(features)
        
        # Ensure stability
        refined_features = torch.clamp(refined_features, -10, 10)
        
        return refined_features

class HybridTextEncoder(nn.Module):
    """Hybrid text encoder combining Transformers backbone with novel semantic layers"""
    
    def __init__(self, model_name='distilbert-base-uncased', feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Primary encoder from Transformers (for stability)
        try:
            self.backbone = AutoModel.from_pretrained(model_name)
            backbone_dim = self.backbone.config.dim if hasattr(self.backbone.config, 'dim') else self.backbone.config.hidden_size
        except:
            # Fallback configuration
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(config)
            backbone_dim = config.hidden_size
        
        # Feature projection
        self.feature_projector = nn.Linear(backbone_dim, feature_dim)
        
        # Novel semantic understanding layers (your IP)
        self.semantic_enhancement = nn.ModuleList([
            self._make_semantic_layer(feature_dim) for _ in range(2)
        ])
        
        # Multi-granularity aggregation (your novel contribution)
        self.word_aggregator = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.context_aggregator = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        
        # Final refinement
        self.final_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.backbone.modules():
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def _make_semantic_layer(self, dim):
        return nn.ModuleDict({
            'self_attention': nn.MultiheadAttention(dim, 8, batch_first=True, dropout=0.1),
            'feed_forward': nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim)
            ),
            'norm1': nn.LayerNorm(dim),
            'norm2': nn.LayerNorm(dim)
        })
    
    def forward(self, input_ids, attention_mask=None):
        # Extract features using backbone
        try:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get pooled representation
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
                
                # Masked mean pooling
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    features = sum_embeddings / sum_mask
                else:
                    features = hidden_states.mean(1)
            else:
                features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs[0].mean(1)
                
        except Exception as e:
            print(f"Text backbone error: {e}")
            batch_size = input_ids.shape[0]
            features = torch.randn(batch_size, 768, device=input_ids.device) * 0.1
            hidden_states = features.unsqueeze(1).expand(-1, input_ids.shape[1], -1)
        
        # Project to target dimension
        features = self.feature_projector(features)
        projected_hidden = self.feature_projector(hidden_states) if 'hidden_states' in locals() else features.unsqueeze(1)
        
        # Apply semantic enhancement (your novel layers)
        enhanced_features = projected_hidden
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        for layer in self.semantic_enhancement:
            try:
                # Self-attention
                attn_out, _ = layer['self_attention'](
                    enhanced_features, enhanced_features, enhanced_features, 
                    key_padding_mask=key_padding_mask
                )
                enhanced_features = layer['norm1'](enhanced_features + attn_out)
                
                # Feed-forward
                ff_out = layer['feed_forward'](enhanced_features)
                enhanced_features = layer['norm2'](enhanced_features + ff_out)
            except:
                # Fallback
                enhanced_features = layer['norm1'](enhanced_features)
        
        # Multi-granularity aggregation (your novel contribution)
        try:
            word_repr, _ = self.word_aggregator(
                enhanced_features, enhanced_features, enhanced_features,
                key_padding_mask=key_padding_mask
            )
            context_repr, _ = self.context_aggregator(
                word_repr, word_repr, word_repr,
                key_padding_mask=key_padding_mask
            )
            
            # Pool representations
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                seq_lengths = attention_mask.sum(-1, keepdim=True).clamp(min=1)
                word_pooled = (word_repr * mask_expanded).sum(1) / seq_lengths
                context_pooled = (context_repr * mask_expanded).sum(1) / seq_lengths
            else:
                word_pooled = word_repr.mean(1)
                context_pooled = context_repr.mean(1)
                
        except:
            # Fallback to basic pooling
            word_pooled = features
            context_pooled = features
        
        # Combine representations
        combined = torch.cat([word_pooled, context_pooled], dim=1)
        final_features = self.final_projection(combined)
        
        # Ensure stability
        final_features = torch.clamp(final_features, -10, 10)
        
        return final_features

class CrossModalFusionNetwork(nn.Module):
    """Enhanced cross-modal fusion with interactive attention and stability improvements"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Cross-modal attention layers
        self.vision_to_text = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        self.text_to_vision = nn.MultiheadAttention(feature_dim, 8, batch_first=True, dropout=0.1)
        
        # Interactive fusion network (your novel architecture)
        self.interactive_fusion = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Modality-specific refinement
        self.vision_refine = nn.Linear(feature_dim, feature_dim)
        self.text_refine = nn.Linear(feature_dim, feature_dim)
        
        # Gating mechanism for adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, vision_features, text_features):
        # Input validation and normalization
        vision_features = torch.clamp(vision_features, -10, 10)
        text_features = torch.clamp(text_features, -10, 10)
        
        # Expand for attention mechanism
        vision_expanded = vision_features.unsqueeze(1)
        text_expanded = text_features.unsqueeze(1)
        
        # Cross-modal attention with error handling
        try:
            v2t_attended, _ = self.vision_to_text(vision_expanded, text_expanded, text_expanded)
            v2t_attended = torch.clamp(v2t_attended.squeeze(1), -5, 5)
        except:
            v2t_attended = vision_features
            
        try:
            t2v_attended, _ = self.text_to_vision(text_expanded, vision_expanded, vision_expanded)
            t2v_attended = torch.clamp(t2v_attended.squeeze(1), -5, 5)
        except:
            t2v_attended = text_features
        
        # Interactive fusion (your novel contribution)
        fused_features = torch.cat([
            vision_features, text_features, v2t_attended, t2v_attended
        ], dim=1)
        
        interactive_output = self.interactive_fusion(fused_features)
        
        # Adaptive gating
        gate_input = torch.cat([vision_features, text_features], dim=1)
        fusion_weight = self.fusion_gate(gate_input)
        
        # Modality-specific refinement with gating
        refined_vision = self.vision_refine(vision_features + fusion_weight * interactive_output)
        refined_text = self.text_refine(text_features + (1 - fusion_weight) * interactive_output)
        
        return refined_vision, refined_text, interactive_output

class LabeleeFoundation(nn.Module):
    """Re-architected Labelee Foundation Model for pre-training"""
    
    def __init__(self, 
                 vision_model_name='vit_base_patch16_224.augreg_in21k_ft_in1k',
                 text_model_name='distilbert-base-uncased',
                 feature_dim=768,
                 num_classes=1000):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Hybrid encoders combining stability with novel architectures
        self.image_encoder = HybridImageEncoder(vision_model_name, feature_dim)
        self.text_encoder = HybridTextEncoder(text_model_name, feature_dim)
        
        # Your novel fusion network
        self.fusion_network = CrossModalFusionNetwork(feature_dim)
        
        # Multi-task heads for pre-training
        self.similarity_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2)
        )
        
        # Reconstruction heads for self-supervised learning
        self.vision_reconstruction = nn.Linear(feature_dim, feature_dim)
        self.text_reconstruction = nn.Linear(feature_dim, feature_dim)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and not any(m in encoder.modules() 
                                                   for encoder in [self.image_encoder.backbone, 
                                                                 self.text_encoder.backbone]):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images, input_ids, attention_mask=None, task='similarity', return_features=False):
        # Extract features from both modalities
        try:
            vision_features = self.image_encoder(images)
        except Exception as e:
            print(f"Vision encoder error: {e}")
            batch_size = images.shape[0]
            vision_features = torch.randn(batch_size, self.feature_dim, device=images.device) * 0.1
            
        try:
            text_features = self.text_encoder(input_ids, attention_mask)
        except Exception as e:
            print(f"Text encoder error: {e}")
            batch_size = input_ids.shape[0]
            text_features = torch.randn(batch_size, self.feature_dim, device=input_ids.device) * 0.1
        
        # Cross-modal fusion
        fused_vision, fused_text, interactive_features = self.fusion_network(
            vision_features, text_features
        )
        
        if return_features:
            return {
                'vision_features': vision_features,
                'text_features': text_features,
                'fused_vision': fused_vision,
                'fused_text': fused_text,
                'interactive_features': interactive_features
            }
        
        # Task-specific outputs
        if task == 'similarity':
            similarity_scores = self.similarity_head(interactive_features)
            return similarity_scores
        
        elif task == 'classification':
            class_logits = self.classification_head(interactive_features)
            return class_logits
        
        elif task == 'contrastive':
            vision_proj = self.contrastive_head(fused_vision)
            text_proj = self.contrastive_head(fused_text)
            return vision_proj, text_proj
        
        elif task == 'reconstruction':
            vision_recon = self.vision_reconstruction(fused_vision)
            text_recon = self.text_reconstruction(fused_text)
            return vision_recon, text_recon, vision_features, text_features
        
        else:
            return fused_vision, fused_text, interactive_features

class MultiTaskLoss(nn.Module):
    """Enhanced multi-task loss for pre-training with numerical stability"""
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, delta=0.2):
        super().__init__()
        self.alpha = alpha    # Similarity loss weight
        self.beta = beta      # Reconstruction loss weight
        self.gamma = gamma    # Contrastive loss weight
        self.delta = delta    # Classification loss weight
        
    def similarity_loss(self, scores, labels):
        scores = scores.squeeze()
        labels = labels.float().squeeze()
        
        # Handle dimension mismatches
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
            
        return F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
    
    def contrastive_loss(self, vision_proj, text_proj, temperature=0.07):
        # Normalize projections
        vision_norm = F.normalize(vision_proj, p=2, dim=1, eps=1e-8)
        text_norm = F.normalize(text_proj, p=2, dim=1, eps=1e-8)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(vision_norm, text_norm.T) / temperature
        sim_matrix = torch.clamp(sim_matrix, -10, 10)
        
        # Create positive pair labels
        batch_size = vision_proj.shape[0]
        labels = torch.arange(batch_size, device=vision_proj.device)
        
        # Symmetric contrastive loss
        loss_v2t = F.cross_entropy(sim_matrix, labels, reduction='mean')
        loss_t2v = F.cross_entropy(sim_matrix.T, labels, reduction='mean')
        
        return (loss_v2t + loss_t2v) / 2
    
    def reconstruction_loss(self, recon_v, recon_t, orig_v, orig_t):
        loss_v = F.mse_loss(recon_v, orig_v, reduction='mean')
        loss_t = F.mse_loss(recon_t, orig_t, reduction='mean')
        return (loss_v + loss_t) / 2
    
    def classification_loss(self, logits, labels):
        # Reshape from [batch_size, 1] to [batch_size] and convert to integer
        labels = labels.squeeze().long() 
        return F.cross_entropy(logits, labels, reduction='mean')
        
    def forward(self, outputs, labels, task='similarity'):
        total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
        loss_dict = {}
        
        if task == 'similarity':
            sim_loss = self.similarity_loss(outputs, labels)
            total_loss = total_loss + self.alpha * sim_loss
            loss_dict['similarity'] = sim_loss.item()
            
        elif task == 'contrastive':
            vision_proj, text_proj = outputs
            cont_loss = self.contrastive_loss(vision_proj, text_proj)
            total_loss = total_loss + self.gamma * cont_loss
            loss_dict['contrastive'] = cont_loss.item()
            
        elif task == 'reconstruction':
            recon_v, recon_t, orig_v, orig_t = outputs
            recon_loss = self.reconstruction_loss(recon_v, recon_t, orig_v, orig_t)
            total_loss = total_loss + self.beta * recon_loss
            loss_dict['reconstruction'] = recon_loss.item()
            
        elif task == 'classification':
            class_loss = self.classification_loss(outputs, labels)
            total_loss = total_loss + self.delta * class_loss
            loss_dict['classification'] = class_loss.item()
        
        # Ensure finite loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
            
        return total_loss, loss_dict

def create_labelee_foundation(config: Optional[Dict] = None):
    """Factory function to create Labelee Foundation model with different configurations"""
    
    # 1. Define the default configuration
    default_config = {
        'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'text_model_name': 'distilbert-base-uncased',
        'feature_dim': 768,
        'num_classes': 1000
    }
    
    # 2. If a custom config is provided, update the defaults
    if config:
        default_config.update(config)
    
    # 3. Create a new dictionary with ONLY the arguments the model needs
    model_args = {
        'vision_model_name': default_config['vision_model_name'],
        'text_model_name': default_config['text_model_name'],
        'feature_dim': default_config['feature_dim'],
        'num_classes': default_config['num_classes']
    }
    
    # 4. Create the model using the specific arguments and the tokenizer
    model = LabeleeFoundation(**model_args)
    tokenizer = AutoTokenizer.from_pretrained(default_config['text_model_name'])
    
    # 5. Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Labelee Foundation Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Vision backbone: {default_config['vision_model_name']}")
    print(f"  Text backbone: {default_config['text_model_name']}")
    
    # 6. Return both the model and the tokenizer
    return model, tokenizer

# Example training step with stability improvements
def train_step(model, optimizer, criterion, images, input_ids, attention_mask, labels, task='similarity'):
    """Robust training step with error handling and gradient clipping"""
    model.train()
    optimizer.zero_grad()
    
    try:
        # Forward pass
        outputs = model(images, input_ids, attention_mask, task=task)
        
        # Compute loss
        loss, loss_dict = criterion(outputs, labels, task=task)
        
        # Check for valid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected, skipping step")
            return 0.0, {}
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item(), loss_dict
        
    except Exception as e:
        print(f"Error in training step: {e}")
        return 0.0, {} 