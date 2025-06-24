import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import logging
import shutil
import json
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import your new model architecture
# Make sure your new_Labelee_model.py file is named something like `labelee_model.py`
# and is in the same directory.
from new_Labelee_model import create_labelee_foundation, MultiTaskLoss

# --- Configuration ---

@dataclass
class TrainingConfig:
    """Modern Training Configuration using a dataclass."""
    # Paths
    checkpoint_dir: str = "checkpoints_foundation"
    log_dir: str = "logs_foundation"
    dataset_root: str = "/Users/immanuelolajuyigbe/Downloads/my_downloaded_data"
    
    # Model Hyperparameters
    vision_model_name: str = 'vit_base_patch16_224.augreg_in21k_ft_in1k'
    text_model_name: str = 'distilbert-base-uncased'
    feature_dim: int = 768

    # Training Hyperparameters
    batch_size: int = 16  # Can be larger with GPU and AMP
    num_epochs: int = 50
    learning_rate: float = 5e-5 # Common starting LR for fine-tuning transformers
    weight_decay: float = 1e-2
    gradient_clip_value: float = 1.0
    patience: int = 5 # Early stopping patience
    
    # System/Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True  # Automatic Mixed Precision
    num_workers: int = 2
    resume_training: bool = True

# --- Core Components ---

class EpochManager:
    """
    ### REFACTOR NOTE ###
    Simplified EpochManager. State is now primarily stored in the .pth checkpoints.
    This class now focuses on tracking metrics, early stopping, and plotting.
    """
    def __init__(self, checkpoint_dir: str, patience: int):
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.metrics_file = os.path.join(checkpoint_dir, 'training_metrics.json')
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        self.latest_model_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = []
        self.should_stop = False

    def end_epoch(self, epoch_metrics: Dict) -> bool:
        """Log epoch metrics and check for improvement."""
        self.current_epoch = epoch_metrics['epoch']
        self.history.append(epoch_metrics)
        self._save_metrics()
        
        improved = False
        if epoch_metrics['val_f1'] > self.best_val_f1:
            self.best_val_f1 = epoch_metrics['val_f1']
            self.best_epoch = self.current_epoch
            self.epochs_no_improve = 0
            improved = True
            print(f"🏆 New best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.should_stop = True
            print(f"⏹️ Early stopping triggered after {self.patience} epochs with no improvement.")

        return improved

    def save_checkpoint(self, model, optimizer, scheduler, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        torch.save(state, self.latest_model_path)
        if is_best:
            shutil.copyfile(self.latest_model_path, self.best_model_path)
            print(f"💾 Best model checkpoint saved to {self.best_model_path}")

    def load_checkpoint(self, model, optimizer, scheduler, load_best=False):
        path = self.best_model_path if load_best else self.latest_model_path
        if not os.path.exists(path):
            return 0 # Start from epoch 0
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.history = checkpoint.get('history', [])
            self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"✅ Checkpoint loaded from {path}. Resuming from epoch {start_epoch}.")
            return start_epoch
        except Exception as e:
            print(f"❌ Could not load checkpoint from {path}: {e}")
            return 0

    def _save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_curves(self):
        if not self.history:
            return
        
        df = pd.DataFrame(self.history)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs. Epochs')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy')
        ax2.plot(df['epoch'], df['val_f1'], label='Val F1-Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric')
        ax2.set_title('Validation Metrics vs. Epochs')
        ax2.legend()
        ax2.grid(True)
        
        save_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"📈 Training curves saved to {save_path}")

class VisionLanguageDataset(Dataset):
    """
    ### REFACTOR NOTE ###
    - Simplified to only yield positive (image, caption) pairs.
    - Uses the __call__ method of the Hugging Face tokenizer for idiomatic encoding.
    - Returns a dictionary for clarity in the training loop.
    """
    def __init__(self, data_entries, tokenizer, transform, max_length):
        self.data_entries = data_entries
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        image_path, caption = entry['image_path'], entry['caption']

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.warning(f"Corrupt image at {image_path}: {e}. Using a black image.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        image = self.transform(image)
        
        # Tokenize text using the Hugging Face tokenizer
        tokenized_output = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': tokenized_output['input_ids'].squeeze(),
            'attention_mask': tokenized_output['attention_mask'].squeeze()
        }

def create_data_transforms(image_size: int, is_training: bool):
    """Create image preprocessing transforms."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# --- Training and Validation Loops ---

def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device, config):
    """
    ### REFACTOR NOTE ###
    - Modernized with `autocast` and `GradScaler` for mixed-precision training.
    - Simplified logic, removed manual safety checks now handled by AMP.
    - Uses in-batch negative sampling via the contrastive loss.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()
        
        images = batch['image'].to(device, non_blocking=True)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        
        with autocast(enabled=config.use_amp):
            # We'll use the contrastive task for pre-training.
            # This task implicitly learns similarity.
            vision_proj, text_proj = model(images, input_ids, attention_mask, task='contrastive')
            loss, _ = criterion(outputs=(vision_proj, text_proj), labels=None, task='contrastive')

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """
    ### REFACTOR NOTE ###
    - Now calculates and returns key classification metrics, not just loss.
    - This provides a much clearer picture of model performance.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            # Create positive and negative pairs for validation
            # For simplicity, we create pairs within the batch
            B = images.shape[0]
            if B < 2: continue # Need at least 2 samples to create a negative pair

            # Get embeddings
            with autocast():
                vision_features, text_features = model(
                    images, 
                    batch['input_ids'].to(device), 
                    batch['attention_mask'].to(device),
                    task='retrieval' # Get base features
                )
            
            # Normalize for cosine similarity
            vision_features = F.normalize(vision_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)
            
            # Calculate all-to-all similarity
            sim_matrix = vision_features @ text_features.T
            
            # Positive pairs are on the diagonal
            pos_sim = torch.diag(sim_matrix)
            # Negative pairs are off-diagonal
            neg_sim = sim_matrix[~torch.eye(B, dtype=bool)].view(B, B - 1)
            
            # We can form a simple binary classification task here
            # Is the correct caption more similar than a random incorrect one?
            random_neg_sim = neg_sim[torch.arange(B), torch.randint(0, B - 1, (B,))]
            
            # Predictions: 1 if positive is more similar, 0 otherwise
            preds = (pos_sim > random_neg_sim).float()
            labels = torch.ones_like(preds) # The goal is always for positive > negative

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate a representative loss (optional but good for tracking)
            # A simple loss could be max(0, margin - (pos_sim - neg_sim))
            margin = 0.2
            loss = torch.mean(torch.clamp(margin - pos_sim + random_neg_sim, min=0))
            total_loss += loss.item()

    if not all_labels:
        return {'val_loss': float('inf'), 'val_accuracy': 0, 'val_precision': 0, 'val_recall': 0, 'val_f1': 0}

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    return {
        'val_loss': total_loss / len(dataloader),
        'val_accuracy': accuracy,
        'val_precision': precision,
        'val_recall': recall,
        'val_f1': f1
    }

# --- Main Execution ---

def main():
    config = TrainingConfig()
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("🚀 Starting Foundation Model Training")
    logger.info(f"Config: {json.dumps(asdict(config), indent=2)}")
    
    # --- Data Loading ---
    ### REFACTOR NOTE ###
    # Simplified data loading. The dataset now only contains positive pairs.
    # Negative pairs are handled "in-batch" by the contrastive loss function.
    try:
        df = pd.read_csv(os.path.join(config.dataset_root, 'results.csv'), delimiter='|')
        df.columns = [col.strip() for col in df.columns]
        all_entries = []
        for _, row in df.iterrows():
            img_path = os.path.join(config.dataset_root, 'flickr30k_images', row['image_name'])
            if os.path.exists(img_path) and isinstance(row['comment'], str) and len(row['comment']) > 5:
                all_entries.append({'image_path': img_path, 'caption': row['comment']})
        logger.info(f"Loaded {len(all_entries)} valid image-caption pairs.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}"); return
        
    random.shuffle(all_entries)
    train_size = int(0.9 * len(all_entries))
    train_entries = all_entries[:train_size]
    val_entries = all_entries[train_size:]
    
    # --- Model, Tokenizer, and Transforms ---
    model, tokenizer = create_labelee_foundation(asdict(config))
    model.to(config.device)
    
    train_transform = create_data_transforms(224, is_training=True)
    val_transform = create_data_transforms(224, is_training=False)
    
    train_dataset = VisionLanguageDataset(train_entries, tokenizer, train_transform, 128)
    val_dataset = VisionLanguageDataset(val_entries, tokenizer, val_transform, 128)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    
    # --- Optimizer, Scheduler, Criterion, and Scaler ---
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.num_epochs)
    criterion = MultiTaskLoss(gamma=1.0) # We are focusing on contrastive loss for pre-training
    scaler = GradScaler(enabled=config.use_amp)
    
    # --- Training Loop ---
    epoch_manager = EpochManager(config.checkpoint_dir, config.patience)
    start_epoch = 0
    if config.resume_training:
        start_epoch = epoch_manager.load_checkpoint(model, optimizer, scheduler)

    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{config.num_epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, config.device, config)
        scheduler.step()
        
        val_metrics = validate(model, val_loader, criterion, config.device)
        
        epoch_summary = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **val_metrics # Unpack validation metrics into the summary
        }
        
        logger.info(f"Epoch {epoch+1} Summary: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val F1: {val_metrics['val_f1']:.4f}")
        
        is_best = epoch_manager.end_epoch(epoch_summary)
        epoch_manager.save_checkpoint(model, optimizer, scheduler, epoch + 1, is_best)
        
        if epoch_manager.should_stop:
            break

    logger.info("🏁 Training finished.")
    epoch_manager.plot_training_curves()

if __name__ == "__main__":
    main()