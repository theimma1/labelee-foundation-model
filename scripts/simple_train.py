#!/usr/bin/env python3
"""
Simplified training script for Labelee Foundation model.
This bypasses config issues and uses hardcoded values for testing.
"""

import os
import sys
import torch
import wandb
from tqdm import tqdm

# Fix for HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_labelee_foundation, MultiTaskLoss, train_step
from src.data_loader import create_dataloaders

def main():
    """Main training function with hardcoded configuration."""
    
    # --- 1. Hardcoded Configuration ---
    config = {
        'model': {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 100
        },
        'train': {
            'epochs': 5,  # Reduced for testing
            'batch_size': 8,  # Reduced for testing
            'learning_rate': 1e-4,
            'task': 'similarity'
        },
        'loss': {
            'alpha': 1.0,  # similarity
            'beta': 0.5,   # reconstruction
            'gamma': 0.7,  # contrastive
            'delta': 0.3   # classification
        },
        'wandb': {
            'project': 'labelee-foundation-model',
            'entity': None  # Set to your username if you have W&B
        }
    }
    
    # --- 2. Initialize W&B (Optional) ---
    wandb_available = False
    try:
        if config['wandb']['entity']:
            wandb.init(
                project=config['wandb']['project'],
                entity=config['wandb']['entity'],
                config=config
            )
            print("W&B initialized successfully")
            wandb_available = True
        else:
            print("W&B entity not set, skipping W&B logging")
    except Exception as e:
        print(f"W&B initialization failed: {e}")
        print("Continuing without W&B logging...")
    
    # --- 3. Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # --- 4. Create Model and Tokenizer ---
    try:
        model, tokenizer = create_labelee_foundation(config['model'])
        model.to(device)
        print("Model and tokenizer created successfully")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # --- 5. Create DataLoaders ---
    try:
        train_loader, val_loader = create_dataloaders(
            tokenizer, 
            batch_size=config['train']['batch_size']
        )
        print(f"DataLoaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return

    # --- 6. Setup Optimizer and Loss ---
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['train']['learning_rate']
    )
    
    criterion = MultiTaskLoss(
        alpha=config['loss']['alpha'],
        beta=config['loss']['beta'],
        gamma=config['loss']['gamma'],
        delta=config['loss']['delta']
    )

    # --- 7. Training Loop ---
    epochs = config['train']['epochs']
    task = config['train']['task']
    
    print(f"Starting training for {epochs} epochs with task: {task}")
    
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0
        num_batches = 0
        
        # Use tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            try:
                # Move batch to device
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                loss, loss_dict = train_step(
                    model, optimizer, criterion, 
                    images, input_ids, attention_mask, labels, 
                    task=task
                )
                
                if loss > 0:
                    train_loss_total += loss
                    num_batches += 1
                    
                    # Log to W&B if available
                    if wandb_available:
                        wandb.log({"train_loss_step": loss, **loss_dict})
                
                # Update progress bar
                progress_bar.set_postfix(loss=f"{loss:.4f}")
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        avg_train_loss = train_loss_total / max(num_batches, 1)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # --- 8. Validation Step ---
        model.eval()
        val_loss_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    images = batch['image'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(images, input_ids, attention_mask, task=task)
                    loss, loss_dict = criterion(outputs, labels, task=task)
                    
                    if loss > 0:
                        val_loss_total += loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_val_loss = val_loss_total / max(val_batches, 1)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        # Log epoch-level metrics to W&B
        if wandb_available:
            wandb.log({
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss
            })

    print("Training finished!")
    if wandb_available:
        wandb.finish()

if __name__ == '__main__':
    main() 