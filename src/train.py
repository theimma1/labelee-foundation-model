import torch
import wandb
from tqdm import tqdm
import os

from src.model import create_labelee_foundation, MultiTaskLoss, train_step
from src.data_loader import create_dataloaders
from configs import base_config

def main():
    config = base_config.get_config()
    
    try:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config 
        )
    except Exception as e:
        print(f"W&B initialization failed: {e}\nContinuing without W&B logging...")
        wandb.init(mode="disabled")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, tokenizer = create_labelee_foundation(config['model'])
    model.to(device)
    print("Model and tokenizer created successfully")
    
    train_loader, val_loader = create_dataloaders(
        tokenizer, 
        batch_size=config['train']['batch_size']
    )
    print(f"DataLoaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['learning_rate'])
    criterion = MultiTaskLoss(
        alpha=config['loss']['alpha'],
        beta=config['loss']['beta'],
        gamma=config['loss']['gamma'],
        delta=config['loss']['delta']
    )

    print("Starting training...")
    for epoch in range(config['train']['epochs']):
        model.train()
        train_loss_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for batch in progress_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            loss, loss_dict = train_step(
                model, optimizer, criterion, 
                images, input_ids, attention_mask, labels, 
                task=config['train']['task']
            )
            
            if loss > 0:
                train_loss_total += loss
                wandb.log({"train_loss_step": loss, **loss_dict})
            
            progress_bar.set_postfix(loss=loss)
        
        avg_train_loss = train_loss_total / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, input_ids, attention_mask, task=config['train']['task'])
                loss, loss_dict = criterion(outputs, labels, task=config['train']['task'])
                
                if loss > 0:
                    val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss
        })

    print("Training finished.")
    wandb.finish()

if __name__ == '__main__':
    main()