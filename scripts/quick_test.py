#!/usr/bin/env python3
"""
Quick test script to verify the Labelee Foundation model works.
This bypasses the config issues and tests the core functionality.
"""

import os
import sys
import torch

# Fix for HuggingFace tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.model import create_labelee_foundation, MultiTaskLoss
        print("✅ Model imports successful")
        return True
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False

def test_model_creation():
    """Test model creation with simple config."""
    print("\nTesting model creation...")
    
    try:
        from src.model import create_labelee_foundation
        
        # Simple config without ml_collections
        config = {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 100
        }
        
        model, tokenizer = create_labelee_foundation(config)
        print("✅ Model creation successful")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None, None

def test_data_loader():
    """Test data loader creation."""
    print("\nTesting data loader...")
    
    try:
        from src.data_loader import create_dataloaders
        
        # Get tokenizer from model creation
        model, tokenizer = test_model_creation()
        if tokenizer is None:
            return False
            
        train_loader, val_loader = create_dataloaders(tokenizer, batch_size=4)
        print(f"✅ Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        # Test getting a batch
        batch = next(iter(train_loader))
        expected_keys = ['image', 'input_ids', 'attention_mask', 'label']
        if all(key in batch for key in expected_keys):
            print("✅ Batch structure correct")
            return True
        else:
            print("❌ Batch structure incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\nTesting forward pass...")
    
    try:
        from src.model import create_labelee_foundation
        from src.data_loader import create_dataloaders
        
        # Create model and data
        config = {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 100
        }
        
        model, tokenizer = create_labelee_foundation(config)
        train_loader, _ = create_dataloaders(tokenizer, batch_size=2)
        batch = next(iter(train_loader))
        
        # Move to device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
        
        # Test different tasks
        tasks = ['similarity', 'contrastive', 'reconstruction', 'classification']
        
        for task in tasks:
            try:
                with torch.no_grad():
                    outputs = model(
                        batch['image'].to(device),
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        task=task
                    )
                print(f"✅ {task} forward pass successful")
            except Exception as e:
                print(f"❌ {task} forward pass failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False

def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    try:
        from src.model import create_labelee_foundation, MultiTaskLoss
        from src.data_loader import create_dataloaders
        
        # Create model, data, and loss
        config = {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 100
        }
        
        model, tokenizer = create_labelee_foundation(config)
        train_loader, _ = create_dataloaders(tokenizer, batch_size=2)
        criterion = MultiTaskLoss()
        batch = next(iter(train_loader))
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)
        
        # Test loss computation for different tasks
        tasks = ['similarity', 'contrastive', 'reconstruction', 'classification']
        
        for task in tasks:
            try:
                with torch.no_grad():
                    outputs = model(
                        batch['image'].to(device),
                        batch['input_ids'].to(device),
                        batch['attention_mask'].to(device),
                        task=task
                    )
                    loss, loss_dict = criterion(outputs, batch['label'].to(device), task=task)
                
                if torch.isfinite(loss):
                    print(f"✅ {task} loss computation successful: {loss.item():.4f}")
                else:
                    print(f"❌ {task} loss is not finite")
                    return False
                    
            except Exception as e:
                print(f"❌ {task} loss computation failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Loss computation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("🚀 Labelee Foundation - Quick Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_data_loader,
        test_forward_pass,
        test_loss_computation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\nYou can now:")
        print("1. Run training: python src/train.py")
        print("2. Or fix the config issues in train.py if needed")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 