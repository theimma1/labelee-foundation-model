#!/usr/bin/env python3
"""
Test script to verify that all components of the Labelee Foundation project work correctly.
"""

import sys
import os
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.model import create_labelee_foundation, MultiTaskLoss
        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from src.data_loader import create_dataloaders
        print("✓ Data loader imports successful")
    except Exception as e:
        print(f"✗ Data loader imports failed: {e}")
        return False
    
    try:
        from configs import base_config
        print("✓ Config imports successful")
    except Exception as e:
        print(f"✗ Config imports failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test that the model can be created successfully."""
    print("\nTesting model creation...")
    
    try:
        from src.model import create_labelee_foundation
        
        # Test with default config
        model, tokenizer = create_labelee_foundation()
        print("✓ Model creation successful")
        
        # Test with custom config
        custom_config = {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 100
        }
        model2, tokenizer2 = create_labelee_foundation(custom_config)
        print("✓ Custom model creation successful")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_loader():
    """Test that data loaders can be created."""
    print("\nTesting data loader creation...")
    
    try:
        from src.model import create_labelee_foundation
        from src.data_loader import create_dataloaders
        
        model, tokenizer = create_labelee_foundation()
        train_loader, val_loader = create_dataloaders(tokenizer, batch_size=4)
        
        print(f"✓ Data loaders created - Train: {len(train_loader)}, Val: {len(val_loader)}")
        
        # Test getting a batch
        batch = next(iter(train_loader))
        expected_keys = ['image', 'input_ids', 'attention_mask', 'label']
        if all(key in batch for key in expected_keys):
            print("✓ Batch structure correct")
        else:
            print("✗ Batch structure incorrect")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Data loader creation failed: {e}")
        return False

def test_forward_pass():
    """Test that the model can perform a forward pass."""
    print("\nTesting forward pass...")
    
    try:
        from src.model import create_labelee_foundation
        from src.data_loader import create_dataloaders
        
        # Create model and data
        model, tokenizer = create_labelee_foundation()
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
                print(f"✓ {task} forward pass successful")
            except Exception as e:
                print(f"✗ {task} forward pass failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        return False

def test_loss_computation():
    """Test that loss computation works."""
    print("\nTesting loss computation...")
    
    try:
        from src.model import create_labelee_foundation, MultiTaskLoss
        from src.data_loader import create_dataloaders
        
        # Create model, data, and loss
        model, tokenizer = create_labelee_foundation()
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
                    print(f"✓ {task} loss computation successful: {loss.item():.4f}")
                else:
                    print(f"✗ {task} loss is not finite")
                    return False
                    
            except Exception as e:
                print(f"✗ {task} loss computation failed: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Loss computation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Labelee Foundation Project - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
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
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Update configs/base_config.py with your W&B username")
        print("2. Run: python src/train.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 