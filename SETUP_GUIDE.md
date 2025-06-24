# Labelee Foundation Model - Setup Guide

This guide will help you set up and run the Labelee Foundation model, addressing all common issues.

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required packages:

```bash
# Install PyTorch and other dependencies
pip install torch torchvision transformers timm wandb tqdm ml-collections numpy Pillow

# Or install from requirements.txt
pip install -r requirements.txt
```

### 2. Fix Common Issues

#### A. LibJPEG Issue (macOS)
If you encounter libjpeg errors:
```bash
conda install -c conda-forge libjpeg-turbo
```

#### B. HuggingFace Tokenizer Warning
This is automatically fixed in our scripts with:
```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

#### C. W&B Configuration
Edit `configs/base_config.py` and set your W&B username:
```python
config.wandb.entity = "your-wandb-username"  # Replace with your username
```

### 3. Test Your Setup

Run the quick test to verify everything works:
```bash
python3 scripts/quick_test.py
```

### 4. Start Training

#### Option A: Simple Training (Recommended for testing)
```bash
python3 scripts/simple_train.py
```

#### Option B: Full Training with Config
```bash
python3 src/train.py
```

## 🔧 Detailed Setup

### Environment Setup

1. **Create a virtual environment** (recommended):
```bash
conda create -n labelee-env python=3.9
conda activate labelee-env
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import timm; print(f'TIMM: {timm.__version__}')"
```

### Configuration

The project uses two configuration approaches:

#### 1. Simple Configuration (scripts/simple_train.py)
- Hardcoded values for quick testing
- No external dependencies
- Easy to modify

#### 2. Advanced Configuration (src/train.py)
- Uses `ml_collections` for structured config
- More flexible and maintainable
- Supports W&B integration

### W&B Setup (Optional)

1. **Install W&B**:
```bash
pip install wandb
```

2. **Login to W&B**:
```bash
wandb login
```

3. **Update configuration**:
Edit `configs/base_config.py`:
```python
config.wandb.entity = "your-wandb-username"
```

## 🧪 Testing

### Quick Test
```bash
python3 scripts/quick_test.py
```

This tests:
- ✅ Model imports
- ✅ Data loader creation
- ✅ Forward pass for all tasks
- ✅ Loss computation

### Full Test Suite
```bash
python3 scripts/test_setup.py
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch torchvision
```

#### 2. "libjpeg.9.dylib not found" (macOS)
**Solution**: Install libjpeg-turbo
```bash
conda install -c conda-forge libjpeg-turbo
```

#### 3. "W&B initialization failed"
**Solution**: 
- Set your W&B username in `configs/base_config.py`
- Or run without W&B using `scripts/simple_train.py`

#### 4. "Zero-element tensor" warnings
**Solution**: These are usually harmless but indicate potential issues. The model includes fallbacks to handle them.

#### 5. "HuggingFace tokenizer" warnings
**Solution**: Already fixed in our scripts with the environment variable.

### Model Architecture Issues

If you encounter model-specific errors:

1. **Check tensor shapes**: The model includes extensive error handling
2. **Verify input format**: Images should be [B, 3, 224, 224], text should be tokenized
3. **Check device placement**: Ensure tensors are on the same device

### Performance Issues

1. **Reduce batch size**: Start with batch_size=4 or 8
2. **Use smaller models**: Change vision_model_name to a smaller variant
3. **Enable mixed precision**: Add `torch.cuda.amp.autocast()` for GPU training

## 📊 Monitoring Training

### Without W&B
Training progress is printed to console with tqdm progress bars.

### With W&B
1. Visit your W&B dashboard
2. Monitor loss curves, validation metrics
3. Track hyperparameters and model artifacts

## 🔄 Next Steps

After successful setup:

1. **Replace dummy data**: Modify `src/data_loader.py` with your real dataset
2. **Adjust hyperparameters**: Edit configuration files
3. **Scale up training**: Increase batch size, epochs, model size
4. **Add custom tasks**: Extend the model for your specific use case

## 📝 File Structure

```
labelee-project/
├── src/
│   ├── model.py              # Main model implementation
│   ├── data_loader.py        # Data loading utilities
│   └── train.py              # Full training script
├── scripts/
│   ├── simple_train.py       # Simplified training (recommended)
│   ├── quick_test.py         # Quick functionality test
│   └── test_setup.py         # Full test suite
├── configs/
│   └── base_config.py        # Configuration management
├── requirements.txt          # Dependencies
└── SETUP_GUIDE.md           # This file
```

## 🆘 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Run the test scripts** to isolate the problem
3. **Check the console output** for specific error messages
4. **Verify your environment** with the quick test

## 🎯 Success Criteria

Your setup is working correctly when:

- ✅ `python3 scripts/quick_test.py` passes all tests
- ✅ `python3 scripts/simple_train.py` runs without errors
- ✅ Training loss decreases over epochs
- ✅ No critical warnings or errors in console

---

**Happy Training! 🚀** 