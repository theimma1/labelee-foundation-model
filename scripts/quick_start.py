#!/usr/bin/env python3
"""
Quick start script for the Labelee Foundation project.
This script will help you set up the environment and run a basic test.
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 60)
    print("🚀 Labelee Foundation Model - Quick Start")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'torchvision', 'transformers', 'timm', 
        'wandb', 'tqdm', 'ml_collections', 'numpy', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_conda_env():
    """Create a conda environment."""
    print("\n🐍 Creating conda environment...")
    
    env_name = "labelee-env"
    
    try:
        # Check if conda is available
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
        
        # Create environment
        subprocess.check_call([
            "conda", "create", "-n", env_name, "python=3.9", "-y"
        ])
        
        print(f"✅ Conda environment '{env_name}' created!")
        print(f"\nTo activate the environment, run:")
        print(f"   conda activate {env_name}")
        print(f"   pip install -r requirements.txt")
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Conda not found or failed to create environment")
        print("   You can install dependencies manually with:")
        print("   pip install -r requirements.txt")
        return False

def test_basic_imports():
    """Test basic imports after installation."""
    print("\n🧪 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        import timm
        print(f"✅ TIMM version: {timm.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def show_next_steps():
    """Show next steps for the user."""
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("=" * 60)
    
    print("1. 📝 Update W&B configuration:")
    print("   Edit configs/base_config.py and change 'your-wandb-username'")
    print("   to your actual Weights & Biases username")
    
    print("\n2. 🧪 Run the test suite:")
    print("   python scripts/test_setup.py")
    
    print("\n3. 🚀 Start training:")
    print("   python src/train.py")
    
    print("\n4. 📊 Monitor training:")
    print("   Visit your W&B dashboard to see training progress")
    
    print("\n5. 📚 Read the documentation:")
    print("   Check README.md for detailed usage instructions")
    
    print("\n" + "=" * 60)
    print("🎉 You're all set! Happy training!")
    print("=" * 60)

def main():
    """Main function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        
        # Ask user if they want to install
        response = input("\nWould you like to install the missing dependencies? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            # Try to create conda environment first
            if create_conda_env():
                print("\nPlease activate the conda environment and run this script again:")
                print("   conda activate labelee-env")
                print("   python scripts/quick_start.py")
                return
            
            # Fallback to pip install
            if install_dependencies():
                # Test imports again
                if test_basic_imports():
                    show_next_steps()
                else:
                    print("❌ Installation completed but import test failed")
            else:
                print("❌ Failed to install dependencies")
        else:
            print("❌ Please install dependencies manually:")
            print("   pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are installed!")
        show_next_steps()

if __name__ == "__main__":
    main() 