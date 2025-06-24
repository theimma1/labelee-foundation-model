from Labelee import Labelee
from labelee_training import TrainingConfig, create_optimizer, EpochManager
import torch

# Initialize config
config = TrainingConfig()

# Load model and checkpoint
model = Labelee(vocab_size=config.vocab_size, feature_dim=config.feature_dim).to(config.device)
optimizer = create_optimizer(model, config)  # Placeholder optimizer
epoch_manager = EpochManager(checkpoint_dir=config.checkpoint_dir, max_epochs=config.num_epochs)
epoch_manager.load_checkpoint(model, optimizer, load_best=True)

# Print epoch to verify
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

# Test TorchScript conversion
dummy_input = (
    torch.randn(1, 3, config.image_size, config.image_size).to(config.device),  # Images
    torch.randint(0, config.vocab_size, (1, config.max_length)).to(config.device),  # Input IDs
    torch.ones(1, config.max_length).to(config.device)  # Attention mask
)
scripted_model = torch.jit.trace(model, dummy_input, strict=False)
scripted_model.save('deployed_labelee.pt')
print("Model exported to deployed_labelee.pt")