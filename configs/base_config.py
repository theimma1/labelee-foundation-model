import ml_collections

def get_config():
    """Returns the base configuration for the entire project."""
    config = {
        # NEW: Added a dedicated section for agent settings
        'agent': {
            'llm_model_name': 'llama3', # The model the agent will use
            'max_steps': 5
        },
        'model': {
            'vision_model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
            'text_model_name': 'distilbert-base-uncased',
            'feature_dim': 768,
            'num_classes': 1000
        },
        'train': {
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'task': 'similarity',
            'model_checkpoint_path': 'models/phoenix_v1.pth' 
        },
        'loss': {
            'alpha': 1.0,  # similarity
            'beta': 0.5,   # reconstruction
            'gamma': 0.7,  # contrastive
            'delta': 0.3   # classification
        },
        'wandb': {
            'project': "labelee-foundation-model",
            'entity': "olliez30"
        }
    }
    return config