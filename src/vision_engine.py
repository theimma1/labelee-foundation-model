import torch
import os
from .model import create_labelee_foundation
from configs import base_config

class VisionEngine:
    def __init__(self, model_checkpoint_path: str):
        print("Initializing Vision Engine...")
        self.config = base_config.get_config()

        # Set up device (with MPS support)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Vision Engine using device: {self.device}")

        # Load the model structure from the config dictionary
        self.model, self.tokenizer = create_labelee_foundation(self.config['model'])

        # Load the trained weights
        if not os.path.exists(model_checkpoint_path):
            print(f"Warning: Checkpoint file not found at {model_checkpoint_path}. The model is using initial weights.")
        else:
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location=self.device))
            print(f"Loaded model weights from {model_checkpoint_path}")
            
        self.model.to(self.device)
        self.model.eval()
        print("Vision Engine ready.")

    @torch.no_grad()
    def describe_image(self, image: torch.Tensor) -> str:
        """
        Placeholder for image-to-text capability.
        For now, analyzes the image and returns feature info.
        """
        features = self.model(images=image.unsqueeze(0).to(self.device), return_features=True)
        # In the future, this would feed into a captioning head.
        return f"Image processed. Vision feature dimension: {features['vision_features'].shape}"

    @torch.no_grad()
    def get_similarity(self, image: torch.Tensor, text: str) -> float:
        """Compares an image and text and returns a similarity score."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        image = image.unsqueeze(0).to(self.device)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        score = self.model(images=image, input_ids=input_ids, attention_mask=attention_mask, task='similarity')
        return torch.sigmoid(score).item() 