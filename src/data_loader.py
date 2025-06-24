import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DummyMultiModalDataset(Dataset):
    """
    A dummy dataset that generates random multimodal data.
    Replace this with your actual dataset loading logic.
    """
    def __init__(self, tokenizer, num_samples=1000, max_text_len=64, image_size=224):
        self.num_samples = num_samples
        self.max_text_len = max_text_len
        self.image_size = image_size
        self.tokenizer = tokenizer

        # Define a simple image transformation
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Generate a dummy image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # 2. Generate dummy text and tokenize it
        dummy_text = "this is a sample sentence for the model"
        tokenized_output = self.tokenizer(
            dummy_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors='pt'
        )
        input_ids = tokenized_output['input_ids'].squeeze(0)
        attention_mask = tokenized_output['attention_mask'].squeeze(0)
        
        # 3. Generate a dummy label (e.g., for similarity or classification)
        # For similarity, 0 = dissimilar, 1 = similar
        # For classification, an integer class index
        label = torch.randint(0, 2, (1,)).float() # Example for similarity

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }

def create_dataloaders(tokenizer, batch_size, num_workers=4):
    """
    Creates training and validation dataloaders with the dummy dataset.
    """
    train_dataset = DummyMultiModalDataset(tokenizer, num_samples=1000)
    val_dataset = DummyMultiModalDataset(tokenizer, num_samples=200)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 