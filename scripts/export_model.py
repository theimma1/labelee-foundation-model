from Labelee import Labelee, EnhancedTokenizer
from labelee_training import TrainingConfig, create_optimizer, EpochManager
import torch
import pandas as pd
import os

config = TrainingConfig()
model = Labelee(vocab_size=config.vocab_size, feature_dim=config.feature_dim).to(config.device)
optimizer = create_optimizer(model, config)
epoch_manager = EpochManager(checkpoint_dir=config.checkpoint_dir, max_epochs=config.num_epochs)

checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

my_dataset_root = "/Users/immanuelolajuyigbe/Downloads/my_downloaded_data"
captions_csv_file = os.path.join(my_dataset_root, 'results.csv')
df = pd.read_csv(captions_csv_file, delimiter='|')
df.columns = df.columns.str.strip()
df = df[['image_name', 'comment']]
df.rename(columns={'comment': 'caption'}, inplace=True)
df['caption'].fillna('', inplace=True)
df['caption'] = df['caption'].astype(str)
all_captions = df['caption'].tolist()

tokenizer = EnhancedTokenizer(vocab_size=config.vocab_size)
tokenizer.build_vocab(all_captions)
vocab_size = len(tokenizer.word2idx)
print(f"Rebuilt vocabulary size: {vocab_size}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    saved_vocab_size = state_dict['text_encoder.word_encoder.word_embedding.weight'].shape[0]
    if saved_vocab_size != vocab_size:
        print(f"Adjusting vocabulary size from {vocab_size} to {saved_vocab_size}")
        model.text_encoder.word_encoder.word_embedding = torch.nn.Embedding(saved_vocab_size, config.feature_dim, padding_idx=0)
        model.text_encoder.word_encoder.word_embedding.weight.data = state_dict['text_encoder.word_encoder.word_embedding.weight']
        model.text_encoder.word_encoder.word_embedding.weight.requires_grad = False
        config.vocab_size = saved_vocab_size
    model.load_state_dict(state_dict, strict=False)

dummy_input = (
    torch.randn(1, 3, config.image_size, config.image_size).to(config.device),
    torch.randint(0, config.vocab_size, (1, config.max_length)).to(config.device),
    torch.ones(1, config.max_length).to(config.device)
)
scripted_model = torch.jit.trace(model, dummy_input, strict=False)
scripted_model.save('deployed_labelee.pt')
print("Model exported to deployed_labelee.pt")