import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ================
# Model Definition
# ================
class ToxicityDataset(Dataset):
    def __init__(self, tokens, labels, comment_nature_features, weights):
        self.tokens = tokens
        self.labels = labels
        self.comment_nature_features = comment_nature_features
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokens.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        item['comment_nature_features'] = torch.tensor(self.comment_nature_features[idx], dtype=torch.float)
        item['weights'] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get token lengths, capping at 512
tokenized_lengths = [min(len(tokenizer.encode(text, truncation=False)), 512) for text in talk_df['text']]

# Sort and find the 90% cutoff
sorted_lengths = np.sort(tokenized_lengths)
percentile = 87.5
cutoff_index = int(len(sorted_lengths) * (percentile / 100))
cutoff_length = sorted_lengths[cutoff_index]

# Tokenize the 'text' field for all splits
def tokenize_data(data, tokenizer,  max_length=128):
    texts = data['text'].astype(str).tolist()
    return tokenizer(
        list(data['text']),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def prepare_data(data, tokenizer, additional_features, max_length = 135):
  tokens = tokenize_data(data, tokenizer, max_length)
  labels = data['Toxicity'].values
  features = data[additional_features].values
  weights = data['weights'].values
  return ToxicityDataset(tokens, labels, features, weights)


# Split into train, validation, and test based on 'split' column
train_df = talk_df[talk_df['split'] == 'train']
val_df = talk_df[talk_df['split'] == 'val']
test_df = talk_df[talk_df['split'] == 'test']

# Prepare datasets
train_dataset = prepare_data(train_df, tokenizer, comment_nature_columns)
val_dataset = prepare_data(val_df, tokenizer, comment_nature_columns)
test_dataset = prepare_data(test_df, tokenizer, comment_nature_columns)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)