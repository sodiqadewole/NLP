import torch
from torch.utils.data import Dataset, DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    

from transformers import AutoTokenizer

# Example data (replace with your actual data)
sample_texts = ["This is a positive review.", "This movie was terrible."]
sample_labels = [1, 0]

# Initialize tokenizer (e.g., BERT tokenizer)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_sequence_length = 64

# Create dataset instance
custom_dataset = CustomTextDataset(
    texts=sample_texts,
    labels=sample_labels,
    tokenizer=tokenizer,
    max_len=max_sequence_length
)

# Create DataLoader for batching and shuffling
data_loader = DataLoader(
    custom_dataset,
    batch_size=2,  # Adjust batch size as needed
    shuffle=True
)

# Iterate through the DataLoader
for batch in data_loader:
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    break