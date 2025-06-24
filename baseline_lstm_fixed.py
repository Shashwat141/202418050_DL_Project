# baseline_lstm_fixed.py - Corrected LSTM baseline
import torch, json, os
import gzip
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np

class FixedLSTMDataset(Dataset):
    def __init__(self, data_file, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        self.reviews = self._load_data(data_file)
    
    def _load_data(self, path):
        reviews = []
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000:  # Limit to 50K samples
                    break
                obj = json.loads(line)
                if obj.get("reviewText"):
                    reviews.append((obj["reviewText"], int(obj["overall"])))
        return reviews
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        text, rating = self.reviews[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(rating-1, dtype=torch.long)
        }

class FixedLSTMModel(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=128, hidden_dim=256, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        return self.classifier(lstm_out[:, -1, :])

def train_fixed_lstm(data_file, epochs=5, batch_size=32, save_dir="models/lstm_baseline"):
    dataset = FixedLSTMDataset(data_file)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FixedLSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Save model and results...

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", default="models/lstm_baseline")
    args = parser.parse_args()
    train_fixed_lstm(args.data_file, args.epochs, args.batch_size, args.save_dir)
