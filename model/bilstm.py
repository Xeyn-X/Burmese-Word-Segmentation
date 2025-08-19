import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=300, hidden_dim=256):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, tagset_size)

    def forward(self, tokens):
        embeds = self.embedding(tokens)        # (B, T, E)
        lstm_out, _ = self.lstm(embeds)        # (B, T, H)
        full1 = F.relu(self.fc1(lstm_out))
        drop = self.dropout(full1)
        logits = self.fc2(drop)
        return logits
