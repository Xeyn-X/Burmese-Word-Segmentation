import torch
import torch.nn as nn

class BiLSTMAttentionTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=300, hidden_dim=256):
        super(BiLSTMAttentionTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.hidden2tag = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, tagset_size)

    def forward(self, tokens):
        embeds = self.embedding(tokens)        # (B, T, E)
        lstm_out, _ = self.lstm(embeds)        # (B, T, H)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (B, T, 1)
        attended = lstm_out * attn_weights     # (B, T, H)
        hid = self.hidden2tag(attended)     # (B, T, C)
        drop = self.dropout(hid)
        logits = self.fc(drop)
        return logits
