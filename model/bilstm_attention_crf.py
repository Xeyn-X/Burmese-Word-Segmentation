import torch
import torch.nn as nn
from torch_crf import CRF

class BiLSTMAttentionCRFModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=300, hidden_dim=256):
        super(BiLSTMAttentionCRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.attention = nn.Linear(hidden_dim, 1)
        self.hidden2tag = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, tokens, tags=None, mask=None):
        embeds = self.embedding(tokens)
        lstm_out, _ = self.lstm(embeds)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = lstm_out * attn_weights

        hid = self.hidden2tag(attended)
        drop = self.dropout(hid)
        emissions = self.fc(drop)
        
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return pred