import torch
import torch.nn as nn
from torch_crf import CRF
import torch.nn.functional as F


class BiLSTMCRFModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=300, hidden_dim=256):
        super(BiLSTMCRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, tokens, tags=None, mask=None):
        embeds = self.embedding(tokens)
        lstm_out, _ = self.lstm(embeds)
        full1 = F.relu(self.fc1(lstm_out))
        drop = self.dropout(full1)
        emissions = self.fc2(drop)

        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
            return loss
        else:
            pred = self.crf.decode(emissions, mask=mask)
            return pred
