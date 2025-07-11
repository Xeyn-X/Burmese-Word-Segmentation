import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=64, hidden_dim=128, pad_idx=4):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
        self.pad_idx = pad_idx

    def forward(self, x, tags=None, mask=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        emissions = self.fc(x)
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emissions, mask=mask)
            return prediction