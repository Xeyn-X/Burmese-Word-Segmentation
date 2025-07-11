import re
import pickle
import torch
import numpy as np

from models.bilstm_crf import BiLSTM_CRF  # Use CRF-based model

class BiLSTMCRFPredict:
    def __init__(self, model_path="weight/word_seg_bilstm_crf.pth", config_path="data_config/data_config_bilstm_crf.pkl"):
        with open(config_path, "rb") as f:
            resources = pickle.load(f)

        self.max_len = resources["max_len"]
        self.syll2idx = resources["syll2idx"]
        self.idx2tag = resources["idx2tag"]
        self.tag2idx = resources["tag2idx"]

        vocab_size = len(self.syll2idx) + 1
        tagset_size = len(self.tag2idx)
        self.model = BiLSTM_CRF(vocab_size, tagset_size, pad_idx=self.tag2idx["PAD"])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def _segment_syllables(self, text):
        pattern = r'(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[^က-၏]+)(?![ှျ]?[့္်]))'
        text = re.sub(pattern, r'|\1', text)
        return [syl for syl in re.split(r'\|', text) if syl]

    def segment_text(self, text):
        text = text.replace(' ', '')
        syll_seq = self._segment_syllables(text)
        x_seq = [self.syll2idx.get(syl, 0) for syl in syll_seq]
        x_seq = x_seq[:self.max_len]  # truncate if needed
        x_padded = x_seq + [0] * (self.max_len - len(x_seq))  # manual padding
        x_tensor = torch.LongTensor([x_padded])
        mask = torch.tensor([[1 if i < len(x_seq) else 0 for i in range(self.max_len)]], dtype=torch.uint8)

        with torch.no_grad():
            pred_tags = self.model(x_tensor, mask=mask)[0]  # decoded output

        result, word = [], ""
        for i, syll in enumerate(syll_seq):
            word += syll
            if i >= len(pred_tags):
                continue
            if self.idx2tag[pred_tags[i]] in ["e", "s"]:
                result.append(word)
                word = ""
        if word:
            result.append(word)

        return " ".join(result)