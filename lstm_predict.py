import re
import pickle
import torch
import torch.nn.functional as F

from models.lstm import LSTM


class LSTMPredict:
    def __init__(self, model_path="weight/word_seg_lstm.pth", config_path="data_config/data_config_lstm.pkl"):
        # Load model configuration
        with open(config_path, "rb") as f:
            resources = pickle.load(f)

        self.max_len = resources["max_len"]
        self.syll2idx = resources["syll2idx"]
        self.idx2tag = resources["idx2tag"]
        self.tag2idx = resources["tag2idx"]

        # Reconstruct and load model
        vocab_size = len(self.syll2idx) + 1
        tagset_size = len(self.tag2idx)
        self.model = LSTM(vocab_size, tagset_size)
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
        x_seq += [0] * (self.max_len - len(x_seq))  # pad manually
        x_tensor = torch.LongTensor([x_seq])

        with torch.no_grad():
            outputs = self.model(x_tensor)
            pred_tags = torch.argmax(F.softmax(outputs, dim=-1), dim=-1).squeeze(0).tolist()

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