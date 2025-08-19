import json
import torch
import re
from model.bilstm_attention_crf import BiLSTMAttentionCRFModel


class BiLSTMAttentionCRFInference:
    def __init__(self, config_path, vocab_path, model_path):
        # Load config
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load vocab
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.token2idx = vocab["token2idx"]
        self.tag2idx = vocab["tag2idx"]
        self.idx2tag = {v: k for k, v in self.tag2idx.items()}

        # Load model
        self.model = BiLSTMAttentionCRFModel(
            vocab_size=len(self.token2idx),
            tagset_size=len(self.tag2idx),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"]
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.config["device"]))
        self.model.to(self.config["device"])
        self.model.eval()

    def segment(self, text):
        specials = r'["\'“”‘’{}()\[\]\.,;:?!\s]'  # punctuation + whitespace
        text = re.sub(
            rf'(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[A-Za-z]+|\d+|{specials}|[^က-၏A-Za-z0-9]+)(?![ှျ]?[့္်]))',
            r'|\1',
            text
        )
        return [seg for seg in re.split(r'\|', text) if seg]

    def predict(self, text):
        tokens = self.segment(text.replace(' ', ''))
        result = []
        buffer = ""

        with torch.no_grad():
            token_ids = [self.token2idx.get(tok, self.token2idx["<UNK>"]) for tok in tokens]
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.config["device"])
            mask = torch.tensor([[1] * len(tokens)], dtype=torch.bool).to(self.config["device"])

            predictions = self.model(input_tensor, mask=mask)
            tag_indices = predictions[0]
            tags = [self.idx2tag[idx] for idx in tag_indices]
            predicted = list(zip(tokens, tags))

            for word, tag in predicted:
                if tag == 'B':
                    if buffer:
                        result.append(buffer)
                    buffer = word
                elif tag == 'I':
                    buffer += word
                elif tag in ('E', 'S'):
                    buffer += word
                    result.append(buffer)
                    buffer = ""

            if buffer:
                result.append(buffer)

        return " ".join(result)
