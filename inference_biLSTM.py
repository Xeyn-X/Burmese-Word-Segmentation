import json
import torch
import re
from model.bilstm import BiLSTMTagger


class BiLSTMInference:
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
        self.model = BiLSTMTagger(
            vocab_size=len(self.token2idx),
            tagset_size=len(self.tag2idx),
            embedding_dim=self.config["embedding_dim"],
            hidden_dim=self.config["hidden_dim"],
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.config["device"])
        )
        self.model.to(self.config["device"])
        self.model.eval()

    def segment(self, text):
        specials = r'["\'“”‘’{}()\[\]\.,;:?!\s]'  # punctuation + whitespace

        # Insert separators around Burmese, English, numbers, and special characters
        text = re.sub(
            rf"(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[A-Za-z]+|\d+|{specials}|[^က-၏A-Za-z0-9]+)(?![ှျ]?[့္်]))",
            r"|\1",
            text,
        )

        return [seg for seg in re.split(r"\|", text) if seg]

    def predict(self, text):
        result = []
        buffer = ""
        segmented = self.segment(text)
        tokens = [t for t in segmented if t.strip() != ""]

        with torch.no_grad():
            token_ids = [
                self.token2idx.get(tok, self.token2idx["<UNK>"]) for tok in tokens
            ]
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(
                self.config["device"]
            )
            # shape: [1, seq_len]

            logits = self.model(input_tensor)  # [1, seq_len, tagset_size]
            pred_indices = torch.argmax(logits, dim=-1)[0].tolist()  # [seq_len]

            tags = [self.idx2tag[idx] for idx in pred_indices[: len(tokens)]]
            predicted = list(zip(tokens, tags))

            for i, (word, tag) in enumerate(predicted):
                if tag == "B":
                    # Look ahead: check next tag if it exists
                    next_tag = predicted[i + 1][1] if i + 1 < len(predicted) else None

                    # Only split if next tag is NOT I or E
                    if next_tag not in ("I", "E"):
                        buffer += word
                        result.append(buffer)
                        buffer = ""
                    else:
                        buffer += word

                elif tag == "I":
                    buffer += word

                elif tag in ("E", "S"):
                    buffer += word
                    result.append(buffer)
                    buffer = ""

            if buffer:
                result.append(buffer)

        return " ".join(result)
