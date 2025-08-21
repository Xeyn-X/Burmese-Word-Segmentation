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

        # Insert separators around Burmese, English, numbers, and special characters
        text = re.sub(
            rf"(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[A-Za-z]+|\d+|{specials}|[^က-၏A-Za-z0-9]+)(?![ှျ]?[့္်]))",
            r"|\1",
            text,
        )

        return [seg for seg in re.split(r"\|", text) if seg]
    
    def name_segment(self, text):
        """Segment Burmese text using regular expressions."""
        text = re.sub(r'(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[^က-၏]+)(?![ှျ]?[့္်]))', r' \1', text)
        return text.strip()
    
    def merge_syllables(self, text):
        special_words = ['ဦး', 'ဒေါ်', 'မောင်', 'မ','ကို']
        dict_file = 'name_dict/name_dict_clean.txt'
        with open(dict_file, 'r', encoding='utf-8') as f:
            syllable_dict = set(self.name_segment(line.strip()).replace(' ', '') for line in f.readlines())

        words = text.split()  
        merged_text = []
        i = 0

        while i < len(words):
            match_found = False

            # Check for longest possible match in dictionary
            for j in range(len(words), i, -1):
                phrase = ''.join(words[i:j])  # Merge words
                if phrase in syllable_dict:
                    merged_text.append(phrase)
                    if merged_text[-2] in special_words:
                        merged_text[-2] += merged_text[-1]
                        merged_text.pop()
                    i = j
                    match_found = True
                    break

            if not match_found:
                merged_text.append(words[i])
                i += 1

        return ' '.join(merged_text)
    
    def predict(self, text):
        segmented = self.segment(text)
        tokens = [t for t in segmented if t.strip() != ""]
        result = []
        buffer = ""
        # Regex patterns
        eng_pattern = re.compile(r"^[A-Za-z]+$")
        num_pattern = re.compile(r"^[0-9]+$")
        special_pattern = re.compile(r"^[^A-Za-z0-9က-ဿ၀-၉]+$")

        with torch.no_grad():
            token_ids = [self.token2idx.get(tok, self.token2idx["<UNK>"]) for tok in tokens]
            input_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.config["device"])
            mask = torch.tensor([[1] * len(tokens)], dtype=torch.bool).to(self.config["device"])

            predictions = self.model(input_tensor, mask=mask)
            tag_indices = predictions[0]
            tags = [self.idx2tag[idx] for idx in tag_indices]
            predicted = list(zip(tokens, tags))
            
            for i, (word, tag) in enumerate(predicted):
                check_word = bool(eng_pattern.match(word) or num_pattern.match(word) or special_pattern.match(word))
                if check_word:
                    # flush buffer first if exists
                    if buffer:
                        result.append(buffer)
                        buffer = ""
                    result.append(word)  # keep split element separately
                    continue

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

            segment_result = " ".join(result)
            final = self.merge_syllables(segment_result)

        return final
