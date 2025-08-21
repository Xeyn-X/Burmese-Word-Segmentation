import re
import pickle

class CRFInference:
    def __init__(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def _segment_syllables(self, text):
        specials = r'["\'“”‘’{}()\[\]\.,;:?!\s]'  # punctuation + whitespace

        # Insert separators around Burmese, English, numbers, and special characters
        text = re.sub(
            rf"(?:(?<!္)([က-ဪဿ၊-၏]|[၀-၉]+|[A-Za-z]+|\d+|{specials}|[^က-၏A-Za-z0-9]+)(?![ှျ]?[့္်]))",
            r"|\1",
            text,
        )

        return [seg for seg in re.split(r"\|", text) if seg]

    def _syllables_to_dicts(self, text):
        syllables = self._segment_syllables(text)
        tokens = [t for t in syllables if t.strip() != ""]
        return [{'syl': syl} for syl in tokens]

    def _extract_features(self, sentence):
        feats_per_syl = []
        for i in range(len(sentence)):
            syl = sentence[i]['syl']
            prev_syl = sentence[i - 1]['syl'] if i > 0 else '<START>'
            next_syl = sentence[i + 1]['syl'] if i < len(sentence) - 1 else '<END>'
            feats_per_syl.append({
                'curr': syl,
                'prev': prev_syl,
                'next': next_syl,
                'is_first': i == 0,
                'is_last': i == len(sentence) - 1,
                'has_digit': bool(re.search(r'[0-9၀-၉]', syl)),
                'has_english': bool(re.search(r'[a-zA-Z]', syl))
            })
        return feats_per_syl

    def _combine_segments(self, syllables, tags):
        result = []
        buffer = ""

        # Regex patterns
        eng_pattern = re.compile(r"^[A-Za-z]+$")
        num_pattern = re.compile(r"^[0-9]+$")
        special_pattern = re.compile(r"^[^A-Za-z0-9က-ဿ၀-၉]+$")

        for i, (syl, tag) in enumerate(zip(syllables, tags)):
            check_word = bool(eng_pattern.match(syl) or num_pattern.match(syl) or special_pattern.match(syl))
            if check_word:
                # flush buffer first if exists
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(syl)  # keep split element separately
                continue

            if tag == "B":
                # Look ahead: check next tag if it exists
                next_tag = tags[i + 1] if i + 1 < len(tags) else None

                # Only split if next tag is NOT I or E
                if next_tag not in ("I", "E"):
                    buffer += syl
                    result.append(buffer)
                    buffer = ""
                else:
                    buffer += syl

            elif tag == "I":
                buffer += syl

            elif tag in ("E", "S"):
                buffer += syl
                result.append(buffer)
                buffer = ""

        if buffer:
            result.append(buffer)

        return result
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
    
    def segment_text(self, text):
        syllable_dicts = self._syllables_to_dicts(text)
        features = [self._extract_features(syllable_dicts)]
        predicted_tags = self.model.predict(features)[0]
        syllables = [s['syl'] for s in syllable_dicts]
        words = self._combine_segments(syllables, predicted_tags)
        segment_result = " ".join(words)
        final = self.merge_syllables(segment_result)

        return final