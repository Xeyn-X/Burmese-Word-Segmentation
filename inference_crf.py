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
        words, word = [], ''
        for syl, tag in zip(syllables, tags):
            if tag == 'B':
                word = syl
            elif tag in ['I', 'E']:
                word += syl
                if tag == 'E':
                    words.append(word)
                    word = ''
            elif tag == 'S':
                words.append(syl)
        if word:
            words.append(word)
        return words

    def segment_text(self, text):
        syllable_dicts = self._syllables_to_dicts(text)
        features = [self._extract_features(syllable_dicts)]
        predicted_tags = self.model.predict(features)[0]
        syllables = [s['syl'] for s in syllable_dicts]
        words = self._combine_segments(syllables, predicted_tags)
        return " ".join(words)