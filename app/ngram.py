# ngram.py
import os
import re
import pickle
import random
from collections import Counter, defaultdict
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def tokenize(text: str, lang: str = "en") -> List[str]:
    """Language-aware tokenizer."""
    text = text.lower().strip()
    if not text:
        return []
    # Indian language scripts
    if lang in ["hi", "te", "ta", "kn"]:
        pattern = r'[\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\w]+|[^\w\s]'
    else:
        pattern = r'\w+|[^\w\s]'
    tokens = re.findall(pattern, text)
    return [t for t in tokens if t.strip()]

class BackoffNGram:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = [defaultdict(Counter) for _ in range(n)]
        self.vocab = Counter()

    def train(self, sentences: List[str], lang: str = "en"):
        for sent in sentences:
            tokens = tokenize(sent, lang)
            if not tokens:
                continue
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            self.vocab.update(tokens)

    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        tokens = tokenize(text, lang)
        if not tokens:
            return [w for w, _ in self.vocab.most_common(k)]

        seen = set()
        predictions = []
        # Try highest order first, then back off
        for order in range(min(self.n, len(tokens) + 1), 0, -1):
            context = tuple(tokens[-(order - 1):]) if order > 1 else ()
            if context in self.ngrams[order - 1]:
                for word, _ in self.ngrams[order - 1][context].most_common(k * 2):
                    if word not in seen and not word.isdigit():
                        predictions.append(word)
                        seen.add(word)
                        if len(predictions) >= k:
                            return predictions
        # Fallback to most common words in vocabulary
        for word, _ in self.vocab.most_common(k):
            if word not in seen and not word.isdigit():
                predictions.append(word)
                if len(predictions) >= k:
                    break
        return predictions

    def predict_continuations(self, text: str, num_continuations: int = 5,
                              max_words: int = 5, lang: str = "en") -> List[str]:
        """Generate possible continuations using random sampling."""
        tokens = tokenize(text, lang)
        if not tokens:
            return []
        continuations = []
        seen = set()
        for _ in range(num_continuations * 3):
            if len(continuations) >= num_continuations:
                break
            generated = []
            current = tokens[:]
            for _ in range(max_words):
                # Try trigram
                if len(current) >= 2:
                    ctx = tuple(current[-2:])
                    if ctx in self.ngrams[2]:
                        next_word = self._random_choice(self.ngrams[2][ctx])
                        if next_word:
                            generated.append(next_word)
                            current.append(next_word)
                            continue
                # Try bigram
                if len(current) >= 1:
                    ctx = tuple(current[-1:])
                    if ctx in self.ngrams[1]:
                        next_word = self._random_choice(self.ngrams[1][ctx])
                        if next_word:
                            generated.append(next_word)
                            current.append(next_word)
                            continue
                # Fallback unigram
                if self.ngrams[0][()]:
                    next_word = self._random_choice(self.ngrams[0][()])
                    if next_word:
                        generated.append(next_word)
                        current.append(next_word)
                else:
                    break
                # Stop early sometimes
                if len(generated) >= 2 and random.random() < 0.3:
                    break
            if generated:
                cont = ' '.join(generated)
                if cont not in seen:
                    continuations.append(cont)
                    seen.add(cont)
        return continuations[:num_continuations]

    def _random_choice(self, counter: Counter) -> str:
        """Pick a random word weighted by frequency."""
        if not counter:
            return ''
        total = sum(counter.values())
        r = random.randint(1, total)
        cumulative = 0
        for word, count in counter.items():
            cumulative += count
            if r <= cumulative:
                return word
        return ''

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngrams': self.ngrams,
                'vocab': self.vocab
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(n=data['n'])
        model.ngrams = data['ngrams']
        model.vocab = data['vocab']
        return model

def _model_path(lang: str) -> str:
    return os.path.join(DATA_DIR, f"ngram_{lang}.pkl")

def _corpus_path(lang: str) -> str:
    return os.path.join(DATA_DIR, f"{lang}.txt")

def load_model(lang: str = "en") -> BackoffNGram:
    path = _model_path(lang)
    if os.path.exists(path):
        try:
            return BackoffNGram.load(path)
        except:
            pass
    corpus_path = _corpus_path(lang)
    if not os.path.exists(corpus_path):
        return BackoffNGram(n=3)  # empty model
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Training {lang} model...")
    model = BackoffNGram(n=3)
    model.train(lines, lang)
    model.save(path)
    return model
