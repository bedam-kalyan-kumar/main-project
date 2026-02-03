# app/ngram.py
import os
import re
import pickle
from collections import Counter, defaultdict
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def tokenize(text: str, lang: str = "en") -> List[str]:
    """Language-aware tokenization"""
    text = text.lower().strip()
    
    # For Indian languages
    if lang in ["hi", "te", "ta", "kn"]:
        # Split on spaces and punctuation
        tokens = re.findall(r'[\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\w]+|[^\w\s]', text)
    else:
        # For European languages
        tokens = re.findall(r'\w+|[^\w\s]', text)
    
    return [t for t in tokens if t.strip()]


class BackoffNGram:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = [defaultdict(Counter) for _ in range(n)]
        self.vocab = Counter()
    
    def train(self, sentences: List[str], lang: str = "en"):
        """Train model on sentences"""
        for sent in sentences:
            tokens = tokenize(sent, lang)
            
            # Build n-grams of different orders
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            
            # Update vocabulary
            for token in tokens:
                self.vocab[token] += 1
    def generate_sentence(self, text: str, max_words: int = 15, lang: str = "en") -> str:
        """Generate a sentence continuation"""
        tokens = self._tokenize(text, lang)
        
        if not tokens:
            return text
        
        generated_tokens = tokens.copy()
        
        for _ in range(max_words):
            # Build context from last 2 tokens
            context = tuple(generated_tokens[-2:]) if len(generated_tokens) >= 2 else tuple()
            
            # Get next word prediction
            next_word = None
            
            # Try trigram first
            if len(context) == 2 and context in self.ngrams[2]:
                candidates = self.ngrams[2][context]
                if candidates:
                    next_word = max(candidates.items(), key=lambda x: x[1])[0]
            
            # Fall back to bigram
            if not next_word and len(generated_tokens) >= 1:
                context = tuple(generated_tokens[-1:])
                if context in self.ngrams[1]:
                    candidates = self.ngrams[1][context]
                    if candidates:
                        next_word = max(candidates.items(), key=lambda x: x[1])[0]
            
            # Fall back to unigram
            if not next_word and self.ngrams[0]:
                next_word = max(self.ngrams[0].items(), key=lambda x: x[1])[0]
            
            if not next_word:
                break
            
            generated_tokens.append(next_word)
            
            # Stop at sentence boundaries
            if next_word in ['.', '!', '?', '।', '॥']:
                break
        
        return ' '.join(generated_tokens)
    
    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        """Predict next words with backoff smoothing"""
        tokens = tokenize(text, lang)
        
        # Try higher order n-grams first, then back off
        predictions = []
        seen = set()
        
        for order in range(min(self.n, len(tokens) + 1), 0, -1):
            if len(tokens) >= order - 1:
                context = tuple(tokens[-(order - 1):]) if order > 1 else tuple()
                
                if context in self.ngrams[order - 1]:
                    candidates = self.ngrams[order - 1][context].most_common(k * 2)
                    
                    for word, count in candidates:
                        if word not in seen:
                            predictions.append(word)
                            seen.add(word)
                            
                            if len(predictions) >= k:
                                break
        
        # Fallback to most common words in vocabulary
        if len(predictions) < k:
            for word, _ in self.vocab.most_common(k * 2):
                if word not in seen:
                    predictions.append(word)
                    seen.add(word)
                    
                    if len(predictions) >= k:
                        break
        
        # Clean predictions - remove numbers
        cleaned_predictions = []
        for pred in predictions:
            # Remove any standalone numbers
            if not pred.isdigit():
                cleaned_predictions.append(pred)
        
        return cleaned_predictions[:k]
    
    def save(self, path: str):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngrams': self.ngrams,
                'vocab': self.vocab
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'])
        model.ngrams = data['ngrams']
        model.vocab = data['vocab']
        return model


def _corpus_path(lang: str) -> str:
    return os.path.join(DATA_DIR, f"{lang}.txt")


def _model_path(lang: str) -> str:
    return os.path.join(DATA_DIR, f"ngram_{lang}.pkl")


def train_and_save(lang: str = "en") -> BackoffNGram:
    """Train and save n-gram model for a language"""
    corpus = _corpus_path(lang)
    if not os.path.exists(corpus):
        raise FileNotFoundError(f"Dataset not found: {corpus}")
    
    with open(corpus, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Training {lang} model with {len(lines)} sentences...")
    model = BackoffNGram(n=3)
    model.train(lines, lang)
    
    model_path = _model_path(lang)
    model.save(model_path)
    print(f"✅ Trained N-gram model for {lang}")
    
    vocab_size = len(model.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    return model


def load_model(lang: str = "en") -> BackoffNGram:
    """Load trained model, train if not exists"""
    path = _model_path(lang)
    
    if not os.path.exists(path):
        print(f"Model for {lang} not found. Training...")
        return train_and_save(lang)
    
    try:
        return BackoffNGram.load(path)
    except Exception as e:
        print(f"Error loading model for {lang}: {e}. Retraining...")
        return train_and_save(lang)
