# app/ngram.py
import os
import re
import pickle
import random
from collections import Counter, defaultdict
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def tokenize(text: str, lang: str = "en") -> List[str]:
    """Language-aware tokenization"""
    text = text.lower().strip()
    if not text:
        return []
    
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
    
    def _tokenize(self, text: str, lang: str) -> List[str]:
        """Internal tokenization method"""
        return tokenize(text, lang)
    
    def train(self, sentences: List[str], lang: str = "en"):
        """Train model on sentences"""
        for sent in sentences:
            tokens = self._tokenize(sent, lang)
            if not tokens:
                continue
                
            # Build n-grams of different orders
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            
            # Update vocabulary
            self.vocab.update(tokens)
    
    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        """Predict next words with backoff smoothing"""
        tokens = self._tokenize(text, lang)
        predictions = []
        seen = set()
        
        # Try higher order n-grams first, then back off
        for order in range(min(self.n, len(tokens) + 1), 0, -1):
            if len(tokens) >= order - 1:
                context = tuple(tokens[-(order - 1):]) if order > 1 else tuple()
                
                if context in self.ngrams[order - 1]:
                    candidates = self.ngrams[order - 1][context].most_common(k * 2)
                    
                    for word, count in candidates:
                        if word not in seen and not word.isdigit():
                            predictions.append(word)
                            seen.add(word)
                            if len(predictions) >= k:
                                return predictions[:k]
        
        # Fallback to most common words in vocabulary
        if len(predictions) < k:
            for word, count in self.vocab.most_common(k * 2):
                if word not in seen and not word.isdigit():
                    predictions.append(word)
                    seen.add(word)
                    if len(predictions) >= k:
                        break
        
        return predictions[:k]
    
    def generate_sentence(self, text: str, max_words: int = 15, lang: str = "en") -> str:
        """Generate a sentence continuation"""
        tokens = self._tokenize(text, lang)
        if not tokens:
            # Start with a common word if no input
            common_words = [w for w, c in self.vocab.most_common(20) if w[0].isalpha()]
            if common_words:
                tokens = [random.choice(common_words)]
            else:
                return ""
        
        generated = tokens.copy()
        
        for _ in range(max_words):
            # Try trigram context first
            if len(generated) >= 2:
                context = tuple(generated[-2:])
                if context in self.ngrams[2]:
                    candidates = list(self.ngrams[2][context].keys())
                    if candidates:
                        next_word = random.choice(candidates[:10])
                        generated.append(next_word)
                        continue
            
            # Fallback to bigram
            if len(generated) >= 1:
                context = tuple(generated[-1:])
                if context in self.ngrams[1]:
                    candidates = list(self.ngrams[1][context].keys())
                    if candidates:
                        next_word = random.choice(candidates[:10])
                        generated.append(next_word)
                        continue
            
            # Fallback to unigram
            candidates = list(self.ngrams[0][()].keys())
            if candidates:
                next_word = random.choice(candidates[:20])
                generated.append(next_word)
            else:
                break
            
            # Random chance to end sentence
            if random.random() < 0.2 and len(generated) > 5:
                if generated[-1] not in '.!?।':
                    generated.append('.')
                break
        
        # Ensure sentence ends with punctuation
        if generated[-1] not in '.!?।':
            generated.append('.')
        
        # Convert to proper string
        sentence = ""
        for i, token in enumerate(generated):
            if i == 0:
                sentence += token.capitalize()
            elif token in '.,!?;:':
                sentence += token
            else:
                sentence += " " + token
        
        return sentence
    
    def predict_sentences(self, text: str, num_sentences: int = 3, max_words: int = 15, lang: str = "en") -> List[str]:
        """Generate multiple sentence completions"""
        sentences = []
        seen = set()
        
        for attempt in range(num_sentences * 3):  # Try multiple times
            if len(sentences) >= num_sentences:
                break
            
            sentence = self.generate_sentence(text, max_words, lang)
            
            # Only add if it's meaningful and unique
            if (sentence and 
                sentence not in seen and 
                len(sentence) > len(text) + 5 and
                not sentence.startswith("import") and  # Filter out code
                not sentence.startswith("def ") and
                not sentence.startswith("class ") and
                "=" not in sentence):
                
                sentences.append(sentence)
                seen.add(sentence)
        
        return sentences[:num_sentences]
    
    def save(self, path: str):
        """Save model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngrams': self.ngrams,
                'vocab': self.vocab
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
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


# Test function
def test_ngram_model():
    """Test the N-gram model"""
    print("Testing N-gram model...")
    
    # Sample training data
    sample_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "the cat in the hat sat on the mat",
        "to be or not to be that is the question",
        "all that glitters is not gold",
        "a journey of a thousand miles begins with a single step"
    ]
    
    # Create and train model
    model = BackoffNGram(n=3)
    model.train(sample_sentences, "en")
    
    # Test predictions
    test_inputs = [
        "the quick",
        "to be or",
        "a journey",
        "the cat"
    ]
    
    for text in test_inputs:
        predictions = model.predict(text, k=3, lang="en")
        print(f"Input: '{text}' -> Predictions: {predictions}")
        
        # Test sentence generation
        sentence = model.generate_sentence(text, max_words=10, lang="en")
        print(f"Generated sentence: {sentence}")
        print()


if __name__ == "__main__":
    test_ngram_model()
