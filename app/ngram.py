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
    text = text.lower().strip()
    if not text:
        return []
    
    if lang in ["hi", "te", "ta", "kn"]:
        tokens = re.findall(r'[\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\w]+|[^\w\s]', text)
    else:
        tokens = re.findall(r'\w+|[^\w\s]', text)
    
    return [t for t in tokens if t.strip()]

class BackoffNGram:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = [defaultdict(Counter) for _ in range(n)]
        self.vocab = Counter()
    
    def _tokenize(self, text: str, lang: str) -> List[str]:
        return tokenize(text, lang)
    
    def train(self, sentences: List[str], lang: str = "en"):
        for sent in sentences:
            tokens = self._tokenize(sent, lang)
            if not tokens:
                continue
            
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            
            self.vocab.update(tokens)
    
    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        tokens = self._tokenize(text, lang)
        if not tokens:
            return [w for w, _ in self.vocab.most_common(k)]
        
        predictions = []
        seen = set()
        
        for order in range(min(self.n, len(tokens) + 1), 0, -1):
            if len(tokens) >= order - 1:
                context = tuple(tokens[-(order - 1):]) if order > 1 else tuple()
                
                if context in self.ngrams[order - 1]:
                    for word, _ in self.ngrams[order - 1][context].most_common(k * 2):   
                        if word not in seen and not word.isdigit():
                            predictions.append(word)
                            seen.add(word)
                            if len(predictions) >= k:
                                return predictions[:k]
        
        if len(predictions) < k:
            for word, _ in self.vocab.most_common(k * 2):
                if word not in seen and not word.isdigit():
                    predictions.append(word)
                    seen.add(word)
                    if len(predictions) >= k:
                        break
        
        return predictions[:k]
    
    def generate_continuation(self, context_tokens: List[str], max_words: int = 5, lang: str = "en") -> str:
        """Generate continuation for ANY language"""
        if not context_tokens:
            common = [w for w, _ in self.vocab.most_common(20) if w[0].isalpha()]
            return random.choice(common) if common else ""
        
        generated = []
        current_context = context_tokens.copy()
        words_added = 0
        
        while words_added < max_words:
            word_found = False
            
            # Try trigram
            if len(current_context) >= 2:
                context = tuple(current_context[-2:])
                if context in self.ngrams[2]:
                    candidates = list(self.ngrams[2][context].keys())
                    if candidates:
                        next_word = random.choice(candidates[:5])
                        generated.append(next_word)
                        current_context.append(next_word)
                        words_added += 1
                        word_found = True
                        continue
            
            # Try bigram
            if not word_found and len(current_context) >= 1:
                context = tuple(current_context[-1:])
                if context in self.ngrams[1]:
                    candidates = list(self.ngrams[1][context].keys())
                    if candidates:
                        next_word = random.choice(candidates[:5])
                        generated.append(next_word)
                        current_context.append(next_word)
                        words_added += 1
                        word_found = True
                        continue
            
            # Fallback to unigram
            if not word_found:
                candidates = list(self.ngrams[0][()].keys())
                if candidates:
                    next_word = random.choice(candidates[:10])
                    generated.append(next_word)
                    current_context.append(next_word)
                    words_added += 1
                else:
                    break
            
            # Random chance to end
            if words_added >= 2 and random.random() < 0.3:
                break
        
        if not generated:
            return ""
        
        # Build continuation
        result = generated[0]
        for word in generated[1:]:
            result += " " + word
        
        # Add punctuation sometimes
        if random.random() < 0.3:
            result += random.choice(['.', '!', '?'])
        
        return result
    
    def predict_continuations(self, text: str, num_continuations: int = 5, max_words: int = 5, lang: str = "en") -> List[str]:
        """Get continuations for ANY language"""
        tokens = self._tokenize(text, lang)
        if not tokens:
            return []
        
        continuations = []
        seen = set()
        
        for _ in range(num_continuations * 3):
            if len(continuations) >= num_continuations:
                break
            
            cont = self.generate_continuation(tokens, max_words, lang)
            
            if cont and cont not in seen and len(cont.split()) <= max_words + 1:
                continuations.append(cont)
                seen.add(cont)
        
        return continuations[:num_continuations]
    
    def add_new_word(self, word: str, context: str = "", lang: str = "en"):
        """
        Add a new word to the vocabulary and update n-grams
        """
        word = word.lower().strip()
        if not word or word in self.vocab:
            return False
        
        # Add to vocabulary with initial count
        self.vocab[word] = 1
        
        # If context provided, update appropriate n-gram
        if context:
            context_tokens = self._tokenize(context, lang)
            if context_tokens:
                # Update unigram (context = ())
                self.ngrams[0][()][word] = 1
                
                # Update bigram if we have context
                if len(context_tokens) >= 1:
                    context_tuple = tuple(context_tokens[-1:])
                    self.ngrams[1][context_tuple][word] = 1
                
                # Update trigram if we have enough context
                if len(context_tokens) >= 2:
                    context_tuple = tuple(context_tokens[-2:])
                    self.ngrams[2][context_tuple][word] = 1
        
        return True
    
    def save_to_corpus(self, word: str, context: str, lang: str = "en"):
        """
        Save the new word to the corpus file for permanent storage
        """
        corpus_path = _corpus_path(lang)
        try:
            # Create a meaningful sentence with the new word
            if context:
                new_sentence = f"{context} {word}".strip()
            else:
                new_sentence = word
            
            # Append to corpus file
            with open(corpus_path, "a", encoding="utf-8") as f:
                f.write(f"\n{new_sentence}")
            
            return True
        except Exception as e:
            print(f"Error saving to corpus: {e}")
            return False
    
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
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Training {lang} model...")
        model = BackoffNGram(n=3)
        model.train(lines, lang)
        model.save(path)
        return model
    
    return BackoffNGram(n=3)
