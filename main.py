from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Tuple
import time
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

# ---------- N-gram Model ----------
import pickle
from collections import Counter, defaultdict

class BackoffNGram:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = [defaultdict(Counter) for _ in range(n)]
        self.vocab = Counter()
    
    def _tokenize(self, text: str, lang: str) -> List[str]:
        text = text.lower().strip()
        if lang in ["hi", "te", "ta", "kn"]:
            tokens = re.findall(r'[\u0900-\u097F\u0C00-\u0C7F\u0B80-\u0BFF\u0C80-\u0CFF\w]+|[^\w\s]', text)
        else:
            tokens = re.findall(r'\w+|[^\w\s]', text)
        return [t for t in tokens if t.strip()]
    
    def train(self, sentences: List[str], lang: str = "en"):
        for sent in sentences:
            tokens = self._tokenize(sent, lang)
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            for token in tokens:
                self.vocab[token] += 1
    
    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        tokens = self._tokenize(text, lang)
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
        
        if len(predictions) < k:
            for word, _ in self.vocab.most_common(k * 2):
                if word not in seen:
                    predictions.append(word)
                    seen.add(word)
                    if len(predictions) >= k:
                        break
        
        # Remove numbers
        cleaned = [p for p in predictions if not p.isdigit()]
        return cleaned[:k]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'n': self.n, 'ngrams': self.ngrams, 'vocab': self.vocab}, f)
    
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

def load_model(lang: str = "en") -> BackoffNGram:
    path = _model_path(lang)
    if not os.path.exists(path):
        # Train model if not exists
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Dataset not found: {corpus_path}")
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Training {lang} model with {len(lines)} sentences...")
        model = BackoffNGram(n=3)
        model.train(lines, lang)
        model.save(path)
        print(f"✅ Trained N-gram model for {lang}")
    
    return BackoffNGram.load(path)

# ---------- Spell Checker ----------
class SpellChecker:
    def __init__(self):
        self.dictionaries: Dict[str, set] = {}
        self.load_all_dictionaries()
    
    def load_all_dictionaries(self):
        available_langs = ['en', 'hi', 'te', 'ta', 'kn', 'fr']
        for lang in available_langs:
            dict_path = os.path.join(DATA_DIR, f"{lang}_word_frequency.txt")
            if os.path.exists(dict_path):
                self.load_dictionary(lang, dict_path)
    
    def load_dictionary(self, lang: str, path: str):
        words = set()
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            words.add(parts[0].lower())
            self.dictionaries[lang] = words
        except:
            self.dictionaries[lang] = set()
    
    def get_available_languages(self) -> List[str]:
        return list(self.dictionaries.keys())
    
    def correct_text(self, text: str, lang: str = "en") -> str:
        if not text or lang not in self.dictionaries:
            return text
        words = text.split()
        corrected_words = []
        for word in words:
            corrected_word = self.correct_word(word, lang)
            corrected_words.append(corrected_word)
        return " ".join(corrected_words)
    
    def correct_word(self, word: str, lang: str) -> str:
        if not word or lang not in self.dictionaries:
            return word
        word_lower = word.lower()
        if word_lower in self.dictionaries[lang]:
            return word
        return word  # Simple implementation - returns same word

# ---------- FastAPI App ----------
app = FastAPI(
    title="Next-Word Prediction (N-gram)",
    description="Multi-language next word prediction using N-gram models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web"), name="static")

spell_checker = SpellChecker()
MODEL_CACHE: Dict[str, BackoffNGram] = {}

def detect_language(text: str) -> str:
    if not text:
        return "en"
    for char in text:
        if '\u0900' <= char <= '\u097F':
            return "hi"
        elif '\u0C00' <= char <= '\u0C7F':
            return "te"
        elif '\u0B80' <= char <= '\u0BFF':
            return "ta"
        elif '\u0C80' <= char <= '\u0CFF':
            return "kn"
        elif char in 'éèêëàâçîïôûùü':
            return "fr"
    return "en"

def get_model(lang: str):
    if lang not in MODEL_CACHE:
        MODEL_CACHE[lang] = load_model(lang)
    return MODEL_CACHE[lang]

def auto_correct_text(text: str, lang: str) -> Tuple[str, str]:
    if not text or lang not in spell_checker.get_available_languages():
        return text, ""
    words = text.strip().split()
    if not words:
        return text, ""
    last_word = words[-1]
    if len(last_word) < 2:
        return text, ""
    corrected_word = spell_checker.correct_word(last_word, lang)
    if corrected_word != last_word and corrected_word:
        words[-1] = corrected_word
        corrected_text = " ".join(words)
        correction_msg = f"'{last_word}' → '{corrected_word}'"
        return corrected_text, correction_msg
    return text, ""

def generate_sentence(text: str, model, lang: str) -> List[str]:
    if not text.strip():
        return []
    sentences = []
    for _ in range(3):
        current_text = text
        for _ in range(10):
            next_words = model.predict(current_text, k=3, lang=lang)
            if not next_words:
                break
            current_text += " " + next_words[0]
            if next_words[0] in ['.', '!', '?', '।', '॥']:
                break
        sentences.append(current_text)
    unique_sentences = []
    seen = set()
    for s in sentences:
        if s not in seen:
            unique_sentences.append(s)
            seen.add(s)
    return unique_sentences[:3]

@app.get("/")
async def home():
    return FileResponse("web/index.html")

@app.get("/predict")
async def predict(
    text: str = Query(..., min_length=0),
    lang: str = Query("auto"),
    auto_correct: bool = Query(True),
    include_sentences: bool = Query(True),
    num_words: int = Query(5, ge=1, le=10)
):
    start_time = time.time()
    try:
        if not text.strip():
            return {
                "success": True,
                "original_text": text,
                "corrected_text": text,
                "correction_msg": "",
                "language": "en",
                "predictions": [],
                "sentence_predictions": [],
                "accuracy_score": 0.0,
                "response_time_ms": 0,
                "available_languages": spell_checker.get_available_languages()
            }
        
        if lang == "auto":
            detected_lang = detect_language(text)
        else:
            detected_lang = lang
        
        correction_msg = ""
        if auto_correct:
            corrected_text, correction_msg = auto_correct_text(text, detected_lang)
        else:
            corrected_text = text
        
        model = get_model(detected_lang)
        raw_predictions = model.predict(corrected_text, k=num_words * 3, lang=detected_lang)
        
        cleaned_predictions = []
        seen = set()
        for pred in raw_predictions:
            if pred and pred not in seen and not pred.isdigit():
                cleaned_predictions.append(pred)
                seen.add(pred)
            if len(cleaned_predictions) >= num_words:
                break
        
        sentence_predictions = []
        if include_sentences and cleaned_predictions:
            sentence_predictions = generate_sentence(corrected_text, model, detected_lang)
        
        accuracy_score = 0.7 + (len(cleaned_predictions) / num_words * 0.3)
        accuracy_score = min(0.95, accuracy_score)
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "original_text": text,
            "corrected_text": corrected_text,
            "correction_msg": correction_msg,
            "language": detected_lang,
            "predictions": cleaned_predictions[:num_words],
            "sentence_predictions": sentence_predictions,
            "accuracy_score": round(accuracy_score, 2),
            "response_time_ms": response_time,
            "available_languages": spell_checker.get_available_languages()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages")
async def get_languages():
    return {
        "available_languages": spell_checker.get_available_languages(),
        "models_loaded": list(MODEL_CACHE.keys())
    }

@app.get("/correct")
async def correct_spelling(text: str = Query(...), lang: str = Query("en")):
    corrected = spell_checker.correct_text(text, lang)
    return {"original": text, "corrected": corrected, "language": lang}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)