# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Tuple
import time
import re
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# ---------- Groq API Model ----------
class GroqModelManager:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        # Language mapping
        self.lang_names = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu",
            "ta": "Tamil",
            "kn": "Kannada",
            "fr": "French",
        }
    
    def predict(self, text: str, num_words: int = 5, num_sentences: int = 3, lang: str = "en") -> Tuple[List[str], List[str]]:
        """Predict next words and sentences using Groq API"""
        language_name = self.lang_names.get(lang, "English")
        
        # Create prompt for Groq API
        prompt = f"""You are a predictive text model. Given the input text below, generate:

1) {num_words} likely next single words in {language_name}
2) {num_sentences} possible next sentences in {language_name}

Rules:
- Do NOT repeat the input text.
- Do NOT produce duplicate items.
- Output strictly in JSON format:
{{
  "words": ["word1", "word2", ...],
  "sentences": ["sentence1", "sentence2", ...]
}}

Input text: "{text}"
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Find JSON pattern
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    words = data.get("words", [])[:num_words]
                    sentences = data.get("sentences", [])[:num_sentences]
                    
                    # Ensure we have lists
                    if isinstance(words, str):
                        words = [words]
                    if isinstance(sentences, str):
                        sentences = [sentences]
                        
                    return words, sentences
            except:
                pass
            
            # Fallback: parse line by line
            lines = content.strip().split('\n')
            words = []
            sentences = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('{') and not line.startswith('}'):
                    if len(line.split()) <= 3 and len(words) < num_words:
                        words.append(line.strip('",.'))
                    elif len(sentences) < num_sentences:
                        sentences.append(line.strip('",.'))
            
            return words[:num_words], sentences[:num_sentences]
            
        except Exception as e:
            print(f"Groq API error: {e}")
            return [], []

# ---------- Spell Checker ----------
class SpellChecker:
    def __init__(self):
        self.dictionaries: Dict[str, Dict[str, int]] = {}
        self.load_all_dictionaries()
    
    def load_all_dictionaries(self):
        available_langs = ['en', 'hi', 'te', 'ta', 'kn', 'fr']
        for lang in available_langs:
            dict_path = os.path.join(DATA_DIR, f"{lang}_word_frequency.txt")
            if os.path.exists(dict_path):
                success = self.load_dictionary(lang, dict_path)
                if success:
                    print(f"✅ Loaded dictionary for {lang}: {len(self.dictionaries[lang])} words")
                else:
                    print(f"❌ Failed to load dictionary for {lang}")
                    self.dictionaries[lang] = {}
            else:
                print(f"⚠️ Dictionary file not found for {lang}: {dict_path}")
                self.dictionaries[lang] = {}
    
    def load_dictionary(self, lang: str, path: str) -> bool:
        words: Dict[str, int] = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0].lower().strip()
                        try:
                            freq = int(parts[1])
                            words[word] = freq
                        except ValueError:
                            # If frequency is not a number, use 1
                            words[word] = 1
                    elif len(parts) == 1:
                        # Just a word without frequency
                        word = parts[0].lower().strip()
                        words[word] = 1
                    else:
                        print(f"  Skipping line {line_count}: '{line}'")
            
            self.dictionaries[lang] = words
            print(f"  Processed {line_count} lines, got {len(words)} unique words")
            return True
            
        except Exception as e:
            print(f"Error loading dictionary {lang}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_available_languages(self) -> List[str]:
        return [lang for lang in self.dictionaries if self.dictionaries[lang]]
    
    def correct_text(self, text: str, lang: str = "en") -> str:
        """Correct all words in text"""
        print(f"\n🔧 Spell checking: '{text}' in {lang}")
        
        if not text or not text.strip():
            return text
        
        if lang not in self.dictionaries or not self.dictionaries[lang]:
            print(f"  No dictionary available for {lang}")
            return text
        
        print(f"  Dictionary size: {len(self.dictionaries[lang])} words")
        
        words = text.split()
        corrected_words = []
        
        for i, word in enumerate(words):
            original_word = word
            corrected_word = self.correct_word(word, lang)
            
            if corrected_word != original_word:
                print(f"  ✓ Corrected '{original_word}' → '{corrected_word}'")
            else:
                print(f"  ✓ '{original_word}' is correct")
            
            corrected_words.append(corrected_word)
        
        result = " ".join(corrected_words)
        print(f"  Result: '{result}'")
        return result
    
    def correct_word(self, word: str, lang: str) -> str:
        """Correct a single word based on frequency"""
        if not word or lang not in self.dictionaries:
            return word
        
        # Clean the word - remove surrounding punctuation but keep it
        original_word = word
        
        # Handle common cases
        word_lower = word.lower()
        
        # Direct match
        if word_lower in self.dictionaries[lang]:
            return word
        
        # Try removing common suffixes/endings
        if word_lower.endswith(('ing', 'ed', 's', 'ly', "'s", "n't")):
            base_word = word_lower[:-3] if word_lower.endswith('ing') else \
                       word_lower[:-2] if word_lower.endswith('ed') else \
                       word_lower[:-1] if word_lower.endswith('s') else \
                       word_lower[:-2] if word_lower.endswith('ly') else \
                       word_lower[:-3] if word_lower.endswith("'s") else \
                       word_lower[:-4] if word_lower.endswith("n't") else word_lower
            
            if base_word in self.dictionaries[lang]:
                return word_lower if word.islower() else word.title() if word.istitle() else base_word
        
        # Try difflib for fuzzy matching
        try:
            import difflib
            
            # Get all dictionary words
            dict_words = list(self.dictionaries[lang].keys())
            
            # Find close matches
            matches = difflib.get_close_matches(
                word_lower, 
                dict_words, 
                n=3, 
                cutoff=0.7
            )
            
            if matches:
                # Get the most frequent match
                best_match = max(matches, key=lambda w: self.dictionaries[lang][w])
                
                # Preserve original capitalization
                if word.isupper():
                    return best_match.upper()
                elif word.istitle():
                    return best_match.title()
                else:
                    return best_match
                    
        except ImportError:
            print("difflib not available")
        except Exception as e:
            print(f"Error in fuzzy matching: {e}")
        
        # If no correction found, return original word
        return word
    
    def is_correct(self, word: str, lang: str) -> bool:
        """Check if a word is spelled correctly"""
        if not word or lang not in self.dictionaries:
            return True
        
        word_lower = word.lower()
        return word_lower in self.dictionaries[lang]

# ---------- FastAPI App ----------
app = FastAPI(
    title="Smart Text Predictor",
    description="Next-word prediction using N-gram models and Groq API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web"), name="static")

# Initialize components
spell_checker = SpellChecker()
MODEL_CACHE: Dict[str, BackoffNGram] = {}
groq_model = None

def get_groq_model():
    """Lazy load Groq model"""
    global groq_model
    if groq_model is None:
        try:
            groq_model = GroqModelManager()
            print("✅ Groq API model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Groq model: {e}")
            groq_model = None
    return groq_model

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

def get_ngram_model(lang: str):
    if lang not in MODEL_CACHE:
        MODEL_CACHE[lang] = load_model(lang)
    return MODEL_CACHE[lang]

def auto_correct_text(text: str, lang: str) -> Tuple[str, str]:
    """Auto-correct the text and return correction message"""
    if not text or not text.strip():
        return text, ""
    
    print(f"\n🔄 Auto-correcting text: '{text}'")
    print(f"   Language: {lang}")
    print(f"   Spell checker available languages: {spell_checker.get_available_languages()}")
    
    if lang not in spell_checker.get_available_languages():
        print(f"   Language {lang} not available for spell checking")
        return text, ""
    
    # Get original text
    original_text = text
    
    # Correct the entire text
    corrected_text = spell_checker.correct_text(text, lang)
    
    # Generate correction message
    corrections = []
    if corrected_text != original_text:
        original_words = original_text.split()
        corrected_words = corrected_text.split()
        
        for orig, corr in zip(original_words, corrected_words):
            if orig != corr:
                corrections.append(f"'{orig}'→'{corr} '")
    
    if corrections:
        correction_msg = "Auto-corrected: " + ", ".join(corrections)
        print(f"   Correction message: {correction_msg}")
    else:
        correction_msg = ""
        print(f"   No corrections needed")
    
    return corrected_text, correction_msg

@app.get("/")
async def home():
    return FileResponse("web/index.html")

@app.get("/predict")
async def predict(
    text: str = Query(..., min_length=0),
    lang: str = Query("auto"),
    model_type: str = Query("ngram"),  # "ngram" or "groq"
    auto_correct: bool = Query(True),
    include_sentences: bool = Query(True),
    num_words: int = Query(5, ge=1, le=10)
):
    start_time = time.time()
    
    try:
        if not text.strip():
            return {
                "success": True,
                "model_type": model_type,
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
        
        # Detect language
        if lang == "auto":
            detected_lang = detect_language(text)
        else:
            detected_lang = lang
        
        # Auto-correct
        correction_msg = ""
        if auto_correct:
            corrected_text, correction_msg = auto_correct_text(text, detected_lang)
        else:
            corrected_text = text
        
        predictions = []
        sentence_predictions = []
        accuracy_score = 0.0
        
        if model_type == "ngram":
            # Use N-gram model
            model = get_ngram_model(detected_lang)
            raw_predictions = model.predict(corrected_text, k=num_words * 3, lang=detected_lang)
            
            # Clean predictions
            cleaned_predictions = []
            seen = set()
            for pred in raw_predictions:
                if pred and pred not in seen and not pred.isdigit():
                    cleaned_predictions.append(pred)
                    seen.add(pred)
                if len(cleaned_predictions) >= num_words:
                    break
            
            predictions = cleaned_predictions[:num_words]
            accuracy_score = 0.7 + (len(predictions) / num_words * 0.3)
            accuracy_score = min(0.95, accuracy_score)
            
        elif model_type == "groq":
            # Use Groq API
            groq_model = get_groq_model()
            if groq_model is None:
                raise HTTPException(status_code=500, detail="Groq API not available. Check API key.")
            
            try:
                words, sentences = groq_model.predict(
                    corrected_text, 
                    num_words=num_words, 
                    num_sentences=3 if include_sentences else 0,
                    lang=detected_lang
                )
                
                # Clean Groq predictions
                predictions = []
                seen = set()
                for word in words:
                    if word and word not in seen and not word.isdigit():
                        clean_word = word.strip('.,!?;:"\'')
                        if clean_word:
                            predictions.append(clean_word)
                            seen.add(clean_word)
                
                # Clean sentence predictions
                if include_sentences:
                    sentence_predictions = []
                    seen_sentences = set()
                    for sentence in sentences:
                        clean_sentence = sentence.strip('",.')
                        if clean_sentence and clean_sentence not in seen_sentences:
                            sentence_predictions.append(clean_sentence)
                            seen_sentences.add(clean_sentence)
                
                # Groq typically has high accuracy
                accuracy_score = 0.85 + (len(predictions) / num_words * 0.15)
                accuracy_score = min(0.98, accuracy_score)
                
            except Exception as e:
                print(f"Groq prediction error: {e}")
                # Fallback to N-gram
                model = get_ngram_model(detected_lang)
                raw_predictions = model.predict(corrected_text, k=num_words * 3, lang=detected_lang)
                predictions = [p for p in raw_predictions if p and not p.isdigit()][:num_words]
                accuracy_score = 0.6
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'ngram' or 'groq'.")
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "model_type": model_type,
            "original_text": text,
            "corrected_text": corrected_text,
            "correction_msg": correction_msg,
            "language": detected_lang,
            "predictions": predictions[:num_words],
            "sentence_predictions": sentence_predictions[:3] if include_sentences else [],
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
        "models_loaded": list(MODEL_CACHE.keys()),
        "groq_available": get_groq_model() is not None
    }

@app.get("/correct")
async def correct_spelling(text: str = Query(...), lang: str = Query("en")):
    corrected = spell_checker.correct_text(text, lang)
    return {"original": text, "corrected": corrected, "language": lang}

@app.get("/groq_status")
async def groq_status():
    """Check if Groq API is available"""
    groq_model = get_groq_model()
    return {
        "available": groq_model is not None,
        "model": "llama-3.1-8b-instant" if groq_model else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
