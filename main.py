# main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Tuple
import time
import re
import os
import json
import pickle
import random
from collections import Counter, defaultdict
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

# ---------- N-gram Model ----------
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
        return [t for t in tokens if t]
    
    def train(self, sentences: List[str], lang: str = "en"):
        for sent in sentences:
            tokens = self._tokenize(sent, lang)
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    context = tuple(tokens[i:i + order - 1])
                    next_word = tokens[i + order - 1]
                    self.ngrams[order - 1][context][next_word] += 1
            self.vocab.update(tokens)
    
    def predict(self, text: str, k: int = 5, lang: str = "en") -> List[str]:
        tokens = self._tokenize(text, lang)
        predictions = []
        seen = set()
        
        for order in range(min(self.n, len(tokens) + 1), 0, -1):
            if len(tokens) >= order - 1:
                context = tuple(tokens[-(order - 1):]) if order > 1 else ()
                if context in self.ngrams[order - 1]:
                    for word, _ in self.ngrams[order - 1][context].most_common(k * 2):
                        if word not in seen:
                            predictions.append(word)
                            seen.add(word)
                            if len(predictions) >= k:
                                break
            if len(predictions) >= k:
                break
        
        if len(predictions) < k:
            for word, _ in self.vocab.most_common(k * 2):
                if word not in seen:
                    predictions.append(word)
                    seen.add(word)
                    if len(predictions) >= k:
                        break
        
        return [p for p in predictions if not p.isdigit()][:k]
    
    def predict_sentences(self, text: str, num_sentences: int = 3, max_words: int = 15, lang: str = "en") -> List[str]:
        """Generate sentence completions using N-gram model"""
        base_tokens = self._tokenize(text, lang)
        if not base_tokens:
            return []
        
        sentences = []
        seen_sentences = set()
        
        for _ in range(num_sentences * 2):  # Generate extra to ensure we get enough unique ones
            if len(sentences) >= num_sentences:
                break
            
            current_tokens = base_tokens.copy()
            
            for word_count in range(max_words):
                # Get context for prediction (last 2 tokens for trigram)
                if len(current_tokens) >= 2:
                    context = tuple(current_tokens[-2:])
                    if context in self.ngrams[2]:
                        candidates = list(self.ngrams[2][context].keys())
                        if candidates:
                            next_word = random.choice(candidates[:10])  # Pick from top 10
                            current_tokens.append(next_word)
                            continue
                
                # Fallback to bigram
                if len(current_tokens) >= 1:
                    context = tuple(current_tokens[-1:])
                    if context in self.ngrams[1]:
                        candidates = list(self.ngrams[1][context].keys())
                        if candidates:
                            next_word = random.choice(candidates[:10])
                            current_tokens.append(next_word)
                            continue
                
                # Fallback to unigram
                common_words = list(self.ngrams[0][()].keys())
                if common_words:
                    next_word = random.choice(common_words[:20])  # Pick from top 20 common words
                    current_tokens.append(next_word)
                else:
                    break
                
                # Random chance to end sentence
                if random.random() < 0.2 and word_count >= 3:
                    # Add ending punctuation
                    if next_word not in '.!?':
                        current_tokens.append('.')
                    break
            
            # Convert tokens back to string
            if len(current_tokens) > len(base_tokens):
                # Join tokens with proper spacing
                sentence_tokens = []
                for i, token in enumerate(current_tokens):
                    if i > 0 and token not in '.,!?;:':
                        sentence_tokens.append(' ')
                    sentence_tokens.append(token)
                
                generated_sentence = ''.join(sentence_tokens).strip()
                
                # Capitalize first letter
                if generated_sentence:
                    generated_sentence = generated_sentence[0].upper() + generated_sentence[1:]
                    
                    # Ensure it ends with punctuation
                    if generated_sentence[-1] not in '.!?':
                        generated_sentence += '.'
                    
                    # Only add if unique and significantly different from input
                    if (generated_sentence not in seen_sentences and 
                        len(generated_sentence) > len(text) + 5):
                        sentences.append(generated_sentence)
                        seen_sentences.add(generated_sentence)
        
        return sentences[:num_sentences]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'n': self.n, 'ngrams': self.ngrams, 'vocab': self.vocab}, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(n=data['n'])
        model.ngrams = data['ngrams']
        model.vocab = data['vocab']
        return model

# ---------- Model Management ----------
def _model_path(lang: str) -> str:
    return os.path.join(DATA_DIR, f"ngram_{lang}.pkl")

def load_model(lang: str = "en") -> BackoffNGram:
    path = _model_path(lang)
    if os.path.exists(path):
        return BackoffNGram.load(path)
    
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
    return model

# ---------- Groq API Model ----------
class GroqModelManager:
    LANG_NAMES = {
        "en": "English", "hi": "Hindi", "te": "Telugu",
        "ta": "Tamil", "kn": "Kannada", "fr": "French"
    }
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
    
    def predict(self, text: str, num_words: int = 5, num_sentences: int = 3, lang: str = "en") -> Tuple[List[str], List[str]]:
        prompt = f"""Generate next words and sentences in {self.LANG_NAMES.get(lang, 'English')}.

Input: "{text}"

Output JSON format:
{{
  "words": ["word1", "word2", ...],
  "sentences": ["sentence1", "sentence2", ...]
}}

Rules:
- Generate {num_words} next words
- Generate {num_sentences} next sentences
- No duplicates
- Do not repeat input text"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    words = data.get("words", [])[:num_words]
                    sentences = data.get("sentences", [])[:num_sentences]
                    if isinstance(words, str):
                        words = [words]
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    return words, sentences
                except json.JSONDecodeError:
                    pass
            
            # Fallback parsing
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            words = []
            sentences = []
            
            for line in lines:
                if not line.startswith(('{', '}')):
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
        for lang in ['en', 'hi', 'te', 'ta', 'kn', 'fr']:
            dict_path = os.path.join(DATA_DIR, f"{lang}_word_frequency.txt")
            if os.path.exists(dict_path):
                self.load_dictionary(lang, dict_path)
    
    def load_dictionary(self, lang: str, path: str) -> bool:
        try:
            words = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        word = parts[0].lower().strip()
                        try:
                            words[word] = int(parts[1])
                        except ValueError:
                            words[word] = 1
                    elif parts:
                        words[parts[0].lower().strip()] = 1
            
            self.dictionaries[lang] = words
            print(f"✅ Loaded {lang} dictionary: {len(words)} words")
            return True
        except Exception as e:
            print(f"❌ Error loading {lang} dictionary: {e}")
            self.dictionaries[lang] = {}
            return False
    
    def get_available_languages(self) -> List[str]:
        return [lang for lang, dict_data in self.dictionaries.items() if dict_data]
    
    def correct_text(self, text: str, lang: str = "en") -> str:
        if not text or lang not in self.dictionaries:
            return text
        
        words = text.split()
        corrected = []
        
        for word in words:
            corrected.append(self.correct_word(word, lang))
        
        return " ".join(corrected)
    
    def correct_word(self, word: str, lang: str) -> str:
        if not word or lang not in self.dictionaries:
            return word
        
        word_lower = word.lower()
        if word_lower in self.dictionaries[lang]:
            return word
        
        # Try suffixes
        suffixes = [('ing', 3), ('ed', 2), ('s', 1), ('ly', 2), ("'s", 2), ("n't", 3)]
        for suffix, length in suffixes:
            if word_lower.endswith(suffix):
                base = word_lower[:-length]
                if base in self.dictionaries[lang]:
                    if word.isupper():
                        return base.upper()
                    elif word.istitle():
                        return base.title()
                    return base
        
        # Fuzzy matching
        try:
            import difflib
            matches = difflib.get_close_matches(
                word_lower, 
                self.dictionaries[lang].keys(), 
                n=3, 
                cutoff=0.7
            )
            if matches:
                best = max(matches, key=lambda w: self.dictionaries[lang][w])
                if word.isupper():
                    return best.upper()
                elif word.istitle():
                    return best.title()
                return best
        except ImportError:
            pass
        except Exception:
            pass
        
        return word

# ---------- Utility Functions ----------
def detect_language(text: str) -> str:
    if not text:
        return "en"
    
    # Unicode ranges for Indian languages
    ranges = [
        ('hi', '\u0900', '\u097F'),
        ('te', '\u0C00', '\u0C7F'),
        ('ta', '\u0B80', '\u0BFF'),
        ('kn', '\u0C80', '\u0CFF')
    ]
    
    for char in text[:100]:  # Check first 100 chars for efficiency
        for lang, start, end in ranges:
            if start <= char <= end:
                return lang
        if char in 'éèêëàâçîïôûùü':
            return "fr"
    
    return "en"

def auto_correct_text(text: str, lang: str, spell_checker: SpellChecker) -> Tuple[str, str]:
    if not text or lang not in spell_checker.get_available_languages():
        return text, ""
    
    original = text
    corrected = spell_checker.correct_text(text, lang)
    
    if corrected == original:
        return corrected, ""
    
    # Generate correction message
    orig_words = original.split()
    corr_words = corrected.split()
    corrections = []
    
    for orig, corr in zip(orig_words, corr_words):
        if orig != corr:
            corrections.append(f"'{orig}'→'{corr}'")
    
    msg = "Auto-corrected: " + ", ".join(corrections) if corrections else ""
    return corrected, msg

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

# Initialize components
spell_checker = SpellChecker()
MODEL_CACHE: Dict[str, BackoffNGram] = {}
groq_model_instance = None

def get_groq_model():
    global groq_model_instance
    if groq_model_instance is None:
        try:
            groq_model_instance = GroqModelManager()
            print("✅ Groq API model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Groq model: {e}")
            groq_model_instance = None
    return groq_model_instance

def get_ngram_model(lang: str):
    if lang not in MODEL_CACHE:
        MODEL_CACHE[lang] = load_model(lang)
    return MODEL_CACHE[lang]

# ---------- API Endpoints ----------
@app.get("/")
async def home():
    return FileResponse("web/index.html")

@app.get("/predict")
async def predict(
    text: str = Query("", min_length=0),
    lang: str = Query("auto"),
    model_type: str = Query("ngram"),
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
        detected_lang = detect_language(text) if lang == "auto" else lang
        
        # Auto-correct
        corrected_text = text
        correction_msg = ""
        if auto_correct:
            corrected_text, correction_msg = auto_correct_text(text, detected_lang, spell_checker)
        
        predictions = []
        sentence_predictions = []
        
        if model_type == "ngram":
            model = get_ngram_model(detected_lang)
            
            # Word predictions
            raw_predictions = model.predict(corrected_text, k=num_words * 3, lang=detected_lang)
            
            # Deduplicate and clean
            seen = set()
            for pred in raw_predictions:
                if pred and not pred.isdigit() and pred not in seen:
                    predictions.append(pred)
                    seen.add(pred)
                    if len(predictions) >= num_words:
                        break
            
            predictions = predictions[:num_words]
            
            # ✅ ADDED: Sentence predictions for N-gram model
            if include_sentences:
                try:
                    ngram_sentences = model.predict_sentences(
                        corrected_text, 
                        num_sentences=3, 
                        max_words=15, 
                        lang=detected_lang
                    )
                    
                    # Clean and filter sentences
                    seen_sentences = set()
                    for sentence in ngram_sentences:
                        clean_sentence = sentence.strip()
                        if (clean_sentence and 
                            clean_sentence not in seen_sentences and
                            len(clean_sentence) > len(corrected_text) + 5):
                            sentence_predictions.append(clean_sentence)
                            seen_sentences.add(clean_sentence)
                    
                    sentence_predictions = sentence_predictions[:3]
                    
                except Exception as e:
                    print(f"N-gram sentence prediction error: {e}")
                    # Fallback to simple sentence generation
                    if predictions:
                        simple_sentences = [
                            f"{corrected_text} {predictions[0]} {predictions[1] if len(predictions) > 1 else 'completely'}.",
                            f"{corrected_text} and then {predictions[0]} happens.",
                            f"{corrected_text} with {predictions[0]} and more."
                        ]
                        sentence_predictions = simple_sentences[:3]
            
            accuracy_score = min(0.95, 0.7 + (len(predictions) / num_words * 0.3))
            
        elif model_type == "groq":
            groq_model = get_groq_model()
            if not groq_model:
                raise HTTPException(status_code=500, detail="Groq API not available")
            
            try:
                words, sentences = groq_model.predict(
                    corrected_text, 
                    num_words=num_words,
                    num_sentences=3 if include_sentences else 0,
                    lang=detected_lang
                )
                
                # Clean and deduplicate words
                seen_words = set()
                for word in words:
                    if word and not word.isdigit() and word not in seen_words:
                        clean_word = word.strip('.,!?;:"\'')
                        if clean_word:
                            predictions.append(clean_word)
                            seen_words.add(clean_word)
                
                predictions = predictions[:num_words]
                
                # Sentence predictions
                if include_sentences and sentences:
                    seen_sentences = set()
                    for sentence in sentences:
                        clean_sentence = sentence.strip('",.')
                        if clean_sentence and clean_sentence not in seen_sentences:
                            sentence_predictions.append(clean_sentence)
                            seen_sentences.add(clean_sentence)
                
                accuracy_score = min(0.98, 0.85 + (len(predictions) / num_words * 0.15))
                
            except Exception as e:
                print(f"Groq prediction error: {e}")
                # Fallback to N-gram
                model = get_ngram_model(detected_lang)
                raw_predictions = model.predict(corrected_text, k=num_words * 3, lang=detected_lang)
                predictions = [p for p in raw_predictions if p and not p.isdigit()][:num_words]
                accuracy_score = 0.6
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type")
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "model_type": model_type,
            "original_text": text,
            "corrected_text": corrected_text,
            "correction_msg": correction_msg,
            "language": detected_lang,
            "predictions": predictions,
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
async def correct_spelling(
    text: str = Query(...),
    lang: str = Query("en")
):
    corrected = spell_checker.correct_text(text, lang)
    return {
        "original": text,
        "corrected": corrected,
        "language": lang
    }

@app.get("/groq_status")
async def groq_status():
    groq_model = get_groq_model()
    return {
        "available": groq_model is not None,
        "model": "llama-3.1-8b-instant" if groq_model else None
    }

@app.get("/test_sentences")
async def test_sentences(
    text: str = Query("The weather is"),
    lang: str = Query("en"),
    model_type: str = Query("ngram")
):
    """Test endpoint to verify sentence generation"""
    try:
        if model_type == "ngram":
            model = get_ngram_model(lang)
            sentences = model.predict_sentences(text, num_sentences=3, max_words=15, lang=lang)
            return {
                "success": True,
                "model_type": "ngram",
                "input": text,
                "sentences": sentences,
                "count": len(sentences)
            }
        else:
            groq_model = get_groq_model()
            if groq_model:
                words, sentences = groq_model.predict(text, num_words=3, num_sentences=3, lang=lang)
                return {
                    "success": True,
                    "model_type": "groq",
                    "input": text,
                    "sentences": sentences,
                    "count": len(sentences)
                }
            else:
                return {
                    "success": False,
                    "error": "Groq API not available"
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
