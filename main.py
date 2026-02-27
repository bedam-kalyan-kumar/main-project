from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Tuple
import time
import re
import os
import json
from groq import Groq
from dotenv import load_dotenv
# Just this one import line!
from app.ngram import BackoffNGram, load_model, tokenize

# ✅ Import from ngram.py instead of redefining
from app.ngram import BackoffNGram, load_model, tokenize

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

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
        prompt = f"""Generate only the next words and continuations in {self.LANG_NAMES.get(lang, 'English')}.

        Input text: "{text}"

          TASK 1 - Next words: Generate {num_words} single words that could come next
         TASK 2 - Continuations: Generate {num_sentences} short phrases (2-5 words) that continue this text, WITHOUT repeating the input

         Output ONLY in this JSON format:
         {{
           "words": ["word1", "word2", "word3"],
         "continuations": ["phrase 1", "phrase 2", "phrase 3"]
          }}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    words = data.get("words", [])[:num_words]
                    conts = data.get("continuations", [])[:num_sentences]
                    return words, conts
                except:
                    pass
            
            return [], []
            
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
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            words[parts[0].lower()] = int(parts[1])
                        except:
                            words[parts[0].lower()] = 1
                    else:
                        words[parts[0].lower()] = 1
            
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
        
        import difflib      #To find similler words
        matches = difflib.get_close_matches(
            word_lower, 
            self.dictionaries[lang].keys(), 
            n=3, 
            cutoff=0.7
        )
        
        if matches:
            best = max(matches, key=lambda w: self.dictionaries[lang].get(w, 1))
            if word.isupper():
                return best.upper()
            elif word.istitle():
                return best.title()
            return best
        
        return word

# ---------- Utility Functions ----------
def detect_language(text: str) -> str:
    if not text:
        return "en"
    
    ranges = [
        ('hi', '\u0900', '\u097F'),
        ('te', '\u0C00', '\u0C7F'),
        ('ta', '\u0B80', '\u0BFF'),
        ('kn', '\u0C80', '\u0CFF')
    ]
    
    for char in text[:100]:
        for lang, start, end in ranges:
            if start <= char <= end:
                return lang
        if char in 'éèêëàâçîïôûùü':
            return "fr"
    
    return "en"

def auto_correct_text(text: str, lang: str, spell_checker: SpellChecker) -> Tuple[str, str]:
    if not text or lang not in spell_checker.get_available_languages():
        return text, ""
    
    corrected = spell_checker.correct_text(text, lang)
    if corrected == text:
        return corrected, ""
    
    return corrected, "Auto-corrected"

# ---------- FastAPI App ----------
app = FastAPI(title="Smart Text Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from ANY website ( * means all)
    allow_credentials=True,  #Allow cookies/auth headers to be sent
    allow_methods=["*"],   #Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],   #Allow all headers in requests
)

spell_checker = SpellChecker()
MODEL_CACHE: Dict[str, BackoffNGram] = {}
groq_model_instance = None

def get_groq_model():
    global groq_model_instance
    if groq_model_instance is None:
        try:
            groq_model_instance = GroqModelManager()
            print("✅ Groq API model loaded")
        except Exception as e:
            print(f"❌ Failed to load Groq model: {e}")
            groq_model_instance = None
    return groq_model_instance

def get_ngram_model(lang: str):
    if lang not in MODEL_CACHE:
        MODEL_CACHE[lang] = load_model(lang)  # ✅ Now using imported function
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
    num_words: int = Query(8, ge=1, le=15)
):
    start_time = time.time()
    
    try:
        if not text.strip():
            return {
                "success": True,
                "predictions": [],
                "sentence_predictions": [],
                "language": "en",
                "response_time_ms": 0
            }
        
        detected_lang = detect_language(text) if lang == "auto" else lang
        
        corrected_text = text
        correction_msg = ""
        if auto_correct:
            corrected_text, correction_msg = auto_correct_text(text, detected_lang, spell_checker)
        
        predictions = []
        sentence_predictions = []
        
        if model_type == "ngram":
            model = get_ngram_model(detected_lang)
            
            # Word predictions
            raw_preds = model.predict(corrected_text, k=num_words * 2, lang=detected_lang)
            seen = set()
            for p in raw_preds:
                if p and p not in seen and not p.isdigit():
                    predictions.append(p)
                    seen.add(p)
                    if len(predictions) >= num_words:
                        break
            
            # Get continuations (5 sentences)
            if include_sentences:
                try:
                    continuations = model.predict_continuations(
                        corrected_text,
                        num_continuations=5,
                        max_words=5,
                        lang=detected_lang
                    )
                    sentence_predictions = continuations[:5]
                except Exception as e:
                    print(f"Continuation error: {e}")
                    # Simple fallback using word predictions
                    if predictions:
                        sentence_predictions = [
                            f"{predictions[0]} to",
                            f"{predictions[0]} and",
                            f"{predictions[0]} with",
                            f"going to {predictions[0]}",
                            f"{predictions[0]} now"
                        ][:5]
            
            accuracy = 0.8
            
        elif model_type == "groq":
            groq_model = get_groq_model()
            if groq_model:
                words, conts = groq_model.predict(
                    corrected_text,
                    num_words=num_words,
                    num_sentences=5,
                    lang=detected_lang
                )
                predictions = words[:num_words]
                sentence_predictions = conts[:5]
                accuracy = 0.9
            else:
                # Fallback to n-gram
                model = get_ngram_model(detected_lang)
                predictions = model.predict(corrected_text, k=num_words, lang=detected_lang)
                accuracy = 0.6
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "model_type": model_type,
            "original_text": text,
            "corrected_text": corrected_text,
            "correction_msg": correction_msg,
            "language": detected_lang,
            "predictions": predictions[:num_words],
            "sentence_predictions": sentence_predictions[:5],
            "accuracy_score": accuracy,
            "response_time_ms": response_time
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            "success": False,
            "error": str(e),
            "predictions": [],
            "sentence_predictions": []
        }

@app.get("/correct")
async def correct_spelling(text: str = Query(...), lang: str = Query("en")):
    corrected = spell_checker.correct_text(text, lang)
    return {"original": text, "corrected": corrected}

@app.get("/groq_status")
async def groq_status():
    return {"available": get_groq_model() is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
