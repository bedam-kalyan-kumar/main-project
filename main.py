from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Tuple
import time
import re
import os
import json
import uuid
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Union

from app.ngram import BackoffNGram, load_model, tokenize
from database import add_history_entry, get_history, delete_history_entry, clear_all_history

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

SESSION_ID = str(uuid.uuid4())[:8]

# ---------- Groq Model ----------
class GroqModelManager:
    LANG_NAMES = {
        "en": "English", "hi": "Hindi", "te": "Telugu",
        "ta": "Tamil", "kn": "Kannada", "fr": "French"
    }
    
    LANGUAGE_PROMPTS = {
        "en": """Generate only the next words and continuations in English.
Input text: "{text}"
TASK 1 - Next words: Generate {num_words} single words that could come next
TASK 2 - Continuations: Generate {num_sentences} short phrases (2-5 words) that continue this text
Output ONLY in JSON: {{"words": ["w1","w2"], "continuations": ["p1","p2"]}}""",
        "hi": """केवल अगले शब्द और वाक्यांश हिंदी में उत्पन्न करें।
इनपुट: "{text}"
कार्य 1 - अगले शब्द: {num_words} शब्द
कार्य 2 - निरंतरता: {num_sentences} वाक्यांश
JSON: {{"words": ["शब्द1"], "continuations": ["वाक्यांश1"]}}""",
        "te": """తదుపరి పదాలు మరియు కొనసాగింపులను తెలుగులో మాత్రమే రూపొందించండి.
ఇన్పుట్: "{text}"
పని 1 - తదుపరి పదాలు: {num_words} పదాలు
పని 2 - కొనసాగింపులు: {num_sentences} వాక్యాలు
JSON: {{"words": ["పదం1"], "continuations": ["వాక్యం1"]}}""",
        "ta": """அடுத்த சொற்கள் மற்றும் தொடர்ச்சிகளை தமிழில் மட்டும் உருவாக்கவும்.
உள்ளீடு: "{text}"
பணி 1 - அடுத்த சொற்கள்: {num_words} சொற்கள்
பணி 2 - தொடர்ச்சிகள்: {num_sentences} சொற்றொடர்கள்
JSON: {{"words": ["சொல்1"], "continuations": ["சொற்றொடர்1"]}}""",
        "kn": """ಮುಂದಿನ ಪದಗಳು ಮತ್ತು ಮುಂದುವರಿಕೆಗಳನ್ನು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ರಚಿಸಿ.
ಇನ್ಪುಟ್: "{text}"
ಕಾರ್ಯ 1 - ಮುಂದಿನ ಪದಗಳು: {num_words} ಪದಗಳು
ಕಾರ್ಯ 2 - ಮುಂದುವರಿಕೆಗಳು: {num_sentences} ನುಡಿಗಟ್ಟುಗಳು
JSON: {{"words": ["ಪದ1"], "continuations": ["ನುಡಿಗಟ್ಟು1"]}}""",
        "fr": """Générez uniquement les prochains mots et continuations en français.
Texte : "{text}"
TÂCHE 1 - Mots suivants : {num_words} mots
TÂCHE 2 - Continuations : {num_sentences} phrases
JSON: {{"words": ["mot1"], "continuations": ["phrase1"]}}"""
    }
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY")
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"
        self.last_request_time = 0
        self.min_request_interval = 1.0
    
    def _wait_for_rate_limit(self):
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _clean_json_response(self, content):
        import re
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```$', '', content, flags=re.MULTILINE)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        return json_match.group(0) if json_match else content
    
    def predict(self, text: str, num_words: int = 5, num_sentences: int = 3, lang: str = "en") -> Tuple[List[str], List[str]]:
        prompt = self.LANGUAGE_PROMPTS.get(lang, self.LANGUAGE_PROMPTS["en"]).format(
            text=text, num_words=num_words, num_sentences=num_sentences
        )
        self._wait_for_rate_limit()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                timeout=30
            )
            content = response.choices[0].message.content
            cleaned = self._clean_json_response(content)
            try:
                data = json.loads(cleaned)
                words = data.get("words", [])[:num_words]
                conts = data.get("continuations", [])[:num_sentences]
                return words, conts
            except:
                return [], []
        except Exception:
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
    
    def load_dictionary(self, lang: str, path: str):
        words = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split()
                    words[parts[0].lower()] = int(parts[1]) if len(parts) >= 2 else 1
            self.dictionaries[lang] = words
        except:
            self.dictionaries[lang] = {}
    
    def get_available_languages(self) -> List[str]:
        return [lang for lang, d in self.dictionaries.items() if d]
    
    def correct_text(self, text: str, lang: str = "en") -> str:
        if not text or lang not in self.dictionaries: return text
        words = text.split()
        corrected = []
        for word in words:
            corrected.append(self.correct_word(word, lang)[0])
        return " ".join(corrected)
    
    def correct_word(self, word: str, lang: str) -> Tuple[str, bool]:
        if not word or lang not in self.dictionaries: return word, False
        word_lower = word.lower()
        if word_lower in self.dictionaries[lang]: return word, False
        import difflib
        matches = difflib.get_close_matches(word_lower, self.dictionaries[lang].keys(), n=3, cutoff=0.7)
        if matches:
            best = max(matches, key=lambda w: self.dictionaries[lang].get(w, 1))
            if word.isupper(): return best.upper(), True
            if word.istitle(): return best.title(), True
            return best, True
        return word, False
    
    def get_suggestions(self, word: str, lang: str, max_suggestions: int = 3) -> List[str]:
        if not word or lang not in self.dictionaries:
            return []
        word_lower = word.lower()
        if word_lower in self.dictionaries[lang]:
            return []
        import difflib
        matches = difflib.get_close_matches(word_lower, self.dictionaries[lang].keys(), n=max_suggestions, cutoff=0.6)
        return matches

# ---------- Utilities ----------
def detect_language(text: str) -> str:
    if not text: return "en"
    ranges = [('hi', '\u0900', '\u097F'), ('te', '\u0C00', '\u0C7F'),
              ('ta', '\u0B80', '\u0BFF'), ('kn', '\u0C80', '\u0CFF')]
    for char in text[:100]:
        for lang, start, end in ranges:
            if start <= char <= end: return lang
        if char in 'éèêëàâçîïôûùü': return "fr"
    return "en"

def auto_correct_text(text: str, lang: str, spell_checker: SpellChecker) -> Tuple[str, str, List[Tuple[str, str, str]]]:
    if not text or lang not in spell_checker.get_available_languages(): return text, "", []
    words = text.split()
    corrected_words = []
    corrections = []
    for word in words:
        corrected, was = spell_checker.correct_word(word, lang)
        corrected_words.append(corrected)
        if was: corrections.append((word, corrected, lang))
    return " ".join(corrected_words), "Auto-corrected" if corrections else "", corrections

# ---------- FastAPI App ----------
app = FastAPI(title="Smart Text Predictor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

spell_checker = SpellChecker()
MODEL_CACHE: Dict[str, BackoffNGram] = {}
groq_model_instance = None

def get_groq_model():
    global groq_model_instance
    if groq_model_instance is None:
        try: groq_model_instance = GroqModelManager()
        except: groq_model_instance = None
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
    text: str = Query(""),
    lang: str = Query("auto"),
    model_type: str = Query("ngram"),
    auto_correct: Union[bool, str] = Query(True),
    include_sentences: Union[bool, str] = Query(True),
    num_words: int = Query(50, ge=1, le=500)
):
    start = time.time()
    if not text.strip():
        return {"success": True, "predictions": [], "sentence_predictions": [], "language": "en", "response_time_ms": 0}
    
    detected_lang = detect_language(text) if lang == "auto" else lang
    corrected_text, correction_msg, corrections = auto_correct_text(text, detected_lang, spell_checker) if auto_correct else (text, "", [])
    
    predictions, sentence_predictions, accuracy = [], [], 0.0
    if model_type == "ngram":
        model = get_ngram_model(detected_lang)
        raw = model.predict(corrected_text, k=num_words * 2, lang=detected_lang)
        seen = set()
        for p in raw:
            if p and p not in seen and not p.isdigit():
                predictions.append(p); seen.add(p)
                if len(predictions) >= num_words: break
        if include_sentences:
            try:
                sentence_predictions = model.predict_continuations(corrected_text, num_continuations=5, max_words=5, lang=detected_lang)
            except:
                sentence_predictions = [f"{p} to" for p in predictions[:3]] if predictions else []
        accuracy = 0.8
    elif model_type == "groq":
        gm = get_groq_model()
        if gm:
            words, conts = gm.predict(corrected_text, num_words=num_words, num_sentences=5, lang=detected_lang)
            predictions, sentence_predictions = words[:num_words], conts[:5]
            accuracy = 0.9
        else:
            model = get_ngram_model(detected_lang)
            predictions = model.predict(corrected_text, k=num_words, lang=detected_lang)
            accuracy = 0.6
    
    return {
        "success": True, "model_type": model_type, "original_text": text, "corrected_text": corrected_text,
        "correction_msg": correction_msg, "corrections": corrections, "language": detected_lang,
        "predictions": predictions[:num_words], "sentence_predictions": sentence_predictions[:5],
        "accuracy_score": accuracy, "response_time_ms": round((time.time() - start) * 1000, 2)
    }

@app.post("/save_history")
async def save_history(request: dict):
    try:
        hid = add_history_entry(
            input_text=request.get("text", ""), language=request.get("language", "en"),
            model_used=request.get("model_used", "ngram"), predictions=request.get("predictions", [])[:10],
            continuations=[], user_session=SESSION_ID
        )
        return {"success": bool(hid), "id": hid}
    except:
        return {"success": False}

@app.get("/history")
async def get_history_endpoint(limit: int = 50):
    return {"success": True, "history": get_history(limit, SESSION_ID)}

@app.delete("/history/{entry_id}")
async def delete_history(entry_id: int):
    return {"success": delete_history_entry(entry_id)}

@app.delete("/history")
async def clear_history():
    return {"success": clear_all_history(SESSION_ID) > 0}

@app.get("/correct")
async def correct_spelling(
    text: str = Query(...),
    lang: str = Query("en"),
    max_suggestions: int = Query(3, ge=1, le=10)
):
    """Return spelling suggestions for a word or correct a phrase."""
    if " " in text:
        corrected = spell_checker.correct_text(text, lang)
        return {"original": text, "corrected": corrected, "suggestions": []}
    else:
        suggestions = spell_checker.get_suggestions(text, lang, max_suggestions)
        # For backward compatibility, also provide the best match as corrected
        corrected = suggestions[0] if suggestions else text
        return {"original": text, "corrected": corrected, "suggestions": suggestions}

@app.get("/groq_status")
async def groq_status():
    return {"available": get_groq_model() is not None}

@app.get("/check_sequence")
async def check_sequence(sequence: str = Query(...), lang: str = "en"):
    model = get_ngram_model(lang)
    words = sequence.lower().split()
    if len(words) >= 3:
        context = tuple(words[-3:-1])
        next_word = words[-1]
        exists = context in model.ngrams[2] and next_word in model.ngrams[2][context]
        return {"exists": exists}
    return {"exists": False}

class AddSequenceRequest(BaseModel):
    sequence: str
    lang: str

@app.post("/add_sequence_newline")
async def add_sequence_newline(request: AddSequenceRequest):
    try:
        sequence = request.sequence.strip().lower()
        lang = request.lang
        words = sequence.split()
        if len(words) < 3:
            return {"success": False, "message": "At least 3 words required"}
        
        model = get_ngram_model(lang)
        for i in range(len(words)):
            word = words[i]
            model.vocab[word] = model.vocab.get(word, 0) + 1
            model.ngrams[0][()][word] = model.ngrams[0][()].get(word, 0) + 1
            if i > 0:
                ctx = tuple([words[i-1]])
                model.ngrams[1][ctx][word] = model.ngrams[1][ctx].get(word, 0) + 1
            if i > 1:
                ctx = tuple(words[i-2:i])
                model.ngrams[2][ctx][word] = model.ngrams[2][ctx].get(word, 0) + 1
        
        model.save(os.path.join(DATA_DIR, f"ngram_{lang}.pkl"))
        with open(os.path.join(DATA_DIR, f"{lang}.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{sequence}")
        return {"success": True, "message": f"Added '{sequence}'"}
    except Exception as e:
        return {"success": False, "message": str(e)}
