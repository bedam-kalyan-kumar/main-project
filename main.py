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
# ... (imports and setup) ...

class GroqModelManager:
    LANG_NAMES = {
        "en": "English", "hi": "Hindi", "te": "Telugu",
        "ta": "Tamil", "kn": "Kannada", "fr": "French"
    }

    # Enhanced prompts with few-shot examples
    LANGUAGE_PROMPTS = {
        "en": """You are a multilingual text predictor. Given the input text, predict the next possible words and short continuations.

Input: "{text}"

Generate {num_words} most likely next single words and {num_sentences} short phrase continuations (2‑5 words) that naturally follow.

Output only valid JSON in this exact format:
{{"words": ["word1", "word2", ...], "continuations": ["phrase1", "phrase2", ...]}}

Examples:
Input: "The weather is"
Output: {{"words": ["nice", "cold", "beautiful", "hot", "good"], "continuations": ["very nice today", "getting colder", "perfect for a walk"]}}

Input: "I am going to"
Output: {{"words": ["the", "work", "school", "store", "sleep"], "continuations": ["the store later", "work tomorrow", "school right now"]}}

Now for the given input:""",

        "hi": """आप एक बहुभाषी पाठ भविष्यवक्ता हैं। दिए गए इनपुट पाठ के आधार पर, अगले संभावित शब्दों और छोटे वाक्यांशों की भविष्यवाणी करें।

इनपुट: "{text}"

{num_words} सबसे संभावित अगले एकल शब्द और {num_sentences} छोटे वाक्यांश (2‑5 शब्द) उत्पन्न करें जो स्वाभाविक रूप से अनुसरण करते हैं।

केवल JSON आउटपुट दें इस सटीक प्रारूप में:
{{"words": ["शब्द1", "शब्द2", ...], "continuations": ["वाक्यांश1", "वाक्यांश2", ...]}}

उदाहरण:
इनपुट: "मौसम आज"
आउटपुट: {{"words": ["बहुत", "अच्छा", "गर्म", "सुहावना", "खराब"], "continuations": ["बहुत अच्छा है", "गर्म हो गया", "सुहावना लग रहा"]}}

इनपुट: "मैं जा रहा"
आउटपुट: {{"words": ["हूँ", "था", "रहा", "चुका", "सकता"], "continuations": ["हूँ बाजार", "था स्कूल", "रहा हूँ घर"]}}

अब दिए गए इनपुट के लिए:""",

        "te": """మీరు బహుభాషా టెక్స్ట్ ప్రిడిక్టర్. ఇచ్చిన ఇన్‌పుట్ టెక్స్ట్ ఆధారంగా, తదుపరి సాధ్యమయ్యే పదాలు మరియు చిన్న కొనసాగింపులను అంచనా వేయండి.

ఇన్‌పుట్: "{text}"

{num_words} అత్యంత సంభావ్య తదుపరి ఒకే పదాలు మరియు {num_sentences} చిన్న పదబంధ కొనసాగింపులు (2‑5 పదాలు) ఉత్పత్తి చేయండి.

ఖచ్చితమైన JSON ఆకృతిలో మాత్రమే అవుట్‌పుట్ ఇవ్వండి:
{{"words": ["పదం1", "పదం2", ...], "continuations": ["పదబంధం1", "పదబంధం2", ...]}}

ఉదాహరణలు:
ఇన్‌పుట్: "వాతావరణం ఈ"
అవుట్‌పుట్: {{"words": ["రోజు", "చాలా", "బాగుంది", "వేడిగా", "చల్లగా"], "continuations": ["రోజు చాలా బాగుంది", "వేడిగా ఉంది", "చల్లగా ఉంది"]}}

ఇప్పుడు ఇచ్చిన ఇన్‌పుట్ కోసం:""",

        "ta": """நீங்கள் ஒரு பன்மொழி உரை முன்கணிப்பாளர். கொடுக்கப்பட்ட உள்ளீட்டு உரையின் அடிப்படையில், அடுத்த சாத்தியமான சொற்கள் மற்றும் குறுகிய தொடர்ச்சிகளைக் கணிக்கவும்.

உள்ளீடு: "{text}"

{num_words} மிகவும் சாத்தியமான அடுத்த ஒற்றைச் சொற்கள் மற்றும் {num_sentences} குறுகிய சொற்றொடர் தொடர்ச்சிகள் (2‑5 சொற்கள்) உருவாக்கவும்.

துல்லியமான JSON வடிவத்தில் மட்டும் வெளியீடு:
{{"words": ["சொல்1", "சொல்2", ...], "continuations": ["சொற்றொடர்1", "சொற்றொடர்2", ...]}}

எடுத்துக்காட்டுகள்:
உள்ளீடு: "வானிலை இன்று"
வெளியீடு: {{"words": ["மிகவும்", "நன்றாக", "வெயிலாக", "குளிர்ச்சியாக", "மோசமாக"], "continuations": ["மிகவும் நன்றாக உள்ளது", "வெயிலாக இருக்கிறது", "குளிர்ச்சியாக உள்ளது"]}}

இப்போது கொடுக்கப்பட்ட உள்ளீட்டிற்கு:""",

        "kn": """ನೀವು ಬಹುಭಾಷಾ ಪಠ್ಯ ಭವಿಷ್ಯಕಾರ. ನೀಡಿದ ಇನ್‌ಪುಟ್ ಪಠ್ಯದ ಆಧಾರದ ಮೇಲೆ, ಮುಂದಿನ ಸಂಭವನೀಯ ಪದಗಳು ಮತ್ತು ಸಣ್ಣ ಮುಂದುವರಿಕೆಗಳನ್ನು ಊಹಿಸಿ.

ಇನ್‌ಪುಟ್: "{text}"

{num_words} ಅತ್ಯಂತ ಸಂಭವನೀಯ ಮುಂದಿನ ಏಕ ಪದಗಳು ಮತ್ತು {num_sentences} ಸಣ್ಣ ನುಡಿಗಟ್ಟು ಮುಂದುವರಿಕೆಗಳನ್ನು (2‑5 ಪದಗಳು) ರಚಿಸಿ.

ನಿಖರವಾದ JSON ಸ್ವರೂಪದಲ್ಲಿ ಮಾತ್ರ ಔಟ್‌ಪುಟ್ ನೀಡಿ:
{{"words": ["ಪದ1", "ಪದ2", ...], "continuations": ["ನುಡಿಗಟ್ಟು1", "ನುಡಿಗಟ್ಟು2", ...]}}

ಉದಾಹರಣೆಗಳು:
ಇನ್‌ಪುಟ್: "ಹವಾಮಾನ ಇಂದು"
ಔಟ್‌ಪುಟ್: {{"words": ["ತುಂಬಾ", "ಚೆನ್ನಾಗಿದೆ", "ಬಿಸಿಯಾಗಿದೆ", "ತಂಪಾಗಿದೆ", "ಕೆಟ್ಟದಾಗಿದೆ"], "continuations": ["ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ", "ಬಿಸಿಯಾಗಿದೆ", "ತಂಪಾಗಿದೆ"]}}

ಈಗ ನೀಡಿದ ಇನ್‌ಪುಟ್‌ಗಾಗಿ:""",

        "fr": """Vous êtes un prédicteur de texte multilingue. Sur la base du texte d'entrée donné, prédisez les prochains mots possibles et les courtes continuations.

Texte d'entrée : "{text}"

Générez {num_words} prochains mots uniques les plus probables et {num_sentences} courtes continuations de phrases (2‑5 mots).

Format de sortie JSON uniquement :
{{"words": ["mot1", "mot2", ...], "continuations": ["phrase1", "phrase2", ...]}}

Exemples :
Entrée : "Le temps est"
Sortie : {{"words": ["beau", "mauvais", "nuageux", "froid", "chaud"], "continuations": ["très beau aujourd'hui", "devenu mauvais", "agréable pour une promenade"]}}

Maintenant pour le texte donné :"""
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
        if current_time - self.last_request_time < self.min_request_interval:
            time.sleep(self.min_request_interval - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def _clean_json_response(self, content):
        import re
        # Remove markdown fences
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```$', '', content, flags=re.MULTILINE)
        # Extract JSON object
        match = re.search(r'\{.*\}', content, re.DOTALL)
        return match.group(0) if match else content

    def predict(self, text: str, num_words: int = 5, num_sentences: int = 3, lang: str = "en") -> Tuple[List[str], List[str]]:
        # Get language-specific prompt (fallback to English if not defined)
        prompt = self.LANGUAGE_PROMPTS.get(lang, self.LANGUAGE_PROMPTS["en"]).format(
            text=text, num_words=num_words, num_sentences=num_sentences
        )
        print(f"Groq request [{lang}]: {text}")
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
            print(f"Groq response (first 200 chars): {content[:200]}")
            cleaned = self._clean_json_response(content)
            try:
                data = json.loads(cleaned)
                words = data.get("words", [])
                conts = data.get("continuations", [])
                # Ensure lists, clean strings
                if isinstance(words, str):
                    words = [words]
                if isinstance(conts, str):
                    conts = [conts]
                # Filter out any empty or placeholder items
                words = [w.strip() for w in words if w and w.strip()][:num_words]
                conts = [c.strip() for c in conts if c and c.strip()][:num_sentences]
                print(f"Parsed words: {words}, continuations: {conts}")
                return words, conts
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # Try to extract manually if JSON failed
                words = self._extract_words_fallback(content, num_words, lang)
                conts = self._extract_sentences_fallback(content, num_sentences, lang)
                return words, conts
        except Exception as e:
            print(f"Groq API error: {e}")
            return [], []

    def _extract_words_fallback(self, text, num_words, lang):
        """Fallback extraction if JSON fails."""
        import re
        # Look for quoted words
        words = re.findall(r'"([^"]+)"', text)
        if not words:
            # Try to find words after numbers or bullets
            lines = text.split('\n')
            for line in lines:
                if re.match(r'^[\d\.\-\*•]\s+', line):
                    word = re.sub(r'^[\d\.\-\*•]\s+', '', line).strip().strip('",.')
                    if word:
                        words.append(word)
                elif line.strip() and not line.startswith('{') and not line.startswith('}'):
                    word = line.strip().strip('",.')
                    if word:
                        words.append(word)
        # Clean non-alphabetic tokens if English
        if lang == "en":
            words = [w for w in words if re.match(r'^[a-zA-Z]+$', w)]
        return words[:num_words]

    def _extract_sentences_fallback(self, text, num_sentences, lang):
        import re
        sentences = re.findall(r'"([^"]+)"', text)
        if not sentences:
            lines = text.split('\n')
            for line in lines:
                if re.match(r'^[\d\.\-\*•]\s+', line):
                    sent = re.sub(r'^[\d\.\-\*•]\s+', '', line).strip().strip('",.')
                    if sent:
                        sentences.append(sent)
                elif line.strip() and not line.startswith('{') and not line.startswith('}'):
                    if len(line.split()) > 2:
                        sentences.append(line.strip().strip('",.'))
        return sentences[:num_sentences]

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
