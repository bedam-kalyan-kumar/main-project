from fastapi import FastAPI, Query
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
from typing import Union  # Add this with other imports

# Import from ngram.py
from app.ngram import BackoffNGram, load_model, tokenize

# Import database functions
from database import add_history_entry, get_history, delete_history_entry, clear_all_history

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "app", "data")

# Generate a session ID for this user
SESSION_ID = str(uuid.uuid4())[:8]

# Track last saved input to prevent duplicates
last_saved_input = {"text": "", "timestamp": 0}

# ---------- Groq API Model ----------
# ---------- Groq API Model with Full Multi-Language Support ----------
class GroqModelManager:
    LANG_NAMES = {
        "en": "English",
        "hi": "Hindi", 
        "te": "Telugu",
        "ta": "Tamil", 
        "kn": "Kannada", 
        "fr": "French"
    }
    
    # Language-specific prompts for better results
    LANGUAGE_PROMPTS = {
        "en": """Generate only the next words and continuations in English.

Input text: "{text}"

TASK 1 - Next words: Generate {num_words} single words that could come next
TASK 2 - Continuations: Generate {num_sentences} short phrases (2-5 words) that continue this text, WITHOUT repeating the input

Output ONLY in this JSON format:
{{
    "words": ["word1", "word2", "word3"],
    "continuations": ["phrase 1", "phrase 2", "phrase 3"]
}}""",

        "hi": """केवल अगले शब्द और वाक्यांश हिंदी में उत्पन्न करें।

इनपुट टेक्स्ट: "{text}"

कार्य 1 - अगले शब्द: {num_words} एकल शब्द उत्पन्न करें जो आगे आ सकते हैं
कार्य 2 - निरंतरता: {num_sentences} छोटे वाक्यांश (2-5 शब्द) उत्पन्न करें जो इस टेक्स्ट को जारी रखते हैं, बिना इनपुट दोहराए

केवल इस JSON फॉर्मेट में आउटपुट दें:
{{
    "words": ["शब्द1", "शब्द2", "शब्द3"],
    "continuations": ["वाक्यांश 1", "वाक्यांश 2", "वाक्यांश 3"]
}}""",

        "te": """తదుపరి పదాలు మరియు కొనసాగింపులను తెలుగులో మాత్రమే రూపొందించండి.

ఇన్పుట్ టెక్స్ట్: "{text}"

పని 1 - తదుపరి పదాలు: {num_words} ఒకే పదాలను రూపొందించండి
పని 2 - కొనసాగింపులు: {num_sentences} చిన్న వాక్యాలు (2-5 పదాలు) రూపొందించండి

ఈ JSON ఫార్మాట్లో మాత్రమే అవుట్పుట్ ఇవ్వండి:
{{
    "words": ["పదం1", "పదం2", "పదం3"],
    "continuations": ["వాక్యం 1", "వాక్యం 2", "వాక్యం 3"]
}}""",

        "ta": """அடுத்த சொற்கள் மற்றும் தொடர்ச்சிகளை தமிழில் மட்டும் உருவாக்கவும்.

உள்ளீடு உரை: "{text}"

பணி 1 - அடுத்த சொற்கள்: {num_words} ஒற்றை சொற்களை உருவாக்கவும்
பணி 2 - தொடர்ச்சிகள்: {num_sentences} குறுகிய சொற்றொடர்களை (2-5 சொற்கள்) உருவாக்கவும்

இந்த JSON வடிவத்தில் மட்டும் வெளியீடு:
{{
    "words": ["சொல்1", "சொல்2", "சொல்3"],
    "continuations": ["சொற்றொடர் 1", "சொற்றொடர் 2", "சொற்றொடர் 3"]
}}""",

        "kn": """ಮುಂದಿನ ಪದಗಳು ಮತ್ತು ಮುಂದುವರಿಕೆಗಳನ್ನು ಕನ್ನಡದಲ್ಲಿ ಮಾತ್ರ ರಚಿಸಿ.

ಇನ್ಪುಟ್ ಪಠ್ಯ: "{text}"

ಕಾರ್ಯ 1 - ಮುಂದಿನ ಪದಗಳು: {num_words} ಒಂದೇ ಪದಗಳನ್ನು ರಚಿಸಿ
ಕಾರ್ಯ 2 - ಮುಂದುವರಿಕೆಗಳು: {num_sentences} ಸಣ್ಣ ನುಡಿಗಟ್ಟುಗಳನ್ನು (2-5 ಪದಗಳು) ರಚಿಸಿ

ಈ JSON ಸ್ವರೂಪದಲ್ಲಿ ಮಾತ್ರ ಔಟ್ಪುಟ್ ನೀಡಿ:
{{
    "words": ["ಪದ1", "ಪದ2", "ಪದ3"],
    "continuations": ["ನುಡಿಗಟ್ಟು 1", "ನುಡಿಗಟ್ಟು 2", "ನುಡಿಗಟ್ಟು 3"]
}}""",

        "fr": """Générez uniquement les prochains mots et continuations en français.

Texte d'entrée : "{text}"

TÂCHE 1 - Mots suivants : Générez {num_words} mots simples qui pourraient suivre
TÂCHE 2 - Continuations : Générez {num_sentences} courtes phrases (2-5 mots) qui continuent ce texte

Format JSON uniquement :
{{
    "words": ["mot1", "mot2", "mot3"],
    "continuations": ["phrase 1", "phrase 2", "phrase 3"]
}}"""
    }
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        self.client = Groq(api_key=api_key)
        # Use a currently active model with good multilingual support
        self.model_name = "llama-3.3-70b-versatile"  # Latest version
        # Alternative: "mixtral-8x7b-32768" also has excellent multilingual performance
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed rate limits"""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _clean_json_response(self, content):
        """Remove markdown fences and extract JSON"""
        import re
        
        # Remove markdown code fences
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'^```\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'```$', '', content, flags=re.MULTILINE)
        
        # Remove XML-like function tags
        content = re.sub(r'<function=\w+>', '', content)
        content = re.sub(r'</function>', '', content)
        
        # Find JSON object in the cleaned content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return content
    
    def _get_language_name(self, lang_code):
        """Get full language name for prompt"""
        return self.LANG_NAMES.get(lang_code, "English")
    
    def _get_prompt(self, text, num_words, num_sentences, lang):
        """Get language-specific prompt"""
        if lang in self.LANGUAGE_PROMPTS:
            return self.LANGUAGE_PROMPTS[lang].format(
                text=text,
                num_words=num_words,
                num_sentences=num_sentences
            )
        else:
            # Fallback to English prompt
            return self.LANGUAGE_PROMPTS["en"].format(
                text=text,
                num_words=num_words,
                num_sentences=num_sentences
            )
    
    def predict(self, text: str, num_words: int = 5, num_sentences: int = 3, lang: str = "en") -> Tuple[List[str], List[str]]:
        """Generate predictions with full multi-language support"""
        
        # Get language-specific prompt
        prompt = self._get_prompt(text, num_words, num_sentences, lang)
        
        # Rate limiting
        self._wait_for_rate_limit()
        
        try:
            # Add timeout to prevent hanging
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,  # Increased for non-English languages
                timeout=30  # 30 second timeout
            )
            
            content = response.choices[0].message.content
            
            # Clean the response
            cleaned_content = self._clean_json_response(content)
            
            try:
                data = json.loads(cleaned_content)
                words = data.get("words", [])[:num_words]
                conts = data.get("continuations", [])[:num_sentences]
                
                # Validate we got actual data
                if words or conts:
                    return words, conts
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {lang}: {e}")
                print(f"Raw content: {content[:200]}...")
                
                # Fallback: Try to extract words manually based on language
                words = self._extract_words_fallback(content, num_words, lang)
                conts = self._extract_sentences_fallback(content, num_sentences, lang)
                if words or conts:
                    return words, conts
            
            return [], []
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle specific error types
            if "rate limit" in error_str or "429" in error_str:
                print(f"⚠️ Groq rate limit hit. Waiting longer...")
                self.min_request_interval = min(self.min_request_interval * 2, 5.0)
                return [], []
                
            elif "timeout" in error_str or "timed out" in error_str:
                print(f"⏱️ Groq timeout for {lang}: {e}")
                return [], []
                
            elif "authentication" in error_str or "api key" in error_str:
                print(f"🔑 Groq authentication error: {e}")
                return [], []
                
            else:
                print(f"❌ Groq API error for {lang}: {e}")
                return [], []
    
    def _extract_words_fallback(self, text, num_words, lang="en"):
        """Fallback method to extract words from text when JSON fails"""
        import re
        
        # Try to find quoted strings (works for all languages)
        words = re.findall(r'"([^"]+)"', text)
        
        if not words:
            # Try to find words in brackets
            match = re.search(r'\[(.*?)\]', text, re.DOTALL)
            if match:
                list_text = match.group(1)
                # Split by commas and clean
                items = [item.strip().strip('"\'') for item in list_text.split(',')]
                words = [item for item in items if item and len(item) > 1]
        
        # For non-English languages, don't filter out non-ascii
        if lang == "en":
            words = [w for w in words if len(w) > 1 and not w.isdigit()]
        else:
            words = [w for w in words if len(w) > 1]
        
        return words[:num_words]
    
    def _extract_sentences_fallback(self, text, num_sentences, lang="en"):
        """Fallback method to extract sentences when JSON fails"""
        import re
        
        # Look for phrases in quotes
        sentences = re.findall(r'"([^"]+)"', text)
        
        # Look for numbered lists
        numbered = re.findall(r'\d+\.\s*([^"\n]+)', text)
        sentences.extend(numbered)
        
        # Look for bullet points
        bullet = re.findall(r'[•\-]\s*([^"\n]+)', text)
        sentences.extend(bullet)
        
        # Filter for reasonable length (2-10 words)
        sentences = [s for s in sentences if 2 <= len(s.split()) <= 10]
        
        return sentences[:num_sentences]
# 
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
    
    def correct_word(self, word: str, lang: str) -> Tuple[str, bool]:
        """Returns (corrected_word, was_corrected)"""
        if not word or lang not in self.dictionaries:
            return word, False
        
        word_lower = word.lower()
        if word_lower in self.dictionaries[lang]:
            return word, False
        
        import difflib
        matches = difflib.get_close_matches(
            word_lower, 
            self.dictionaries[lang].keys(), 
            n=3, 
            cutoff=0.7
        )
        
        if matches:
            best = max(matches, key=lambda w: self.dictionaries[lang].get(w, 1))
            if word.isupper():
                return best.upper(), True
            elif word.istitle():
                return best.title(), True
            return best, True
        
        return word, False

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

def auto_correct_text(text: str, lang: str, spell_checker: SpellChecker) -> Tuple[str, str, List[Tuple[str, str, str]]]:
    """Returns (corrected_text, message, corrections_list)"""
    if not text or lang not in spell_checker.get_available_languages():
        return text, "", []
    
    words = text.split()
    corrected_words = []
    corrections = []
    
    for word in words:
        corrected, was_corrected = spell_checker.correct_word(word, lang)
        corrected_words.append(corrected)
        if was_corrected:
            corrections.append((word, corrected, lang))
    
    corrected_text = " ".join(corrected_words)
    
    if corrections:
        return corrected_text, "Auto-corrected", corrections
    return corrected_text, "", []

# ---------- FastAPI App ----------
app = FastAPI(title="Smart Text Predictor")

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
            print("✅ Groq API model loaded")
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
    auto_correct: Union[bool, str] = Query(True),
    include_sentences: Union[bool, str] = Query(True),
    num_words: int = Query(15, ge=1, le=500)  # Changed from le=15 to le=50
):
    start_time = time.time()
    
    try:
        # Convert string parameters to boolean
        if isinstance(auto_correct, str):
            auto_correct = auto_correct.lower() == 'true'
        if isinstance(include_sentences, str):
            include_sentences = include_sentences.lower() == 'true'
        
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
        corrections = []
        if auto_correct:
            corrected_text, correction_msg, corrections = auto_correct_text(text, detected_lang, spell_checker)
        
        predictions = []
        sentence_predictions = []
        accuracy = 0.0
        
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
            
            # Get continuations
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
            "corrections": corrections,
            "language": detected_lang,
            "predictions": predictions[:num_words],
            "sentence_predictions": sentence_predictions[:5],
            "accuracy_score": accuracy,
            "response_time_ms": response_time
        }
        
    except Exception as e:
        print(f"Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "predictions": [],
            "sentence_predictions": []
        }

@app.post("/save_history")
async def save_history(request: dict):
    """Save prediction to history (called explicitly by frontend)"""
    global last_saved_input
    
    try:
        text = request.get("text", "")
        language = request.get("language", "en")
        model_used = request.get("model_used", "ngram")
        predictions = request.get("predictions", [])
        continuations = request.get("continuations", [])
        
        if not text.strip():
            return {"success": False, "message": "Empty text"}
        
        # Check if this is a duplicate (same text within last 10 seconds)
        current_time = time.time()
        if (text == last_saved_input["text"] and 
            current_time - last_saved_input["timestamp"] < 10):
            return {"success": False, "message": "Duplicate entry"}
        
        # Save to history
        history_id = add_history_entry(
            input_text=text,
            language=language,
            model_used=model_used,
            predictions=predictions[:10],
            continuations=continuations[:5],
            user_session=SESSION_ID
        )
        
        if history_id:
            last_saved_input = {"text": text, "timestamp": current_time}
            return {"success": True, "message": "Saved to history", "id": history_id}
        else:
            return {"success": False, "message": "Failed to save"}
            
    except Exception as e:
        print(f"Error in save_history: {e}")
        return {"success": False, "message": str(e)}

@app.get("/history")
async def get_history_endpoint(limit: int = Query(50, ge=1, le=200)):
    """Get prediction history"""
    try:
        history = get_history(limit=limit, user_session=SESSION_ID)
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        print(f"Error getting history: {e}")
        return {
            "success": False,
            "error": str(e),
            "history": []
        }

@app.delete("/history/{entry_id}")
async def delete_history(entry_id: int):
    """Delete a specific history entry"""
    try:
        deleted = delete_history_entry(entry_id)
        if deleted:
            return {
                "success": True,
                "message": f"Entry {entry_id} deleted successfully"
            }
        else:
            return {
                "success": False,
                "message": f"Entry {entry_id} not found"
            }
    except Exception as e:
        print(f"Error deleting history: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/history")
async def clear_history():
    """Clear all history"""
    try:
        count = clear_all_history(user_session=SESSION_ID)
        return {
            "success": True,
            "message": f"Cleared {count} history entries"
        }
    except Exception as e:
        print(f"Error clearing history: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/correct")
async def correct_spelling(text: str = Query(...), lang: str = Query("en")):
    """Correct spelling of a word or text"""
    try:
        if " " in text:
            # Multiple words
            corrected, was_corrected = spell_checker.correct_text(text, lang)
            return {"original": text, "corrected": corrected}
        else:
            # Single word
            corrected, was_corrected = spell_checker.correct_word(text, lang)
            return {"original": text, "corrected": corrected, "was_corrected": was_corrected}
    except Exception as e:
        print(f"Error in correct_spelling: {e}")
        return {"original": text, "corrected": text}

@app.get("/groq_status")
async def groq_status():
    """Check if Groq API is available"""
    return {"available": get_groq_model() is not None}

# Request model for add_word
class AddWordRequest(BaseModel):
    word: str
    context: str = ""
    lang: str
    corrected_word: str = ""  # Add this for spell-corrected words

@app.post("/add_word")
async def add_word(request: AddWordRequest):
    """
    Add a new word to the dataset
    """
    try:
        lang = request.lang
        # Use corrected word if provided, otherwise use original
        word_to_add = request.corrected_word.strip().lower() if request.corrected_word else request.word.strip().lower()
        original_word = request.word.strip().lower()
        context = request.context.strip()
        
        # Get the model for this language
        model = get_ngram_model(lang)
        
        # Check if word already exists
        if word_to_add in model.vocab:
            return {
                "success": False,
                "message": f"Word '{word_to_add}' already exists in {lang} dataset",
                "already_exists": True
            }
        
        # Add to model vocabulary
        model.vocab[word_to_add] = 1
        
        # If context provided, update appropriate n-gram
        if context:
            context_tokens = tokenize(context, lang)
            if context_tokens:
                # Update unigram
                model.ngrams[0][()][word_to_add] = 1
                
                # Update bigram if we have context
                if len(context_tokens) >= 1:
                    context_tuple = tuple(context_tokens[-1:])
                    model.ngrams[1][context_tuple][word_to_add] = 1
                
                # Update trigram if we have enough context
                if len(context_tokens) >= 2:
                    context_tuple = tuple(context_tokens[-2:])
                    model.ngrams[2][context_tuple][word_to_add] = 1
        
        # Save the updated model
        model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
        model.save(model_path)
        
        # Also append to corpus file for persistence
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        try:
            with open(corpus_path, "a", encoding="utf-8") as f:
                if context:
                    f.write(f"\n{context} {word_to_add}")
                else:
                    f.write(f"\n{word_to_add}")
        except Exception as e:
            print(f"Error saving to corpus: {e}")
        
        message = f"✅ Word '{word_to_add}' added to {lang} dataset"
        if original_word != word_to_add:
            message += f" (corrected from '{original_word}')"
        
        return {
            "success": True,
            "message": message,
            "word": word_to_add,
            "original_word": original_word if original_word != word_to_add else None,
            "language": lang
        }
            
    except Exception as e:
        print(f"Error adding word: {e}")
        return {
            "success": False,
            "message": str(e)
        }
@app.post("/add_word_with_context")
async def add_word_with_context(request: dict):
    """
    Add a new word with its full sentence context to the dataset
    """
    try:
        word = request.get("word", "").lower().strip()
        full_text = request.get("full_text", "").strip()
        lang = request.get("lang", "en")
        
        if not word or not full_text:
            return {"success": False, "message": "Invalid word or text"}
        
        model = get_ngram_model(lang)
        
        # Check if word already exists
        if word in model.vocab:
            return {"success": False, "message": f"Word '{word}' already exists"}
        
        # Tokenize the full text
        tokens = tokenize(full_text, lang)
        
        if len(tokens) < 2:
            # Just a single word
            model.vocab[word] = 1
            model.ngrams[0][()][word] = 1
        else:
            # Add all n-grams from the sentence
            for i in range(len(tokens)):
                current_word = tokens[i]
                model.vocab[current_word] = model.vocab.get(current_word, 0) + 1
                
                # Unigram
                model.ngrams[0][()][current_word] = model.ngrams[0][()].get(current_word, 0) + 1
                
                # Bigram
                if i > 0:
                    context = tuple([tokens[i-1]])
                    model.ngrams[1][context][current_word] = model.ngrams[1][context].get(current_word, 0) + 1
                
                # Trigram
                if i > 1:
                    context = tuple(tokens[i-2:i])
                    model.ngrams[2][context][current_word] = model.ngrams[2][context].get(current_word, 0) + 1
        
        # Save the updated model
        model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
        model.save(model_path)
        
        # Append the full sentence to corpus
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{full_text}")
        
        return {
            "success": True, 
            "message": f"✅ Word '{word}' added with context to {lang} dataset"
        }
        
    except Exception as e:
        print(f"Error adding word with context: {e}")
        return {"success": False, "message": str(e)}
@app.get("/check_word")
async def check_word(word: str = Query(...), lang: str = Query("en")):
    """Check if a word exists in the dataset"""
    try:
        model = get_ngram_model(lang)
        word_lower = word.lower().strip()
        
        exists = word_lower in model.vocab
        
        # Get correction if word doesn't exist
        correction = None
        if not exists and lang in spell_checker.dictionaries:
            corrected, was_corrected = spell_checker.correct_word(word, lang)
            if was_corrected:
                correction = corrected
        
        return {
            "exists": exists,
            "word": word,
            "language": lang,
            "correction": correction
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

@app.post("/add_single_word")
async def add_single_word(request: dict):
    """Add a single word to the dataset"""
    try:
        word = request.get("word", "").lower().strip()
        lang = request.get("lang", "en")
        
        if not word:
            return {"success": False, "message": "Invalid word"}
        
        model = get_ngram_model(lang)
        
        # Check if word already exists
        if word in model.vocab:
            return {"success": False, "message": f"Word '{word}' already exists"}
        
        # Add to vocabulary
        model.vocab[word] = 1
        
        # Add to unigram
        model.ngrams[0][()][word] = 1
        
        # Save model
        model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
        model.save(model_path)
        
        # Append to corpus
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{word}")
        
        return {"success": True, "message": f"✅ Word '{word}' added to {lang} dataset"}
        
    except Exception as e:
        return {"success": False, "message": str(e)}
@app.post("/add_sequence_inline")
async def add_sequence_inline(request: dict):
    """
    Add a 3-word sequence to the dataset (inline, not new line)
    """
    try:
        sequence = request.get("sequence", "").strip().lower()
        lang = request.get("lang", "en")
        
        if not sequence:
            return {"success": False, "message": "Invalid sequence"}
        
        words = sequence.split()
        if len(words) != 3:
            return {"success": False, "message": "Sequence must be exactly 3 words"}
        
        model = get_ngram_model(lang)
        
        # Check if sequence already exists
        context = tuple(words[:2])
        next_word = words[2]
        
        if context in model.ngrams[2] and next_word in model.ngrams[2][context]:
            return {"success": False, "message": f"Sequence '{sequence}' already exists"}
        
        # Add to trigram
        model.ngrams[2][context][next_word] = model.ngrams[2][context].get(next_word, 0) + 1
        
        # Update vocabulary for all words
        for word in words:
            model.vocab[word] = model.vocab.get(word, 0) + 1
        
        # Update bigrams
        model.ngrams[1][tuple(words[:1])][words[1]] = model.ngrams[1][tuple(words[:1])].get(words[1], 0) + 1
        model.ngrams[1][tuple(words[1:2])][words[2]] = model.ngrams[1][tuple(words[1:2])].get(words[2], 0) + 1
        
        # Update unigrams
        for word in words:
            model.ngrams[0][()][word] = model.ngrams[0][()].get(word, 0) + 1
        
        # Save the updated model
        model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
        model.save(model_path)
        
        # Append to corpus file IN THE SAME LINE (with a space before)
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        
        # Check if file exists and has content
        if os.path.exists(corpus_path) and os.path.getsize(corpus_path) > 0:
            # Append with a space at the beginning to add to the same line
            with open(corpus_path, "a", encoding="utf-8") as f:
                f.write(f" {sequence}")
        else:
            # New file, just write the sequence
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write(sequence)
        
        return {
            "success": True, 
            "message": f"✅ Sequence '{sequence}' added inline to {lang} dataset"
        }
        
    except Exception as e:
        print(f"Error adding sequence inline: {e}")
        return {"success": False, "message": str(e)}
@app.get("/dictionary")
async def get_dictionary(
    lang: str = Query("en"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=500),
    search: str = Query("")
):
    """
    Get dictionary-style view of vocabulary with pagination and search
    """
    try:
        model = get_ngram_model(lang)
        
        # Get all words with their frequencies
        all_words = list(model.vocab.items())
        all_words.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by search if provided
        if search:
            search_lower = search.lower()
            all_words = [(w, c) for w, c in all_words if search_lower in w]
        
        # Paginate
        total = len(all_words)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_words = all_words[start:end]
        
        # Get n-gram statistics for each word
        word_details = []
        for word, count in paginated_words:
            # Find common contexts for this word
            contexts = []
            for order in range(3):
                for context, counter in model.ngrams[order].items():
                    if word in counter:
                        if order == 0:
                            contexts.append("(unigram)")
                        elif order == 1 and context:
                            contexts.append(f"after '{context[0]}'")
                        elif order == 2 and context:
                            contexts.append(f"after '{context[0]} {context[1]}'")
                        break
            
            word_details.append({
                "word": word,
                "frequency": count,
                "example_context": contexts[0] if contexts else "",
                "is_common": count > 5
            })
        
        return {
            "success": True,
            "language": lang,
            "total_words": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "words": word_details
        }
        
    except Exception as e:
        print(f"Error in dictionary: {e}")
        return {
            "success": False,
            "error": str(e),
            "words": []
        }
@app.post("/add_sequence_newline")
async def add_sequence_newline(request: dict):
    """
    Add a sequence (any length) to the dataset as a new line
    """
    try:
        sequence = request.get("sequence", "").strip().lower()
        lang = request.get("lang", "en")
        
        if not sequence:
            return {"success": False, "message": "Invalid sequence"}
        
        words = sequence.split()
        if len(words) < 3:
            return {"success": False, "message": "Sequence must be at least 3 words"}
        
        model = get_ngram_model(lang)
        
        # Add all n-grams from the sequence
        for i in range(len(words)):
            current_word = words[i]
            
            # Update vocabulary
            model.vocab[current_word] = model.vocab.get(current_word, 0) + 1
            
            # Update unigram
            model.ngrams[0][()][current_word] = model.ngrams[0][()].get(current_word, 0) + 1
            
            # Update bigram
            if i > 0:
                context = tuple([words[i-1]])
                model.ngrams[1][context][current_word] = model.ngrams[1][context].get(current_word, 0) + 1
            
            # Update trigram
            if i > 1:
                context = tuple(words[i-2:i])
                model.ngrams[2][context][current_word] = model.ngrams[2][context].get(current_word, 0) + 1
        
        # Save the updated model
        model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
        model.save(model_path)
        
        # Append to corpus file as a NEW LINE
        corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
        
        with open(corpus_path, "a", encoding="utf-8") as f:
            f.write(f"\n{sequence}")
        
        return {
            "success": True, 
            "message": f"✅ Sequence '{sequence}' ({len(words)} words) added as new line to {lang} dataset"
        }
        
    except Exception as e:
        print(f"Error adding sequence: {e}")
        return {"success": False, "message": str(e)}
@app.get("/check_sequence")
async def check_sequence(sequence: str = Query(...), lang: str = Query("en")):
    """Check if a 3-word sequence exists in the model"""
    try:
        model = get_ngram_model(lang)
        words = sequence.lower().split()
        
        if len(words) >= 3:
            context = tuple(words[-3:-1])
            next_word = words[-1]
            
            exists = (context in model.ngrams[2] and 
                     next_word in model.ngrams[2][context])
            
            return {"exists": exists, "sequence": sequence}
        
        return {"exists": False, "sequence": sequence}
    except Exception as e:
        return {"exists": False, "error": str(e)}

@app.post("/add_sequence")
async def add_sequence(request: dict):
    """Add a new 3-word sequence to the dataset"""
    try:
        sequence = request.get("sequence", "")
        context = request.get("context", "")
        word = request.get("word", "")
        lang = request.get("lang", "en")
        
        model = get_ngram_model(lang)
        words = sequence.lower().split()
        
        if len(words) >= 3:
            context_tuple = tuple(words[-3:-1])
            next_word = words[-1]
            
            # Add to trigram
            model.ngrams[2][context_tuple][next_word] = model.ngrams[2][context_tuple].get(next_word, 0) + 1
            
            # Update vocabulary
            for w in words:
                model.vocab[w] = model.vocab.get(w, 0) + 1
            
            # Save model
            model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
            model.save(model_path)
            
            # Append to corpus
            corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
            with open(corpus_path, "a", encoding="utf-8") as f:
                f.write(f"\n{sequence}")
            
            return {"success": True, "message": f"✅ Sequence '{sequence}' added to dataset"}
        
        return {"success": False, "message": "Invalid sequence"}
    except Exception as e:
        return {"success": False, "message": str(e)}
@app.get("/all_words")
async def get_all_words(
    lang: str = Query("en"),
    search: str = Query(""),
    limit: int = Query(1000, ge=10, le=10000)
):
    """
    Get all words for dictionary view (no pagination, for "show more" feature)
    """
    try:
        model = get_ngram_model(lang)
        
        # Get all words with their frequencies
        all_words = list(model.vocab.items())
        all_words.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by search if provided
        if search:
            search_lower = search.lower()
            all_words = [(w, c) for w, c in all_words if search_lower in w]
        
        # Limit results
        limited_words = all_words[:limit]
        
        word_list = [{"word": w, "frequency": c} for w, c in limited_words]
        
        return {
            "success": True,
            "language": lang,
            "total_count": len(all_words),
            "display_count": len(word_list),
            "words": word_list
        }
        
    except Exception as e:
        print(f"Error in all_words: {e}")
        return {
            "success": False,
            "error": str(e),
            "words": []
        }

@app.get("/word_details/{word}")
async def get_word_details(word: str, lang: str = Query("en")):
    """
    Get detailed information about a specific word
    """
    try:
        model = get_ngram_model(lang)
        word_lower = word.lower()
        
        if word_lower not in model.vocab:
            return {
                "success": False,
                "message": f"Word '{word}' not found in {lang} vocabulary"
            }
        
        # Get frequency
        frequency = model.vocab[word_lower]
        total_words = sum(model.vocab.values())
        
        # Find contexts where this word appears
        contexts = []
        
        # Unigram
        contexts.append({
            "type": "unigram",
            "context": "any",
            "probability": frequency / total_words if total_words > 0 else 0
        })
        
        # Bigrams where this word is the target
        for context, counter in model.ngrams[1].items():
            if word_lower in counter:
                prev_word = context[0] if context else "START"
                total_in_context = sum(counter.values())
                contexts.append({
                    "type": "bigram",
                    "context": f"after '{prev_word}'",
                    "count": counter[word_lower],
                    "probability": counter[word_lower] / total_in_context if total_in_context > 0 else 0
                })
        
        # Trigrams
        for context, counter in model.ngrams[2].items():
            if word_lower in counter:
                prev_words = f"{context[0]} {context[1]}" if context else "START"
                total_in_context = sum(counter.values())
                contexts.append({
                    "type": "trigram",
                    "context": f"after '{prev_words}'",
                    "count": counter[word_lower],
                    "probability": counter[word_lower] / total_in_context if total_in_context > 0 else 0
                })
        
        # Sort contexts by probability
        contexts.sort(key=lambda x: x.get('probability', 0), reverse=True)
        
        # Get common next words
        next_words = []
        for context, counter in model.ngrams[1].items():
            if context and context[0] == word_lower:
                for next_word, count in counter.most_common(5):
                    next_words.append({
                        "word": next_word,
                        "count": count
                    })
        
        # Get word rank
        sorted_words = sorted(model.vocab.items(), key=lambda x: x[1], reverse=True)
        rank = 1
        for i, (w, _) in enumerate(sorted_words):
            if w == word_lower:
                rank = i + 1
                break
        
        return {
            "success": True,
            "word": word_lower,
            "frequency": frequency,
            "total_vocab_size": len(model.vocab),
            "rank": rank,
            "contexts": contexts[:10],
            "common_next_words": next_words[:10]
        }
        
    except Exception as e:
        print(f"Error in word_details: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
