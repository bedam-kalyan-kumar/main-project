# app/spellchecker.py
import os
from typing import Dict, List, Set

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class SpellChecker:
    def __init__(self):
        self.dictionaries: Dict[str, Set[str]] = {}
        self.load_all_dictionaries()
    
    def load_all_dictionaries(self):
        """Load all available language dictionaries"""
        available_langs = ['en', 'hi', 'te', 'ta', 'kn', 'fr']
        
        for lang in available_langs:
            dict_path = os.path.join(DATA_DIR, f"{lang}_word_frequency.txt")
            if os.path.exists(dict_path):
                self.load_dictionary(lang, dict_path)
                print(f"Loaded dictionary for {lang}: {len(self.dictionaries.get(lang, set()))} words")
    
    def load_dictionary(self, lang: str, path: str):
        """Load language dictionary from file"""
        words = set()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if parts:
                        words.add(parts[0].lower())
            
            self.dictionaries[lang] = words
            
        except Exception as e:
            print(f"Error loading dictionary for {lang}: {e}")
            self.dictionaries[lang] = set()
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language dictionaries"""
        return list(self.dictionaries.keys())
    
    def correct_text(self, text: str, lang: str = "en") -> str:
        """Correct spelling in text"""
        if not text or lang not in self.dictionaries:
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = self.correct_word(word, lang)
            corrected_words.append(corrected_word)
        
        return " ".join(corrected_words)
    
    def correct_word(self, word: str, lang: str) -> str:
        """Correct a single word"""
        if not word or lang not in self.dictionaries:
            return word
        
        word_lower = word.lower()
        
        # Check if word is already correct
        if word_lower in self.dictionaries[lang]:
            return word
        
        # Simple correction: find similar words in dictionary
        best_match = None
        best_score = 0
        
        for dict_word in self.dictionaries[lang]:
            score = self._similarity(word_lower, dict_word)
            if score > best_score:
                best_score = score
                best_match = dict_word
        
        return best_match if best_match and best_score > 0.7 else word
    
    def _similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words (Levenshtein distance based)"""
        if not word1 or not word2:
            return 0.0
        
        # Simple character overlap similarity
        set1 = set(word1)
        set2 = set(word2)
        
        if len(set1) == 0 or len(set2) == 0:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
