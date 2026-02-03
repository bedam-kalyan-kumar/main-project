#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import BackoffNGram, DATA_DIR
import pickle

def train_and_save(lang: str = "en"):
    """Train and save n-gram model for a language"""
    corpus_path = os.path.join(DATA_DIR, f"{lang}.txt")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Dataset not found: {corpus_path}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Training {lang} model with {len(lines)} sentences...")
    model = BackoffNGram(n=3)
    model.train(lines, lang)
    
    model_path = os.path.join(DATA_DIR, f"ngram_{lang}.pkl")
    model.save(model_path)
    print(f"✅ Trained N-gram model for {lang}")
    
    vocab_size = len(model.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    return model

def main():
    """Train models for all supported languages"""
    languages = ['en', 'hi', 'te', 'ta', 'kn', 'fr']
    
    for lang in languages:
        print(f"\n{'='*50}")
        print(f"Training model for: {lang}")
        print('='*50)
        
        try:
            train_and_save(lang)
        except Exception as e:
            print(f"Failed to train {lang}: {e}")
    
    print("\n✅ All models trained successfully!")

if __name__ == "__main__":
    main()
