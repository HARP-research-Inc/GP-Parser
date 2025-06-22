#!/usr/bin/env python3
"""
Compare different lemmatization approaches
"""
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download required data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

lemmatizer = WordNetLemmatizer()

test_words = [
    "running", "ran", "runs", "run",
    "better", "best", "good", 
    "mice", "mouse", "feet", "foot",
    "children", "child", "women", "woman",
    "ate", "eating", "eaten", "eat",
    "drove", "driving", "driven", "drive"
]

print("=== NLTK WordNet Lemmatizer (Dictionary + Rules) ===")
for word in test_words:
    # Try different POS tags
    noun_lemma = lemmatizer.lemmatize(word, pos='n')
    verb_lemma = lemmatizer.lemmatize(word, pos='v')
    adj_lemma = lemmatizer.lemmatize(word, pos='a')
    
    print(f"{word:10} → noun: {noun_lemma:8} verb: {verb_lemma:8} adj: {adj_lemma:8}")

print("\n=== Simple Suffix Stripping (Current Approach) ===")
for word in test_words:
    # Simulate our current basic approach
    lemma = word
    if word.endswith('ing'):
        lemma = word[:-3]
    elif word.endswith('ed'):
        lemma = word[:-2]
    elif word.endswith('s') and not word.endswith('ss'):
        lemma = word[:-1]
    
    print(f"{word:10} → {lemma}")

print("\n=== WordNet Morphy (NLTK's underlying algorithm) ===")
for word in test_words:
    # This is what NLTK uses internally
    morphy_result = wordnet.morphy(word)
    print(f"{word:10} → {morphy_result or word}") 