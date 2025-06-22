#!/usr/bin/env python3
import json
import os
from collections import Counter
import nltk

def get_cache_path():
    """Get the path to the cache JSON file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'pos_cache.json')

def load_cache():
    """Load the cache from JSON file, return empty dict if file doesn't exist."""
    cache_path = get_cache_path()
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_cache(cache):
    """Save the cache to JSON file."""
    cache_path = get_cache_path()
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_pos_distribution(word: str):
    """
    Return a dict mapping Universal POS tags to counts (and relative freq)
    for all occurrences of `word` in the Brown corpus.
    Uses caching to avoid recomputing for previously processed words.
    """
    word = word.lower()
    
    # Check cache first
    cache = load_cache()
    if word in cache:
        print(f"Using cached result for '{word}'")
        return cache[word]
    
    print(f"Computing POS distribution for '{word}' from Brown corpus...")
    
    # Ensure the corpora/tagset are downloaded
    nltk.download('brown', quiet=True)
    nltk.download('universal_tagset', quiet=True)

    # Counter over POS tags
    tag_counts = Counter()
    for w, pos in nltk.corpus.brown.tagged_words(tagset='universal'):
        if w.lower() == word:
            tag_counts[pos] += 1

    total = sum(tag_counts.values())
    if total == 0:
        result = {}
    else:
        # Return both raw counts and relative frequencies
        result = {
            tag: {
                'count': cnt,
                'relative': round(cnt / total, 4)
            }
            for tag, cnt in tag_counts.items()
        }
    
    # Save to cache
    cache[word] = result
    save_cache(cache)
    
    return result


