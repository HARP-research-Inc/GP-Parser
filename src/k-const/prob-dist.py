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

def process_text_file_bulk(text_file_path: str, output_file_path: str = None):
    """
    Process all unique tokens from a text file and generate POS distributions.
    
    Args:
        text_file_path: Path to the input text file
        output_file_path: Optional path for output JSON file. If None, prints to console.
    
    Returns:
        dict: Mapping of words to their POS distributions
    """
    # Ensure tokenizer is available
    nltk.download('punkt', quiet=True)
    
    # Read and tokenize the text file
    print(f"Reading and tokenizing '{text_file_path}'...")
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    # Tokenize and get unique words (case-insensitive)
    tokens = nltk.word_tokenize(text)
    # Filter out punctuation and get unique words
    unique_words = set()
    for token in tokens:
        if token.isalpha():  # Only include alphabetic tokens
            unique_words.add(token.lower())
    
    print(f"Found {len(unique_words)} unique alphabetic tokens to process...")
    
    # Process each unique word
    results = {}
    processed = 0
    cached = 0
    
    # Load cache once to check for existing entries
    cache = load_cache()
    
    for word in sorted(unique_words):
        if word in cache:
            cached += 1
            results[word] = cache[word]
        else:
            results[word] = get_pos_distribution(word)
            processed += 1
    
    print(f"\nProcessing complete:")
    print(f"  - Used cached results: {cached}")
    print(f"  - Computed from scratch: {processed}")
    print(f"  - Total unique words: {len(unique_words)}")
    
    # Output results
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to '{output_file_path}'")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return results


