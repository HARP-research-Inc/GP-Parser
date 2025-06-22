#!/usr/bin/env python3
import json
import os
from collections import Counter, defaultdict
import nltk
import pickle

def get_cache_path():
    """Get the path to the cache JSON file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'pos_cache.json')

def get_checkpoint_path():
    """Get the path to the checkpoint file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'bulk_checkpoint.pkl')

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

def load_checkpoint():
    """Load checkpoint data if it exists."""
    checkpoint_path = get_checkpoint_path()
    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError):
        return None

def save_checkpoint(data):
    """Save checkpoint data."""
    checkpoint_path = get_checkpoint_path()
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(data, f)

def cleanup_checkpoint():
    """Remove checkpoint file."""
    checkpoint_path = get_checkpoint_path()
    try:
        os.remove(checkpoint_path)
    except FileNotFoundError:
        pass

def get_best_pos_tagger(fast: bool = False):
    """
    Get the best available POS tagger that outputs universal tags.
    
    Args:
        fast: If True, prioritize speed over accuracy
    
    Returns:
        tuple: (tagger_function, tagger_name)
    """
    # Just use NLTK's averaged perceptron tagger - it's good and converts easily to universal
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
    if fast:
        return nltk.pos_tag, "NLTK Averaged Perceptron (fast mode)"
    else:
        return nltk.pos_tag, "NLTK Averaged Perceptron (accurate mode)"

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

def process_text_file_bulk_efficient(text_file_path: str, output_file_path: str = None, resume: bool = True, fast: bool = False):
    """
    Efficiently process all unique tokens from a text file using a smarter approach:
    1. POS tag the entire input text first
    2. Use checkpoints for resumability
    3. Count running totals for all words
    4. Calculate probabilities at the end
    
    Args:
        text_file_path: Path to the input text file
        output_file_path: Optional path for output JSON file
        resume: Whether to resume from checkpoint if available
        fast: If True, use faster but less accurate POS tagger
    
    Returns:
        dict: Mapping of words to their POS distributions
    """
    # Ensure required NLTK data is available
    nltk.download('punkt', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    nltk.download('brown', quiet=True)
    
    # Get the best POS tagger for accuracy/speed preference
    pos_tagger, tagger_name = get_best_pos_tagger(fast)
    print(f"Using POS tagger: {tagger_name}")
    
    # Check for existing cache and checkpoint
    cache = load_cache()
    checkpoint = load_checkpoint() if resume else None
    
    print(f"Processing '{text_file_path}' with efficient bulk method...")
    
    # Step 1: Check if we have a checkpoint to resume from
    if checkpoint and checkpoint.get('text_file_path') == text_file_path:
        print(f"Resuming from checkpoint at token {checkpoint['current_token']} of {checkpoint['total_tokens']}")
        word_pos_counts = checkpoint['word_pos_counts']
        brown_word_pos_counts = checkpoint['brown_word_pos_counts']
        text_tokens = checkpoint['text_tokens']
        start_token = checkpoint['current_token']
        unique_words = checkpoint['unique_words']
    else:
        # Step 1: Read and tokenize the input text
        print("Reading and tokenizing input text...")
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{text_file_path}' not found.")
            return {}
        except Exception as e:
            print(f"Error reading file: {e}")
            return {}
        
        # Tokenize the text
        raw_tokens = nltk.word_tokenize(text)
        # Filter to only alphabetic tokens
        alphabetic_tokens = [token for token in raw_tokens if token.isalpha()]
        
        print(f"POS tagging {len(alphabetic_tokens)} tokens...")
        # POS tag the input text
        tagged_tokens = pos_tagger(alphabetic_tokens)
        
        # Convert to universal tagset and lowercase
        text_tokens = []
        for token, pos_tag in tagged_tokens:
            # Convert to universal tagset
            universal_tag = nltk.tag.mapping.map_tag('en-ptb', 'universal', pos_tag)
            text_tokens.append((token.lower(), universal_tag))
        
        unique_words = set(word for word, _ in text_tokens)
        print(f"Found {len(text_tokens)} alphabetic tokens, {len(unique_words)} unique words")
        
        # Step 2: Load Brown corpus data once (much more efficient)
        print("Loading Brown corpus data...")
        brown_word_pos_counts = defaultdict(lambda: defaultdict(int))
        for word, pos in nltk.corpus.brown.tagged_words(tagset='universal'):
            if word.isalpha():  # Only consider alphabetic words
                brown_word_pos_counts[word.lower()][pos] += 1
        
        print(f"Loaded Brown corpus data for {len(brown_word_pos_counts)} unique words")
        
        # Initialize counters
        word_pos_counts = defaultdict(lambda: defaultdict(int))
        start_token = 0
    
    # Step 3: Process text tokens and count POS occurrences
    print("Counting POS tag occurrences in input text...")
    checkpoint_interval = 1000  # Save checkpoint every 1000 tokens
    
    for i, (word, pos_tag) in enumerate(text_tokens[start_token:], start_token):
        # Count POS tag occurrences for each word from the input text
        word_pos_counts[word][pos_tag] += 1
        
        if i % checkpoint_interval == 0 and i > start_token:
            # Save checkpoint
            checkpoint_data = {
                'text_file_path': text_file_path,
                'current_token': i,
                'total_tokens': len(text_tokens),
                'word_pos_counts': dict(word_pos_counts),
                'brown_word_pos_counts': dict(brown_word_pos_counts),
                'text_tokens': text_tokens,
                'unique_words': unique_words
            }
            save_checkpoint(checkpoint_data)
            print(f"Checkpoint saved at token {i}/{len(text_tokens)} ({i/len(text_tokens)*100:.1f}%)")
    
    # Step 4: Calculate final distributions by combining input text + Brown corpus + cache
    print("Calculating final POS distributions...")
    results = {}
    cached_count = 0
    computed_count = 0
    
    for word in unique_words:
        # Get counts from different sources
        input_text_counts = word_pos_counts.get(word, {})
        brown_counts = brown_word_pos_counts.get(word, {})
        
        # Combine all counts
        combined_counts = defaultdict(int)
        
        # Add existing cached counts if available
        if word in cache and cache[word]:
            existing_result = cache[word]
            for tag, info in existing_result.items():
                combined_counts[tag] += info['count']
            cached_count += 1
        else:
            computed_count += 1
        
        # Add input text counts (from POS tagging the input)
        for tag, count in input_text_counts.items():
            combined_counts[tag] += count
        
        # Add Brown corpus counts
        for tag, count in brown_counts.items():
            combined_counts[tag] += count
        
        # Calculate final percentages from combined counts
        total = sum(combined_counts.values())
        if total > 0:
            result = {
                tag: {
                    'count': cnt,
                    'relative': round(cnt / total, 4)
                }
                for tag, cnt in combined_counts.items()
            }
        else:
            result = {}
        
        results[word] = result
        cache[word] = result  # Update cache with accumulated result
    
    # Save updated cache
    save_cache(cache)
    
    # Clean up checkpoint since we're done
    cleanup_checkpoint()
    
    print(f"\nProcessing complete:")
    print(f"  - Words with existing cache: {cached_count}")
    print(f"  - New words processed: {computed_count}")
    print(f"  - Total unique words: {len(unique_words)}")
    print(f"  - Input text tokens processed: {len(text_tokens)}")
    print(f"  - POS tagger used: {tagger_name}")
    
    # Output results
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to '{output_file_path}'")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return results

# Keep the old function for backward compatibility but mark it as deprecated
def process_text_file_bulk(text_file_path: str, output_file_path: str = None):
    """
    DEPRECATED: Use process_text_file_bulk_efficient instead.
    This function is kept for backward compatibility.
    """
    print("Warning: Using deprecated bulk processing method. Consider using the efficient version.")
    return process_text_file_bulk_efficient(text_file_path, output_file_path)


