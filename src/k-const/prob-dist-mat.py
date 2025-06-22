#!/usr/bin/env python3
import json
import os
import numpy as np
from typing import Dict, List, Tuple
import importlib.util

# Import from prob-dist.py (handle hyphen in filename)
spec = importlib.util.spec_from_file_location("prob_dist", os.path.join(os.path.dirname(__file__), "prob-dist.py"))
prob_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prob_dist)

# Universal POS tags (excluding X)
UNIVERSAL_POS_TAGS = [
    'ADJ',    # Adjective
    'ADP',    # Adposition (prepositions, postpositions)
    'ADV',    # Adverb
    'AUX',    # Auxiliary verb
    'CCONJ',  # Coordinating conjunction
    'DET',    # Determiner
    'INTJ',   # Interjection
    'NOUN',   # Noun
    'NUM',    # Numeral
    'PART',   # Particle
    'PRON',   # Pronoun
    'PROPN',  # Proper noun
    'PUNCT',  # Punctuation
    'SCONJ',  # Subordinating conjunction
    'SYM',    # Symbol
    'VERB'    # Verb
]

def get_matrix_cache_path():
    """Get the path to the matrix cache JSON file in the same directory as this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'pos_mat_cache.json')

def load_matrix_cache():
    """Load the matrix cache from JSON file, return empty dict if file doesn't exist."""
    cache_path = get_matrix_cache_path()
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_matrix_cache(cache):
    """Save the matrix cache to JSON file."""
    cache_path = get_matrix_cache_path()
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def pos_distribution_to_vector(pos_dist: Dict) -> List[float]:
    """
    Convert a POS distribution dictionary to a 16-dimensional vector.
    
    Args:
        pos_dist: Dictionary with POS tags as keys and {'count': int, 'relative': float} as values
    
    Returns:
        List of 16 floats representing the relative frequencies for each universal POS tag
    """
    vector = [0.0] * len(UNIVERSAL_POS_TAGS)
    
    for i, tag in enumerate(UNIVERSAL_POS_TAGS):
        if tag in pos_dist:
            vector[i] = pos_dist[tag]['relative']
    
    return vector

def vector_to_pos_distribution(vector: List[float]) -> Dict:
    """
    Convert a 16-dimensional vector back to a POS distribution dictionary.
    
    Args:
        vector: List of 16 floats representing relative frequencies
    
    Returns:
        Dictionary with POS tags as keys and relative frequencies as values
    """
    pos_dist = {}
    
    for i, tag in enumerate(UNIVERSAL_POS_TAGS):
        if i < len(vector) and vector[i] > 0:
            pos_dist[tag] = {'relative': vector[i]}
    
    return pos_dist

def get_word_vector(word: str, use_existing_cache: bool = True) -> List[float]:
    """
    Get the POS distribution vector for a word.
    
    Args:
        word: The word to get the vector for
        use_existing_cache: Whether to use existing prob-dist cache first
    
    Returns:
        16-dimensional vector representing POS distribution
    """
    word = word.lower()
    
    # Check matrix cache first
    matrix_cache = load_matrix_cache()
    if word in matrix_cache:
        return matrix_cache[word]
    
    # If not in matrix cache, get from prob-dist system
    if use_existing_cache:
        # Use the existing prob-dist system
        pos_dist = prob_dist.get_pos_distribution(word)
    else:
        # Force recomputation
        pos_dist = {}
    
    # Convert to vector
    vector = pos_distribution_to_vector(pos_dist)
    
    # Cache the vector
    matrix_cache[word] = vector
    save_matrix_cache(matrix_cache)
    
    return vector

def convert_pos_cache_to_matrix():
    """
    Convert the entire pos_cache.json to pos_mat_cache.json format.
    """
    print("Converting POS cache to matrix format...")
    
    # Load existing POS cache
    pos_cache = prob_dist.load_cache()
    if not pos_cache:
        print("No POS cache found. Run prob-dist first to build the cache.")
        return
    
    # Load existing matrix cache
    matrix_cache = load_matrix_cache()
    
    converted_count = 0
    updated_count = 0
    
    for word, pos_dist in pos_cache.items():
        vector = pos_distribution_to_vector(pos_dist)
        
        if word in matrix_cache:
            updated_count += 1
        else:
            converted_count += 1
        
        matrix_cache[word] = vector
    
    # Save updated matrix cache
    save_matrix_cache(matrix_cache)
    
    print(f"Conversion complete:")
    print(f"  - New words converted: {converted_count}")
    print(f"  - Existing words updated: {updated_count}")
    print(f"  - Total words in matrix cache: {len(matrix_cache)}")
    print(f"  - Matrix cache saved to: {get_matrix_cache_path()}")

def get_similar_words(target_word: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Find words with similar POS distributions using cosine similarity.
    
    Args:
        target_word: The word to find similar words for
        top_k: Number of similar words to return
    
    Returns:
        List of (word, similarity_score) tuples, sorted by similarity
    """
    target_vector = get_word_vector(target_word)
    target_array = np.array(target_vector)
    
    # Load all cached vectors
    matrix_cache = load_matrix_cache()
    similarities = []
    
    for word, vector in matrix_cache.items():
        if word == target_word.lower():
            continue
        
        word_array = np.array(vector)
        
        # Calculate cosine similarity
        if np.linalg.norm(target_array) > 0 and np.linalg.norm(word_array) > 0:
            similarity = np.dot(target_array, word_array) / (np.linalg.norm(target_array) * np.linalg.norm(word_array))
            similarities.append((word, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def print_word_vector(word: str):
    """Print the POS vector for a word in a readable format."""
    vector = get_word_vector(word)
    
    print(f"POS vector for '{word}':")
    print("Tag    | Probability")
    print("-------|------------")
    
    for i, tag in enumerate(UNIVERSAL_POS_TAGS):
        prob = vector[i]
        if prob > 0:
            print(f"{tag:6} | {prob:.4f}")
    
    return vector

def sentence_to_matrix(sentence: str, max_length: int = 32) -> np.ndarray:
    """
    Convert a sentence to a fixed-size matrix (max_length x 16).
    
    Args:
        sentence: The input sentence
        max_length: Maximum number of words (rows) in the matrix (default: 32)
    
    Returns:
        numpy.ndarray: A matrix of shape (max_length, 16) where:
        - First n rows contain POS vectors for the n words in the sentence
        - Remaining rows are filled with zeros
        - Each column represents one of the 16 universal POS tags
    """
    import nltk
    
    # Tokenize the sentence
    nltk.download('punkt', quiet=True)
    tokens = nltk.word_tokenize(sentence)
    
    # Filter to only alphabetic tokens and convert to lowercase
    words = [token.lower() for token in tokens if token.isalpha()]
    
    # Create the fixed-size matrix
    matrix = np.zeros((max_length, 16), dtype=np.float32)
    
    # Fill the first n rows with word vectors
    for i, word in enumerate(words):
        if i >= max_length:
            break  # Don't exceed matrix size
        vector = get_word_vector(word)
        matrix[i] = vector
    
    return matrix, words[:max_length]  # Return matrix and list of words used 