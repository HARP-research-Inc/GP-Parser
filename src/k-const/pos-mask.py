#!/usr/bin/env python3
import numpy as np
from typing import List, Union, Set
import importlib.util
import os

# Import from prob-dist-mat.py
spec = importlib.util.spec_from_file_location("prob_dist_mat", os.path.join(os.path.dirname(__file__), "prob-dist-mat.py"))
prob_dist_mat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prob_dist_mat)

UNIVERSAL_POS_TAGS = prob_dist_mat.UNIVERSAL_POS_TAGS

def create_pos_mask(pos_tags: Union[str, List[str]]) -> np.ndarray:
    """
    Create a boolean mask for specific POS tags.
    
    Args:
        pos_tags: Single POS tag string or list of POS tag strings to mask
    
    Returns:
        numpy.ndarray: Boolean array of length 16 with True for specified POS tags
    """
    if isinstance(pos_tags, str):
        pos_tags = [pos_tags]
    
    mask = np.zeros(16, dtype=bool)
    
    for tag in pos_tags:
        tag = tag.upper()
        if tag in UNIVERSAL_POS_TAGS:
            idx = UNIVERSAL_POS_TAGS.index(tag)
            mask[idx] = True
        else:
            print(f"Warning: '{tag}' is not a valid universal POS tag")
    
    return mask

def apply_mask_to_matrix(matrix: np.ndarray, mask: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
    """
    Apply a mask to a sentence matrix, setting masked positions to mask_value.
    
    Args:
        matrix: Input matrix (m x 16)
        mask: Boolean mask (16,) - True positions will be masked
        mask_value: Value to set for masked positions (default: 0.0)
    
    Returns:
        numpy.ndarray: Masked matrix
    """
    masked_matrix = matrix.copy()
    masked_matrix[:, mask] = mask_value
    return masked_matrix

def apply_inverse_mask_to_matrix(matrix: np.ndarray, mask: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
    """
    Apply inverse mask - keep only the specified POS tags, mask everything else.
    
    Args:
        matrix: Input matrix (m x 16)
        mask: Boolean mask (16,) - True positions will be kept, False will be masked
        mask_value: Value to set for masked positions (default: 0.0)
    
    Returns:
        numpy.ndarray: Inverse masked matrix
    """
    masked_matrix = matrix.copy()
    masked_matrix[:, ~mask] = mask_value
    return masked_matrix

def get_mask_for_pos_tags(pos_tags: Union[str, List[str]]) -> Set[int]:
    """
    Get column indices for specific POS tags (for highlighting).
    
    Args:
        pos_tags: Single POS tag string or list of POS tag strings
    
    Returns:
        Set[int]: Set of column indices to highlight
    """
    if isinstance(pos_tags, str):
        pos_tags = [pos_tags]
    
    indices = set()
    
    for tag in pos_tags:
        tag = tag.upper()
        if tag in UNIVERSAL_POS_TAGS:
            idx = UNIVERSAL_POS_TAGS.index(tag)
            indices.add(idx)
        else:
            print(f"Warning: '{tag}' is not a valid universal POS tag")
    
    return indices

def print_available_pos_tags():
    """Print all available universal POS tags."""
    print("Available Universal POS Tags:")
    print("=" * 30)
    
    descriptions = {
        'ADJ': 'Adjective',
        'ADP': 'Adposition (prepositions, postpositions)', 
        'ADV': 'Adverb',
        'AUX': 'Auxiliary verb',
        'CCONJ': 'Coordinating conjunction',
        'DET': 'Determiner',
        'INTJ': 'Interjection',
        'NOUN': 'Noun',
        'NUM': 'Numeral',
        'PART': 'Particle',
        'PRON': 'Pronoun',
        'PROPN': 'Proper noun',
        'PUNCT': 'Punctuation',
        'SCONJ': 'Subordinating conjunction',
        'SYM': 'Symbol',
        'VERB': 'Verb'
    }
    
    for i, tag in enumerate(UNIVERSAL_POS_TAGS):
        desc = descriptions.get(tag, 'Unknown')
        print(f"{i:2}: {tag:6} - {desc}")

# Preset masks for common use cases
def get_content_words_mask():
    """Get mask for content words (NOUN, VERB, ADJ, ADV)."""
    return create_pos_mask(['NOUN', 'VERB', 'ADJ', 'ADV'])

def get_function_words_mask():
    """Get mask for function words (DET, ADP, PRON, etc.)."""
    return create_pos_mask(['DET', 'ADP', 'PRON', 'CCONJ', 'SCONJ', 'AUX', 'PART'])

def get_nouns_and_verbs_mask():
    """Get mask for nouns and verbs only."""
    return create_pos_mask(['NOUN', 'VERB', 'PROPN'])

if __name__ == '__main__':
    print_available_pos_tags()
    print("\nExample usage:")
    print("mask = create_pos_mask(['NOUN', 'VERB'])")
    print("masked_matrix = apply_mask_to_matrix(matrix, mask)") 