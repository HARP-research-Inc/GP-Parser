#!/usr/bin/env python3
"""
Adjective Position Analysis

This script analyzes the positional distribution of adjectives around nouns in text.
For every noun in the text, it counts adjectives at different displacements 
(e.g., -2, -1, +1, +2) and calculates probabilities.

Usage:
    python adjective-position-analysis.py input.txt
    python adjective-position-analysis.py input.txt --range 3 --output results.json
"""

import argparse
import json
import os
from collections import defaultdict, Counter
import nltk
import numpy as np

def ensure_nltk_data():
    """Download required NLTK data."""
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)

def analyze_adjective_positions(text_file_path: str, displacement_range: int = 2):
    """
    Analyze adjective positions around nouns in a text file.
    
    Args:
        text_file_path: Path to the input text file
        displacement_range: How many positions to check before/after each noun
    
    Returns:
        dict: Analysis results with counts and probabilities
    """
    ensure_nltk_data()
    
    print(f"Analyzing adjective positions around nouns in '{text_file_path}'")
    print(f"Displacement range: -{displacement_range} to +{displacement_range}")
    
    # Read the text file
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    # Tokenize and POS tag the text
    print("Tokenizing and POS tagging text...")
    tokens = nltk.word_tokenize(text)
    
    # Filter to only alphabetic tokens
    alphabetic_tokens = [token for token in tokens if token.isalpha()]
    
    # POS tag the tokens
    tagged_tokens = nltk.pos_tag(alphabetic_tokens)
    
    # Convert to universal tagset
    universal_tagged = []
    for token, pos_tag in tagged_tokens:
        universal_tag = nltk.tag.mapping.map_tag('en-ptb', 'universal', pos_tag)
        universal_tagged.append((token.lower(), universal_tag))
    
    print(f"Found {len(universal_tagged)} alphabetic tokens")
    
    # Find all nouns and their positions
    noun_positions = []
    for i, (token, tag) in enumerate(universal_tagged):
        if tag in ['NOUN', 'PROPN']:  # Both common and proper nouns
            noun_positions.append(i)
    
    print(f"Found {len(noun_positions)} nouns")
    
    # Count adjectives at each displacement from nouns
    displacement_counts = defaultdict(int)  # displacement -> count of adjectives
    total_positions = defaultdict(int)      # displacement -> total positions checked
    
    # For each noun, check surrounding positions
    for noun_pos in noun_positions:
        for displacement in range(-displacement_range, displacement_range + 1):
            if displacement == 0:  # Skip the noun position itself
                continue
            
            target_pos = noun_pos + displacement
            
            # Check if target position is within bounds
            if 0 <= target_pos < len(universal_tagged):
                total_positions[displacement] += 1
                
                # Check if there's an adjective at this position
                token, tag = universal_tagged[target_pos]
                if tag == 'ADJ':
                    displacement_counts[displacement] += 1
    
    # Calculate probabilities
    probabilities = {}
    for displacement in range(-displacement_range, displacement_range + 1):
        if displacement == 0:
            continue
        
        count = displacement_counts.get(displacement, 0)
        total = total_positions.get(displacement, 0)
        
        if total > 0:
            probability = count / total
        else:
            probability = 0.0
        
        probabilities[displacement] = {
            'adjective_count': count,
            'total_positions': total,
            'probability': probability
        }
    
    # Compile results
    results = {
        'text_file': text_file_path,
        'total_tokens': len(universal_tagged),
        'total_nouns': len(noun_positions),
        'displacement_range': displacement_range,
        'position_analysis': probabilities,
        'summary': {
            'total_adjectives_found': sum(displacement_counts.values()),
            'total_positions_checked': sum(total_positions.values())
        }
    }
    
    return results

def print_results(results: dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*60)
    print("ADJECTIVE POSITION ANALYSIS RESULTS")
    print("="*60)
    
    print(f"Text file: {results['text_file']}")
    print(f"Total tokens: {results['total_tokens']:,}")
    print(f"Total nouns: {results['total_nouns']:,}")
    print(f"Displacement range: ±{results['displacement_range']}")
    print(f"Total adjectives found: {results['summary']['total_adjectives_found']:,}")
    print(f"Total positions checked: {results['summary']['total_positions_checked']:,}")
    
    print("\nPOSITION ANALYSIS:")
    print("Position | Adj Count | Total Pos | Probability | Visualization")
    print("---------|-----------|-----------|-------------|" + "-"*20)
    
    # Sort by displacement (negative first, then positive)
    sorted_positions = sorted(results['position_analysis'].items())
    
    for displacement, data in sorted_positions:
        count = data['adjective_count']
        total = data['total_positions']
        prob = data['probability']
        
        # Create a simple bar visualization
        bar_length = int(prob * 50)  # Scale to 50 characters max
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"{displacement:+8} | {count:9,} | {total:9,} | {prob:11.4f} | {bar}")
    
    # Show most likely positions
    print("\nMOST LIKELY ADJECTIVE POSITIONS:")
    sorted_by_prob = sorted(results['position_analysis'].items(), 
                           key=lambda x: x[1]['probability'], reverse=True)
    
    for i, (displacement, data) in enumerate(sorted_by_prob[:3]):
        prob = data['probability']
        count = data['adjective_count']
        total = data['total_positions']
        
        position_desc = f"{abs(displacement)} position{'s' if abs(displacement) > 1 else ''}"
        direction = "before" if displacement < 0 else "after"
        
        print(f"{i+1}. {position_desc} {direction} noun: {prob:.1%} "
              f"({count:,} adjectives / {total:,} positions)")

def create_visualization_data(results: dict):
    """Create data suitable for plotting."""
    positions = []
    probabilities = []
    counts = []
    
    for displacement in sorted(results['position_analysis'].keys()):
        data = results['position_analysis'][displacement]
        positions.append(displacement)
        probabilities.append(data['probability'])
        counts.append(data['adjective_count'])
    
    return {
        'positions': positions,
        'probabilities': probabilities,
        'counts': counts
    }

def main():
    parser = argparse.ArgumentParser(
        description="Analyze positional distribution of adjectives around nouns"
    )
    
    parser.add_argument('input_file', 
                       help="Path to the input text file")
    parser.add_argument('--range', '-r', type=int, default=2,
                       help="Displacement range to analyze (default: 2)")
    parser.add_argument('--output', '-o',
                       help="Save results to JSON file")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Only show summary, not detailed results")
    
    args = parser.parse_args()
    
    # Validate displacement range
    if args.range < 1 or args.range > 10:
        print("Error: Displacement range must be between 1 and 10")
        return
    
    # Run analysis
    results = analyze_adjective_positions(args.input_file, args.range)
    
    if not results:
        print("Analysis failed or no results generated.")
        return
    
    # Print results
    if not args.quiet:
        print_results(results)
    else:
        # Just show summary
        print(f"Analysis complete: {results['total_nouns']:,} nouns, "
              f"{results['summary']['total_adjectives_found']:,} adjectives found")
        
        # Show top position
        sorted_by_prob = sorted(results['position_analysis'].items(), 
                               key=lambda x: x[1]['probability'], reverse=True)
        if sorted_by_prob:
            top_pos, top_data = sorted_by_prob[0]
            direction = "before" if top_pos < 0 else "after"
            print(f"Most likely position: {abs(top_pos)} {direction} noun ({top_data['probability']:.1%})")
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == '__main__':
    main() 