#!/usr/bin/env python3
"""
Dependency-Based Adjective-Noun Analysis

This script uses spaCy's dependency parser to find actual syntactic relationships
between adjectives and nouns, then calculates statistics about their distances.

Usage:
    python dependency-adjective-analysis.py input.txt
    python dependency-adjective-analysis.py input.txt --output results.json
"""

import argparse
import json
import os
from collections import defaultdict, Counter
import numpy as np

def ensure_spacy():
    """Ensure spaCy is available and download model if needed."""
    try:
        import spacy
    except ImportError:
        print("Error: spaCy is not installed. Please install it with:")
        print("pip install spacy")
        print("python -m spacy download en_core_web_sm")
        return None
    
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Error: spaCy English model not found. Please download it with:")
        print("python -m spacy download en_core_web_sm")
        return None

def analyze_adjective_noun_dependencies(text_file_path: str):
    """
    Analyze adjective-noun dependencies using spaCy's dependency parser.
    
    Args:
        text_file_path: Path to the input text file
    
    Returns:
        dict: Analysis results with dependency relationships and distances
    """
    # Load spaCy model
    nlp = ensure_spacy()
    if nlp is None:
        return {}
    
    print(f"Analyzing adjective-noun dependencies in '{text_file_path}'")
    
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
    
    print("Processing text with spaCy dependency parser...")
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Find adjective-noun relationships
    adj_noun_relationships = []
    dependency_types = Counter()
    distances = []
    
    # Look for different types of adjective-noun relationships
    for token in doc:
        if token.pos_ == "ADJ":
            # Check if this adjective modifies a noun
            if token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Adjectival modifier relationship
                adj_noun_relationships.append({
                    'adjective': token.text.lower(),
                    'noun': token.head.text.lower(),
                    'adj_index': token.i,
                    'noun_index': token.head.i,
                    'distance': abs(token.i - token.head.i),
                    'dependency': token.dep_,
                    'direction': 'before' if token.i < token.head.i else 'after'
                })
                dependency_types[token.dep_] += 1
                distances.append(abs(token.i - token.head.i))
            
            elif token.head.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nmod", "compound"]:
                # Other adjective-noun relationships
                adj_noun_relationships.append({
                    'adjective': token.text.lower(),
                    'noun': token.head.text.lower(),
                    'adj_index': token.i,
                    'noun_index': token.head.i,
                    'distance': abs(token.i - token.head.i),
                    'dependency': token.dep_,
                    'direction': 'before' if token.i < token.head.i else 'after'
                })
                dependency_types[token.dep_] += 1
                distances.append(abs(token.i - token.head.i))
    
    # Also check for predicative adjectives (adjectives that are predicates of nouns)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            for child in token.children:
                if child.pos_ == "ADJ" and child.dep_ in ["acomp", "xcomp"]:
                    adj_noun_relationships.append({
                        'adjective': child.text.lower(),
                        'noun': token.text.lower(),
                        'adj_index': child.i,
                        'noun_index': token.i,
                        'distance': abs(child.i - token.i),
                        'dependency': child.dep_,
                        'direction': 'before' if child.i < token.i else 'after'
                    })
                    dependency_types[child.dep_] += 1
                    distances.append(abs(child.i - token.i))
    
    print(f"Found {len(adj_noun_relationships)} adjective-noun dependency relationships")
    
    # Calculate statistics
    if distances:
        distance_stats = {
            'mean': np.mean(distances),
            'median': np.median(distances),
            'std': np.std(distances),
            'min': min(distances),
            'max': max(distances)
        }
    else:
        distance_stats = {
            'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0
        }
    
    # Analyze by direction
    before_count = sum(1 for rel in adj_noun_relationships if rel['direction'] == 'before')
    after_count = sum(1 for rel in adj_noun_relationships if rel['direction'] == 'after')
    
    # Analyze by distance
    distance_distribution = Counter(distances)
    
    # Analyze by dependency type
    dependency_distances = defaultdict(list)
    for rel in adj_noun_relationships:
        dependency_distances[rel['dependency']].append(rel['distance'])
    
    dependency_stats = {}
    for dep_type, dist_list in dependency_distances.items():
        if dist_list:
            dependency_stats[dep_type] = {
                'count': len(dist_list),
                'mean_distance': np.mean(dist_list),
                'median_distance': np.median(dist_list),
                'std_distance': np.std(dist_list) if len(dist_list) > 1 else 0
            }
    
    # Most common adjective-noun pairs
    pair_counts = Counter()
    for rel in adj_noun_relationships:
        pair_counts[(rel['adjective'], rel['noun'])] += 1
    
    # Convert tuple keys to strings for JSON serialization
    most_common_pairs = {}
    for (adj, noun), count in pair_counts.most_common(10):
        key = f"{adj} + {noun}"
        most_common_pairs[key] = count
    
    results = {
        'text_file': text_file_path,
        'total_tokens': len(doc),
        'total_relationships': len(adj_noun_relationships),
        'dependency_types': dict(dependency_types),
        'direction_analysis': {
            'adjectives_before_nouns': before_count,
            'adjectives_after_nouns': after_count,
            'before_percentage': (before_count / len(adj_noun_relationships) * 100) if adj_noun_relationships else 0,
            'after_percentage': (after_count / len(adj_noun_relationships) * 100) if adj_noun_relationships else 0
        },
        'distance_statistics': distance_stats,
        'distance_distribution': dict(distance_distribution),
        'dependency_statistics': dependency_stats,
        'most_common_pairs': most_common_pairs,
        'sample_relationships': adj_noun_relationships[:20]  # First 20 for inspection
    }
    
    return results

def print_results(results: dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*70)
    print("DEPENDENCY-BASED ADJECTIVE-NOUN ANALYSIS RESULTS")
    print("="*70)
    
    print(f"Text file: {results['text_file']}")
    print(f"Total tokens: {results['total_tokens']:,}")
    print(f"Total adj-noun relationships found: {results['total_relationships']:,}")
    
    if results['total_relationships'] == 0:
        print("No adjective-noun relationships found.")
        return
    
    print(f"\nDEPENDENCY TYPES:")
    for dep_type, count in sorted(results['dependency_types'].items()):
        print(f"  {dep_type:10} → {count:4} relationships")
    
    print(f"\nDIRECTIONAL ANALYSIS:")
    direction = results['direction_analysis']
    print(f"  Adjectives before nouns: {direction['adjectives_before_nouns']:4} ({direction['before_percentage']:.1f}%)")
    print(f"  Adjectives after nouns:  {direction['adjectives_after_nouns']:4} ({direction['after_percentage']:.1f}%)")
    
    print(f"\nDISTANCE STATISTICS:")
    stats = results['distance_statistics']
    print(f"  Average distance: {stats['mean']:.2f} tokens")
    print(f"  Median distance:  {stats['median']:.1f} tokens")
    print(f"  Std deviation:    {stats['std']:.2f}")
    print(f"  Range:           {stats['min']}-{stats['max']} tokens")
    
    print(f"\nDISTANCE DISTRIBUTION:")
    dist_dist = results['distance_distribution']
    for distance in sorted(dist_dist.keys())[:10]:  # Show first 10 distances
        count = dist_dist[distance]
        percentage = (count / results['total_relationships']) * 100
        bar = "█" * int(percentage / 2)  # Scale to reasonable bar length
        print(f"  {distance:2} tokens: {count:4} relationships ({percentage:5.1f}%) {bar}")
    
    print(f"\nDEPENDENCY TYPE STATISTICS:")
    for dep_type, stats in results['dependency_statistics'].items():
        print(f"  {dep_type}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg distance: {stats['mean_distance']:.2f} tokens")
        print(f"    Median distance: {stats['median_distance']:.1f} tokens")
    
    print(f"\nMOST COMMON ADJECTIVE-NOUN PAIRS:")
    for pair_desc, count in list(results['most_common_pairs'].items())[:10]:
        print(f"  {pair_desc}: {count} times")
    
    print(f"\nSAMPLE RELATIONSHIPS:")
    print("  Adjective     | Noun          | Distance | Dependency | Direction")
    print("  --------------|---------------|----------|------------|----------")
    for rel in results['sample_relationships'][:10]:
        print(f"  {rel['adjective']:13} | {rel['noun']:13} | {rel['distance']:8} | {rel['dependency']:10} | {rel['direction']}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze adjective-noun dependencies using spaCy's dependency parser"
    )
    
    parser.add_argument('input_file', 
                       help="Path to the input text file")
    parser.add_argument('--output', '-o',
                       help="Save results to JSON file")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Only show summary, not detailed results")
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_adjective_noun_dependencies(args.input_file)
    
    if not results:
        print("Analysis failed or no results generated.")
        return
    
    # Print results
    if not args.quiet:
        print_results(results)
    else:
        # Just show summary
        print(f"Analysis complete: {results['total_relationships']:,} adj-noun relationships found")
        if results['total_relationships'] > 0:
            stats = results['distance_statistics']
            print(f"Average distance: {stats['mean']:.2f} tokens")
            direction = results['direction_analysis']
            print(f"Direction: {direction['before_percentage']:.1f}% before, {direction['after_percentage']:.1f}% after")
    
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