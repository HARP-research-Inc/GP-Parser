#!/usr/bin/env python3
"""
Comprehensive Dependency Distance Analysis

This script uses spaCy's dependency parser to analyze ALL dependency relationships
in text and calculates signed distance distributions for each dependency type.

Signed distance = dependent_token_index - head_token_index
- Negative: dependent comes before head
- Positive: dependent comes after head

Usage:
    python dependency-distance-analysis.py input.txt
    python dependency-distance-analysis.py input.txt --output results.json
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

def analyze_dependency_distances(text_file_path: str):
    """
    Analyze signed distances for all dependency relationships.
    
    Args:
        text_file_path: Path to the input text file
    
    Returns:
        dict: Analysis results with dependency distances organized by POS and dependency type
    """
    # Load spaCy model
    nlp = ensure_spacy()
    if nlp is None:
        return {}
    
    print(f"Analyzing dependency distances in '{text_file_path}'")
    
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
    
    # Collect relationships organized by POS -> dependency -> offset counts
    pos_dependency_offsets = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Also collect the original format for backward compatibility
    dependency_distances = defaultdict(list)  # dep_type -> list of signed distances
    total_relationships = 0
    
    for token in doc:
        # Skip ROOT dependencies (they don't have meaningful distances)
        if token.dep_ == "ROOT":
            continue
            
        # Calculate signed distance: dependent - head
        signed_distance = token.i - token.head.i
        
        # Get POS tags
        dependent_pos = token.pos_
        head_pos = token.head.pos_
        
        # Store in the new structured format: POS -> dependency -> offset -> count
        pos_dependency_offsets[dependent_pos][token.dep_][str(signed_distance)] += 1
        
        # Store the relationship for backward compatibility
        dependency_distances[token.dep_].append(signed_distance)
        total_relationships += 1
    
    print(f"Found {total_relationships} dependency relationships across {len(dependency_distances)} dependency types")
    
    # Calculate statistics for each dependency type
    dependency_stats = {}
    
    for dep_type, distances in dependency_distances.items():
        if not distances:
            continue
            
        distances_array = np.array(distances)
        
        # Calculate statistics
        stats = {
            'count': len(distances),
            'mean_signed_distance': float(np.mean(distances_array)),
            'median_signed_distance': float(np.median(distances_array)),
            'std_signed_distance': float(np.std(distances_array)),
            'min_distance': int(np.min(distances_array)),
            'max_distance': int(np.max(distances_array)),
            'before_count': int(np.sum(distances_array < 0)),  # Dependent before head
            'after_count': int(np.sum(distances_array > 0)),   # Dependent after head
            'same_position': int(np.sum(distances_array == 0)), # Same position (shouldn't happen)
            'before_percentage': float(np.sum(distances_array < 0) / len(distances_array) * 100),
            'after_percentage': float(np.sum(distances_array > 0) / len(distances_array) * 100),
            'distance_distribution': dict(Counter(distances))
        }
        
        dependency_stats[dep_type] = stats
    
    # Overall statistics
    all_distances = []
    for distances in dependency_distances.values():
        all_distances.extend(distances)
    
    if all_distances:
        overall_stats = {
            'total_relationships': total_relationships,
            'total_dependency_types': len(dependency_distances),
            'mean_signed_distance': float(np.mean(all_distances)),
            'median_signed_distance': float(np.median(all_distances)),
            'std_signed_distance': float(np.std(all_distances)),
            'min_distance': int(np.min(all_distances)),
            'max_distance': int(np.max(all_distances)),
            'before_count': int(np.sum(np.array(all_distances) < 0)),
            'after_count': int(np.sum(np.array(all_distances) > 0)),
            'before_percentage': float(np.sum(np.array(all_distances) < 0) / len(all_distances) * 100),
            'after_percentage': float(np.sum(np.array(all_distances) > 0) / len(all_distances) * 100)
        }
    else:
        overall_stats = {}
    
    # Convert nested defaultdicts to regular dicts for JSON serialization
    pos_dependency_data = {}
    for pos, dep_dict in pos_dependency_offsets.items():
        pos_dependency_data[pos] = {}
        for dep_type, offset_dict in dep_dict.items():
            pos_dependency_data[pos][dep_type] = dict(offset_dict)
    
    results = {
        'text_file': text_file_path,
        'total_tokens': len(doc),
        'overall_statistics': overall_stats,
        'dependency_statistics': dependency_stats,
        'pos_dependency_offsets': pos_dependency_data  # New structured format
    }
    
    return results

def print_results(results: dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*80)
    print("COMPREHENSIVE DEPENDENCY DISTANCE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Text file: {results['text_file']}")
    print(f"Total tokens: {results['total_tokens']:,}")
    
    if not results['overall_statistics']:
        print("No dependency relationships found.")
        return
    
    overall = results['overall_statistics']
    print(f"Total relationships: {overall['total_relationships']:,}")
    print(f"Dependency types: {overall['total_dependency_types']}")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Average signed distance: {overall['mean_signed_distance']:+.2f}")
    print(f"  Median signed distance:  {overall['median_signed_distance']:+.1f}")
    print(f"  Range: {overall['min_distance']:+d} to {overall['max_distance']:+d}")
    print(f"  Dependent before head: {overall['before_count']:,} ({overall['before_percentage']:.1f}%)")
    print(f"  Dependent after head:  {overall['after_count']:,} ({overall['after_percentage']:.1f}%)")
    
    print(f"\nDEPENDENCY TYPE ANALYSIS:")
    print("-" * 80)
    
    # Sort dependency types by frequency
    sorted_deps = sorted(results['dependency_statistics'].items(), 
                        key=lambda x: x[1]['count'], reverse=True)
    
    for dep_type, stats in sorted_deps:
        print(f"\n{dep_type.upper()} ({stats['count']} relationships):")
        print(f"  Average signed distance: {stats['mean_signed_distance']:+.2f}")
        print(f"  Range: {stats['min_distance']:+d} to {stats['max_distance']:+d}")
        print(f"  Before/After: {stats['before_count']}/{stats['after_count']} "
              f"({stats['before_percentage']:.1f}%/{stats['after_percentage']:.1f}%)")
        
        # Show distance distribution (most common distances)
        dist_items = sorted(stats['distance_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]  # Top 10 distances
        
        if len(dist_items) > 1:
            print(f"  Most common distances:")
            for distance, count in dist_items:
                percentage = (count / stats['count']) * 100
                direction = "before" if distance < 0 else "after" if distance > 0 else "same"
                print(f"    {distance:+3d}: {count:4d} ({percentage:5.1f}%) - dependent {direction} head")

def print_summary(results: dict):
    """Print a condensed summary of results."""
    if not results['overall_statistics']:
        print("No dependency relationships found.")
        return
    
    overall = results['overall_statistics']
    print(f"Analysis complete: {overall['total_relationships']:,} relationships, "
          f"{overall['total_dependency_types']} dependency types")
    print(f"Average signed distance: {overall['mean_signed_distance']:+.2f}")
    print(f"Direction bias: {overall['before_percentage']:.1f}% before, "
          f"{overall['after_percentage']:.1f}% after")
    
    # Show top 5 most frequent dependency types
    sorted_deps = sorted(results['dependency_statistics'].items(), 
                        key=lambda x: x[1]['count'], reverse=True)[:5]
    
    print("Top dependency types:")
    for dep_type, stats in sorted_deps:
        print(f"  {dep_type}: {stats['count']} ({stats['mean_signed_distance']:+.2f} avg distance)")

def print_pos_structured_results(results: dict):
    """Print results organized by POS tag."""
    print("\n" + "="*80)
    print("POS-ORGANIZED DEPENDENCY DISTANCE RESULTS")
    print("="*80)
    
    pos_data = results.get('pos_dependency_offsets', {})
    if not pos_data:
        print("No POS-organized data available.")
        return
    
    # Sort POS tags by total frequency
    pos_totals = {}
    for pos, deps in pos_data.items():
        total = sum(sum(offset_counts.values()) for offset_counts in deps.values())
        pos_totals[pos] = total
    
    sorted_pos = sorted(pos_totals.items(), key=lambda x: x[1], reverse=True)
    
    for pos, total_count in sorted_pos:
        print(f"\n{pos} ({total_count} relationships):")
        print("-" * 50)
        
        deps = pos_data[pos]
        # Sort dependencies by frequency within this POS
        dep_totals = {dep: sum(offset_counts.values()) for dep, offset_counts in deps.items()}
        sorted_deps = sorted(dep_totals.items(), key=lambda x: x[1], reverse=True)
        
        for dep_type, dep_count in sorted_deps:
            offset_counts = deps[dep_type]
            print(f"  {dep_type} ({dep_count} cases):")
            
            # Sort offsets numerically
            sorted_offsets = sorted(offset_counts.items(), key=lambda x: int(x[0]))
            
            for offset_str, count in sorted_offsets:
                offset = int(offset_str)
                percentage = (count / dep_count) * 100
                direction = "before" if offset < 0 else "after" if offset > 0 else "same"
                print(f"    {offset:+3d}: {count:3d} ({percentage:5.1f}%) - dependent {direction} head")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze signed distances for all dependency relationships using spaCy"
    )
    
    parser.add_argument('input_file', 
                       help="Path to the input text file")
    parser.add_argument('--output', '-o',
                       help="Save results to JSON file")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Only show summary, not detailed results")
    parser.add_argument('--pos-only', action='store_true',
                       help="Only show POS-organized results, not dependency-type analysis")
    parser.add_argument('--pos-json', metavar='FILE',
                       help="Save only the POS-organized data structure to a clean JSON file")
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_dependency_distances(args.input_file)
    
    if not results:
        print("Analysis failed or no results generated.")
        return
    
    # Print results
    if args.pos_only:
        print_pos_structured_results(results)
    elif not args.quiet:
        print_results(results)
        print_pos_structured_results(results)
    else:
        print_summary(results)
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Save just the POS-organized structure if requested
    if args.pos_json:
        try:
            pos_data = results.get('pos_dependency_offsets', {})
            with open(args.pos_json, 'w', encoding='utf-8') as f:
                json.dump(pos_data, f, indent=2, ensure_ascii=False)
            print(f"\nPOS-organized data saved to {args.pos_json}")
        except Exception as e:
            print(f"Error saving POS data: {e}")

if __name__ == '__main__':
    main() 