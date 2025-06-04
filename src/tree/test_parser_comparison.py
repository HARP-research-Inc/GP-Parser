#!/usr/bin/env python3
"""
test_parser_comparison.py
------------------------
Test module that compares DepCCG and Benepar parsers on a corpus of sentences
and reports when outputs don't match or when parsers disagree.

Usage:
    python test_parser_comparison.py
    python test_parser_comparison.py --corpus custom_sentences.txt
    python test_parser_comparison.py --quick  # Run on a small test set
"""

import argparse
import sys
import time
from typing import List, Dict, Any, Tuple
import json

# Import our parsers
from depccg_treeviz import CCGTreeVisualizer as DepCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


def get_test_corpus(corpus_type: str = "default") -> List[str]:
    """Get a corpus of test sentences."""
    if corpus_type == "quick":
        return [
            "The cat sleeps.",
            "Dogs bark loudly.",
            "She reads books.",
            "Birds fly south.",
            "Rain falls down."
        ]
    elif corpus_type == "medium":
        return [
            "The cat sleeps quietly.",
            "The old man walked slowly.",
            "She reads interesting books daily.",
            "Birds migrate south in winter.",
            "Heavy rain falls during storms.",
            "The quick brown fox jumps.",
            "Students study hard for exams.",
            "The teacher explains complex concepts.",
            "Children play games outside.",
            "Scientists discover new planets."
        ]
    else:  # default - comprehensive test set
        return [
            # Simple sentences
            "The cat sleeps.",
            "Dogs bark loudly.",
            "She reads books.",
            "Birds fly south.",
            "Rain falls down.",
            
            # Complex sentences
            "The cat sleeps quietly on the mat.",
            "The old man who lived next door walked slowly.",
            "She reads interesting books about science daily.",
            "Birds migrate south when winter approaches.",
            "Heavy rain falls during severe storms.",
            
            # Very complex sentences
            "The quick brown fox jumps over the lazy dog.",
            "Students who study hard usually perform well on difficult exams.",
            "The teacher carefully explains complex mathematical concepts to confused students.",
            "Small children often play imaginative games outside during sunny afternoons.",
            "Dedicated scientists continuously discover fascinating new planets in distant galaxies.",
            
            # Edge cases
            "Yes.",
            "Hello world!",
            "What time is it?",
            "The book that I read yesterday was fascinating.",
            "Although it was raining, we decided to go for a walk.",
            
            # Challenging syntax
            "The man whom I saw yesterday called me today.",
            "Having finished his homework, John went to play.",
            "The car, which was red, drove away quickly.",
            "Neither John nor Mary came to the party.",
            "The more you practice, the better you become."
        ]


def extract_key_features(parse_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key features from parse data for comparison."""
    if not parse_data or not parse_data.get('success', False):
        return {'success': False, 'error': parse_data.get('error', 'Unknown error')}
    
    def extract_leaves(node):
        """Extract leaf words from parse tree."""
        if isinstance(node, dict):
            if 'word' in node:
                return [node['word']]
            elif 'children' in node:
                leaves = []
                for child in node['children']:
                    leaves.extend(extract_leaves(child))
                return leaves
        return []
    
    def count_nodes(node):
        """Count total nodes in parse tree."""
        if isinstance(node, dict):
            if 'word' in node:
                return 1
            elif 'children' in node:
                return 1 + sum(count_nodes(child) for child in node['children'])
        return 1
    
    data = parse_data.get('parse_data', {})
    if not data:
        return {'success': False, 'error': 'No parse data'}
    
    leaves = extract_leaves(data)
    node_count = count_nodes(data)
    
    return {
        'success': True,
        'parser': parse_data.get('parser', 'unknown'),
        'parse_time': parse_data.get('parse_time', 0.0),
        'word_count': len(leaves),
        'node_count': node_count,
        'leaves': leaves,
        'tree_depth': calculate_tree_depth(data),
        'root_category': data.get('cat', 'unknown')
    }


def calculate_tree_depth(node) -> int:
    """Calculate the maximum depth of the parse tree."""
    if isinstance(node, dict):
        if 'word' in node:
            return 1
        elif 'children' in node and node['children']:
            return 1 + max(calculate_tree_depth(child) for child in node['children'])
    return 1


def compare_parse_results(dep_result: Dict[str, Any], ben_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compare results from both parsers and identify differences."""
    dep_features = extract_key_features(dep_result)
    ben_features = extract_key_features(ben_result)
    
    comparison = {
        'sentence': dep_result.get('sentence', ''),
        'both_successful': dep_features.get('success', False) and ben_features.get('success', False),
        'dep_success': dep_features.get('success', False),
        'ben_success': ben_features.get('success', False),
        'time_difference': 0.0,
        'differences': []
    }
    
    if not comparison['both_successful']:
        if not dep_features.get('success', False):
            comparison['differences'].append(f"DepCCG failed: {dep_features.get('error', 'Unknown error')}")
        if not ben_features.get('success', False):
            comparison['differences'].append(f"Benepar failed: {ben_features.get('error', 'Unknown error')}")
        return comparison
    
    # Compare timing
    dep_time = dep_features.get('parse_time', 0.0)
    ben_time = ben_features.get('parse_time', 0.0)
    comparison['time_difference'] = dep_time - ben_time
    
    # Compare word counts
    dep_words = dep_features.get('word_count', 0)
    ben_words = ben_features.get('word_count', 0)
    if dep_words != ben_words:
        comparison['differences'].append(f"Word count mismatch: DepCCG={dep_words}, Benepar={ben_words}")
    
    # Compare tree complexity
    dep_nodes = dep_features.get('node_count', 0)
    ben_nodes = ben_features.get('node_count', 0)
    node_diff = abs(dep_nodes - ben_nodes)
    if node_diff > dep_words * 0.2:  # Allow some variation but flag major differences
        comparison['differences'].append(f"Tree complexity mismatch: DepCCG={dep_nodes} nodes, Benepar={ben_nodes} nodes")
    
    # Compare tree depth
    dep_depth = dep_features.get('tree_depth', 0)
    ben_depth = ben_features.get('tree_depth', 0)
    depth_diff = abs(dep_depth - ben_depth)
    if depth_diff > 2:  # Flag significant depth differences
        comparison['differences'].append(f"Tree depth mismatch: DepCCG={dep_depth}, Benepar={ben_depth}")
    
    # Compare actual words (tokenization differences)
    dep_leaves = dep_features.get('leaves', [])
    ben_leaves = ben_features.get('leaves', [])
    if dep_leaves != ben_leaves:
        comparison['differences'].append(f"Tokenization mismatch: DepCCG={dep_leaves}, Benepar={ben_leaves}")
    
    return comparison


def run_parser_comparison(corpus: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Run comparison test on the corpus."""
    print(f"🔬 Starting parser comparison on {len(corpus)} sentences...")
    
    dep_parser = DepCCG()
    ben_parser = Benepar()
    
    results = {
        'total_sentences': len(corpus),
        'both_successful': 0,
        'dep_only_successful': 0,
        'ben_only_successful': 0,
        'both_failed': 0,
        'mismatches': [],
        'timing_stats': {'dep_total': 0.0, 'ben_total': 0.0},
        'detailed_results': []
    }
    
    for i, sentence in enumerate(corpus, 1):
        if verbose:
            print(f"\n📝 Testing {i}/{len(corpus)}: {sentence}")
        
        # Parse with both systems
        dep_result = dep_parser.parse_only(sentence)
        ben_result = ben_parser.parse_only(sentence)
        
        # Compare results
        comparison = compare_parse_results(dep_result, ben_result)
        results['detailed_results'].append(comparison)
        
        # Update statistics
        if comparison['both_successful']:
            results['both_successful'] += 1
            results['timing_stats']['dep_total'] += dep_result.get('parse_time', 0.0)
            results['timing_stats']['ben_total'] += ben_result.get('parse_time', 0.0)
        elif comparison['dep_success'] and not comparison['ben_success']:
            results['dep_only_successful'] += 1
        elif comparison['ben_success'] and not comparison['dep_success']:
            results['ben_only_successful'] += 1
        else:
            results['both_failed'] += 1
        
        # Report differences
        if comparison['differences']:
            results['mismatches'].append(comparison)
            if verbose:
                print(f"⚠️  Differences found:")
                for diff in comparison['differences']:
                    print(f"   - {diff}")
        elif verbose:
            print("✅ Parsers agree (no significant differences)")
    
    return results


def print_summary_report(results: Dict[str, Any]):
    """Print a summary report of the comparison."""
    total = results['total_sentences']
    both_ok = results['both_successful'] 
    dep_only = results['dep_only_successful']
    ben_only = results['ben_only_successful']
    both_fail = results['both_failed']
    
    print("\n" + "="*60)
    print("📊 PARSER COMPARISON SUMMARY")
    print("="*60)
    
    print(f"📈 Overall Statistics:")
    print(f"   Total sentences tested: {total}")
    print(f"   Both parsers successful: {both_ok} ({both_ok/total*100:.1f}%)")
    print(f"   Only DepCCG successful: {dep_only} ({dep_only/total*100:.1f}%)")
    print(f"   Only Benepar successful: {ben_only} ({ben_only/total*100:.1f}%)")
    print(f"   Both parsers failed: {both_fail} ({both_fail/total*100:.1f}%)")
    
    if both_ok > 0:
        dep_avg = results['timing_stats']['dep_total'] / both_ok
        ben_avg = results['timing_stats']['ben_total'] / both_ok
        print(f"\n⏱️  Average Timing (successful parses only):")
        print(f"   DepCCG: {dep_avg:.3f}s per sentence")
        print(f"   Benepar: {ben_avg:.3f}s per sentence")
        print(f"   Speed ratio: Benepar is {dep_avg/ben_avg:.1f}x faster")
    
    mismatches = len(results['mismatches'])
    if mismatches > 0:
        print(f"\n🔍 Mismatches and Differences:")
        print(f"   Total sentences with differences: {mismatches} ({mismatches/total*100:.1f}%)")
        print(f"\n📋 Detailed Mismatches:")
        for i, mismatch in enumerate(results['mismatches'][:10], 1):  # Show first 10
            print(f"   {i}. \"{mismatch['sentence'][:50]}{'...' if len(mismatch['sentence']) > 50 else ''}\"")
            for diff in mismatch['differences']:
                print(f"      - {diff}")
        if len(results['mismatches']) > 10:
            print(f"      ... and {len(results['mismatches']) - 10} more")
    else:
        print(f"\n✅ Perfect Agreement: No significant differences found!")


def main():
    parser = argparse.ArgumentParser(description="Compare DepCCG and Benepar parsers")
    parser.add_argument("--corpus", choices=["quick", "medium", "default"], default="default",
                      help="Size of test corpus")
    parser.add_argument("--corpus-file", help="Path to custom corpus file (one sentence per line)")
    parser.add_argument("--quiet", action="store_true", help="Only show summary, not per-sentence details")
    parser.add_argument("--save-results", help="Save detailed results to JSON file")
    
    args = parser.parse_args()
    
    # Get corpus
    if args.corpus_file:
        try:
            with open(args.corpus_file, 'r') as f:
                corpus = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Corpus file not found: {args.corpus_file}")
            sys.exit(1)
    else:
        corpus = get_test_corpus(args.corpus)
    
    # Run comparison
    results = run_parser_comparison(corpus, verbose=not args.quiet)
    
    # Print summary
    print_summary_report(results)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {args.save_results}")


if __name__ == "__main__":
    main() 