#!/usr/bin/env python3
import argparse
import importlib.util
import sys
import os

# Import from prob-dist.py (handle hyphen in filename)
spec = importlib.util.spec_from_file_location("prob_dist", os.path.join(os.path.dirname(__file__), "prob-dist.py"))
prob_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prob_dist)
get_pos_distribution = prob_dist.get_pos_distribution
process_text_file_bulk = prob_dist.process_text_file_bulk

def main():
    p = argparse.ArgumentParser(
        description="Estimate POS-tag distributions (Universal tags) from the Brown corpus."
    )
    
    # Create subparsers for different modes
    subparsers = p.add_subparsers(dest='mode', help='Processing mode')
    
    # Single word mode
    single_parser = subparsers.add_parser('word', help='Process a single word')
    single_parser.add_argument('word', help="The word to analyze (case-insensitive).")
    
    # Bulk mode for text files
    bulk_parser = subparsers.add_parser('bulk', help='Process all tokens from a text file')
    bulk_parser.add_argument('input_file', help="Path to the input text file.")
    bulk_parser.add_argument('-o', '--output', help="Optional output JSON file path.")
    bulk_parser.add_argument('--print-summary', action='store_true', 
                           help="Print a summary of results to console.")
    
    # If no subcommand provided, default to single word mode for backward compatibility
    if len(sys.argv) == 2 and not sys.argv[1].startswith('-'):
        # Assume it's a single word
        args = argparse.Namespace(mode='word', word=sys.argv[1])
    else:
        args = p.parse_args()
        if args.mode is None:
            p.print_help()
            return

    if args.mode == 'word':
        # Single word processing
        dist = get_pos_distribution(args.word)
        if not dist:
            print(f"No occurrences of '{args.word}' found in the Brown corpus.")
        else:
            print(f"POS distribution for '{args.word}' (Brown corpus):")
            for tag, info in sorted(dist.items(), key=lambda x: -x[1]['count']):
                print(f"  {tag:6} → {info['count']:5} times  ({info['relative']*100:.1f}%)")
    
    elif args.mode == 'bulk':
        # Bulk text file processing
        results = process_text_file_bulk(args.input_file, args.output)
        
        if args.print_summary and results:
            print(f"\nSample results:")
            # Show first 5 words with distributions
            sample_words = list(results.keys())[:5]
            for word in sample_words:
                dist = results[word]
                if dist:
                    print(f"\n'{word}':")
                    for tag, info in sorted(dist.items(), key=lambda x: -x[1]['count']):
                        print(f"  {tag:6} → {info['count']:5} times  ({info['relative']*100:.1f}%)")
                else:
                    print(f"'{word}': No occurrences found")

if __name__ == '__main__':
    main() 