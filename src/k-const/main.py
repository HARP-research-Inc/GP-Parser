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

def main():
    p = argparse.ArgumentParser(
        description="Estimate a word's POS-tag distribution (Universal tags) in the Brown corpus."
    )
    p.add_argument('word', help="The word to analyze (case-insensitive).")
    args = p.parse_args()

    dist = get_pos_distribution(args.word)
    if not dist:
        print(f"No occurrences of '{args.word}' found in the Brown corpus.")
    else:
        print(f"POS distribution for '{args.word}' (Brown corpus):")
        for tag, info in sorted(dist.items(), key=lambda x: -x[1]['count']):
            print(f"  {tag:6} → {info['count']:5} times  ({info['relative']*100:.1f}%)")

if __name__ == '__main__':
    main() 