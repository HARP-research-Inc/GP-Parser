#!/usr/bin/env python3
"""
main.py – CCG tree visualizer with direct PNG output

    python main.py "I placed my red hat in Johnny's hand"
    python main.py -o tree.png "The fox jumped over the dog"
"""

import argparse
import sys
import os
import re
import time
import json

from depccg_treeviz import CCGTreeVisualizer as DepCCG
#from easyccg_treeviz import EasyCCGTreeVisualizer as EasyCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="Sentence (if empty, read stdin).")
    p.add_argument("-o", "--output", help="Output PNG file path.")
    p.add_argument("--width", type=int, default=12, help="Image width in inches (default: 12)")
    p.add_argument("--height", type=int, default=8, help="Image height in inches (default: 8)")
    p.add_argument("--no-save", action="store_true", help="Don't save files, just parse and display")
    return p.parse_args()


def sanitize_filename(text: str, max_length: int = 20) -> str:
    """Convert text to a safe filename by removing punctuation and limiting length."""
    # Remove/replace problematic characters
    safe = re.sub(r'[^\w\s-]', '', text.strip())
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    # Limit length
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe or "sentence"


def main():
    args = cli()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    sentences = [" ".join(args.sentence)] if args.sentence else [line.strip() for line in sys.stdin if line.strip()]
    if not sentences:
        print("No sentences provided.")
        return

    # instantiate each backend
    vis_dep   = DepCCG()
    #vis_easy  = EasyCCG()
    vis_bene  = Benepar()

    for sentence in sentences:
        if args.no_save:
            # Parse-only mode - no file output
            print("\n=== DepCCG ===")
            start_time = time.time()
            dep_data = vis_dep.parse_only(sentence)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")

            print("\n=== EasyCCG ===")
            #vis_easy.parse_only(sentence)

            print("\n=== Benepar ===")
            start_time = time.time()
            ben_data = vis_bene.parse_only(sentence)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
        else:
            # Full mode with file output
            # build separate output filenames in the output folder
            base = args.output or f"output/{sanitize_filename(sentence[:20])}"
            dep_out  = base + "_depccg.png"
            dep_json = base + "_depccg.json"
            #easy_out = base + "_easyccg.png"
            ben_out  = base + "_benepar.png"
            ben_json = base + "_benepar.json"

            print("\n=== DepCCG ===")
            start_time = time.time()
            dep_data = vis_dep.save_tree_image(sentence, dep_out, width=args.width, height=args.height)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            # Save DepCCG JSON
            if dep_data:
                with open(dep_json, 'w') as f:
                    json.dump(dep_data, f, indent=2)
                print(f"💾 Saved JSON: {dep_json}")

            print("\n=== EasyCCG ===")
            #vis_easy.save_tree_image(sentence, easy_out, width=args.width, height=args.height)

            print("\n=== Benepar ===")
            start_time = time.time()
            ben_data = vis_bene.save_tree_image(sentence, ben_out, width=args.width, height=args.height)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            
            # Save Benepar JSON
            if ben_data:
                with open(ben_json, 'w') as f:
                    json.dump(ben_data, f, indent=2)
                print(f"💾 Saved JSON: {ben_json}")


if __name__ == "__main__":
    main()
