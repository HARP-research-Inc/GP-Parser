#!/usr/bin/env python3
"""
main.py – CCG tree visualizer with direct PNG output

    python main.py "I placed my red hat in Johnny's hand"
    python main.py -o tree.png "The fox jumped over the dog"
"""

import argparse
import sys
import os

from depccg_treeviz import CCGTreeVisualizer


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="Sentence (if empty, read stdin).")
    p.add_argument("-o", "--output", help="Output PNG file path.")
    p.add_argument("--width", type=int, default=12, help="Image width in inches (default: 12)")
    p.add_argument("--height", type=int, default=8, help="Image height in inches (default: 8)")
    return p.parse_args()


def main() -> None:
    args = cli()
    sentences = [" ".join(args.sentence)] if args.sentence else [
        line.strip() for line in sys.stdin if line.strip()
    ]
    
    if not sentences:
        print("No sentences provided. Use: python main.py 'Your sentence here'")
        return
    
    vis = CCGTreeVisualizer()
    
    for i, sentence in enumerate(sentences):
        # Determine output path
        if args.output:
            if len(sentences) == 1:
                output_path = args.output
                # Force PNG extension
                if not output_path.lower().endswith('.png'):
                    output_path = os.path.splitext(output_path)[0] + '.png'
            else:
                # Multiple sentences: add index
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_{i+1}.png"
        else:
            # Auto-generate filename
            safe_sentence = "".join(c for c in sentence if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_sentence = safe_sentence[:50].replace(' ', '_')
            output_path = f"/workspace/output/syntax_tree_{safe_sentence}.png"
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate and save the tree image
        vis.save_tree_image(sentence, output_path, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
