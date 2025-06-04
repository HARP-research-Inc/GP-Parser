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

from depccg_treeviz import CCGTreeVisualizer as DepCCG
#from easyccg_treeviz import EasyCCGTreeVisualizer as EasyCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="Sentence (if empty, read stdin).")
    p.add_argument("-o", "--output", help="Output PNG file path.")
    p.add_argument("--width", type=int, default=12, help="Image width in inches (default: 12)")
    p.add_argument("--height", type=int, default=8, help="Image height in inches (default: 8)")
    return p.parse_args()


def main():
    args = cli()
    sentences = [" ".join(args.sentence)] if args.sentence else [line.strip() for line in sys.stdin if line.strip()]

    # instantiate each backend
    vis_dep   = DepCCG()
    #vis_easy  = EasyCCG()
    vis_bene  = Benepar()

    for sentence in sentences:
        # build 3 separate output filenames
        base = args.output or f"output_{sentence[:20].replace(' ','_')}"
        dep_out  = base + "_depccg.png"
        #easy_out = base + "_easyccg.png"
        ben_out  = base + "_benepar.png"

        print("\n=== DepCCG ===")
        vis_dep.save_tree_image(sentence, dep_out, width=args.width, height=args.height)

        print("\n=== EasyCCG ===")
        #vis_easy.save_tree_image(sentence, easy_out, width=args.width, height=args.height)

        print("\n=== Benepar ===")
        vis_bene.save_tree_image(sentence, ben_out, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
