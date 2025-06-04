#!/usr/bin/env python3
"""
main.py – quick driver for depccg_treeviz.

    python main.py "I placed my red hat in Johnny's hand"
    python main.py -o tree.html "The fox jumped over the dog"
"""

import argparse
import sys
import webbrowser
from tempfile import NamedTemporaryFile

import holoviews as hv
hv.extension('bokeh')

from depccg_treeviz import CCGTreeVisualizer


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="Sentence (if empty, read stdin).")
    p.add_argument("-o", "--output", help="Write HTML instead of auto-opening.")
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
    
    # Create visualizations
    hv_objs = []
    for sentence in sentences:
        try:
            graph = vis.visualize(sentence)
            # Add title to each graph
            titled_graph = graph.opts(title=sentence[:50] + "..." if len(sentence) > 50 else sentence)
            hv_objs.append(titled_graph)
        except Exception as e:
            print(f"Error visualizing '{sentence}': {e}")
            continue
    
    if not hv_objs:
        print("No successful visualizations generated.")
        return
    
    # Create layout
    if len(hv_objs) == 1:
        layout = hv_objs[0]
    else:
        layout = hv.Layout(hv_objs).cols(1)

    # Save to file
    if args.output:
        hv.save(layout, args.output)
        print(f"wrote {args.output}")
    else:
        # Save to temporary file and open
        tmp = NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()
        hv.save(layout, tmp.name)
        webbrowser.open(f"file://{tmp.name}")
        print(f"Opened visualization in browser: {tmp.name}")


if __name__ == "__main__":
    main()
