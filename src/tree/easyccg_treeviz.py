#!/usr/bin/env python3
"""
easyccg_treeviz.py
------------------
Same interface as depccg_treeviz, but uses EasyCCG to build a CCG
derivation and converts it to the same “subphrase‐only” JSON tree
format.  Everything else (layout + drawing) is identical.

Dependencies:
    pip install easyccg nltk matplotlib networkx

Usage (in place of depccg_treeviz):
    from easyccg_treeviz import EasyCCGTreeVisualizer as CCGTreeVisualizer
    vis = CCGTreeVisualizer()
    vis.save_tree_image("I placed my red hat in Johnny's hand", "out.png")
"""

import time
from typing import Any, Dict, List, Tuple, Optional

import easyccg
import nltk  # EasyCCG’s output is an nltk.Tree
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import patches, patheffects

# --------------------------------------------------------------------
#  Load EasyCCG model (one-time; ~35 MB download)
# --------------------------------------------------------------------
#   By default this downloads “model/” under ~/.easyccg
_parser = easyccg.Parser.load_model("model")


# --------------------------------------------------------------------
#  Helper: convert an nltk.Tree (from EasyCCG) → our JSON‐dict tree
# --------------------------------------------------------------------
def _flatten_leaves_eccg(node: Any) -> List[str]:
    """Given a node that is either an nltk.Tree or a string leaf, return its leaves."""
    if isinstance(node, str):
        return [node]
    return [w for leaf in node.leaves() for w in [leaf]]


def _convert_nltk_tree(nltk_node: Any) -> Dict[str, Any]:
    """
    Convert an nltk.Tree from EasyCCG into our Dict format:
        { "cat": <CCG category>, "children": [ child1, child2 ] }
    And at leaves: { "word": "..." }.
    """
    if isinstance(nltk_node, str):
        return {"word": nltk_node}

    label = nltk_node.label()  # e.g. "S[dcl]" or "NP" or "S\NP"
    children = [ _convert_nltk_tree(c) for c in nltk_node ]
    return {"cat": label, "children": children}


# --------------------------------------------------------------------
#  The “drop-in” visualizer class replacing depcccg_treeviz's version
# --------------------------------------------------------------------
class EasyCCGTreeVisualizer:
    """Tree visualizer using EasyCCG + matplotlib for direct image output."""

    def __init__(self):
        # nothing to configure
        pass

    def _parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Run EasyCCG on a single sentence (whitespace‐split).  Returns a dict:
            {
              'success': True/False,
              'parse_data': <dict tree> or None,
              'parse_time': float seconds,
              'error': <str> or None
            }
        """
        words = sentence.strip().split()
        try:
            start = time.time()
            derivations = _parser.parse(words)       # returns List[nltk.Tree]
            duration = time.time() - start

            if derivations:
                best = derivations[0]
                tree_dict = _convert_nltk_tree(best)
                return {
                    "success": True,
                    "parse_data": tree_dict,
                    "parse_time": duration,
                    "error": None,
                }
            else:
                return {
                    "success": False,
                    "parse_data": None,
                    "parse_time": duration,
                    "error": "EasyCCG returned no derivation",
                }

        except Exception as e:
            return {
                "success": False,
                "parse_data": None,
                "parse_time": 0.0,
                "error": str(e),
            }

    # …—————————————————————————————————————————————————————————————————
    #  The methods below are *identical* to depccg_treeviz.py
    #  except for spelling “EasyCCG” in the banner.
    # ——————————————————————————————————————————————————————————————————
    def _flatten_leaves_from_dict(self, node: Dict[str, Any]) -> List[str]:
        """Extract leaf tokens from a tree dictionary structure (same as DepCCG version)."""
        if isinstance(node, dict):
            if "word" in node:
                return [node["word"]]
            if "children" in node and node["children"]:
                out: List[str] = []
                for child in node["children"]:
                    out.extend(self._flatten_leaves_from_dict(child))
                return out
        elif isinstance(node, str):
            return [node]
        return []

    def _hierarchy_pos(
        self,
        G: nx.DiGraph,
        root: int,
        width: float = 1.0,
        vert_gap: float = 0.25,
        y0: float = 0.0,
    ) -> Dict[int, Tuple[float, float]]:
        if root not in G:
            return {root: (0.5, y0)}

        def _place(
            n: int, x0: float, x1: float, y: float, pos: Dict[int, Tuple[float, float]]
        ):
            kids = list(G.successors(n))
            if not kids:
                pos[n] = ((x0 + x1) * 0.5, y)
                return
            slice_w = (x1 - x0) / len(kids)
            for i, k in enumerate(kids):
                _place(k, x0 + i * slice_w, x0 + (i + 1) * slice_w, y - vert_gap, pos)
            xs = [pos[c][0] for c in kids]
            pos[n] = (sum(xs) / len(xs) / len(xs), y)

        pos: Dict[int, Tuple[float, float]] = {}
        _place(root, 0.0, width, y0, pos)
        return pos

    def _dict_tree_to_graph(
        self, tree_data: Dict[str, Any]
    ) -> Tuple[nx.DiGraph, int, Dict[int, str], Dict[int, str]]:
        spans: Dict[int, str] = {}
        edges: List[Tuple[int, int]] = {}
        labels: Dict[int, str] = {}
        # (no categories needed, we only show spans)

        def walk(node: Any, parent_id: Optional[int] = None) -> int:
            my_id = len(spans)
            if isinstance(node, dict) and "children" in node:
                # internal node
                span_text = " ".join(self._flatten_leaves_from_dict(node)) or "ε"
                spans[my_id] = span_text
                labels[my_id] = span_text
                if parent_id is not None:
                    edges.append((parent_id, my_id))
                for ch in node["children"]:
                    walk(ch, my_id)
            else:
                # leaf assumed to be {"word": "..."}
                if isinstance(node, dict) and "word" in node:
                    w = node["word"]
                else:
                    w = str(node)
                spans[my_id] = w
                labels[my_id] = w
                if parent_id is not None:
                    edges.append((parent_id, my_id))
            return my_id

        root_id = walk(tree_data)
        G = nx.DiGraph()
        G.add_nodes_from(spans.keys())
        G.add_edges_from(edges)
        return G, root_id, labels, spans

    def _print_tree(
        self,
        G: nx.DiGraph,
        root_id: int,
        labels: Dict[int, str],
        spans: Dict[int, str],
        indent: str = "",
        is_last: bool = True,
    ):
        if root_id not in G:
            return

        span = spans[root_id]
        children = list(G.successors(root_id))
        is_leaf = len(children) == 0
        branch = "└── " if is_last else "├── "

        if is_leaf:
            print(f"{indent}{branch}{span}")
        else:
            print(f"{indent}{branch}{span}")

        for i, ch in enumerate(children):
            last = i == len(children) - 1
            nxt_indent = indent + ("    " if is_last else "│   ")
            self._print_tree(G, ch, labels, spans, nxt_indent, last)

    def save_tree_image(
        self, sentence: str, output_path: str, width: int = 12, height: int = 8
    ):
        print(f"[easyccg_treeviz] Parsing: {sentence}")
        parse_result = self._parse_sentence(sentence)

        if not parse_result["success"]:
            print(f"❌ Parse failed: {parse_result['error']}")
            print(f"⏱️  Parse time: {parse_result['parse_time']:.3f} seconds")
        else:
            print(f"⏱️  Parse time: {parse_result['parse_time']:.3f} seconds")
            G, root_id, labels, spans = self._dict_tree_to_graph(parse_result["parse_data"])
            if G.number_of_nodes() == 0:
                print("❌ No tree structure found")
            else:
                print("🌳 Parse tree structure:")
                print("=" * 50)
                self._print_tree(G, root_id, labels, spans)
                print("=" * 50)

        # Draw the PNG exactly as in depccg_treeviz
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.axis("off")

        if not parse_result["success"]:
            msg = f"Parse failed: {sentence}\nError: {parse_result['error']}"
            ax.text(
                0.5,
                0.5,
                msg,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
            plt.title("Parse Failed", fontsize=16, color="red")
        else:
            if G.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "No tree structure", ha="center", va="center")
            else:
                pos = self._hierarchy_pos(G, root_id, width=0.9, vert_gap=0.15, y0=0.9)
                # draw edges + nodes
                for (u, v) in G.edges():
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    ax.plot([x0, x1], [y0, y1], "k-", linewidth=1.5, alpha=0.7)

                for nid in G.nodes():
                    x, y = pos[nid]
                    text = spans[nid]
                    is_leaf = G.out_degree(nid) == 0
                    face = "lightblue" if is_leaf else "lightgreen"
                    ax.text(
                        x,
                        y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=10 if is_leaf else 9,
                        fontweight="bold" if is_leaf else "normal",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=face, alpha=0.7),
                    )
            plt.title(f"EasyCCG Parse: {sentence}", fontsize=14, pad=20)

        plt.subplots_adjust(top=0.85)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"💾 Saved image: {output_path}")
