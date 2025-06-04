#!/usr/bin/env python3
"""
spacy_benepar_treeviz.py
------------------------
Same interface as depccg_treeviz, but uses spaCy + Benepar
(constituency) to build a phrase tree and converts it into
the identical “subphrase‐only” dict‐tree.  The PNG output
follows exactly the same layout + styling.

Dependencies:
    pip install spacy benepar matplotlib networkx
    python -m spacy download en_core_web_sm
    python -m benepar.download_en3

Usage (instead of depccg_treeviz):
    from spacy_benepar_treeviz import BeneparTreeVisualizer as CCGTreeVisualizer
    vis = CCGTreeVisualizer()
    vis.save_tree_image("I placed my red hat in Johnny's hand", "out.png")
"""

import time
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import spacy
import benepar

# --------------------------------------------------------------------
#  Load spaCy + Benepar (one-time)
# --------------------------------------------------------------------
_nlp = spacy.load("en_core_web_sm")
_nlp.add_pipe("benepar", config={"model": "benepar_en3"})


# --------------------------------------------------------------------
#  Helpers to convert an nltk.Tree → our dict format
# --------------------------------------------------------------------
def _flatten_leaves_benepar(node: Any) -> List[str]:
    if isinstance(node, str):
        return [node]
    return [t for leaf in node.leaves() for t in [leaf]]


def _convert_nltk_tree(nltk_node: Any) -> Dict[str, Any]:
    if isinstance(nltk_node, str):
        return {"word": nltk_node}
    label = nltk_node.label()
    children = [ _convert_nltk_tree(c) for c in nltk_node ]
    return {"cat": label, "children": children}


# --------------------------------------------------------------------
#  The “drop-in” visualizer class using spaCy + Benepar
# --------------------------------------------------------------------
class BeneparTreeVisualizer:
    """Tree visualizer using spaCy + Benepar for direct image output."""

    def __init__(self):
        pass

    def _parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Run spaCy+Benepar on the sentence.  Returns:
            {
              'success': True/False,
              'parse_data': <dict tree> or None,
              'parse_time': float seconds,
              'error': <str> or None
            }
        """
        try:
            start = time.time()
            doc = _nlp(sentence.strip())
            duration = time.time() - start

            # take the first sentence
            if not list(doc.sents):
                return {
                    "success": False,
                    "parse_data": None,
                    "parse_time": duration,
                    "error": "spaCy found no sentence",
                }

            sp = list(doc.sents)[0]                  # a spacy Span
            # Benepar parse is stored as an attribute
            #            ""    0           1
            #    sp._.parse_string → "(ROOT (S (NP ...) (VP ...)))"
            # Convert to nltk.Tree
            import nltk

            nltk_tree = nltk.Tree.fromstring(sp._.parse_string)
            tree_dict = _convert_nltk_tree(nltk_tree)
            return {
                "success": True,
                "parse_data": tree_dict,
                "parse_time": duration,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "parse_data": None,
                "parse_time": 0.0,
                "error": str(e),
            }

    # …—————————————————————————————————————————————————————————————————
    # The methods below are identical to depccg_treeviz.py
    # (block‐for‐block), except for “Benepar” in the banner.
    # ——————————————————————————————————————————————————————————————————
    def _flatten_leaves_from_dict(self, node: Dict[str, Any]) -> List[str]:
        if isinstance(node, dict):
            if "word" in node:
                return [node["word"]]
            if "children" in node and node["children"]:
                out: List[str] = []
                for c in node["children"]:
                    out.extend(self._flatten_leaves_from_dict(c))
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
        # note: no category storage—we only show spans

        def walk(node: Any, parent_id: Optional[int] = None) -> int:
            my_id = len(spans)
            if isinstance(node, dict) and "children" in node:
                span_text = " ".join(self._flatten_leaves_from_dict(node)) or "ε"
                spans[my_id] = span_text
                labels[my_id] = span_text
                if parent_id is not None:
                    edges.append((parent_id, my_id))
                for ch in node["children"]:
                    walk(ch, my_id)
            else:
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
        print(f"[benepar_treeviz] Parsing: {sentence}")
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

        # Draw identical PNG style
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
            plt.title(f"Benepar Parse: {sentence}", fontsize=14, pad=20)

        plt.subplots_adjust(top=0.85)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"💾 Saved image: {output_path}")
