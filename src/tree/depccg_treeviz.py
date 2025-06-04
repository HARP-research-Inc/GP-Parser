#!/usr/bin/env python3
"""
DepCCG Tree Visualizer - Direct image output using matplotlib

Uses subprocess calls to depccg and matplotlib for direct PNG generation.
"""

import subprocess
import tempfile
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
def _flatten_leaves_from_dict(node: Dict[str, Any]) -> List[str]:
    """Extract leaf tokens from a tree dictionary structure."""
    if isinstance(node, dict):
        if 'word' in node:
            return [node['word']]
        if 'children' in node and node['children']:
            words = []
            for child in node['children']:
                words.extend(_flatten_leaves_from_dict(child))
            return words
    elif isinstance(node, str):
        return [node]
    return []

def _hierarchy_pos(
    G: nx.DiGraph,
    root: int,
    width: float = 1.0,
    vert_gap: float = 0.25,
    y0: float = 0.0,
) -> Dict[int, Tuple[float, float]]:
    """Calculate hierarchical positions for tree layout."""
    if root not in G:
        return {root: (0.5, y0)}

    def _place(n: int, x0: float, x1: float, y: float, pos: Dict[int, Tuple[float, float]]):
        kids = list(G.successors(n))
        if not kids:
            pos[n] = ((x0 + x1) * 0.5, y)
            return
        slice_w = (x1 - x0) / len(kids)
        for i, k in enumerate(kids):
            _place(k, x0 + i * slice_w, x0 + (i + 1) * slice_w, y - vert_gap, pos)
        xs = [pos[c][0] for c in kids]
        pos[n] = (sum(xs) / len(xs), y)

    out: Dict[int, Tuple[float, float]] = {}
    _place(root, 0.0, width, y0, out)
    return out

# --------------------------------------------------------------------
# Main visualizer class  
# --------------------------------------------------------------------
class CCGTreeVisualizer:
    """Tree visualizer using matplotlib for direct image output."""

    def __init__(self, lang: str = "en", model: str = None):
        self.lang = lang
        self.model = model

    def _parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """Parse a sentence using subprocess call to depccg."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(sentence.strip() + '\n')
            temp_input = f.name

        try:
            cmd = ['python3', '-m', 'depccg', self.lang]
            if self.model:
                cmd.extend(['--model', self.model])
            cmd.extend(['--input', temp_input, '--format', 'json'])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if "1" in data and data["1"]:
                    return {
                        'success': True,
                        'parse_data': data["1"][0],
                        'error': None
                    }
            
            return {
                'success': False,
                'parse_data': None,
                'error': result.stderr.strip() if result.stderr else "Parse failed"
            }

        except Exception as e:
            return {
                'success': False,
                'parse_data': None,
                'error': str(e)
            }
        finally:
            if os.path.exists(temp_input):
                os.unlink(temp_input)

    def _dict_tree_to_graph(self, tree_data: Dict[str, Any]) \
            -> Tuple[nx.DiGraph, int, Dict[int, str], Dict[int, str]]:

        spans: Dict[int, str] = {}
        edges: List[Tuple[int, int]] = []
        labels: Dict[int, str] = {}
        categories: Dict[int, str] = {}

        def walk(node: Any, parent_id: Optional[int] = None) -> int:
            my_id = len(spans)

            if isinstance(node, dict):
                span_text = " ".join(_flatten_leaves_from_dict(node)) or "ε"
                cat       = node.get("cat", "")
                spans[my_id]       = span_text
                categories[my_id]  = cat
                labels[my_id]      = cat or span_text
                if parent_id is not None:
                    edges.append((parent_id, my_id))
                for child in node.get("children", []):
                    walk(child, my_id)
            else:                                  # raw string leaf
                spans[my_id]  = str(node)
                labels[my_id] = str(node)
                categories[my_id] = ""
                if parent_id is not None:
                    edges.append((parent_id, my_id))
            return my_id

        root_id = walk(tree_data)          # <-- walk unconditionally

        G = nx.DiGraph()
        G.add_nodes_from(spans.keys())
        G.add_edges_from(edges)
        return G, root_id, labels, spans

    def _print_tree(self, G: nx.DiGraph, root_id: int, labels: Dict[int, str], spans: Dict[int, str], indent: str = "", is_last: bool = True):
        """Print tree structure to console."""
        if root_id not in G:
            return
            
        # Print current node
        label = labels.get(root_id, str(root_id))
        span = spans.get(root_id, "")
        
        # Determine if this is a leaf
        children = list(G.successors(root_id))
        is_leaf = len(children) == 0
        
        # Print node with tree structure
        branch = "└── " if is_last else "├── "
        if is_leaf:
            print(f"{indent}{branch}{span} [{label}]")
        else:
            print(f"{indent}{branch}{label} → '{span}'")
        
        # Print children
        for i, child_id in enumerate(children):
            child_is_last = (i == len(children) - 1)
            next_indent = indent + ("    " if is_last else "│   ")
            self._print_tree(G, child_id, labels, spans, next_indent, child_is_last)

    def save_tree_image(self, sentence: str, output_path: str, width: int = 12, height: int = 8):
        """Parse sentence and save tree visualization as PNG."""
        print(f"[depccg_treeviz] Parsing: {sentence}")
        
        parse_result = self._parse_sentence(sentence)
        
        if not parse_result['success']:
            print(f"❌ Parse failed: {parse_result['error']}")
        else:
            # Convert parse to graph
            G, root_id, labels, spans = self._dict_tree_to_graph(parse_result['parse_data'])
            
            if not G.nodes():
                print("❌ No tree structure found")
            else:
                print("🌳 Parse tree structure:")
                print("=" * 50)
                self._print_tree(G, root_id, labels, spans)
                print("=" * 50)
        
        # Now create the image
        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if not parse_result['success']:
            # Draw error message
            ax.text(0.5, 0.5, f"Parse failed: {sentence}\nError: {parse_result['error']}", 
                   ha='center', va='center', fontsize=12, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title("Parse Failed", fontsize=16, color='red')
        else:
            # Draw successful parse tree (reuse the same graph we built for printing)
            if not G.nodes():
                ax.text(0.5, 0.5, "No tree structure found", ha='center', va='center')
            else:
                # Calculate positions
                pos = _hierarchy_pos(G, root_id, width=0.9, vert_gap=0.15, y0=0.9)
                
                # Draw edges
                for edge in G.edges():
                    start_pos = pos[edge[0]]
                    end_pos = pos[edge[1]]
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           'k-', linewidth=1.5, alpha=0.7)
                
                # Draw nodes
                for node_id in G.nodes():
                    x, y = pos[node_id]
                    label = labels.get(node_id, str(node_id))
                    span = spans.get(node_id, "")
                    
                    # Determine if this is a leaf (terminal) node
                    is_leaf = len(list(G.successors(node_id))) == 0
                    
                    if is_leaf:
                        # Terminal nodes (words) - simple text
                        ax.text(x, y, span, ha='center', va='center', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
                    else:
                        # Non-terminal nodes (categories) - show category
                        ax.text(x, y, label, ha='center', va='center', 
                               fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
            
            plt.title(f"CCG Parse: {sentence}", fontsize=14, pad=20)
        
        # Remove tight_layout to avoid the warning
        plt.subplots_adjust(top=0.85)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"💾 Saved image: {output_path}")

print("[depccg_treeviz] Using matplotlib for direct image output")
