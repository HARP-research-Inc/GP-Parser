#!/usr/bin/env python3
"""
DepCCG Tree Visualizer - Works with current DepCCG versions

Uses subprocess calls to depccg (like the working parser module) instead of 
trying to import non-existent API classes.
"""

import subprocess
import tempfile
import os
import json
from typing import Dict, List, Tuple, Any
from pathlib import Path
import importlib.util

import networkx as nx
import holoviews as hv

# Activate bokeh backend
hv.extension("bokeh")

# -------------------------------------------------------------------- 
# Helper functions
# --------------------------------------------------------------------
def _flatten_leaves_from_dict(node: Dict[str, Any]) -> List[str]:
    """Extract leaf tokens from a tree dictionary structure."""
    if isinstance(node, dict):
        # If this is a leaf node with a word
        if 'word' in node:
            return [node['word']]
        # If it has children, recurse
        if 'children' in node and node['children']:
            words = []
            for child in node['children']:
                words.extend(_flatten_leaves_from_dict(child))
            return words
    # If it's a string (sometimes leaves are just strings)
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
    """Classic recursive tree layout (top-down)."""
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
    """
    Tree visualizer that uses subprocess calls to DepCCG.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        model: str = None,
        beam_size: int = 1,
        nbest: int = 1,
        max_sent_len: int = 200,
    ) -> None:
        """
        Initialize the visualizer.
        
        Args:
            lang: Language code (default: "en")
            model: Model variant (None for default, "elmo" for ELMo)
            beam_size: Beam size for parsing
            nbest: Number of best parses to return
            max_sent_len: Maximum sentence length
        """
        self.lang = lang
        self.model = model
        self.beam_size = beam_size
        self.nbest = nbest
        self.max_sent_len = max_sent_len

    def _parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """Parse a sentence using subprocess call to depccg."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(sentence.strip() + '\n')
            temp_input = f.name

        try:
            # Build command
            cmd = ['python3', '-m', 'depccg', self.lang]
            if self.model:
                cmd.extend(['--model', self.model])
            cmd.extend(['--input', temp_input, '--format', 'json'])

            # Run parser
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

    def _dict_tree_to_graph(self, tree_data: Dict[str, Any]) -> Tuple[nx.DiGraph, int]:
        """Convert dictionary tree structure to NetworkX graph."""
        spans: Dict[int, str] = {}
        edges: List[Tuple[int, int]] = []
        labels: Dict[int, str] = {}

        def walk(node: Any, parent_id: int = None) -> int:
            """Walk the tree structure and assign IDs."""
            my_id = len(spans)
            
            # Extract text and label information
            if isinstance(node, dict):
                # Get the span text from flattened leaves
                span_text = " ".join(_flatten_leaves_from_dict(node)) or "ε"
                
                # Get category/label if available
                label = node.get('cat', node.get('category', node.get('label', span_text)))
                
                spans[my_id] = span_text
                labels[my_id] = f"{label}\\n{span_text}" if label != span_text else span_text
                
                # Add edge from parent
                if parent_id is not None:
                    edges.append((parent_id, my_id))
                
                # Process children
                children = node.get('children', [])
                if children:
                    for child in children:
                        walk(child, my_id)
                        
            elif isinstance(node, str):
                # Leaf node that's just a string
                spans[my_id] = node
                labels[my_id] = node
                if parent_id is not None:
                    edges.append((parent_id, my_id))
            
            return my_id

        # Start walking from the tree data
        if 'tree' in tree_data:
            root_id = walk(tree_data['tree'])
        elif 'deriv' in tree_data:
            root_id = walk(tree_data['deriv'])
        else:
            # Fallback: create a simple node with available info
            root_id = 0
            spans[0] = str(tree_data)
            labels[0] = str(tree_data)

        # Create graph
        G = nx.DiGraph()
        for i in spans:
            G.add_node(i, label=labels.get(i, spans[i]), span=spans[i])
        G.add_edges_from(edges)
        
        return G, root_id

    def visualize(self, sentence: str) -> hv.Graph:
        """
        Parse sentence and return HoloViews Graph visualization.
        """
        try:
            parse_result = self._parse_sentence(sentence)
            
            if not parse_result['success']:
                # Create fallback graph
                G = nx.DiGraph()
                G.add_node(0, label=f"Parse failed:\\n{sentence}", span=sentence)
                layout = {0: (0.5, 0.5)}
                hv_graph = hv.Graph.from_networkx(G, layout)
                return hv_graph.opts(
                    node_size=20, 
                    tools=["hover"],
                    bgcolor="lightgray"
                )

            # Convert parse to graph
            G, root_id = self._dict_tree_to_graph(parse_result['parse_data'])
            
            # Create layout
            layout = _hierarchy_pos(G, root_id, width=1.0, vert_gap=0.3)

            # Create HoloViews graph
            hv_graph = hv.Graph.from_networkx(G, layout)
            
            # Apply styling
            hv_graph = hv_graph.opts(
                node_size=15,
                node_color="white",
                node_line_color="black",
                edge_line_width=1.5,
                tools=["hover", "box_zoom", "reset"],
                xaxis=None,
                yaxis=None,
                bgcolor="white",
                width=800,
                height=600
            )
            
            return hv_graph

        except Exception as e:
            # Create error graph
            G = nx.DiGraph()
            G.add_node(0, label=f"Error: {str(e)}\\n{sentence}", span=sentence)
            layout = {0: (0.5, 0.5)}
            hv_graph = hv.Graph.from_networkx(G, layout)
            return hv_graph.opts(
                node_size=20,
                tools=["hover"],
                bgcolor="lightgray"
            )

print("[depccg_treeviz] Using subprocess approach (compatible with current DepCCG)")
