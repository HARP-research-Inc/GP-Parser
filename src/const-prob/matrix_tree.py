import spacy
import benepar
import numpy as np
import pandas as pd
from nltk.tree import Tree

# Initialize spaCy + benepar
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def get_spans_with_depth(tree: Tree):
    spans = {}
    def helper(node, depth):
        if isinstance(node, Tree):
            start = helper.idx
            for child in node:
                helper(child, depth + 1)
            end = helper.idx
            spans[(start, end)] = depth
        else:
            helper.idx += 1
    helper.idx = 0
    helper(tree, 0)
    return spans

def parse_to_matrices(sentence: str):
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    tree = Tree.fromstring(sent._.parse_string)
    tokens = tree.leaves()
    n = len(tokens)

    # Modifier matrices
    noun_mod = np.zeros((n, n), dtype=np.uint8)
    verb_mod = np.zeros((n, n), dtype=np.uint8)
    adj_mod  = np.zeros((n, n), dtype=np.uint8)
    for i, tok in enumerate(sent):
        if tok.tag_.startswith("JJ"):
            noun_mod[i, i] = 1
        if tok.tag_.startswith("RB"):
            verb_mod[i, i] = 1
            adj_mod[i, i]  = 1

    # Span matrices
    spans = get_spans_with_depth(tree)
    max_depth = max(spans.values(), default=0)
    span_mats = np.zeros((max_depth + 1, n, n), dtype=np.uint8)
    for (start, end), depth in spans.items():
        if end == start + 1:
            span_mats[depth, start, start] = 3
        elif end > start + 1:
            span_mats[depth, start, end - 1] = 2
            span_mats[depth, end - 1, start] = 1

    return tokens, {"noun_mod": noun_mod, "verb_mod": verb_mod, "adj_mod": adj_mod}, span_mats

def visualize_matrix(mat: np.ndarray, tokens: list, title: str):
    """
    Print a matrix with row/column labels from tokens.
    """
    df = pd.DataFrame(mat, index=tokens, columns=tokens)
    print(f"\n{title}")
    print(df)

if __name__ == "__main__":
    sentence = "The quick brown fox jumps over the lazy dog."
    tokens, mods, spans = parse_to_matrices(sentence)

    # Visualize modifier matrices
    for name, mat in mods.items():
        visualize_matrix(mat, tokens, f"Modifier: {name}")

    # Visualize span matrices by depth
    for depth in range(spans.shape[0]):
        visualize_matrix(spans[depth], tokens, f"Span matrix at depth {depth}")
