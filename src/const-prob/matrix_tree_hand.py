import spacy
import benepar
import numpy as np
import pandas as pd
from nltk.tree import Tree
import time
import warnings
from time import sleep
from tqdm import tqdm
import threading


# Suppress Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_struct")


# Initialize spaCy + benepar

print("Loading models...")
loading_complete = False
estimated_time = 5.5

def load_models():
    global loading_complete, nlp
    start = time.time()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # benepar.download('benepar_en3_large')
    # nlp = spacy.load("en_core_web_trf")
    # nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})

    end = time.time()
    loading_complete = True
    actual_time = end - start
    print(f"\nModels loaded in {actual_time:10.1f}s.")

def show_progress():
    # Create progress bar based on estimated time
    steps = 100
    step_duration = estimated_time / steps
    
    with tqdm(total=steps, desc="Loading") as pbar:
        for i in range(steps):
            if loading_complete:
                pbar.update(steps - i)  # Complete the bar
                break
            time.sleep(step_duration)
            pbar.update(1)

# Start both threads
loading_thread = threading.Thread(target=load_models)
progress_thread = threading.Thread(target=show_progress)

loading_thread.start()
progress_thread.start()

# Wait for both to complete
loading_thread.join()
progress_thread.join()


class TreeTrix:
    def __init__(self, tree: Tree):
        self.tree = tree
        self.tokens = tree.leaves()
        self.noun_phrases = self.noun_phrases()

    def noun_phrases(self):
        noun_phrases = []
        for s in self.tree.subtrees():
            if s.label() == "NP":
                noun_phrases.append(s)
        return noun_phrases
    
    def childz(self, t):
        immediate_subtrees = [t[i] for i in range(len(t)) if isinstance(t[i], Tree)]
        return immediate_subtrees
    
    def noun_phrases_to_layer_0(self):
        noun_phrases = self.noun_phrases
        
        t = self.tree
        for np in noun_phrases:
            print("NP: Checking - ", " ".join(np.leaves()))
            print(np)
            nouns = []
            modifiers = []
            nonmodifiers = []
            ic = self.childz(np)
            cands = [t for t in ic if t.height() == 2]
            for s in ic:
                if ("NN" in s.label()) or s.label() == "PRP" or s.label() == "WP":
                    nouns.append(s)
                    print('  Found static noun "'+s.leaves()[0]+'" in '+'"'+" ".join(np.leaves())+'"')
                     
                if (s.label() in ["WP$", "WDT", "PRP$", "POS", "PDT", "IN", "DT", "CD" ]) or ("JJ" in s.label()):
                    modifiers.append(s)
                    print('  Found noun modifer "'+s.leaves()[0]+'" of type '+s.label())
                
                if (s.label)

      
def parse_to_matrices(sentence: str):
    doc = nlp(sentence)
    sent = list(doc.sents)[0]
    tree = Tree.fromstring(sent._.parse_string)
    tokens = tree.leaves()
    n = len(tokens)

    print("Tokens:")
    print(tokens)

    print("Tree:")
    print(tree)


    tree_trix = TreeTrix(tree)
    tree_trix.noun_phrases_to_layer_0()
    

    return None, None, None

def visualize_matrix(mat: np.ndarray, tokens: list, title: str):
    """
    Print a matrix with row/column labels from tokens.
    """
    df = pd.DataFrame(mat, index=tokens, columns=tokens)
    print(f"\n{title}")
    print(df)

if __name__ == "__main__":
    # sentence = "The quick brown fox linguistically jumps over the lazy dog."
    # parse_to_matrices(sentence)
    sentence2 = "John's dog lazily basked in his own greed."
    parse_to_matrices(sentence2)

    # Visualize modifier matrices
    # for name, mat in mods.items():
    #     visualize_matrix(mat, tokens, f"Modifier: {name}")

