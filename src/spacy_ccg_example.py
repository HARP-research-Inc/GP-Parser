#!/usr/bin/env python3
"""
Alternative CCG Parsing Example

Since depccg can be difficult to install on Windows, this example demonstrates
alternative approaches for CCG-style parsing and syntactic analysis.
"""

import spacy
import json
from typing import List, Dict, Any

def setup_spacy_model():
    """Setup spaCy for syntactic parsing."""
    try:
        # Try to load the model
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy English model loaded successfully")
        return nlp
    except OSError:
        print("spaCy model not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
        print("✓ spaCy English model downloaded and loaded")
        return nlp

def spacy_syntactic_analysis(nlp, sentence: str) -> Dict[str, Any]:
    """
    Perform syntactic analysis using spaCy.
    While not exactly CCG, provides similar structural information.
    """
    doc = nlp(sentence)
    
    analysis = {
        'sentence': sentence,
        'tokens': [],
        'dependencies': [],
        'constituency_approx': []
    }
    
    # Token-level analysis
    for token in doc:
        token_info = {
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'tag': token.tag_,
            'dep': token.dep_,
            'shape': token.shape_,
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop,
        }
        analysis['tokens'].append(token_info)
    
    # Dependency relations
    for token in doc:
        if token.dep_ != 'ROOT':
            dep_relation = {
                'dependent': token.text,
                'dependent_id': token.i,
                'head': token.head.text,
                'head_id': token.head.i,
                'relation': token.dep_
            }
            analysis['dependencies'].append(dep_relation)
    
    # Noun chunks (constituency approximation)
    for chunk in doc.noun_chunks:
        chunk_info = {
            'text': chunk.text,
            'root': chunk.root.text,
            'label': 'NP',  # Approximating as Noun Phrase
            'start': chunk.start,
            'end': chunk.end
        }
        analysis['constituency_approx'].append(chunk_info)
    
    return analysis

def simple_ccg_like_categories():
    """
    Demonstrate simple CCG-like category assignment based on POS tags.
    This is a simplified approximation, not a full CCG parser.
    """
    
    # Basic CCG category mappings
    pos_to_ccg = {
        'NOUN': 'N',
        'VERB': 'S\\NP',  # Simplified - intransitive verb
        'ADJ': 'N/N',
        'DET': 'NP/N',
        'ADP': '(S\\NP)\\(S\\NP)',  # Preposition
        'ADV': '(S\\NP)\\(S\\NP)',  # Adverb
        'PRON': 'NP',
        'CONJ': 'conj',
        'NUM': 'N/N'
    }
    
    return pos_to_ccg

def assign_ccg_categories(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Assign basic CCG-like categories to tokens."""
    ccg_mapping = simple_ccg_like_categories()
    
    for token in analysis['tokens']:
        pos = token['pos']
        
        # Basic category assignment
        basic_category = ccg_mapping.get(pos, 'X')  # X for unknown
        
        # Refine based on dependency relation
        dep = token['dep']
        if pos == 'VERB':
            if dep == 'ROOT':
                # Check if it takes objects
                basic_category = 'S'  # Root verb
            elif 'obj' in dep.lower():
                basic_category = '(S\\NP)/NP'  # Transitive verb
        
        token['ccg_category'] = basic_category
    
    return analysis

def demonstrate_parsing():
    """Demonstrate the alternative parsing approach."""
    print("CCG-Alternative Parsing Example")
    print("=" * 50)
    
    # Setup
    nlp = setup_spacy_model()
    
    sentences = [
        "The cat sat on the mat.",
        "John loves Mary.",
        "The quick brown fox jumps over the lazy dog.",
        "She gave him a book yesterday."
    ]
    
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Analyzing: '{sentence}'")
        
        # Perform analysis
        analysis = spacy_syntactic_analysis(nlp, sentence)
        analysis = assign_ccg_categories(analysis)
        
        # Display results
        print("   Tokens with CCG-like categories:")
        for token in analysis['tokens']:
            print(f"     '{token['text']:12s}' -> {token['ccg_category']:15s} (POS: {token['pos']}, DEP: {token['dep']})")
        
        print("   Dependencies:")
        for dep in analysis['dependencies']:
            print(f"     {dep['dependent']} --{dep['relation']}--> {dep['head']}")
        
        results.append(analysis)
    
    return results

def export_results(results: List[Dict[str, Any]], filename: str = "ccg_alternative_results.json"):
    """Export results to JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results exported to {filename}")
    except Exception as e:
        print(f"❌ Failed to export results: {e}")

def compare_with_ccg_concepts():
    """Explain how this relates to actual CCG parsing."""
    print("\n" + "="*60)
    print("COMPARISON WITH ACTUAL CCG PARSING")
    print("="*60)
    
    explanation = """
    While this example doesn't use a full CCG parser like depccg, it demonstrates:
    
    1. **Syntactic Categories**: Basic category assignment similar to CCG
       - N (noun), NP (noun phrase), S (sentence)
       - S\\NP (intransitive verb), (S\\NP)/NP (transitive verb)
    
    2. **Compositional Structure**: How words combine to form larger units
       - Dependency relations show head-dependent relationships
       - Noun chunks approximate constituency structure
    
    3. **Grammatical Relations**: Similar to CCG's semantic dependencies
       - Subject-verb, verb-object relationships
       - Modifier-head relationships
    
    **Advantages of this approach:**
    - ✓ Easy to install on Windows
    - ✓ Fast and reliable
    - ✓ Good documentation and support
    - ✓ Integrates well with other NLP tools
    
    **Limitations compared to full CCG:**
    - Limited compositional semantics
    - No lambda calculus representations
    - Simplified category system
    - No formal semantic parsing
    
    **When to use each:**
    - Use this approach for: practical NLP applications, quick prototyping
    - Use full CCG for: formal semantic analysis, research, complex reasoning
    """
    
    print(explanation)

def main():
    """Main execution function."""
    try:
        # Run the demonstration
        results = demonstrate_parsing()
        
        # Export results
        export_results(results)
        
        # Show comparison with CCG
        compare_with_ccg_concepts()
        
        print("\n" + "="*60)
        print("✅ CCG-Alternative parsing completed successfully!")
        print("💡 Consider this approach if depccg installation continues to be problematic.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Make sure spaCy is installed: pip install spacy")

if __name__ == "__main__":
    main() 