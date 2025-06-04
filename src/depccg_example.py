#!/usr/bin/env python3
"""
DepCCG Example: Combinatory Categorial Grammar Parser

This example demonstrates how to use depccg for syntactic parsing of English text.
DepCCG is a dependency-based CCG parser that produces syntactic parse trees and
dependency structures for natural language sentences.
"""

import depccg
from depccg import EnglishModel
from depccg.download import download_model
import json
import os

def setup_model():
    """Download and setup the English CCG model if not already available."""
    try:
        # Try to load the model
        model = EnglishModel.from_pretrained()
        print("✓ English CCG model loaded successfully")
        return model
    except Exception as e:
        print(f"Model not found: {e}")
        print("Downloading English CCG model...")
        try:
            download_model('en')
            model = EnglishModel.from_pretrained()
            print("✓ English CCG model downloaded and loaded successfully")
            return model
        except Exception as download_error:
            print(f"Failed to download model: {download_error}")
            return None

def basic_parsing_example(model):
    """Demonstrate basic CCG parsing functionality."""
    print("\n" + "="*50)
    print("BASIC PARSING EXAMPLE")
    print("="*50)
    
    sentences = [
        "The cat sat on the mat.",
        "John loves Mary.",
        "The quick brown fox jumps over the lazy dog.",
        "She gave him a book yesterday.",
        "Complex sentences can be parsed effectively."
    ]
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Parsing: '{sentence}'")
        try:
            # Parse the sentence
            parsed = model(sentence)
            
            # Get the parse tree
            tree = parsed[0]  # First (best) parse
            
            print(f"   CCG Category: {tree.cat}")
            print(f"   Parse successful: ✓")
            
        except Exception as e:
            print(f"   Parse failed: {e}")

def detailed_parse_analysis(model):
    """Show detailed analysis of a parse tree."""
    print("\n" + "="*50)
    print("DETAILED PARSE ANALYSIS")
    print("="*50)
    
    sentence = "The linguist studies syntactic structures carefully."
    print(f"Analyzing: '{sentence}'")
    
    try:
        parsed = model(sentence)
        tree = parsed[0]
        
        print(f"\nRoot category: {tree.cat}")
        print(f"Number of words: {len(tree.words)}")
        
        # Print word-by-word analysis
        print("\nWord-by-word analysis:")
        for i, (word, cat) in enumerate(zip(tree.words, tree.cats)):
            print(f"  {i+1:2d}. '{word:12s}' -> {cat}")
        
        # Show the tree structure
        print(f"\nTree structure:")
        print(tree.format_tree())
        
    except Exception as e:
        print(f"Analysis failed: {e}")

def dependency_extraction(model):
    """Extract and display dependency relationships."""
    print("\n" + "="*50)
    print("DEPENDENCY EXTRACTION")
    print("="*50)
    
    sentences = [
        "The teacher explains grammar rules.",
        "Students ask interesting questions about linguistics.",
        "Modern computers process language efficiently."
    ]
    
    for sentence in sentences:
        print(f"\nSentence: '{sentence}'")
        try:
            parsed = model(sentence)
            tree = parsed[0]
            
            # Extract dependencies
            dependencies = tree.to_conll()
            
            print("Dependencies (CoNLL format):")
            print("ID\tWORD\t\tHEAD\tREL")
            print("-" * 40)
            
            for dep in dependencies:
                word_id, word, head, rel = dep['id'], dep['form'], dep['head'], dep['deprel']
                print(f"{word_id}\t{word:12s}\t{head}\t{rel}")
                
        except Exception as e:
            print(f"Dependency extraction failed: {e}")

def multiple_parses_example(model):
    """Show how to get multiple parse candidates."""
    print("\n" + "="*50)
    print("MULTIPLE PARSES EXAMPLE")
    print("="*50)
    
    # Ambiguous sentence that might have multiple valid parses
    sentence = "I saw the man with the telescope."
    print(f"Parsing ambiguous sentence: '{sentence}'")
    
    try:
        # Get multiple parse candidates (if available)
        parsed = model(sentence, nbest=3)  # Get top 3 parses
        
        print(f"Number of parses found: {len(parsed)}")
        
        for i, tree in enumerate(parsed, 1):
            print(f"\nParse {i}:")
            print(f"  Root category: {tree.cat}")
            print(f"  Score: {getattr(tree, 'score', 'N/A')}")
            
    except Exception as e:
        print(f"Multiple parsing failed: {e}")

def batch_processing_example(model):
    """Demonstrate batch processing of multiple sentences."""
    print("\n" + "="*50)
    print("BATCH PROCESSING EXAMPLE")
    print("="*50)
    
    sentences = [
        "Natural language processing is fascinating.",
        "Computers understand human language better now.",
        "Machine learning improves parsing accuracy.",
        "CCG represents syntactic categories elegantly.",
        "Dependency parsing reveals grammatical relationships."
    ]
    
    print(f"Processing {len(sentences)} sentences in batch:")
    
    try:
        # Process all sentences
        results = []
        for sentence in sentences:
            parsed = model(sentence)
            results.append((sentence, parsed[0]))
        
        # Display results summary
        print("\nBatch processing results:")
        for i, (sentence, tree) in enumerate(results, 1):
            print(f"{i:2d}. {sentence[:40]:40s} -> {str(tree.cat)[:20]}")
            
    except Exception as e:
        print(f"Batch processing failed: {e}")

def save_parse_results(model):
    """Save parse results to a JSON file."""
    print("\n" + "="*50)
    print("SAVING PARSE RESULTS")
    print("="*50)
    
    sentences = [
        "The parser generates syntactic trees.",
        "CCG combines categories systematically."
    ]
    
    results = []
    
    for sentence in sentences:
        try:
            parsed = model(sentence)
            tree = parsed[0]
            
            # Convert to dictionary for JSON serialization
            result = {
                'sentence': sentence,
                'root_category': str(tree.cat),
                'words': tree.words,
                'categories': [str(cat) for cat in tree.cats],
                'dependencies': tree.to_conll()
            }
            results.append(result)
            
        except Exception as e:
            print(f"Failed to parse '{sentence}': {e}")
    
    # Save to JSON file
    output_file = "parse_results.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ Parse results saved to {output_file}")
        
        # Display a sample
        print(f"\nSample result:")
        if results:
            sample = results[0]
            print(f"  Sentence: {sample['sentence']}")
            print(f"  Root category: {sample['root_category']}")
            print(f"  Word count: {len(sample['words'])}")
            
    except Exception as e:
        print(f"Failed to save results: {e}")

def main():
    """Main function to run all examples."""
    print("DepCCG Example - CCG Parsing Demonstration")
    print("=" * 60)
    
    # Setup the model
    model = setup_model()
    if model is None:
        print("❌ Could not load CCG model. Please check your installation.")
        return
    
    try:
        # Run all examples
        basic_parsing_example(model)
        detailed_parse_analysis(model)
        dependency_extraction(model)
        multiple_parses_example(model)
        batch_processing_example(model)
        save_parse_results(model)
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("✨ Check the generated 'parse_results.json' file for saved results.")
        
    except Exception as e:
        print(f"\n❌ An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
