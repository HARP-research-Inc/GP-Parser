#!/usr/bin/env python3
"""
Main script to demonstrate DepCCG ELMo parsing

This script uses the depccg_parser module to parse test sentences
with the ELMo model and display the results in a structured format.
"""

import json
import sys
import os

# Add src directory to path so we can import our module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from depccg_parser import DepCCGParser, parse_with_elmo

def print_parse_result(result):
    """Pretty print the parse result."""
    print(f"Sentence: {result['sentence']}")
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"Root Category: {result['category']}")
        print(f"Log Probability: {result['log_probability']:.4f}" if result['log_probability'] else "Log Probability: N/A")
        
        if result['parse_tree']:
            print(f"Parse Tree: {result['parse_tree']}")
        
        if result['derivation']:
            print("Derivation structure available")
            
    else:
        print(f"Error: {result['error']}")
    
    print("-" * 80)

def main():
    """Main function to test the DepCCG ELMo parser."""
    print("DepCCG ELMo Parser Test")
    print("=" * 80)
    
    # Test sentences covering different linguistic phenomena
    test_sentences = [
        "The cat sleeps.",
        "The book that John read was interesting.",
        "Birds fly south in winter, yet some stay north all year.",
        "Although it was raining, the children played outside.",
        "I saw the man with the telescope."
    ]
    
    # Create parser instance
    parser = DepCCGParser(model="elmo")
    
    print(f"Testing {len(test_sentences)} sentences with ELMo model:")
    print()
    
    # Parse each sentence
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Test {i}:")
        result = parser.parse_sentence(sentence)
        print_parse_result(result)
    
    # Example of using the convenience function
    print("\nUsing convenience function:")
    print("-" * 40)
    result = parse_with_elmo("The quick brown fox jumps over the lazy dog.")
    print_parse_result(result)
    
    # Example of parsing multiple sentences at once
    print("\nParsing multiple sentences:")
    print("-" * 40)
    batch_sentences = [
        "John loves Mary.",
        "She gave him a book.",
        "The teacher explained the lesson clearly."
    ]
    
    results = parser.parse_sentences(batch_sentences)
    
    for i, result in enumerate(results, 1):
        print(f"Batch result {i}:")
        print(f"  Sentence: {result['sentence']}")
        print(f"  Success: {result['success']}")
        if result['success']:
            print(f"  Category: {result['category']}")
            print(f"  Log Prob: {result['log_probability']:.4f}" if result['log_probability'] else "  Log Prob: N/A")
        else:
            print(f"  Error: {result['error']}")
        print()
    
    # Save results to JSON file
    print("Saving results to parse_results.json...")
    all_results = {
        'individual_tests': [parser.parse_sentence(s) for s in test_sentences],
        'convenience_test': parse_with_elmo("The quick brown fox jumps over the lazy dog."),
        'batch_tests': results
    }
    
    with open('parse_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Results saved to parse_results.json")
    print("\nDone!")

if __name__ == "__main__":
    main() 