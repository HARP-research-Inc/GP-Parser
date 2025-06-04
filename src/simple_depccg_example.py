#!/usr/bin/env python3
"""
Simple DepCCG Example using Command Line Interface

This example demonstrates how to use depccg for parsing by using its command line interface.
"""

import subprocess
import tempfile
import os
import json

def parse_sentences_with_depccg(sentences):
    """Parse sentences using depccg command line interface."""
    
    # Create a temporary file with sentences
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
        temp_input = f.name
    
    try:
        # Run depccg parser
        result = subprocess.run([
            'python3', '-m', 'depccg', 'en', 
            '--input', temp_input,
            '--format', 'auto'
        ], capture_output=True, text=True)
        
        print("=" * 60)
        print("DEPCCG PARSING RESULTS")
        print("=" * 60)
        
        if result.stdout:
            print("Parse Results:")
            print(result.stdout)
        
        if result.stderr:
            print("\nWarnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to run depccg: {e}")
        return False
    finally:
        os.unlink(temp_input)

def parse_with_different_formats(sentence):
    """Try parsing a sentence with different output formats."""
    print(f"\n{'='*60}")
    print(f"PARSING: '{sentence}'")
    print(f"{'='*60}")
    
    formats = ['auto', 'deriv', 'conll', 'json']
    
    for fmt in formats:
        print(f"\n--- Format: {fmt} ---")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(sentence + '\n')
            temp_input = f.name
        
        try:
            result = subprocess.run([
                'python3', '-m', 'depccg', 'en', 
                '--input', temp_input,
                '--format', fmt
            ], capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout.strip())
            else:
                print("No output")
                
            if result.stderr:
                print(f"Errors: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"Failed: {e}")
        finally:
            os.unlink(temp_input)

def test_tokenization():
    """Test if depccg can handle different sentence structures."""
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT SENTENCE STRUCTURES")
    print(f"{'='*60}")
    
    test_sentences = [
        "The cat sleeps.",
        "Dogs run quickly.",
        "John loves Mary.",
        "The quick brown fox jumps.",
        "She gave him a book.",
        "Birds fly south in winter."
    ]
    
    return parse_sentences_with_depccg(test_sentences)

def main():
    """Main function to run all examples."""
    print("Simple DepCCG Example")
    print("Using Command Line Interface")
    print("=" * 60)
    
    # Test basic functionality
    success = test_tokenization()
    
    if success:
        print("\n✓ Basic parsing successful!")
        
        # Try different formats with a simple sentence
        parse_with_different_formats("The cat sat on the mat.")
    else:
        print("\n✗ Basic parsing failed. Check depccg installation.")

if __name__ == "__main__":
    main() 