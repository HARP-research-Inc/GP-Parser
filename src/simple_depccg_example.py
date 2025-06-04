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

def parse_with_model_variant(sentence, variant=None, output_format='auto'):
    """Parse a sentence with a specific model variant."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(sentence + '\n')
        temp_input = f.name
    
    try:
        # Build command - add variant if specified
        cmd = ['python3', '-m', 'depccg', 'en']
        if variant:
            cmd.extend(['--model', variant])
        cmd.extend(['--input', temp_input, '--format', output_format])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return result.stdout.strip() if result.stdout else "No output", result.stderr.strip() if result.stderr else ""
        
    except Exception as e:
        return f"Error: {e}", ""
    finally:
        os.unlink(temp_input)

def compare_models_on_compound_sentences():
    """Compare all three English models on compound sentences."""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON: EN vs ELMO")
    print("Testing on compound and complex sentences")
    print(f"{'='*80}")
    
    # Test sentences that should show differences in parsing accuracy
    test_sentences = [
        "The cat sat on the mat, and the dog barked loudly.",
        "John loves Mary, but she prefers pizza over pasta.",
        "The teacher explained the lesson, so the students understood quickly.",
        "Birds fly south in winter, yet some stay north all year.",
        "The book that I read yesterday was fascinating and educational."
    ]
    
    models = [
        ("Default (en)", None),
        ("ELMo", "elmo")
    ]
    
    for sentence in test_sentences:
        print(f"\n{'-'*80}")
        print(f"SENTENCE: {sentence}")
        print(f"{'-'*80}")
        
        for model_name, variant in models:
            print(f"\n--- {model_name} Model ---")
            
            # Get parse result
            output, errors = parse_with_model_variant(sentence, variant, 'auto')
            
            if output and not output.startswith("Error"):
                # Extract just the parse tree structure (remove ID and log prob for cleaner comparison)
                lines = output.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('ID='):
                        if 'log probability=' in line:
                            # Extract and display log probability for comparison
                            prob_part = line.split('log probability=')[1].split()[0]
                            print(f"Log Probability: {prob_part}")
                        else:
                            # This is the parse tree
                            print(f"Parse: {line.strip()}")
                            break
            else:
                print(f"Failed: {output}")
                
            if errors:
                print(f"Notes: {errors.split('parsing')[0]}...")  # Show just setup info, not full log

def compare_models_detailed_analysis():
    """Detailed analysis comparing models on specific linguistic phenomena."""
    print(f"\n{'='*80}")
    print("DETAILED LINGUISTIC ANALYSIS")
    print(f"{'='*80}")
    
    # Test cases for specific linguistic phenomena
    test_cases = [
        {
            "name": "Coordination (AND/OR)",
            "sentence": "The quick brown fox jumps and the lazy dog sleeps."
        },
        {
            "name": "Prepositional Phrase Attachment", 
            "sentence": "I saw the man with the telescope."
        },
        {
            "name": "Relative Clause",
            "sentence": "The book that John read was interesting."
        },
        {
            "name": "Complex Subordination",
            "sentence": "Although it was raining, the children played outside because they were excited."
        }
    ]
    
    models = [("Default", None), ("ELMo", "elmo")]
    
    for test_case in test_cases:
        print(f"\n{'-'*60}")
        print(f"TESTING: {test_case['name']}")
        print(f"Sentence: {test_case['sentence']}")
        print(f"{'-'*60}")
        
        for model_name, variant in models:
            print(f"\n{model_name}:")
            output, _ = parse_with_model_variant(test_case['sentence'], variant, 'json')
            
            if output and not output.startswith("Error"):
                try:
                    # Parse JSON to extract key information
                    data = json.loads(output)
                    if "1" in data and data["1"]:
                        parse_info = data["1"][0]
                        log_prob = parse_info.get("log_prob", "N/A")
                        root_cat = parse_info.get("cat", "N/A")
                        print(f"  Root Category: {root_cat}")
                        print(f"  Log Probability: {log_prob}")
                        print(f"  Confidence: {'High' if isinstance(log_prob, float) and log_prob > -1.0 else 'Medium' if isinstance(log_prob, float) and log_prob > -5.0 else 'Low'}")
                except json.JSONDecodeError:
                    print(f"  Parse: {output[:100]}...")
            else:
                print(f"  Failed to parse")

def parse_with_different_formats(sentence):
    """Try parsing a sentence with different output formats."""
    print(f"\n{'='*60}")
    print(f"PARSING: '{sentence}'")
    print(f"{'='*60}")
    
    formats = ['json'] # ['auto', 'deriv', 'conll', 'json']
    
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
    print("DepCCG Model Comparison Example")
    print("Comparing Default and ELMo models")
    print("=" * 80)
    
    # Test basic functionality first
    success = test_tokenization()
    
    if success:
        print("\n✓ Basic parsing successful!")
        
        # Compare models on compound sentences
        compare_models_on_compound_sentences()
        
        # Detailed linguistic analysis
        compare_models_detailed_analysis()
        
    else:
        print("\n✗ Basic parsing failed. Check depccg installation.")

if __name__ == "__main__":
    main() 