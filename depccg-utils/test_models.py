#!/usr/bin/env python3
"""Simple test script to check if ELMo models work"""

import subprocess
import tempfile
import os

def test_model(model_name):
    """Test a specific model variant"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write('The cat sat on the mat.\n')
        temp_input = f.name
    
    try:
        cmd = ['python3', '-m', 'depccg', 'en', '--input', temp_input, '--format', 'auto']
        if model_name:
            cmd.extend(['--model', model_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print(f"=== {model_name or 'default'} model ===")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Error: {result.stderr[:400]}...")
        print()
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to test {model_name}: {e}")
        return False
    finally:
        os.unlink(temp_input)

if __name__ == "__main__":
    models = [None, 'elmo', 'elmo_rebank']
    for model in models:
        test_model(model) 