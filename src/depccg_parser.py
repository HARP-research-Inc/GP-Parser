#!/usr/bin/env python3
"""
DepCCG Parser Module

A clean interface for parsing sentences using DepCCG with the ELMo model.
Returns structured Python dictionaries containing parse information.
"""

import subprocess
import tempfile
import os
import json
from typing import Dict, List, Optional, Any

class DepCCGParser:
    """Parser class for DepCCG with ELMo model."""
    
    def __init__(self, model: str = "elmo", language: str = "en"):
        """
        Initialize the DepCCG parser.
        
        Args:
            model: Model variant to use ("elmo" for ELMo model, None for default)
            language: Language code (default: "en")
        """
        self.model = model
        self.language = language
    
    def parse_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Parse a single sentence and return structured data.
        
        Args:
            sentence: The sentence to parse
            
        Returns:
            Dictionary containing parse results with keys:
            - success: bool indicating if parsing succeeded
            - sentence: original sentence
            - parse_tree: the parsed tree structure
            - log_probability: confidence score
            - category: root syntactic category
            - error: error message if parsing failed
        """
        # Create temporary file with the sentence
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(sentence.strip() + '\n')
            temp_input = f.name
        
        try:
            # Build command
            cmd = ['python3', '-m', 'depccg', self.language]
            if self.model:
                cmd.extend(['--model', self.model])
            cmd.extend(['--input', temp_input, '--format', 'json'])
            
            # Run parser
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                return self._parse_json_output(sentence, result.stdout)
            else:
                return {
                    'success': False,
                    'sentence': sentence,
                    'parse_tree': None,
                    'log_probability': None,
                    'category': None,
                    'error': result.stderr.strip() if result.stderr else "Unknown error"
                }
        
        except Exception as e:
            return {
                'success': False,
                'sentence': sentence,
                'parse_tree': None,
                'log_probability': None,
                'category': None,
                'error': str(e)
            }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_input):
                os.unlink(temp_input)
    
    def parse_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple sentences.
        
        Args:
            sentences: List of sentences to parse
            
        Returns:
            List of dictionaries, one for each sentence
        """
        return [self.parse_sentence(sentence) for sentence in sentences]
    
    def _parse_json_output(self, sentence: str, json_output: str) -> Dict[str, Any]:
        """Parse the JSON output from DepCCG."""
        try:
            data = json.loads(json_output)
            
            # DepCCG returns results indexed by sentence number (starting from "1")
            if "1" in data and data["1"]:
                parse_info = data["1"][0]  # Take the best parse
                
                return {
                    'success': True,
                    'sentence': sentence,
                    'parse_tree': parse_info.get('tree', None),
                    'log_probability': parse_info.get('log_prob', None),
                    'category': parse_info.get('cat', None),
                    'derivation': parse_info.get('deriv', None),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'sentence': sentence,
                    'parse_tree': None,
                    'log_probability': None,
                    'category': None,
                    'error': "No parse found"
                }
                
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'sentence': sentence,
                'parse_tree': None,
                'log_probability': None,
                'category': None,
                'error': f"JSON decode error: {str(e)}"
            }

def create_elmo_parser() -> DepCCGParser:
    """Convenience function to create an ELMo parser."""
    return DepCCGParser(model="elmo")

def parse_with_elmo(sentence: str) -> Dict[str, Any]:
    """
    Convenience function to parse a sentence with ELMo model.
    
    Args:
        sentence: Sentence to parse
        
    Returns:
        Dictionary with parse results
    """
    parser = create_elmo_parser()
    return parser.parse_sentence(sentence) 