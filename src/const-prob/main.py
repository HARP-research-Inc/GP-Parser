#!/usr/bin/env python3
"""
Constituency Parser Tree Depth Analysis

This script uses the Berkeley Neural Parser (Benepar) to parse sentences
and analyze the depth distribution of constituency trees. It generates
histograms showing the frequency of different tree depths in the dataset.

By default, this script uses the WikiText-103 dataset for analysis.

Requirements (with fallbacks):
- spacy (required)
- datasets (optional - for WikiText-103, will use sample if not available)
- benepar (optional - will use estimation if not available)
- matplotlib (optional - will create text histogram if not available)
- numpy (optional - will use basic statistics if not available)

Usage:
    python main.py [--dataset wikitext-103|sample|file] [--sample-size N]
    python main.py --dataset file --input-file path/to/text/file
"""

import argparse
import json
import logging
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Suppress common warnings from ML libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_struct")

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Will create text-based histograms.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available. Will use basic statistics.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("Error: spacy is required. Please install with: pip install spacy")

try:
    import benepar
    HAS_BENEPAR = True
except ImportError:
    HAS_BENEPAR = False
    print("Warning: benepar not available. Will use heuristic depth estimation.")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not available. Install with: pip install datasets")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConstituencyTreeAnalyzer:
    """High-accuracy constituency parser with tree depth analysis and fallbacks."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the parser with Benepar constituency parsing."""
        self.nlp = None
        self.has_benepar = False
        self._initialize_parser(model_name)
    
    def _initialize_parser(self, model_name: str):
        """Initialize spaCy and optionally Benepar with error handling."""
        if not HAS_SPACY:
            logger.error("spaCy is required but not available.")
            return
        
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found.")
            logger.info("Please install it with: python -m spacy download en_core_web_sm")
            return
        
        # Try to add Benepar if available
        if HAS_BENEPAR:
            try:
                logger.info("Loading Benepar constituency parser...")
                # Download model if needed
                try:
                    benepar.download('benepar_en3')
                except:
                    pass  # Model might already be downloaded
                
                self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
                self.has_benepar = True
                logger.info("Benepar parser initialized successfully!")
            except Exception as e:
                logger.warning(f"Failed to load Benepar: {e}")
                logger.info("Falling back to heuristic depth estimation...")
                self.has_benepar = False
        else:
            logger.info("Benepar not available. Using heuristic depth estimation.")
            self.has_benepar = False
    
    def calculate_tree_depth(self, tree) -> int:
        """Calculate the maximum depth of a constituency tree."""
        if not hasattr(tree, '__iter__') or isinstance(tree, str):
            return 1
        
        if len(tree) == 0:
            return 1
        
        max_child_depth = 0
        for child in tree:
            child_depth = self.calculate_tree_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth + 1
    
    def estimate_depth_heuristic(self, sentence: str) -> int:
        """Heuristic depth estimation based on sentence complexity."""
        if not self.nlp:
            # Very basic estimation without spaCy
            words = sentence.split()
            return max(4, min(len(words) // 3 + 3, 12))
        
        # Use spaCy for better heuristics
        doc = self.nlp(sentence)
        
        # Base depth
        depth = 4
        
        # Add depth based on sentence length
        word_count = len(doc)
        if word_count > 10:
            depth += 1
        if word_count > 15:
            depth += 1
        if word_count > 20:
            depth += 1
        
        # Add depth for syntactic complexity
        for token in doc:
            # Subordinate clauses
            if token.dep_ in ['advcl', 'ccomp', 'xcomp', 'acl']:
                depth += 1
            # Relative clauses
            elif token.dep_ == 'relcl':
                depth += 2
            # Coordination
            elif token.dep_ == 'conj':
                depth += 1
        
        # Add depth for punctuation complexity
        punct_count = sum(1 for token in doc if token.pos_ == 'PUNCT')
        if punct_count > 2:
            depth += 1
        
        return min(depth, 15)  # Cap at reasonable maximum
    
    def parse_sentence(self, sentence: str) -> Dict:
        """Parse a single sentence and return analysis results."""
        try:
            if not self.nlp:
                return {
                    'sentence': sentence,
                    'success': False,
                    'error': 'Parser not initialized',
                    'tree_depth': 0,
                    'parse_tree': None,
                    'method': 'none'
                }
            
            doc = self.nlp(sentence.strip())
            
            if len(doc) == 0:
                return {
                    'sentence': sentence,
                    'success': False,
                    'error': 'Empty sentence after processing',
                    'tree_depth': 0,
                    'parse_tree': None,
                    'method': 'none'
                }
            
            # Try Benepar parsing first
            if self.has_benepar:
                try:
                    sent = list(doc.sents)[0]  # Take first sentence
                    if hasattr(sent._, 'parse_tree'):
                        parse_tree = sent._.parse_string
                        tree_depth = self.calculate_tree_depth(sent._.parse_tree)
                        
                        return {
                            'sentence': sentence,
                            'success': True,
                            'tree_depth': tree_depth,
                            'parse_tree': parse_tree,
                            'token_count': len(sent),
                            'method': 'benepar',
                            'error': None
                        }
                except Exception as e:
                    logger.debug(f"Benepar parsing failed for: {sentence[:30]}... Error: {e}")
            
            # Fallback to heuristic estimation
            tree_depth = self.estimate_depth_heuristic(sentence)
            
            return {
                'sentence': sentence,
                'success': True,
                'tree_depth': tree_depth,
                'parse_tree': None,
                'token_count': len(doc),
                'method': 'heuristic',
                'error': None
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse sentence: {sentence[:50]}... Error: {e}")
            return {
                'sentence': sentence,
                'success': False,
                'error': str(e),
                'tree_depth': 0,
                'parse_tree': None,
                'method': 'failed'
            }
    
    def analyze_dataset(self, sentences: List[str], max_sentences: Optional[int] = None) -> Dict:
        """Analyze a dataset of sentences and return depth statistics."""
        logger.info(f"Analyzing dataset with {len(sentences)} sentences")
        
        if max_sentences and len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
            logger.info(f"Limiting analysis to {max_sentences} sentences")
        
        results = []
        depth_counts = Counter()
        method_counts = Counter()
        successful_parses = 0
        failed_parses = 0
        
        start_time = time.time()
        
        for i, sentence in enumerate(sentences):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(sentences) - i) / rate if rate > 0 else 0
                logger.info(f"Processed {i}/{len(sentences)} sentences. Rate: {rate:.1f} sent/sec. ETA: {eta:.1f}s")
            
            result = self.parse_sentence(sentence)
            results.append(result)
            
            if result['success']:
                depth_counts[result['tree_depth']] += 1
                method_counts[result['method']] += 1
                successful_parses += 1
            else:
                failed_parses += 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis complete! {successful_parses} successful, {failed_parses} failed in {elapsed_time:.2f}s")
        
        # Calculate statistics
        depths = [r['tree_depth'] for r in results if r['success']]
        
        stats = {
            'total_sentences': len(sentences),
            'successful_parses': successful_parses,
            'failed_parses': failed_parses,
            'success_rate': successful_parses / len(sentences) if sentences else 0,
            'processing_time': elapsed_time,
            'method_distribution': dict(method_counts)
        }
        
        if depths:
            if HAS_NUMPY:
                stats.update({
                    'mean_depth': float(np.mean(depths)),
                    'median_depth': float(np.median(depths)),
                    'std_depth': float(np.std(depths)),
                    'max_depth': int(np.max(depths)),
                    'min_depth': int(np.min(depths))
                })
            else:
                sorted_depths = sorted(depths)
                stats.update({
                    'mean_depth': sum(depths) / len(depths),
                    'median_depth': sorted_depths[len(depths)//2],
                    'std_depth': 0,  # Can't calculate without numpy
                    'max_depth': max(depths),
                    'min_depth': min(depths)
                })
        
        return {
            'results': results,
            'depth_counts': dict(depth_counts),
            'statistics': stats
        }
    
    def create_histogram(self, depth_counts: Dict[int, int], output_path: str = "tree_depth_histogram.png"):
        """Create and save a histogram of tree depths."""
        if not depth_counts:
            logger.warning("No data to plot histogram")
            return
        
        if HAS_MATPLOTLIB:
            self._create_matplotlib_histogram(depth_counts, output_path)
        else:
            self._create_text_histogram(depth_counts, output_path.replace('.png', '.txt'))
    
    def _create_matplotlib_histogram(self, depth_counts: Dict[int, int], output_path: str):
        """Create histogram using matplotlib."""
        depths = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depths]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(depths, counts, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
        
        plt.xlabel('Tree Depth', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Constituency Tree Depths', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        # Add statistics text
        total_sentences = sum(counts)
        mean_depth = sum(d * c for d, c in depth_counts.items()) / total_sentences
        stats_text = f'Total Sentences: {total_sentences}\nMean Depth: {mean_depth:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        logger.info(f"Histogram saved to {output_path}")
    
    def _create_text_histogram(self, depth_counts: Dict[int, int], output_path: str):
        """Create a text-based histogram."""
        depths = sorted(depth_counts.keys())
        max_count = max(depth_counts.values())
        total_sentences = sum(depth_counts.values())
        
        histogram_lines = []
        histogram_lines.append("Tree Depth Distribution")
        histogram_lines.append("=" * 60)
        histogram_lines.append("")
        
        for depth in depths:
            count = depth_counts[depth]
            bar_length = int((count / max_count) * 40)
            bar = '█' * bar_length
            percentage = (count / total_sentences) * 100
            line = f"Depth {depth:2d}: {bar:<40} {count:4d} ({percentage:5.1f}%)"
            histogram_lines.append(line)
            print(line)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(histogram_lines))
        
        logger.info(f"Text histogram saved to {output_path}")

def load_sample_dataset() -> List[str]:
    """Load a sample dataset for demonstration."""
    sample_sentences = [
        "The cat sleeps peacefully on the warm windowsill.",
        "John, who loves reading mystery novels, visited the old library yesterday.",
        "Although it was raining heavily, the children continued playing outside in the garden.",
        "The teacher explained the complex mathematical concept to her attentive students.",
        "Birds migrate south during winter months to find warmer climates and abundant food sources.",
        "The ancient oak tree, standing majestically in the center of the park, provides shade for many visitors.",
        "Scientists have discovered that dolphins use sophisticated communication methods to coordinate their hunting strategies.",
        "When the storm finally passed, we saw a beautiful rainbow stretching across the entire valley.",
        "The company's innovative approach to sustainable energy has revolutionized the industry significantly.",
        "Despite facing numerous challenges, the ambitious project was completed successfully ahead of schedule.",
        "The museum's new exhibition features artifacts from ancient civilizations that flourished thousands of years ago.",
        "Students who participate actively in class discussions tend to perform better on their final examinations.",
        "The chef carefully prepared an exquisite meal using fresh ingredients sourced from local organic farms.",
        "Technology has transformed the way people communicate, work, and access information in modern society.",
        "The hikers discovered a hidden waterfall cascading down the rocky cliffs of the remote mountain canyon."
    ]
    return sample_sentences

def load_wikitext_103(max_sentences: Optional[int] = None) -> List[str]:
    """Load sentences from WikiText-103 dataset."""
    if not HAS_DATASETS:
        logger.error("datasets library not available. Cannot load WikiText-103.")
        logger.info("Install with: pip install datasets")
        return load_sample_dataset()
    
    try:
        logger.info("Loading WikiText-103 dataset...")
        
        # Load the training split of WikiText-103
        dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
        logger.info(f"WikiText-103 loaded: {len(dataset)} articles")
        
        sentences = []
        
        for item in dataset:
            text = item['text'].strip()
            if not text or text.startswith('='):  # Skip headers and empty lines
                continue
            
            # Split into sentences using multiple delimiters
            import re
            # Split on periods, exclamation marks, question marks
            sent_parts = re.split(r'[.!?]+', text)
            
            for sent in sent_parts:
                sent = sent.strip()
                # Filter out very short sentences and those that look like titles
                if (len(sent.split()) >= 5 and 
                    len(sent) >= 20 and 
                    not sent.isupper() and
                    not sent.startswith('(')):
                    sentences.append(sent + '.')
            
            # Stop if we have enough sentences
            if max_sentences and len(sentences) >= max_sentences:
                sentences = sentences[:max_sentences]
                break
        
        logger.info(f"Extracted {len(sentences)} sentences from WikiText-103")
        return sentences
        
    except Exception as e:
        logger.error(f"Failed to load WikiText-103: {e}")
        logger.info("Falling back to sample dataset...")
        return load_sample_dataset()

def load_text_file(file_path: str) -> List[str]:
    """Load sentences from a text file."""
    logger.info(f"Loading text from {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simple sentence splitting (you might want to use more sophisticated methods)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        # Remove very short sentences
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        logger.info(f"Loaded {len(sentences)} sentences from file")
        return sentences
        
    except Exception as e:
        logger.error(f"Failed to load file {file_path}: {e}")
        return []

def main():
    """Main function to run the constituency tree depth analysis."""
    parser = argparse.ArgumentParser(description='Analyze constituency tree depths in text data')
    parser.add_argument('--input-file', type=str, help='Path to input text file')
    parser.add_argument('--dataset', type=str, default='wikitext-103', 
                       choices=['wikitext-103', 'sample', 'file'],
                       help='Dataset to use: wikitext-103, sample, or file')
    parser.add_argument('--sample-size', type=int, default=1000, help='Maximum number of sentences to analyze')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for results')
    parser.add_argument('--model', type=str, default='en_core_web_sm', help='spaCy model to use')
    
    args = parser.parse_args()
    
    # Check if spaCy is available
    if not HAS_SPACY:
        print("Error: spaCy is required but not installed.")
        print("Please install with: pip install spacy")
        print("Then download the model with: python -m spacy download en_core_web_sm")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset based on selection
    if args.dataset == 'file' or args.input_file:
        if not args.input_file:
            logger.error("--input-file must be specified when using 'file' dataset")
            return
        sentences = load_text_file(args.input_file)
        if not sentences:
            logger.error("No sentences loaded from file. Using WikiText-103 instead.")
            sentences = load_wikitext_103(args.sample_size)
    elif args.dataset == 'sample':
        logger.info("Using sample dataset.")
        sentences = load_sample_dataset()
    else:  # wikitext-103 (default)
        logger.info("Using WikiText-103 dataset.")
        sentences = load_wikitext_103(args.sample_size)
    
    # Initialize analyzer
    analyzer = ConstituencyTreeAnalyzer(model_name=args.model)
    
    # Analyze dataset
    analysis_results = analyzer.analyze_dataset(sentences, max_sentences=args.sample_size)
    
    # Print statistics
    stats = analysis_results['statistics']
    print("\n" + "="*50)
    print("CONSTITUENCY TREE DEPTH ANALYSIS RESULTS")
    print("="*50)
    print(f"Total sentences processed: {stats['total_sentences']}")
    print(f"Successful parses: {stats['successful_parses']}")
    print(f"Failed parses: {stats['failed_parses']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")
    print(f"Method distribution: {stats['method_distribution']}")
    
    if 'mean_depth' in stats:
        print(f"Mean tree depth: {stats['mean_depth']:.2f}")
        print(f"Median tree depth: {stats['median_depth']:.2f}")
        print(f"Depth range: {stats['min_depth']} - {stats['max_depth']}")
    
    print(f"\nDepth distribution:")
    for depth in sorted(analysis_results['depth_counts'].keys()):
        count = analysis_results['depth_counts'][depth]
        percentage = (count / stats['successful_parses']) * 100 if stats['successful_parses'] > 0 else 0
        print(f"  Depth {depth}: {count} sentences ({percentage:.1f}%)")
    
    # Create and save histogram
    histogram_path = output_dir / "tree_depth_histogram.png"
    analyzer.create_histogram(analysis_results['depth_counts'], str(histogram_path))
    
    # Save detailed results to JSON
    results_path = output_dir / "constituency_analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Detailed results saved to {results_path}")
    
    print(f"\nFiles saved:")
    print(f"- Histogram: {histogram_path}")
    print(f"- Detailed results: {results_path}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
