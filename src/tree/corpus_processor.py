#!/usr/bin/env python3
"""
corpus_processor.py
------------------
High-performance corpus processing for 64-core systems.
Processes large corpora with multiprocessing and saves individual JSON parse results.

Usage Examples:
    # Process corpus file with all cores
    python corpus_processor.py --corpus sentences.txt --save-json
    
    # Use 64 cores explicitly
    python corpus_processor.py --corpus large_corpus.txt --cores 64 --save-json --batch-size 1000
    
    # Process with filtering and limits
    python corpus_processor.py --corpus corpus.txt --cores 64 --save-json \
        --min-words 3 --max-words 20 --max-sentences 10000
    
    # Quiet processing for massive datasets
    python corpus_processor.py --corpus huge_corpus.txt --cores 64 --save-json \
        --quiet --batch-size 2000 --output-dir results/
"""

import argparse
import os
import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator
import re
from functools import partial

# Import our parsers
from depccg_treeviz import CCGTreeVisualizer as DepCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Convert text to a safe filename by removing punctuation and limiting length."""
    safe = re.sub(r'[^\w\s-]', '', text.strip())
    safe = re.sub(r'\s+', '_', safe)
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe or "sentence"


def load_corpus(corpus_path: str, max_sentences: int = None, 
                min_words: int = 1, max_words: int = 100) -> List[str]:
    """Load sentences from corpus file with filtering options.
    
    Args:
        corpus_path: Path to corpus file (one sentence per line)
        max_sentences: Maximum number of sentences to load
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence
    
    Returns:
        List of filtered sentences
    """
    sentences = []
    
    print(f"📖 Loading corpus from: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_sentences and len(sentences) >= max_sentences:
                break
                
            sentence = line.strip()
            if not sentence:
                continue
                
            # Filter by word count
            word_count = len(sentence.split())
            if word_count < min_words or word_count > max_words:
                continue
                
            sentences.append(sentence)
            
            # Progress for large files
            if line_num % 10000 == 0:
                print(f"   Loaded {len(sentences)} sentences (line {line_num})")
    
    print(f"✅ Loaded {len(sentences)} sentences from corpus")
    return sentences


def process_sentence_batch(batch_data: Tuple[List[Tuple[int, str]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of sentences in a worker process.
    
    Args:
        batch_data: Tuple of (sentence_list, config)
        
    Returns:
        List of processing results
    """
    sentence_batch, config = batch_data
    worker_id = mp.current_process().pid
    
    # Create parser instances for this worker
    dep_parser = DepCCG()
    ben_parser = Benepar()
    
    results = []
    
    for sentence_index, sentence in sentence_batch:
        if config.get('verbose', False):
            print(f"🔄 Worker {worker_id}: Processing sentence {sentence_index}")
        
        result = {
            'index': sentence_index,
            'sentence': sentence,
            'success': False,
            'timing': {},
            'json_files_created': [],
            'errors': []
        }
        
        try:
            # Parse with DepCCG
            dep_start = time.time()
            dep_result = dep_parser.parse_only(sentence)
            dep_time = time.time() - dep_start
            result['timing']['depccg'] = dep_time
            
            # Parse with Benepar
            ben_start = time.time()
            ben_result = ben_parser.parse_only(sentence)
            ben_time = time.time() - ben_start
            result['timing']['benepar'] = ben_time
            
            if dep_result.get('success') and ben_result.get('success'):
                result['success'] = True
                result['parse_results'] = {
                    'depccg': dep_result,
                    'benepar': ben_result
                }
                
                # Save JSON files if requested
                if config.get('save_json', False):
                    json_files = save_parse_json(sentence, sentence_index, 
                                               dep_result, ben_result, config)
                    result['json_files_created'] = json_files
                    
            else:
                result['errors'].append("One or both parsers failed")
                
        except Exception as e:
            result['errors'].append(f"Processing error: {str(e)}")
        
        results.append(result)
    
    if config.get('verbose', False):
        successful = sum(1 for r in results if r['success'])
        print(f"✅ Worker {worker_id}: Completed {successful}/{len(results)} sentences")
    
    return results


def save_parse_json(sentence: str, index: int, dep_result: Dict, ben_result: Dict, 
                   config: Dict[str, Any]) -> List[str]:
    """Save parse results as JSON files.
    
    Args:
        sentence: The input sentence
        index: Sentence index
        dep_result: DepCCG parse result
        ben_result: Benepar parse result
        config: Configuration options
        
    Returns:
        List of created file paths
    """
    output_dir = Path(config.get('output_dir', 'output'))
    output_dir.mkdir(exist_ok=True)
    
    safe_name = sanitize_filename(sentence[:40])
    timestamp = int(time.time())
    
    json_files = []
    
    # Save DepCCG result
    dep_file = output_dir / f"{safe_name}_{index:06d}_depccg_{timestamp}.json"
    with open(dep_file, 'w') as f:
        json.dump({
            'sentence': sentence,
            'index': index,
            'parser': 'depccg',
            'parse_data': dep_result.get('parse_data'),
            'success': dep_result.get('success'),
            'timing': dep_result.get('timing', {}),
            'metadata': {
                'timestamp': time.time(),
                'processing_mode': 'corpus_batch',
                'corpus_index': index
            }
        }, f, indent=2)
    json_files.append(str(dep_file))
    
    # Save Benepar result
    ben_file = output_dir / f"{safe_name}_{index:06d}_benepar_{timestamp}.json"
    with open(ben_file, 'w') as f:
        json.dump({
            'sentence': sentence,
            'index': index,
            'parser': 'benepar',
            'parse_data': ben_result.get('parse_data'),
            'success': ben_result.get('success'),
            'timing': ben_result.get('timing', {}),
            'metadata': {
                'timestamp': time.time(),
                'processing_mode': 'corpus_batch',
                'corpus_index': index
            }
        }, f, indent=2)
    json_files.append(str(ben_file))
    
    return json_files


def create_batches(sentences: List[str], batch_size: int) -> Iterator[List[Tuple[int, str]]]:
    """Create batches of sentences for processing.
    
    Args:
        sentences: List of sentences
        batch_size: Number of sentences per batch
        
    Yields:
        Batches of (index, sentence) tuples
    """
    for i in range(0, len(sentences), batch_size):
        batch = [(i + j + 1, sentence) 
                for j, sentence in enumerate(sentences[i:i + batch_size])]
        yield batch


def process_corpus(corpus_path: str, cores: int = None, batch_size: int = 100,
                  save_json: bool = False, output_dir: str = "output",
                  verbose: bool = True, max_sentences: int = None,
                  min_words: int = 1, max_words: int = 100) -> Dict[str, Any]:
    """Process a corpus with multiprocessing.
    
    Args:
        corpus_path: Path to corpus file
        cores: Number of CPU cores to use
        batch_size: Number of sentences per batch
        save_json: Whether to save individual JSON files
        output_dir: Output directory for JSON files
        verbose: Whether to show detailed progress
        max_sentences: Maximum sentences to process
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence
        
    Returns:
        Processing statistics and results
    """
    if cores is None:
        cores = mp.cpu_count()
    
    # Load corpus
    sentences = load_corpus(corpus_path, max_sentences, min_words, max_words)
    
    if not sentences:
        print("❌ No sentences loaded from corpus")
        return {}
    
    print(f"🚀 Starting corpus processing:")
    print(f"   📊 {len(sentences)} sentences")
    print(f"   🖥️  {cores} CPU cores") 
    print(f"   📦 {batch_size} sentences per batch")
    print(f"   💾 JSON output: {'enabled' if save_json else 'disabled'}")
    
    # Create configuration
    config = {
        'save_json': save_json,
        'output_dir': output_dir,
        'verbose': verbose and len(sentences) <= 100  # Reduce verbosity for large corpora
    }
    
    # Create batches
    batches = list(create_batches(sentences, batch_size))
    batch_data = [(batch, config) for batch in batches]
    
    print(f"   🔄 Processing {len(batches)} batches...")
    
    # Process batches in parallel
    start_time = time.time()
    
    with mp.Pool(processes=cores) as pool:
        batch_results = pool.map(process_sentence_batch, batch_data)
    
    processing_time = time.time() - start_time
    
    # Aggregate results
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    # Calculate statistics
    total_depccg_time = sum(r['timing'].get('depccg', 0) for r in successful_results)
    total_benepar_time = sum(r['timing'].get('benepar', 0) for r in successful_results)
    avg_depccg_time = total_depccg_time / len(successful_results) if successful_results else 0
    avg_benepar_time = total_benepar_time / len(successful_results) if successful_results else 0
    
    # Count JSON files created
    total_json_files = sum(len(r.get('json_files_created', [])) for r in successful_results)
    
    stats = {
        'corpus_path': corpus_path,
        'total_sentences': len(sentences),
        'processed_sentences': len(all_results),
        'successful_sentences': len(successful_results),
        'failed_sentences': len(failed_results),
        'processing_time': processing_time,
        'sentences_per_second': len(successful_results) / processing_time if processing_time > 0 else 0,
        'cores_used': cores,
        'batch_size': batch_size,
        'avg_depccg_time': avg_depccg_time,
        'avg_benepar_time': avg_benepar_time,
        'total_json_files': total_json_files,
        'json_output_enabled': save_json
    }
    
    # Print results
    print(f"\n⚡ Corpus processing completed!")
    print(f"📊 Results:")
    print(f"   ✅ Successfully processed: {len(successful_results)}/{len(sentences)} sentences")
    print(f"   ⚠️  Failed: {len(failed_results)} sentences")
    print(f"   ⏱️  Total time: {processing_time:.2f} seconds")
    print(f"   🚀 Speed: {stats['sentences_per_second']:.1f} sentences/second")
    print(f"   📈 Average DepCCG time: {avg_depccg_time:.2f}s per sentence")
    print(f"   📈 Average Benepar time: {avg_benepar_time:.2f}s per sentence")
    
    if save_json:
        print(f"   💾 Created {total_json_files} JSON files in {output_dir}/")
    
    if failed_results and verbose:
        print(f"\n❌ Failed sentences:")
        for result in failed_results[:5]:  # Show first 5 failures
            print(f"   {result['index']}: {result['sentence'][:50]}... - {', '.join(result['errors'])}")
        if len(failed_results) > 5:
            print(f"   ... and {len(failed_results) - 5} more failures")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process corpus with 64-core multiprocessing")
    parser.add_argument("--corpus", required=True, help="Path to corpus file (one sentence per line)")
    parser.add_argument("--cores", type=int, default=None, 
                      help="Number of CPU cores (default: auto-detect all cores)")
    parser.add_argument("--batch-size", type=int, default=100,
                      help="Number of sentences per batch (default: 100)")
    parser.add_argument("--save-json", action="store_true",
                      help="Save individual parse results as JSON files")
    parser.add_argument("--output-dir", default="output",
                      help="Output directory for JSON files (default: output)")
    parser.add_argument("--quiet", action="store_true",
                      help="Minimal output (recommended for large corpora)")
    parser.add_argument("--max-sentences", type=int, default=None,
                      help="Maximum number of sentences to process")
    parser.add_argument("--min-words", type=int, default=1,
                      help="Minimum words per sentence (default: 1)")
    parser.add_argument("--max-words", type=int, default=100,
                      help="Maximum words per sentence (default: 100)")
    parser.add_argument("--save-stats", help="Save processing statistics to JSON file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.corpus):
        print(f"❌ Corpus file not found: {args.corpus}")
        sys.exit(1)
    
    # Process the corpus
    stats = process_corpus(
        corpus_path=args.corpus,
        cores=args.cores,
        batch_size=args.batch_size,
        save_json=args.save_json,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        max_sentences=args.max_sentences,
        min_words=args.min_words,
        max_words=args.max_words
    )
    
    # Save statistics if requested
    if args.save_stats and stats:
        with open(args.save_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"📈 Statistics saved to: {args.save_stats}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    main() 