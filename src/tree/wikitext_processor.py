#!/usr/bin/env python3
"""
wikitext_processor.py
--------------------
Download and process WikiText corpus with 64-core multiprocessing.

Supports WikiText-2, WikiText-103 datasets with automatic downloading
and sentence extraction from Wikipedia articles.

Usage Examples:
    # Download and process WikiText-2 with all cores
    python wikitext_processor.py --dataset wikitext-2 --cores 64 --save-json
    
    # Process WikiText-103 with custom settings
    python wikitext_processor.py --dataset wikitext-103 --cores 64 --save-json \
        --batch-size 500 --max-sentences 10000 --quiet
    
    # Process specific split (train/valid/test)
    python wikitext_processor.py --dataset wikitext-2 --split train \
        --cores 64 --save-json --output-dir wikitext_results/
"""

import argparse
import os
import sys
import time
import json
import multiprocessing as mp
import urllib.request
import zipfile
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from corpus_processor import process_sentence_batch, save_parse_json


def download_wikitext(dataset: str, data_dir: str = "data") -> Path:
    """Download WikiText dataset if not already present.
    
    Args:
        dataset: 'wikitext-2' or 'wikitext-103'
        data_dir: Directory to store data
        
    Returns:
        Path to extracted dataset directory
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    dataset_urls = {
        'wikitext-2': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
        'wikitext-103': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
    }
    
    if dataset not in dataset_urls:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(dataset_urls.keys())}")
    
    url = dataset_urls[dataset]
    zip_path = data_path / f"{dataset}.zip"
    extract_path = data_path / dataset
    
    # Check if already downloaded and extracted
    if extract_path.exists():
        print(f"✅ {dataset} already exists at {extract_path}")
        return extract_path
    
    print(f"📥 Downloading {dataset} from {url}")
    
    # Download the dataset
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"✅ Downloaded {zip_path}")
    except Exception as e:
        print(f"❌ Failed to download {dataset}: {e}")
        sys.exit(1)
    
    # Extract the dataset
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"✅ Extracted to {data_path}")
        
        # Remove zip file to save space
        zip_path.unlink()
        
    except Exception as e:
        print(f"❌ Failed to extract {dataset}: {e}")
        sys.exit(1)
    
    return extract_path


def extract_sentences_from_wikitext(file_path: Path, max_sentences: int = None,
                                   min_words: int = 3, max_words: int = 100) -> List[str]:
    """Extract sentences from WikiText format.
    
    WikiText format has:
    - Article titles starting with ' = '
    - Paragraph breaks as empty lines
    - Some markup that needs cleaning
    
    Args:
        file_path: Path to WikiText file
        max_sentences: Maximum sentences to extract
        min_words: Minimum words per sentence
        max_words: Maximum words per sentence
        
    Returns:
        List of cleaned sentences
    """
    sentences = []
    
    print(f"📖 Extracting sentences from {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_sentences and len(sentences) >= max_sentences:
                break
                
            line = line.strip()
            
            # Skip empty lines and article headers
            if not line or line.startswith(' = ') or line.startswith(' == '):
                continue
            
            # Clean up WikiText markup
            line = clean_wikitext_line(line)
            
            # Split into sentences (basic sentence splitting)
            line_sentences = split_into_sentences(line)
            
            for sentence in line_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Filter by word count
                word_count = len(sentence.split())
                if word_count < min_words or word_count > max_words:
                    continue
                
                sentences.append(sentence)
                
                if max_sentences and len(sentences) >= max_sentences:
                    break
            
            # Progress for large files
            if line_num % 1000 == 0:
                print(f"   Processed {line_num} lines, found {len(sentences)} sentences")
    
    print(f"✅ Extracted {len(sentences)} sentences from WikiText")
    return sentences


def clean_wikitext_line(line: str) -> str:
    """Clean WikiText markup from a line.
    
    Args:
        line: Raw line from WikiText
        
    Returns:
        Cleaned line
    """
    # Remove common WikiText markup
    line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
    line = re.sub(r'\[\[([^|\]]+)\|([^|\]]+)\]\]', r'\2', line)  # [[link|text]] -> text
    line = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', line)  # [[link]] -> link
    line = re.sub(r"'''([^']+)'''", r'\1', line)  # Remove bold markup
    line = re.sub(r"''([^']+)''", r'\1', line)  # Remove italic markup
    line = re.sub(r'@-@ ', '', line)  # Remove @-@ tokens
    line = re.sub(r'@,@ ', ', ', line)  # Replace @,@ with comma
    line = re.sub(r'@\.@ ', '. ', line)  # Replace @.@ with period
    line = re.sub(r' +', ' ', line)  # Collapse multiple spaces
    
    return line.strip()


def split_into_sentences(text: str) -> List[str]:
    """Basic sentence splitting for WikiText.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting on periods, exclamation marks, question marks
    # This is basic - for production use, consider using spaCy or NLTK
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Clean up sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Skip very short fragments
            # Ensure sentence ends with punctuation
            if not sentence[-1] in '.!?':
                sentence += '.'
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def process_wikitext(dataset: str, split: str = "train", cores: int = None,
                    batch_size: int = 200, save_json: bool = False,
                    output_dir: str = "wikitext_output", verbose: bool = True,
                    max_sentences: int = None, min_words: int = 3,
                    max_words: int = 100, data_dir: str = "data") -> Dict[str, Any]:
    """Process WikiText dataset with multiprocessing.
    
    Args:
        dataset: 'wikitext-2' or 'wikitext-103'
        split: 'train', 'valid', or 'test'
        cores: Number of CPU cores
        batch_size: Sentences per batch
        save_json: Save individual JSON files
        output_dir: Output directory
        verbose: Show detailed progress
        max_sentences: Max sentences to process
        min_words: Min words per sentence
        max_words: Max words per sentence
        data_dir: Directory for WikiText data
        
    Returns:
        Processing statistics
    """
    if cores is None:
        cores = mp.cpu_count()
    
    # Download WikiText if needed
    dataset_path = download_wikitext(dataset, data_dir)
    
    # Find the appropriate file
    if dataset == 'wikitext-2':
        file_path = dataset_path / f"wiki.{split}.tokens"
    else:  # wikitext-103
        file_path = dataset_path / f"wiki.{split}.tokens"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        print(f"Available files in {dataset_path}:")
        for f in dataset_path.iterdir():
            print(f"   {f.name}")
        sys.exit(1)
    
    # Extract sentences
    sentences = extract_sentences_from_wikitext(
        file_path, max_sentences, min_words, max_words
    )
    
    if not sentences:
        print("❌ No sentences extracted from WikiText")
        return {}
    
    print(f"🚀 Starting WikiText processing:")
    print(f"   📚 Dataset: {dataset} ({split} split)")
    print(f"   📊 {len(sentences)} sentences")
    print(f"   🖥️  {cores} CPU cores")
    print(f"   📦 {batch_size} sentences per batch")
    print(f"   💾 JSON output: {'enabled' if save_json else 'disabled'}")
    
    # Use the corpus processor's batch processing function
    from corpus_processor import create_batches
    
    # Create configuration
    config = {
        'save_json': save_json,
        'output_dir': output_dir,
        'verbose': verbose and len(sentences) <= 100
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
        'dataset': dataset,
        'split': split,
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
    print(f"\n⚡ WikiText processing completed!")
    print(f"📚 Dataset: {dataset} ({split})")
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
        print(f"\n❌ Sample failed sentences:")
        for result in failed_results[:3]:
            print(f"   {result['index']}: {result['sentence'][:50]}...")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Process WikiText corpus with 64-core multiprocessing")
    parser.add_argument("--dataset", choices=['wikitext-2', 'wikitext-103'], default='wikitext-2',
                      help="WikiText dataset to use (default: wikitext-2)")
    parser.add_argument("--split", choices=['train', 'valid', 'test'], default='train',
                      help="Dataset split to process (default: train)")
    parser.add_argument("--cores", type=int, default=None,
                      help="Number of CPU cores (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=200,
                      help="Sentences per batch (default: 200)")
    parser.add_argument("--save-json", action="store_true",
                      help="Save individual parse results as JSON files")
    parser.add_argument("--output-dir", default="wikitext_output",
                      help="Output directory for JSON files (default: wikitext_output)")
    parser.add_argument("--quiet", action="store_true",
                      help="Minimal output")
    parser.add_argument("--max-sentences", type=int, default=None,
                      help="Maximum sentences to process")
    parser.add_argument("--min-words", type=int, default=3,
                      help="Minimum words per sentence (default: 3)")
    parser.add_argument("--max-words", type=int, default=100,
                      help="Maximum words per sentence (default: 100)")
    parser.add_argument("--data-dir", default="data",
                      help="Directory to store WikiText data (default: data)")
    parser.add_argument("--save-stats", help="Save processing statistics to JSON file")
    
    args = parser.parse_args()
    
    # Process WikiText
    stats = process_wikitext(
        dataset=args.dataset,
        split=args.split,
        cores=args.cores,
        batch_size=args.batch_size,
        save_json=args.save_json,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        max_sentences=args.max_sentences,
        min_words=args.min_words,
        max_words=args.max_words,
        data_dir=args.data_dir
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