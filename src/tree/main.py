#!/usr/bin/env python3
"""
main.py – CCG tree visualizer with direct PNG output

    python main.py "I placed my red hat in Johnny's hand"
    python main.py -o tree.png "The fox jumped over the dog"
    python main.py --cores 4 "Sentence 1" "Sentence 2" "Sentence 3"  # parallel processing
"""

import argparse
import sys
import os
import re
import time
import json
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict, Any

from depccg_treeviz import CCGTreeVisualizer as DepCCG
#from easyccg_treeviz import EasyCCGTreeVisualizer as EasyCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("sentence", nargs="*", help="Sentence (if empty, read stdin).")
    p.add_argument("-o", "--output", help="Output PNG file path.")
    p.add_argument("--width", type=int, default=12, help="Image width in inches (default: 12)")
    p.add_argument("--height", type=int, default=8, help="Image height in inches (default: 8)")
    p.add_argument("--no-save", action="store_true", help="Don't save files, just parse and display")
    p.add_argument("--cores", type=int, default=None, 
                  help="Number of CPU cores to use for parallel processing (default: auto-detect)")
    return p.parse_args()


def sanitize_filename(text: str, max_length: int = 20) -> str:
    """Convert text to a safe filename by removing punctuation and limiting length."""
    # Remove/replace problematic characters
    safe = re.sub(r'[^\w\s-]', '', text.strip())
    # Replace spaces with underscores
    safe = re.sub(r'\s+', '_', safe)
    # Limit length
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe or "sentence"


def process_single_sentence_worker(task_data: Tuple[int, str, Dict[str, Any]]) -> Dict[str, Any]:
    """Process a single sentence in a worker process.
    
    Args:
        task_data: Tuple of (index, sentence, config_dict)
    
    Returns:
        Dict with processing results and timing information
    """
    index, sentence, config = task_data
    
    # Create parser instances in each worker process
    vis_dep = DepCCG()
    vis_bene = Benepar()
    
    results = {
        'index': index,
        'sentence': sentence,
        'success': True,
        'files_created': [],
        'timing': {}
    }
    
    try:
        if config['no_save']:
            # Parse-only mode - no file output
            print(f"Worker {index}: Processing sentence {index} (parse-only)")
            
            # DepCCG parsing
            start_time = time.time()
            dep_data = vis_dep.parse_only(sentence)
            dep_time = time.time() - start_time
            results['timing']['depccg'] = dep_time
            
            # Benepar parsing
            start_time = time.time()
            ben_data = vis_bene.parse_only(sentence)
            ben_time = time.time() - start_time
            results['timing']['benepar'] = ben_time
            
        else:
            # Full mode with file output
            base = config['output_base'] or f"output/{sanitize_filename(sentence[:20])}"
            if len(config['all_sentences']) > 1:
                # Add index for multiple sentences
                base = f"{base}_{index:03d}"
            
            dep_out = base + "_depccg.png"
            dep_json = base + "_depccg.json"
            ben_out = base + "_benepar.png"
            ben_json = base + "_benepar.json"
            
            print(f"Worker {index}: Processing sentence {index} -> {base}_*.png")
            
            # DepCCG processing
            start_time = time.time()
            dep_data = vis_dep.save_tree_image(sentence, dep_out, 
                                             width=config['width'], height=config['height'])
            dep_time = time.time() - start_time
            results['timing']['depccg'] = dep_time
            
            if dep_data:
                with open(dep_json, 'w') as f:
                    json.dump(dep_data, f, indent=2)
                results['files_created'].extend([dep_out, dep_json])
            
            # Benepar processing
            start_time = time.time()
            ben_data = vis_bene.save_tree_image(sentence, ben_out, 
                                              width=config['width'], height=config['height'])
            ben_time = time.time() - start_time
            results['timing']['benepar'] = ben_time
            
            if ben_data:
                with open(ben_json, 'w') as f:
                    json.dump(ben_data, f, indent=2)
                results['files_created'].extend([ben_out, ben_json])
        
        print(f"Worker {index}: Completed in {sum(results['timing'].values()):.2f}s")
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        print(f"Worker {index}: Error processing sentence - {e}")
    
    return results


def main():
    args = cli()
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    sentences = [" ".join(args.sentence)] if args.sentence else [line.strip() for line in sys.stdin if line.strip()]
    if not sentences:
        print("No sentences provided.")
        return

    # Determine number of cores
    num_cores = args.cores or mp.cpu_count()
    
    print(f"🔬 Processing {len(sentences)} sentence(s) using {num_cores} core(s)")
    
    if len(sentences) == 1:
        # Single sentence - no need for multiprocessing overhead
        print("📝 Single sentence mode - processing directly")
        
        sentence = sentences[0]
        vis_dep = DepCCG()
        vis_bene = Benepar()
        
        if args.no_save:
            # Parse-only mode
            print("\n=== DepCCG ===")
            start_time = time.time()
            dep_data = vis_dep.parse_only(sentence)
            dep_time = time.time() - start_time
            print(f"Time taken: {dep_time:.2f} seconds")

            print("\n=== Benepar ===")
            start_time = time.time()
            ben_data = vis_bene.parse_only(sentence)
            ben_time = time.time() - start_time
            print(f"Time taken: {ben_time:.2f} seconds")
            
            print(f"\n🚀 Total processing time: {dep_time + ben_time:.2f}s")
        else:
            # File output mode
            base = args.output or f"output/{sanitize_filename(sentence[:20])}"
            dep_out = base + "_depccg.png"
            dep_json = base + "_depccg.json"
            ben_out = base + "_benepar.png" 
            ben_json = base + "_benepar.json"

            print("\n=== DepCCG ===")
            start_time = time.time()
            dep_data = vis_dep.save_tree_image(sentence, dep_out, width=args.width, height=args.height)
            dep_time = time.time() - start_time
            print(f"Time taken: {dep_time:.2f} seconds")
            
            if dep_data:
                with open(dep_json, 'w') as f:
                    json.dump(dep_data, f, indent=2)
                print(f"💾 Saved: {dep_out}, {dep_json}")

            print("\n=== Benepar ===")
            start_time = time.time()
            ben_data = vis_bene.save_tree_image(sentence, ben_out, width=args.width, height=args.height)
            ben_time = time.time() - start_time
            print(f"Time taken: {ben_time:.2f} seconds")
            
            if ben_data:
                with open(ben_json, 'w') as f:
                    json.dump(ben_data, f, indent=2)
                print(f"💾 Saved: {ben_out}, {ben_json}")
            
            print(f"\n🚀 Total processing time: {dep_time + ben_time:.2f}s")
    
    else:
        # Multiple sentences - use multiprocessing
        print("📋 Multiple sentence mode - using parallel processing")
        
        # Prepare task data
        config = {
            'no_save': args.no_save,
            'output_base': args.output,
            'width': args.width,
            'height': args.height,
            'all_sentences': sentences
        }
        
        task_data = [(i+1, sentence, config) for i, sentence in enumerate(sentences)]
        
        # Process in parallel
        start_time = time.time()
        with mp.Pool(processes=num_cores) as pool:
            worker_results = pool.map(process_single_sentence_worker, task_data)
        
        total_time = time.time() - start_time
        
        # Aggregate and report results
        successful_results = [r for r in worker_results if r['success']]
        failed_results = [r for r in worker_results if not r['success']]
        
        print(f"\n⚡ Parallel processing completed in {total_time:.2f} seconds")
        print(f"📊 Successfully processed {len(successful_results)}/{len(sentences)} sentences")
        
        if successful_results:
            # Calculate timing statistics
            all_depccg_times = [r['timing'].get('depccg', 0) for r in successful_results]
            all_benepar_times = [r['timing'].get('benepar', 0) for r in successful_results]
            
            avg_depccg = sum(all_depccg_times) / len(all_depccg_times)
            avg_benepar = sum(all_benepar_times) / len(all_benepar_times)
            
            print(f"📈 Timing Statistics:")
            print(f"   Average DepCCG time: {avg_depccg:.2f}s per sentence")
            print(f"   Average Benepar time: {avg_benepar:.2f}s per sentence")
            print(f"   Processing speed: {len(successful_results)/total_time:.1f} sentences/second")
        
        if not args.no_save and successful_results:
            # Report created files
            all_files = []
            for result in successful_results:
                all_files.extend(result['files_created'])
            print(f"💾 Created {len(all_files)} files in output/ directory")
        
        if failed_results:
            print(f"❌ Failed to process {len(failed_results)} sentences:")
            for result in failed_results:
                print(f"   - Sentence {result['index']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    
    main()
