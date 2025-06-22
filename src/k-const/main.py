#!/usr/bin/env python3
import argparse
import importlib.util
import sys
import os

# Import from prob-dist.py (handle hyphen in filename)
spec = importlib.util.spec_from_file_location("prob_dist", os.path.join(os.path.dirname(__file__), "prob-dist.py"))
prob_dist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prob_dist)
get_pos_distribution = prob_dist.get_pos_distribution
process_text_file_bulk_efficient = prob_dist.process_text_file_bulk_efficient
cleanup_checkpoint = prob_dist.cleanup_checkpoint

# Import from prob-dist-mat.py (handle hyphen in filename)
spec_mat = importlib.util.spec_from_file_location("prob_dist_mat", os.path.join(os.path.dirname(__file__), "prob-dist-mat.py"))
prob_dist_mat = importlib.util.module_from_spec(spec_mat)
spec_mat.loader.exec_module(prob_dist_mat)
get_word_vector = prob_dist_mat.get_word_vector
convert_pos_cache_to_matrix = prob_dist_mat.convert_pos_cache_to_matrix
get_similar_words = prob_dist_mat.get_similar_words
print_word_vector = prob_dist_mat.print_word_vector
sentence_to_matrix = prob_dist_mat.sentence_to_matrix

# Import from pos-mask.py
spec_mask = importlib.util.spec_from_file_location("pos_mask", os.path.join(os.path.dirname(__file__), "pos-mask.py"))
pos_mask = importlib.util.module_from_spec(spec_mask)
spec_mask.loader.exec_module(pos_mask)
get_mask_for_pos_tags = pos_mask.get_mask_for_pos_tags

# Import from adjective-position-analysis.py
spec_adj = importlib.util.spec_from_file_location("adjective_position_analysis", os.path.join(os.path.dirname(__file__), "adjective-position-analysis.py"))
adjective_position_analysis = importlib.util.module_from_spec(spec_adj)
spec_adj.loader.exec_module(adjective_position_analysis)
analyze_adjective_positions = adjective_position_analysis.analyze_adjective_positions
print_results_pos = adjective_position_analysis.print_results

# Import from dependency-adjective-analysis.py
spec_dep = importlib.util.spec_from_file_location("dependency_adjective_analysis", os.path.join(os.path.dirname(__file__), "dependency-adjective-analysis.py"))
dependency_adjective_analysis = importlib.util.module_from_spec(spec_dep)
spec_dep.loader.exec_module(dependency_adjective_analysis)
analyze_adjective_noun_dependencies = dependency_adjective_analysis.analyze_adjective_noun_dependencies
print_results_dep = dependency_adjective_analysis.print_results

# Import from dependency-distance-analysis.py
spec_dist = importlib.util.spec_from_file_location("dependency_distance_analysis", os.path.join(os.path.dirname(__file__), "dependency-distance-analysis.py"))
dependency_distance_analysis = importlib.util.module_from_spec(spec_dist)
spec_dist.loader.exec_module(dependency_distance_analysis)
analyze_dependency_distances = dependency_distance_analysis.analyze_dependency_distances
print_results_dist = dependency_distance_analysis.print_results
print_summary_dist = dependency_distance_analysis.print_summary
print_pos_structured_results = dependency_distance_analysis.print_pos_structured_results

# Import from dependency-distribution-modeling.py
spec_distrib = importlib.util.spec_from_file_location("dependency_distribution_modeling", os.path.join(os.path.dirname(__file__), "dependency-distribution-modeling.py"))
dependency_distribution_modeling = importlib.util.module_from_spec(spec_distrib)
spec_distrib.loader.exec_module(dependency_distribution_modeling)
analyze_dependency_distributions = dependency_distribution_modeling.analyze_dependency_distributions
print_distribution_results = dependency_distribution_modeling.print_distribution_results
save_compact_distributions = dependency_distribution_modeling.save_compact_distributions

def analyze_sentence(text: str, output_file: str = None, raw: bool = False, format_type: str = 'table', highlight_pos: list = None):
    """
    Analyze a sentence and get POS vectors for each word.
    
    Args:
        text: The sentence or text to analyze
        output_file: Optional file to save results
        raw: Whether to show raw vectors
        format_type: Output format ('table', 'json', 'compact')
    """
    import nltk
    import json
    
    # Tokenize the text
    nltk.download('punkt', quiet=True)
    tokens = nltk.word_tokenize(text)
    
    # Filter to only alphabetic tokens
    words = [token.lower() for token in tokens if token.isalpha()]
    
    print(f"Analyzing sentence: \"{text}\"")
    print(f"Found {len(words)} alphabetic words: {', '.join(words)}")
    print()
    
    # Get vectors for each word
    results = {}
    for word in words:
        vector = get_word_vector(word)
        results[word] = vector
    
    # Output based on format
    if format_type == 'table':
        if raw:
            print("Word           | Raw Vector")
            print("---------------|" + "-" * 80)
            for word, vector in results.items():
                print(f"{word:14} | {vector}")
        else:
            # Show non-zero POS tags for each word
            for word, vector in results.items():
                print(f"'{word}':")
                print("  Tag    | Probability")
                print("  -------|------------")
                for i, tag in enumerate(prob_dist_mat.UNIVERSAL_POS_TAGS):
                    prob = vector[i]
                    if prob > 0:
                        print(f"  {tag:6} | {prob:.4f}")
                print()
    
    elif format_type == 'json':
        if raw:
            json_output = results
        else:
            # Convert to POS distribution format
            json_output = {}
            for word, vector in results.items():
                pos_dist = {}
                for i, tag in enumerate(prob_dist_mat.UNIVERSAL_POS_TAGS):
                    if vector[i] > 0:
                        pos_dist[tag] = vector[i]
                json_output[word] = pos_dist
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        else:
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
    
    elif format_type == 'compact':
        for word, vector in results.items():
            if raw:
                print(f"{word}: {vector}")
            else:
                # Show only non-zero tags
                active_tags = []
                for i, tag in enumerate(prob_dist_mat.UNIVERSAL_POS_TAGS):
                    if vector[i] > 0:
                        active_tags.append(f"{tag}:{vector[i]:.3f}")
                print(f"{word}: {', '.join(active_tags)}")
    
    elif format_type == 'matrix':
        import numpy as np
        
        # Create matrix: n words x 16 POS dimensions
        words_list = list(results.keys())
        matrix = np.array([results[word] for word in words_list])
        
        # Get highlight indices if specified
        highlight_indices = set()
        if highlight_pos:
            highlight_indices = get_mask_for_pos_tags(highlight_pos)
        
        print(f"Matrix shape: {len(words_list)} words × 16 POS dimensions")
        if highlight_pos:
            print(f"Highlighting: {', '.join(highlight_pos)}")
        print()
        
        # Print header with POS tags
        print("Word".ljust(12), end=" ")
        for i, tag in enumerate(prob_dist_mat.UNIVERSAL_POS_TAGS):
            if i in highlight_indices:
                print(f"\033[91m{tag:>4}\033[0m", end=" ")  # Red header for highlighted columns
            else:
                print(f"{tag:>4}", end=" ")
        print()
        print("-" * 12 + " " + "-" * (5 * 16))
        
        # Print each word's vector as a row
        for i, word in enumerate(words_list):
            print(f"{word[:11]:<11}", end=" ")
            for j, value in enumerate(matrix[i]):
                # Format as .xx (remove leading 0 from 0.xx)
                if value >= 1.0:
                    formatted_value = f"{value:.2f}"  # Keep 1.00 format for values >= 1
                else:
                    formatted_value = f"{value:.2f}"[1:]  # Remove leading 0 for 0.xx -> .xx
                
                # Apply highlighting
                if j in highlight_indices and value > 0.001:
                    # Highlighted non-zero values in red
                    print(f"\033[91m{formatted_value:>4}\033[0m", end=" ")
                elif value > 0.001:  # Non-zero values in normal color
                    print(f"{formatted_value:>4}", end=" ")
                else:  # Zero values in very dark grey (almost black)
                    print(f"\033[2;30m{formatted_value:>4}\033[0m", end=" ")
            print()
    
    if output_file and format_type != 'json':
        # Save raw results to file for non-json formats
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Raw results also saved to {output_file}")

def main():
    p = argparse.ArgumentParser(
        description="Estimate POS-tag distributions (Universal tags) from the Brown corpus."
    )
    
    # Create subparsers for different modes
    subparsers = p.add_subparsers(dest='mode', help='Processing mode')
    
    # Single word mode
    single_parser = subparsers.add_parser('word', help='Process a single word')
    single_parser.add_argument('word', help="The word to analyze (case-insensitive).")
    
    # Bulk mode for text files (now with efficient processing)
    bulk_parser = subparsers.add_parser('bulk', help='Process all tokens from a text file (efficient method)')
    bulk_parser.add_argument('input_file', help="Path to the input text file.")
    bulk_parser.add_argument('-o', '--output', help="Optional output JSON file path.")
    bulk_parser.add_argument('--print-summary', action='store_true', 
                           help="Print a summary of results to console.")
    bulk_parser.add_argument('--no-resume', action='store_true',
                           help="Don't resume from checkpoint, start fresh.")
    bulk_parser.add_argument('--fast', action='store_true',
                           help="Use faster but less accurate POS tagger.")
    
    # Checkpoint management
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Manage processing checkpoints')
    checkpoint_parser.add_argument('action', choices=['clean'], help='Checkpoint action')
    
    # Matrix mode subcommands
    # Convert mode - convert pos_cache.json to pos_mat_cache.json
    convert_parser = subparsers.add_parser('convert', help='Convert POS cache to matrix format')
    
    # Vector mode - get vector for a single word
    vector_parser = subparsers.add_parser('vector', help='Get POS vector for a word')
    vector_parser.add_argument('word', help="The word to get vector for.")
    vector_parser.add_argument('--raw', action='store_true', help="Output raw vector numbers.")
    
    # Similar mode - find words with similar POS distributions
    similar_parser = subparsers.add_parser('similar', help='Find words with similar POS distributions')
    similar_parser.add_argument('word', help="The target word.")
    similar_parser.add_argument('-k', '--top-k', type=int, default=10, help="Number of similar words to return.")
    
    # Batch mode - get vectors for multiple words
    batch_parser = subparsers.add_parser('batch', help='Get vectors for multiple words')
    batch_parser.add_argument('words', nargs='+', help="Words to get vectors for.")
    batch_parser.add_argument('--output', '-o', help="Save results to JSON file.")
    
    # Sentence mode - analyze a sentence and get vectors for each word
    sentence_parser = subparsers.add_parser('sentence', help='Analyze a sentence and get vectors for each word')
    sentence_parser.add_argument('text', help="The sentence or text to analyze.")
    sentence_parser.add_argument('--output', '-o', help="Save results to JSON file.")
    sentence_parser.add_argument('--raw', action='store_true', help="Output raw vector numbers.")
    sentence_parser.add_argument('--format', choices=['table', 'json', 'compact', 'matrix'], default='table', 
                               help="Output format (default: table).")
    sentence_parser.add_argument('--highlight', nargs='+', help="POS tags to highlight in red (e.g. NOUN VERB).")
    
    # Matrix mode - get fixed-size matrix for a sentence
    matrix_parser = subparsers.add_parser('matrix', help='Get fixed-size matrix (m×16) for a sentence')
    matrix_parser.add_argument('text', help="The sentence or text to convert to matrix.")
    matrix_parser.add_argument('--length', '-l', type=int, default=32, help="Matrix length (default: 32).")
    matrix_parser.add_argument('--output', '-o', help="Save matrix to numpy file (.npy).")
    matrix_parser.add_argument('--print', action='store_true', help="Print the matrix values.")
    matrix_parser.add_argument('--highlight', nargs='+', help="POS tags to highlight in red (e.g. NOUN VERB).")
    
    # Adjective position analysis - analyze positional distribution of adjectives around nouns
    adj_parser = subparsers.add_parser('adjpos', help='Analyze positional distribution of adjectives around nouns')
    adj_parser.add_argument('input_file', help="Path to the input text file")
    adj_parser.add_argument('--range', '-r', type=int, default=2, help="Displacement range to analyze (default: 2)")
    adj_parser.add_argument('--output', '-o', help="Save results to JSON file")
    adj_parser.add_argument('--quiet', '-q', action='store_true', help="Only show summary, not detailed results")
    
    # Dependency adjective analysis - analyze syntactic adjective-noun relationships using spaCy
    dep_parser = subparsers.add_parser('depadj', help='Analyze adjective-noun dependencies using spaCy parser')
    dep_parser.add_argument('input_file', help="Path to the input text file")
    dep_parser.add_argument('--output', '-o', help="Save results to JSON file")
    dep_parser.add_argument('--quiet', '-q', action='store_true', help="Only show summary, not detailed results")
    
    # Comprehensive dependency distance analysis - analyze signed distances for ALL dependency types
    dist_parser = subparsers.add_parser('depdist', help='Analyze signed distances for all dependency relationships')
    dist_parser.add_argument('input_file', help="Path to the input text file")
    dist_parser.add_argument('--output', '-o', help="Save results to JSON file")
    dist_parser.add_argument('--quiet', '-q', action='store_true', help="Only show summary, not detailed results")
    dist_parser.add_argument('--pos-only', action='store_true', help="Only show POS-organized results")
    dist_parser.add_argument('--pos-json', metavar='FILE', help="Save only the POS-organized data structure to a clean JSON file")
    
    # Dependency distribution modeling - model distances as Gaussian distributions
    distrib_parser = subparsers.add_parser('depgauss', help='Model dependency distances as Gaussian distributions')
    distrib_parser.add_argument('input_file', help="Path to the input text file")
    distrib_parser.add_argument('--output', '-o', help="Save full results to JSON file")
    distrib_parser.add_argument('--compact', '-c', help="Save compact distribution models to JSON file")
    distrib_parser.add_argument('--min-samples', type=int, default=10, help="Minimum samples needed to fit distributions")
    distrib_parser.add_argument('--quiet', '-q', action='store_true', help="Only show summary, not detailed results")
    
    args = p.parse_args()
    if args.mode is None:
        p.print_help()
        return

    if args.mode == 'word':
        # Single word processing
        dist = get_pos_distribution(args.word)
        if not dist:
            print(f"No occurrences of '{args.word}' found in the Brown corpus.")
        else:
            print(f"POS distribution for '{args.word}' (Brown corpus):")
            for tag, info in sorted(dist.items(), key=lambda x: -x[1]['count']):
                print(f"  {tag:6} → {info['count']:5} times  ({info['relative']*100:.1f}%)")
    
    elif args.mode == 'bulk':
        # Efficient bulk text file processing
        resume = not args.no_resume
        results = process_text_file_bulk_efficient(args.input_file, args.output, resume=resume, fast=args.fast)
        
        if args.print_summary and results:
            print(f"\nSample results:")
            # Show first 5 words with distributions
            sample_words = list(results.keys())[:5]
            for word in sample_words:
                dist = results[word]
                if dist:
                    print(f"\n'{word}':")
                    for tag, info in sorted(dist.items(), key=lambda x: -x[1]['count']):
                        print(f"  {tag:6} → {info['count']:5} times  ({info['relative']*100:.1f}%)")
                else:
                    print(f"'{word}': No occurrences found")
    
    elif args.mode == 'checkpoint':
        if args.action == 'clean':
            cleanup_checkpoint()
            print("Checkpoint cleaned.")
    
    elif args.mode == 'convert':
        convert_pos_cache_to_matrix()
    
    elif args.mode == 'vector':
        if args.raw:
            vector = get_word_vector(args.word)
            print(f"Vector for '{args.word}': {vector}")
        else:
            print_word_vector(args.word)
    
    elif args.mode == 'similar':
        print(f"Finding words similar to '{args.word}'...")
        similar_words = get_similar_words(args.word, args.top_k)
        
        if not similar_words:
            print("No similar words found (make sure to run 'convert' first).")
        else:
            print(f"\nTop {len(similar_words)} words similar to '{args.word}':")
            print("Word              | Similarity")
            print("------------------|------------")
            for word, similarity in similar_words:
                print(f"{word:17} | {similarity:.4f}")
    
    elif args.mode == 'batch':
        import json
        results = {}
        
        print(f"Getting vectors for {len(args.words)} words...")
        for word in args.words:
            vector = get_word_vector(word)
            results[word] = vector
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            # Print results
            for word, vector in results.items():
                print(f"\n{word}: {vector}")
    
    elif args.mode == 'sentence':
        analyze_sentence(args.text, args.output, args.raw, args.format, args.highlight)
    
    elif args.mode == 'matrix':
        import numpy as np
        matrix, words = sentence_to_matrix(args.text, args.length)
        
        # Get highlight indices if specified
        highlight_indices = set()
        if args.highlight:
            highlight_indices = get_mask_for_pos_tags(args.highlight)
        
        print(f"Generated {args.length}×16 matrix for: \"{args.text}\"")
        print(f"Words used: {', '.join(words)} ({len(words)} words)")
        print(f"Matrix shape: {matrix.shape}")
        if args.highlight:
            print(f"Highlighting: {', '.join(args.highlight)}")
        print()
        
        # Print visual matrix format
        print("Word".ljust(12), end=" ")
        for i, tag in enumerate(prob_dist_mat.UNIVERSAL_POS_TAGS):
            if i in highlight_indices:
                print(f"\033[91m{tag:>4}\033[0m", end=" ")  # Red header for highlighted columns
            else:
                print(f"{tag:>4}", end=" ")
        print()
        print("-" * 12 + " " + "-" * (5 * 16))
        
        # Print each row with word name (or empty for padding rows)
        for i in range(args.length):
            if i < len(words):
                word_name = words[i][:11]
            else:
                word_name = f"[{i+1}]"  # Show row number for empty rows
            
            print(f"{word_name:<11}", end=" ")
            for j, value in enumerate(matrix[i]):
                # Format as .xx (remove leading 0 from 0.xx)
                if value >= 1.0:
                    formatted_value = f"{value:.2f}"  # Keep 1.00 format for values >= 1
                else:
                    formatted_value = f"{value:.2f}"[1:]  # Remove leading 0 for 0.xx -> .xx
                
                # Apply highlighting
                if j in highlight_indices and value > 0.001:
                    # Highlighted non-zero values in red
                    print(f"\033[91m{formatted_value:>4}\033[0m", end=" ")
                elif value > 0.001:  # Non-zero values in normal color
                    print(f"{formatted_value:>4}", end=" ")
                else:  # Zero values in very dark grey (almost black)
                    print(f"\033[2;30m{formatted_value:>4}\033[0m", end=" ")
            print()
        
        if args.print:
            print("\nRaw matrix (numpy array):")
            print(matrix)
        
        if args.output:
            np.save(args.output, matrix)
            print(f"Matrix saved to {args.output}")
        
        # Always show a summary of non-zero rows
        non_zero_rows = np.count_nonzero(matrix, axis=1)
        filled_rows = np.sum(non_zero_rows > 0)
        print(f"Filled rows: {filled_rows}/{args.length}")
        print(f"Empty rows: {args.length - filled_rows}")
        
        # Show shape for easy copying to other code
        print(f"Usage: matrix = np.load('{args.output or 'matrix.npy'}')")
        print(f"Shape: {matrix.shape} (dtype: {matrix.dtype})")
    
    elif args.mode == 'adjpos':
        # Adjective position analysis
        import json
        
        # Validate displacement range
        if args.range < 1 or args.range > 10:
            print("Error: Displacement range must be between 1 and 10")
            return
        
        # Run analysis
        results = analyze_adjective_positions(args.input_file, args.range)
        
        if not results:
            print("Analysis failed or no results generated.")
            return
        
        # Print results
        if not args.quiet:
            print_results_pos(results)
        else:
            # Just show summary
            print(f"Analysis complete: {results['total_nouns']:,} nouns, "
                  f"{results['summary']['total_adjectives_found']:,} adjectives found")
            
            # Show top position
            sorted_by_prob = sorted(results['position_analysis'].items(), 
                                   key=lambda x: x[1]['probability'], reverse=True)
            if sorted_by_prob:
                top_pos, top_data = sorted_by_prob[0]
                direction = "before" if top_pos < 0 else "after"
                print(f"Most likely position: {abs(top_pos)} {direction} noun ({top_data['probability']:.1%})")
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
    
    elif args.mode == 'depadj':
        # Dependency adjective analysis
        import json
        
        # Run analysis
        results = analyze_adjective_noun_dependencies(args.input_file)
        
        if not results:
            print("Analysis failed or no results generated.")
            return
        
        # Print results
        if not args.quiet:
            print_results_dep(results)
        else:
            # Just show summary
            print(f"Analysis complete: {results['total_relationships']:,} adj-noun relationships found")
            if results['total_relationships'] > 0:
                stats = results['distance_statistics']
                print(f"Average distance: {stats['mean']:.2f} tokens")
                direction = results['direction_analysis']
                print(f"Direction: {direction['before_percentage']:.1f}% before, {direction['after_percentage']:.1f}% after")
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
    
    elif args.mode == 'depdist':
        # Comprehensive dependency distance analysis
        import json
        
        # Run analysis
        results = analyze_dependency_distances(args.input_file)
        
        if not results:
            print("Analysis failed or no results generated.")
            return
        
        # Print results
        if args.pos_only:
            print_pos_structured_results(results)
        elif not args.quiet:
            print_results_dist(results)
            print_pos_structured_results(results)
        else:
            print_summary_dist(results)
        
        # Save to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nResults saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
        
        # Save just the POS-organized structure if requested
        if args.pos_json:
            try:
                pos_data = results.get('pos_dependency_offsets', {})
                with open(args.pos_json, 'w', encoding='utf-8') as f:
                    json.dump(pos_data, f, indent=2, ensure_ascii=False)
                print(f"\nPOS-organized data saved to {args.pos_json}")
            except Exception as e:
                print(f"Error saving POS data: {e}")
    
    elif args.mode == 'depgauss':
        # Dependency distribution modeling
        import json
        
        # Run analysis
        results = analyze_dependency_distributions(args.input_file, args.min_samples)
        
        if not results:
            print("Analysis failed or no results generated.")
            return
        
        # Print results
        if not args.quiet:
            print_distribution_results(results)
        else:
            summary = results['summary']
            print(f"Analysis complete: {summary['total_relationships']:,} relationships analyzed")
            print(f"Fitted {summary['fitted_distributions']} distributions from {summary['dependency_combinations']} combinations")
            print(f"Distribution types: {summary['distribution_types']['single_gaussian']} Gaussian, "
                  f"{summary['distribution_types']['gaussian_mixture']} mixtures")
        
        # Save results
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nFull results saved to {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")
        
        if args.compact:
            try:
                save_compact_distributions(results, args.compact)
                print(f"Compact distribution models saved to {args.compact}")
            except Exception as e:
                print(f"Error saving compact results: {e}")

if __name__ == '__main__':
    main() 