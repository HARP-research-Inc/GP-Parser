#!/usr/bin/env python3
"""
Dependency Distance Distribution Modeling

This script analyzes dependency relationships and models them as Gaussian distributions
for much more compact storage and statistical analysis.

Usage:
    python dependency-distribution-modeling.py input.txt
    python dependency-distribution-modeling.py input.txt --output distributions.json
"""

import argparse
import json
import os
from collections import defaultdict, Counter
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings

def ensure_dependencies():
    """Ensure required dependencies are available."""
    try:
        import spacy
        import scipy
        import numpy
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install with: pip install spacy scipy numpy")
        print("And download spaCy model: python -m spacy download en_core_web_sm")
        return None
    
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        print("Error: spaCy English model not found. Please download it with:")
        print("python -m spacy download en_core_web_sm")
        return None

def fit_gaussian_mixture(distances, max_components=3, min_samples=10):
    """
    Fit a Gaussian mixture model to distance data.
    
    Args:
        distances: List of signed distances
        max_components: Maximum number of Gaussian components to try
        min_samples: Minimum samples required to fit a distribution
    
    Returns:
        dict: Best-fit distribution parameters
    """
    if len(distances) < min_samples:
        return {
            'type': 'insufficient_data',
            'sample_count': len(distances),
            'raw_mean': float(np.mean(distances)),
            'raw_std': float(np.std(distances)) if len(distances) > 1 else 0.0
        }
    
    distances = np.array(distances)
    
    # Try single Gaussian first
    mean_est = np.mean(distances)
    std_est = np.std(distances)
    
    if std_est == 0:  # All values are the same
        return {
            'type': 'point_mass',
            'location': float(mean_est),
            'sample_count': len(distances),
            'probability': 1.0
        }
    
    # Single Gaussian log-likelihood
    single_ll = np.sum(stats.norm.logpdf(distances, mean_est, std_est))
    
    best_model = {
        'type': 'gaussian',
        'components': [{
            'weight': 1.0,
            'mean': float(mean_est),
            'std': float(std_est)
        }],
        'log_likelihood': float(single_ll),
        'aic': float(-2 * single_ll + 2 * 2),  # 2 parameters (mean, std)
        'sample_count': len(distances)
    }
    
    # Try mixture models only if we have enough data
    if len(distances) >= max_components * 5:  # At least 5 samples per component
        for n_components in range(2, min(max_components + 1, len(distances) // 3)):
            try:
                # Simple EM-style fitting for mixture of Gaussians
                mixture_params = fit_gaussian_mixture_em(distances, n_components)
                if mixture_params:
                    # Calculate log-likelihood
                    ll = calculate_mixture_loglikelihood(distances, mixture_params)
                    n_params = n_components * 3 - 1  # (weight, mean, std) per component, minus 1 constraint
                    aic = -2 * ll + 2 * n_params
                    
                    if aic < best_model['aic']:
                        best_model = {
                            'type': 'gaussian_mixture',
                            'components': mixture_params,
                            'log_likelihood': float(ll),
                            'aic': float(aic),
                            'sample_count': len(distances)
                        }
            except:
                continue  # Skip if fitting fails
    
    return best_model

def fit_gaussian_mixture_em(data, n_components, max_iter=50, tol=1e-6):
    """Simple EM algorithm for Gaussian mixture fitting."""
    n_samples = len(data)
    data = np.array(data)
    
    # Initialize parameters
    np.random.seed(42)  # For reproducibility
    weights = np.ones(n_components) / n_components
    means = np.random.choice(data, n_components, replace=False)
    stds = np.full(n_components, np.std(data))
    
    for iteration in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = np.zeros((n_samples, n_components))
        for k in range(n_components):
            if stds[k] > 0:
                responsibilities[:, k] = weights[k] * stats.norm.pdf(data, means[k], stds[k])
        
        # Normalize responsibilities
        total_resp = np.sum(responsibilities, axis=1, keepdims=True)
        total_resp[total_resp == 0] = 1e-8  # Avoid division by zero
        responsibilities = responsibilities / total_resp
        
        # M-step: update parameters
        old_means = means.copy()
        
        for k in range(n_components):
            resp_k = responsibilities[:, k]
            weights[k] = np.mean(resp_k)
            
            if weights[k] > 1e-8:
                means[k] = np.sum(resp_k * data) / np.sum(resp_k)
                variance = np.sum(resp_k * (data - means[k])**2) / np.sum(resp_k)
                stds[k] = np.sqrt(max(variance, 1e-8))
        
        # Check convergence
        if np.allclose(old_means, means, atol=tol):
            break
    
    # Return components sorted by weight
    components = []
    for k in range(n_components):
        if weights[k] > 1e-8 and stds[k] > 1e-8:
            components.append({
                'weight': float(weights[k]),
                'mean': float(means[k]),
                'std': float(stds[k])
            })
    
    # Sort by weight descending
    components.sort(key=lambda x: x['weight'], reverse=True)
    
    # Renormalize weights
    total_weight = sum(c['weight'] for c in components)
    if total_weight > 0:
        for c in components:
            c['weight'] /= total_weight
    
    return components if components else None

def calculate_mixture_loglikelihood(data, components):
    """Calculate log-likelihood of data under mixture model."""
    ll = 0.0
    for x in data:
        prob = sum(c['weight'] * stats.norm.pdf(x, c['mean'], c['std']) for c in components)
        ll += np.log(max(prob, 1e-8))  # Avoid log(0)
    return ll

def analyze_dependency_distributions(text_file_path: str, min_samples: int = 10):
    """
    Analyze dependency relationships and model them as distributions.
    
    Args:
        text_file_path: Path to the input text file
        min_samples: Minimum samples needed to fit a distribution
    
    Returns:
        dict: Distribution models for each POS-dependency combination
    """
    # Load spaCy model
    nlp = ensure_dependencies()
    if nlp is None:
        return {}
    
    print(f"Analyzing dependency distance distributions in '{text_file_path}'")
    
    # Read the text file
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found.")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}
    
    print("Processing text with spaCy dependency parser...")
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Collect dependency relationships by POS and dependency type
    pos_dep_distances = defaultdict(lambda: defaultdict(list))
    total_relationships = 0
    
    for token in doc:
        # Skip ROOT dependencies
        if token.dep_ == "ROOT":
            continue
            
        # Calculate signed distance: dependent - head
        signed_distance = token.i - token.head.i
        
        # Store by POS and dependency type
        pos_dep_distances[token.pos_][token.dep_].append(signed_distance)
        total_relationships += 1
    
    print(f"Found {total_relationships} dependency relationships")
    print("Fitting distributions...")
    
    # Fit distributions for each POS-dependency combination
    distribution_models = {}
    
    for pos, dep_dict in pos_dep_distances.items():
        distribution_models[pos] = {}
        
        for dep_type, distances in dep_dict.items():
            print(f"  Fitting {pos}-{dep_type} ({len(distances)} samples)...")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress fitting warnings
                model = fit_gaussian_mixture(distances, min_samples=min_samples)
            
            distribution_models[pos][dep_type] = model
    
    # Calculate summary statistics
    total_pos_types = len(distribution_models)
    total_dep_combinations = sum(len(deps) for deps in distribution_models.values())
    
    fitted_distributions = 0
    gaussian_count = 0
    mixture_count = 0
    insufficient_data = 0
    
    for pos_models in distribution_models.values():
        for model in pos_models.values():
            if model['type'] == 'gaussian':
                fitted_distributions += 1
                gaussian_count += 1
            elif model['type'] == 'gaussian_mixture':
                fitted_distributions += 1
                mixture_count += 1
            elif model['type'] == 'insufficient_data':
                insufficient_data += 1
    
    summary = {
        'text_file': text_file_path,
        'total_tokens': len(doc),
        'total_relationships': total_relationships,
        'pos_types': total_pos_types,
        'dependency_combinations': total_dep_combinations,
        'fitted_distributions': fitted_distributions,
        'distribution_types': {
            'single_gaussian': gaussian_count,
            'gaussian_mixture': mixture_count,
            'insufficient_data': insufficient_data,
            'point_mass': sum(1 for pos_models in distribution_models.values() 
                            for model in pos_models.values() 
                            if model['type'] == 'point_mass')
        },
        'min_samples_threshold': min_samples
    }
    
    results = {
        'summary': summary,
        'distributions': distribution_models
    }
    
    return results

def print_distribution_results(results: dict):
    """Print distribution analysis results."""
    print("\n" + "="*80)
    print("DEPENDENCY DISTANCE DISTRIBUTION MODELING RESULTS")
    print("="*80)
    
    summary = results['summary']
    print(f"Text file: {summary['text_file']}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total relationships: {summary['total_relationships']:,}")
    print(f"POS types: {summary['pos_types']}")
    print(f"Dependency combinations: {summary['dependency_combinations']}")
    
    print(f"\nDISTRIBUTION FITTING SUMMARY:")
    dist_types = summary['distribution_types']
    print(f"  Successfully fitted: {summary['fitted_distributions']}")
    print(f"  Single Gaussians: {dist_types['single_gaussian']}")
    print(f"  Gaussian Mixtures: {dist_types['gaussian_mixture']}")
    print(f"  Point masses: {dist_types['point_mass']}")
    print(f"  Insufficient data: {dist_types['insufficient_data']}")
    
    print(f"\nTOP DISTRIBUTION MODELS BY POS:")
    distributions = results['distributions']
    
    # Sort POS types by number of relationships
    pos_counts = []
    for pos, dep_models in distributions.items():
        total_samples = sum(model.get('sample_count', 0) for model in dep_models.values())
        pos_counts.append((pos, total_samples, len(dep_models)))
    
    pos_counts.sort(key=lambda x: x[1], reverse=True)
    
    for pos, total_samples, dep_count in pos_counts[:10]:  # Top 10 POS types
        print(f"\n{pos} ({total_samples} samples, {dep_count} dependency types):")
        
        dep_models = distributions[pos]
        # Sort dependencies by sample count
        sorted_deps = sorted(dep_models.items(), 
                           key=lambda x: x[1].get('sample_count', 0), reverse=True)
        
        for dep_type, model in sorted_deps[:5]:  # Top 5 dependencies per POS
            sample_count = model.get('sample_count', 0)
            model_type = model['type']
            
            if model_type == 'gaussian':
                comp = model['components'][0]
                print(f"  {dep_type:12} ({sample_count:4d}): Gaussian(μ={comp['mean']:+.2f}, σ={comp['std']:.2f})")
            
            elif model_type == 'gaussian_mixture':
                print(f"  {dep_type:12} ({sample_count:4d}): Mixture of {len(model['components'])} Gaussians")
                for i, comp in enumerate(model['components']):
                    print(f"    Component {i+1}: w={comp['weight']:.3f}, μ={comp['mean']:+.2f}, σ={comp['std']:.2f}")
            
            elif model_type == 'point_mass':
                print(f"  {dep_type:12} ({sample_count:4d}): Point mass at {model['location']:+.1f}")
            
            elif model_type == 'insufficient_data':
                mean = model.get('raw_mean', 0)
                print(f"  {dep_type:12} ({sample_count:4d}): Insufficient data (raw μ={mean:+.2f})")

def save_compact_distributions(results: dict, output_path: str):
    """Save a compact version with just the distribution parameters."""
    compact_data = {}
    
    for pos, dep_models in results['distributions'].items():
        compact_data[pos] = {}
        
        for dep_type, model in dep_models.items():
            if model['type'] in ['gaussian', 'gaussian_mixture', 'point_mass']:
                # Only save successfully fitted models
                compact_data[pos][dep_type] = {
                    'type': model['type'],
                    'sample_count': model.get('sample_count', 0)
                }
                
                if model['type'] == 'gaussian':
                    comp = model['components'][0]
                    compact_data[pos][dep_type].update({
                        'mean': comp['mean'],
                        'std': comp['std']
                    })
                
                elif model['type'] == 'gaussian_mixture':
                    compact_data[pos][dep_type]['components'] = model['components']
                
                elif model['type'] == 'point_mass':
                    compact_data[pos][dep_type]['location'] = model['location']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(compact_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description="Model dependency distances as Gaussian distributions"
    )
    
    parser.add_argument('input_file', help="Path to the input text file")
    parser.add_argument('--output', '-o', help="Save full results to JSON file")
    parser.add_argument('--compact', '-c', help="Save compact distribution models to JSON file")
    parser.add_argument('--min-samples', type=int, default=10, 
                       help="Minimum samples needed to fit distributions (default: 10)")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Only show summary, not detailed results")
    
    args = parser.parse_args()
    
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