#!/usr/bin/env python3
"""
tree_structure_comparison.py
---------------------------
Converts DepCCG and Benepar outputs to standardized tree structures
for direct structural comparison, ignoring linguistic category differences.

The goal is to test whether parsers produce equivalent hierarchical structures
regardless of their different category systems (CCG vs constituency).

Features multiprocessing support for faster processing of large sentence sets.

Usage Examples:
    # Basic usage with auto-detected cores
    python tree_structure_comparison.py --sentences "The cat sleeps." "She reads books."
    
    # Use specific number of cores
    python tree_structure_comparison.py --cores 4 --sentences "Hello world." "How are you?"
    
    # Process quietly and save results
    python tree_structure_comparison.py --quiet --save-results results.json --sentences "Test sentence."
"""

import argparse
from typing import Dict, List, Any, Tuple, Optional
import json
import re
import multiprocessing as mp
import time
from functools import partial

from depccg_treeviz import CCGTreeVisualizer as DepCCG
from spacy_treeviz import BeneparTreeVisualizer as Benepar


class StandardTree:
    """Standardized tree representation for parser comparison."""
    
    def __init__(self, text: str, children: Optional[List['StandardTree']] = None, 
                 original_category: str = "", parser_source: str = ""):
        self.text = text  # The text span this node covers
        self.children = children or []  # Child nodes
        self.original_category = original_category  # Original parser category (for reference)
        self.parser_source = parser_source  # Which parser this came from
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (terminal)."""
        return len(self.children) == 0
    
    def get_leaves(self) -> List[str]:
        """Get all leaf text in left-to-right order."""
        if self.is_leaf():
            return [self.text]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves
    
    def get_depth(self) -> int:
        """Get the maximum depth of this tree."""
        if self.is_leaf():
            return 1
        return 1 + max(child.get_depth() for child in self.children)
    
    def get_node_count(self) -> int:
        """Count total nodes in tree."""
        return 1 + sum(child.get_node_count() for child in self.children)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "text": self.text,
            "is_leaf": self.is_leaf(),
            "original_category": self.original_category,
            "parser_source": self.parser_source
        }
        if self.children:
            result["children"] = [child.to_dict() for child in self.children]
        return result
    
    def get_structure_signature(self) -> str:
        """Get a signature representing the tree structure (ignoring categories)."""
        if self.is_leaf():
            return f"LEAF({self.text})"
        child_sigs = [child.get_structure_signature() for child in self.children]
        return f"NODE({len(self.children)}:[{','.join(child_sigs)}])"
    
    def print_tree(self, indent: str = "", is_last: bool = True) -> None:
        """Print tree structure in readable format."""
        branch = "└── " if is_last else "├── "
        
        if self.is_leaf():
            print(f"{indent}{branch}{self.text} [{self.original_category}]")
        else:
            print(f"{indent}{branch}{self.text} ({self.original_category})")
            
        for i, child in enumerate(self.children):
            child_is_last = (i == len(self.children) - 1)
            next_indent = indent + ("    " if is_last else "│   ")
            child.print_tree(next_indent, child_is_last)


def depccg_to_standard_tree(parse_data: Dict[str, Any]) -> StandardTree:
    """Convert DepCCG parse data to StandardTree format."""
    def extract_leaves(node):
        """Extract leaf words from DepCCG tree."""
        if isinstance(node, dict):
            if 'word' in node:
                return [node['word']]
            elif 'children' in node:
                leaves = []
                for child in node['children']:
                    leaves.extend(extract_leaves(child))
                return leaves
        return []
    
    def convert_node(node) -> StandardTree:
        if isinstance(node, dict):
            if 'word' in node:
                # Leaf node
                return StandardTree(
                    text=node['word'],
                    children=[],
                    original_category=node.get('cat', ''),
                    parser_source="depccg"
                )
            else:
                # Internal node
                span_text = " ".join(extract_leaves(node))
                children = [convert_node(child) for child in node.get('children', [])]
                return StandardTree(
                    text=span_text,
                    children=children,
                    original_category=node.get('cat', ''),
                    parser_source="depccg"
                )
        else:
            # Fallback for string nodes
            return StandardTree(
                text=str(node),
                children=[],
                original_category="",
                parser_source="depccg"
            )
    
    return convert_node(parse_data)


def benepar_to_standard_tree(parse_data: Dict[str, Any]) -> StandardTree:
    """Convert Benepar parse data to StandardTree format."""
    def extract_leaves(node):
        """Extract leaf words from Benepar tree."""
        if isinstance(node, dict):
            if 'word' in node:
                return [node['word']]
            elif 'children' in node:
                leaves = []
                for child in node['children']:
                    leaves.extend(extract_leaves(child))
                return leaves
        return []
    
    def convert_node(node) -> StandardTree:
        if isinstance(node, dict):
            if 'word' in node:
                # Leaf node
                return StandardTree(
                    text=node['word'],
                    children=[],
                    original_category=node.get('cat', ''),
                    parser_source="benepar"
                )
            else:
                # Internal node
                span_text = " ".join(extract_leaves(node))
                children = [convert_node(child) for child in node.get('children', [])]
                return StandardTree(
                    text=span_text,
                    children=children,
                    original_category=node.get('cat', ''),
                    parser_source="benepar"
                )
        else:
            # Fallback for string nodes
            return StandardTree(
                text=str(node),
                children=[],
                original_category="",
                parser_source="benepar"
            )
    
    return convert_node(parse_data)


def compare_tree_structures(tree1: StandardTree, tree2: StandardTree) -> Dict[str, Any]:
    """Compare two StandardTree structures and report differences."""
    comparison = {
        "identical_structure": False,
        "identical_leaves": False,
        "identical_depth": False,
        "identical_branching": False,
        "differences": [],
        "tree1_info": {
            "parser": tree1.parser_source,
            "leaves": tree1.get_leaves(),
            "depth": tree1.get_depth(),
            "node_count": tree1.get_node_count(),
            "structure_signature": tree1.get_structure_signature()
        },
        "tree2_info": {
            "parser": tree2.parser_source,
            "leaves": tree2.get_leaves(),
            "depth": tree2.get_depth(),
            "node_count": tree2.get_node_count(),
            "structure_signature": tree2.get_structure_signature()
        }
    }
    
    # Compare leaves (tokenization)
    leaves1 = tree1.get_leaves()
    leaves2 = tree2.get_leaves()
    comparison["identical_leaves"] = leaves1 == leaves2
    if not comparison["identical_leaves"]:
        comparison["differences"].append(f"Different leaves: {tree1.parser_source}={leaves1}, {tree2.parser_source}={leaves2}")
    
    # Compare depth
    depth1 = tree1.get_depth()
    depth2 = tree2.get_depth()
    comparison["identical_depth"] = depth1 == depth2
    if not comparison["identical_depth"]:
        comparison["differences"].append(f"Different depth: {tree1.parser_source}={depth1}, {tree2.parser_source}={depth2}")
    
    # Compare structure signatures (most important test)
    sig1 = tree1.get_structure_signature()
    sig2 = tree2.get_structure_signature()
    identical_structure_sig = sig1 == sig2
    if not identical_structure_sig:
        comparison["differences"].append(f"Different structure signatures")
        # Don't print full signatures as they're very long
    
    # Compare branching factors at each level
    def get_branching_pattern(node, pattern=None):
        if pattern is None:
            pattern = []
        if not node.is_leaf():
            pattern.append(len(node.children))
            for child in node.children:
                get_branching_pattern(child, pattern)
        return pattern
    
    branch1 = get_branching_pattern(tree1)
    branch2 = get_branching_pattern(tree2)
    comparison["identical_branching"] = branch1 == branch2
    if not comparison["identical_branching"]:
        comparison["differences"].append(f"Different branching patterns: {tree1.parser_source}={branch1}, {tree2.parser_source}={branch2}")
    
    # Overall assessment
    comparison["identical_structure"] = (
        comparison["identical_leaves"] and 
        identical_structure_sig  # Use the local variable, not the key
    )
    
    return comparison


def process_single_sentence(sentence_data: Tuple[int, str, bool]) -> Optional[Dict[str, Any]]:
    """Process a single sentence in a worker process.
    
    Args:
        sentence_data: Tuple of (index, sentence, verbose)
    
    Returns:
        Dict with comparison results or None if parsing failed
    """
    index, sentence, verbose = sentence_data
    
    # Create parser instances in each worker process
    dep_parser = DepCCG()
    ben_parser = Benepar()
    assessor = ParsingAccuracyAssessor()
    
    if verbose:
        print(f"📝 Worker processing {index}: {sentence}")
    
    # Parse with both systems
    dep_result = dep_parser.parse_only(sentence)
    ben_result = ben_parser.parse_only(sentence)
    
    if not (dep_result.get('success') and ben_result.get('success')):
        if verbose:
            print(f"❌ Worker {index}: One or both parsers failed - skipping comparison")
        return None
    
    # Convert to standard trees
    dep_tree = depccg_to_standard_tree(dep_result['parse_data'])
    ben_tree = benepar_to_standard_tree(ben_result['parse_data'])
    
    # Compare structures
    comparison = compare_tree_structures(dep_tree, ben_tree)
    
    # Assess parsing quality
    quality_comparison = assessor.compare_parsing_quality(dep_tree, ben_tree)
    comparison['quality_assessment'] = quality_comparison
    comparison['sentence'] = sentence
    comparison['index'] = index
    
    if verbose:
        print(f"✅ Worker {index}: Completed parsing and comparison")
    
    return comparison


def run_structure_comparison_test(sentences: List[str], verbose: bool = True, num_cores: int = None) -> Dict[str, Any]:
    """Run structure comparison test on multiple sentences with multiprocessing.
    
    Args:
        sentences: List of sentences to process
        verbose: Whether to show detailed progress
        num_cores: Number of CPU cores to use (None = auto-detect)
    """
    if num_cores is None:
        num_cores = mp.cpu_count()
    
    print(f"🔬 Starting tree structure comparison on {len(sentences)} sentences using {num_cores} cores...")
    start_time = time.time()
    
    results = {
        "total_sentences": len(sentences),
        "cores_used": num_cores,
        "processing_time": 0.0,
        "sentences_per_second": 0.0,
        "identical_structures": 0,
        "identical_leaves": 0,
        "identical_depths": 0,
        "identical_branching": 0,
        "depccg_better": 0,
        "benepar_better": 0,
        "quality_ties": 0,
        "detailed_results": []
    }
    
    # Prepare sentence data for parallel processing
    sentence_data = [(i+1, sentence, verbose) for i, sentence in enumerate(sentences)]
    
    # Process sentences in parallel
    with mp.Pool(processes=num_cores) as pool:
        parallel_results = pool.map(process_single_sentence, sentence_data)
    
    # Filter out None results (failed parses) and sort by index
    valid_results = [r for r in parallel_results if r is not None]
    valid_results.sort(key=lambda x: x['index'])
    
    processing_time = time.time() - start_time
    results["processing_time"] = processing_time
    results["sentences_per_second"] = len(valid_results) / processing_time if processing_time > 0 else 0
    
    print(f"⚡ Parallel processing completed in {processing_time:.2f} seconds")
    print(f"📊 Successfully processed {len(valid_results)}/{len(sentences)} sentences")
    print(f"🚀 Processing speed: {results['sentences_per_second']:.1f} sentences/second")
    
    # Process results and update statistics
    for comparison in valid_results:
        results['detailed_results'].append(comparison)
        
        # Update statistics
        if comparison['identical_structure']:
            results['identical_structures'] += 1
        if comparison['identical_leaves']:
            results['identical_leaves'] += 1
        if comparison['identical_depth']:
            results['identical_depths'] += 1
        if comparison['identical_branching']:
            results['identical_branching'] += 1
        
        # Update quality statistics
        quality_comparison = comparison['quality_assessment']
        if quality_comparison['better_parser'] == 'depccg':
            results['depccg_better'] += 1
        elif quality_comparison['better_parser'] == 'benepar':
            results['benepar_better'] += 1
        else:
            results['quality_ties'] += 1
        
        # Report individual results if verbose
        if verbose:
            sentence = comparison['sentence']
            index = comparison['index']
            print(f"\n📝 Results for {index}: {sentence}")
            print(f"🔍 Structure Analysis:")
            if comparison['identical_structure']:
                print("   ✅ Identical tree structures!")
            else:
                print("   ⚠️  Different tree structures:")
                for diff in comparison['differences']:
                    print(f"      - {diff}")
            
            print(f"🎯 Parsing Quality Analysis:")
            dep_assessment = quality_comparison['tree1_assessment']
            ben_assessment = quality_comparison['tree2_assessment']
            
            print(f"   DepCCG Score: {quality_comparison['tree1_score']:.1f}/10")
            if dep_assessment['total_errors'] > 0:
                print(f"     🚨 Errors ({dep_assessment['total_errors']}):")
                for error in dep_assessment['pos_details'][:2]:  # Show first 2
                    print(f"       - {error}")
                for error in dep_assessment['subcat_details'][:2]:
                    print(f"       - {error}")
                for error in dep_assessment['structural_details'][:2]:
                    print(f"       - {error}")
            
            print(f"   Benepar Score: {quality_comparison['tree2_score']:.1f}/10")
            if ben_assessment['total_errors'] > 0:
                print(f"     🚨 Errors ({ben_assessment['total_errors']}):")
                for error in ben_assessment['pos_details'][:2]:
                    print(f"       - {error}")
                for error in ben_assessment['subcat_details'][:2]:
                    print(f"       - {error}")
                for error in ben_assessment['structural_details'][:2]:
                    print(f"       - {error}")
            
            if quality_comparison['better_parser']:
                print(f"   🏆 Better Parser: {quality_comparison['better_parser'].upper()} (confidence: {quality_comparison['confidence']})")
            else:
                print(f"   🤝 Tie - similar quality")
    
    return results


def print_structure_summary(results: Dict[str, Any]):
    """Print summary of structure comparison results."""
    total = results['total_sentences']
    
    print("\n" + "="*60)
    print("📊 TREE STRUCTURE COMPARISON SUMMARY")
    print("="*60)
    
    # Performance metrics
    if 'processing_time' in results:
        print(f"⚡ Performance Metrics:")
        print(f"   CPU cores used: {results.get('cores_used', 'N/A')}")
        print(f"   Processing time: {results['processing_time']:.2f} seconds")
        print(f"   Processing speed: {results['sentences_per_second']:.1f} sentences/second")
        print()
    
    print(f"📈 Structure Agreement Statistics:")
    print(f"   Total sentences tested: {total}")
    print(f"   Identical tree structures: {results['identical_structures']} ({results['identical_structures']/total*100:.1f}%)")
    print(f"   Identical leaf sequences: {results['identical_leaves']} ({results['identical_leaves']/total*100:.1f}%)")
    print(f"   Identical tree depths: {results['identical_depths']} ({results['identical_depths']/total*100:.1f}%)")
    print(f"   Identical branching patterns: {results['identical_branching']} ({results['identical_branching']/total*100:.1f}%)")
    
    print(f"\n🎯 Parsing Quality Assessment:")
    print(f"   DepCCG better: {results['depccg_better']} ({results['depccg_better']/total*100:.1f}%)")
    print(f"   Benepar better: {results['benepar_better']} ({results['benepar_better']/total*100:.1f}%)")
    print(f"   Quality ties: {results['quality_ties']} ({results['quality_ties']/total*100:.1f}%)")
    
    # Determine overall winner
    if results['benepar_better'] > results['depccg_better']:
        winner = "Benepar"
        margin = results['benepar_better'] - results['depccg_better']
    elif results['depccg_better'] > results['benepar_better']:
        winner = "DepCCG"  
        margin = results['depccg_better'] - results['benepar_better']
    else:
        winner = None
        margin = 0
    
    if winner and margin > 0:
        print(f"   🏆 Overall Winner: {winner} (+{margin} sentences)")
    else:
        print(f"   🤝 Overall: Tie")
    
    # Show common issues if available
    if results['detailed_results']:
        depccg_issues = []
        benepar_issues = []
        
        for result in results['detailed_results']:
            if 'quality_assessment' in result:
                qa = result['quality_assessment']
                depccg_issues.extend(qa['tree1_assessment'].get('pos_details', []))
                depccg_issues.extend(qa['tree1_assessment'].get('subcat_details', []))
                depccg_issues.extend(qa['tree1_assessment'].get('structural_details', []))
                
                benepar_issues.extend(qa['tree2_assessment'].get('pos_details', []))
                benepar_issues.extend(qa['tree2_assessment'].get('subcat_details', []))
                benepar_issues.extend(qa['tree2_assessment'].get('structural_details', []))
        
        if depccg_issues or benepar_issues:
            print(f"\n🔍 Common Parsing Issues Detected:")
            
            if depccg_issues:
                print(f"   DepCCG Issues ({len(depccg_issues)} total):")
                # Show most common issues (simple frequency count)
                issue_freq = {}
                for issue in depccg_issues:
                    # Extract key pattern from issue
                    if "incorrectly tagged as" in issue:
                        pattern = "POS tag errors"
                    elif "incorrectly treated as" in issue:
                        pattern = "Subcategorization errors"
                    elif "analyzed as" in issue and "instead of" in issue:
                        pattern = "Structural analysis errors"
                    else:
                        pattern = "Other errors"
                    issue_freq[pattern] = issue_freq.get(pattern, 0) + 1
                
                for pattern, count in sorted(issue_freq.items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {pattern}: {count} occurrences")
            
            if benepar_issues:
                print(f"   Benepar Issues ({len(benepar_issues)} total):")
                issue_freq = {}
                for issue in benepar_issues:
                    if "incorrectly tagged as" in issue:
                        pattern = "POS tag errors"
                    elif "incorrectly treated as" in issue:
                        pattern = "Subcategorization errors"
                    elif "analyzed as" in issue and "instead of" in issue:
                        pattern = "Structural analysis errors"
                    else:
                        pattern = "Other errors"
                    issue_freq[pattern] = issue_freq.get(pattern, 0) + 1
                
                for pattern, count in sorted(issue_freq.items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {pattern}: {count} occurrences")
    
    if results['identical_structures'] == total:
        print(f"\n🎉 PERFECT STRUCTURAL AGREEMENT!")
        print(f"   Both parsers produce identical tree structures for all sentences.")
    elif results['identical_leaves'] == total:
        print(f"\n✅ TOKENIZATION AGREEMENT ACHIEVED")
        print(f"   All sentences have identical tokenization, but different tree structures.")
        print(f"   This reflects genuine differences in parsing approaches (CCG vs constituency).")
    else:
        print(f"\n🔍 STRUCTURAL DIFFERENCES DETECTED")
        print(f"   Parsers disagree on tree structure and/or tokenization.")


class ParsingAccuracyAssessor:
    """Assess parsing accuracy and detect common parsing errors."""
    
    def __init__(self):
        # Known word categories for basic validation
        self.common_determiners = {"the", "a", "an", "this", "that", "these", "those"}
        self.common_adjectives = {"old", "new", "big", "small", "good", "bad", "quick", "slow", "happy", "sad"}
        self.common_adverbs = {"quickly", "slowly", "carefully", "loudly", "quietly", "well", "badly"}
        self.common_prepositions = {"in", "on", "at", "to", "from", "with", "by", "for", "of"}
        self.intransitive_verbs = {"sleep", "walk", "run", "sit", "stand", "arrive", "go", "come"}
        self.transitive_verbs = {"read", "write", "see", "hear", "make", "take", "give", "put"}
    
    def assess_pos_accuracy(self, tree: 'StandardTree') -> Dict[str, Any]:
        """Assess POS tag accuracy for leaf nodes."""
        errors = []
        warnings = []
        
        def check_node(node):
            if node.is_leaf():
                word = node.text.lower().rstrip(".,!?;:")  # Remove punctuation
                category = node.original_category.upper() if node.original_category else ""
                
                # Skip empty categories - they're not errors, just missing data
                if not category:
                    return
                
                # Check common POS mismatches
                if word in self.common_determiners and category not in ["DT", "NP[NB]/N"]:
                    errors.append(f"'{node.text}' tagged as {category}, expected determiner")
                
                if word in self.common_adjectives and category not in ["JJ", "N/N"]:
                    errors.append(f"'{node.text}' tagged as {category}, expected adjective")
                
                if word in self.common_adverbs and category not in ["RB", "ADVP", "ADV", "(S\\NP)/(S\\NP)"]:
                    if category in ["N", "NP"]:  # This is a serious error
                        errors.append(f"'{node.text}' incorrectly tagged as {category}, should be adverb")
                    else:
                        warnings.append(f"'{node.text}' tagged as {category}, possibly should be adverb")
                
                if word in self.common_prepositions and category not in ["IN", "PP", "((S\\NP)\\(S\\NP))/NP"]:
                    errors.append(f"'{node.text}' tagged as {category}, expected preposition")
            else:
                # Recursively check children
                for child in node.children:
                    check_node(child)
        
        check_node(tree)
        
        return {
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
    
    def assess_subcategorization(self, tree: 'StandardTree') -> Dict[str, Any]:
        """Assess verb subcategorization accuracy."""
        errors = []
        warnings = []
        
        def find_verb_phrases(node, path=""):
            """Find verb phrases and analyze their structure."""
            if node.is_leaf():
                return
            
            # Look for verb-related patterns
            category = node.original_category.upper()
            
            if category in ["VP", "S[DCL]\\NP", "S\\NP"]:
                # This is a verb phrase, analyze its structure
                verb_analysis = self._analyze_verb_phrase(node)
                errors.extend(verb_analysis['errors'])
                warnings.extend(verb_analysis['warnings'])
            
            for child in node.children:
                find_verb_phrases(child, path + f"/{child.original_category}")
        
        find_verb_phrases(tree)
        
        return {
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
    
    def _analyze_verb_phrase(self, vp_node: 'StandardTree') -> Dict[str, Any]:
        """Analyze a verb phrase for subcategorization errors."""
        errors = []
        warnings = []
        
        # Get the main verb and its complements
        verb = None
        complements = []
        
        for child in vp_node.children:
            if child.is_leaf():
                word = child.text.lower().rstrip(".,!?;:")
                category = child.original_category.upper()
                
                # Check if this looks like a verb
                if any(verb_marker in category for verb_marker in ["VB", "V", "(S", "S[DCL]"]):
                    verb = word
                elif category in ["NP", "N"] and word in self.common_adverbs:
                    # Adverb incorrectly categorized as NP/N
                    errors.append(f"Adverb '{child.text}' incorrectly treated as object (category: {category})")
                else:
                    complements.append((word, category))
            else:
                # Look at complement phrases
                comp_category = child.original_category.upper()
                comp_text = child.text.lower()
                
                if comp_category in ["NP", "N"] and any(adv in comp_text for adv in self.common_adverbs):
                    # Check if this NP actually contains an adverb
                    adverb_leaves = [leaf.text.lower().rstrip(".,!?;:") for leaf in self._get_leaves(child) 
                                   if leaf.text.lower().rstrip(".,!?;:") in self.common_adverbs]
                    if adverb_leaves:
                        errors.append(f"Adverb(s) {adverb_leaves} incorrectly treated as noun phrase")
        
        # Check verb subcategorization
        if verb:
            if verb in self.intransitive_verbs:
                object_complements = [comp for comp in complements if comp[1] in ["NP", "N"]]
                object_phrases = [child for child in vp_node.children 
                                if not child.is_leaf() and child.original_category.upper() in ["NP"]]
                
                if object_complements or object_phrases:
                    warnings.append(f"Intransitive verb '{verb}' appears to have object: {object_complements + [p.text for p in object_phrases]}")
        
        return {
            'errors': errors,
            'warnings': warnings
        }
    
    def _get_leaves(self, node: 'StandardTree') -> List['StandardTree']:
        """Get all leaf nodes from a tree."""
        if node.is_leaf():
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves
    
    def assess_structural_plausibility(self, tree: 'StandardTree') -> Dict[str, Any]:
        """Assess overall structural plausibility."""
        issues = []
        
        # Check for unusual root categories
        root_category = tree.original_category.upper()
        sentence_text = tree.text.lower()
        
        # If sentence ends with period and has subject-verb structure, should be S not NP
        if (sentence_text.endswith('.') and 
            any(word in sentence_text for word in ['sleeps', 'walks', 'runs', 'reads']) and
            root_category in ['NP', 'N']):
            issues.append(f"Complete sentence incorrectly analyzed as {root_category} instead of sentence")
        
        # Check for overly deep trees with little content
        depth = tree.get_depth()
        word_count = len(tree.get_leaves())
        if depth > word_count + 1:  # Suspiciously deep
            issues.append(f"Tree depth ({depth}) seems excessive for {word_count} words")
        
        return {
            'issues': issues,
            'issue_count': len(issues)
        }
    
    def compare_parsing_quality(self, tree1: 'StandardTree', tree2: 'StandardTree') -> Dict[str, Any]:
        """Compare parsing quality between two trees."""
        assessment1 = self.assess_tree_quality(tree1)
        assessment2 = self.assess_tree_quality(tree2)
        
        score1 = self._calculate_quality_score(assessment1)
        score2 = self._calculate_quality_score(assessment2)
        
        better_parser = None
        confidence = "low"
        
        if abs(score1 - score2) > 2:  # Significant difference
            better_parser = tree1.parser_source if score1 > score2 else tree2.parser_source
            confidence = "high" if abs(score1 - score2) > 5 else "medium"
        
        return {
            'tree1_assessment': assessment1,
            'tree2_assessment': assessment2,
            'tree1_score': score1,
            'tree2_score': score2,
            'better_parser': better_parser,
            'confidence': confidence,
            'quality_difference': abs(score1 - score2)
        }
    
    def assess_tree_quality(self, tree: 'StandardTree') -> Dict[str, Any]:
        """Comprehensive quality assessment of a parse tree."""
        pos_assessment = self.assess_pos_accuracy(tree)
        subcat_assessment = self.assess_subcategorization(tree)
        structural_assessment = self.assess_structural_plausibility(tree)
        
        return {
            'parser': tree.parser_source,
            'pos_errors': pos_assessment['error_count'],
            'pos_warnings': pos_assessment['warning_count'],
            'pos_details': pos_assessment['errors'] + pos_assessment['warnings'],
            'subcat_errors': subcat_assessment['error_count'],
            'subcat_warnings': subcat_assessment['warning_count'], 
            'subcat_details': subcat_assessment['errors'] + subcat_assessment['warnings'],
            'structural_issues': structural_assessment['issue_count'],
            'structural_details': structural_assessment['issues'],
            'total_errors': pos_assessment['error_count'] + subcat_assessment['error_count'] + structural_assessment['issue_count'],
            'total_warnings': pos_assessment['warning_count'] + subcat_assessment['warning_count']
        }
    
    def _calculate_quality_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate a quality score (higher = better)."""
        # Start with base score
        score = 10.0
        
        # Deduct for errors (more severe)
        score -= assessment['total_errors'] * 2.0
        
        # Deduct for warnings (less severe)
        score -= assessment['total_warnings'] * 0.5
        
        # Ensure score doesn't go below 0
        return max(0.0, score)


def main():
    parser = argparse.ArgumentParser(description="Compare tree structures between DepCCG and Benepar")
    parser.add_argument("--sentences", nargs="+", default=["The cat sleeps.", "She reads books."], 
                      help="Sentences to test")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    parser.add_argument("--save-results", help="Save detailed results to JSON file")
    parser.add_argument("--cores", type=int, default=None, 
                      help="Number of CPU cores to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Run comparison with multiprocessing
    results = run_structure_comparison_test(
        args.sentences, 
        verbose=not args.quiet, 
        num_cores=args.cores
    )
    
    # Print summary
    print_structure_summary(results)
    
    # Save results if requested
    if args.save_results:
        # Convert StandardTree objects to dictionaries for JSON serialization
        json_results = results.copy()
        json_results['detailed_results'] = [
            {k: v for k, v in result.items() if k != 'tree_objects'}  # Remove non-serializable objects
            for result in results['detailed_results']
        ]
        with open(args.save_results, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\n💾 Results saved to: {args.save_results}")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    
    main() 