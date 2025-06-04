#!/usr/bin/env python3
"""
Performance Test Script for GPU-Accelerated CCG Parsing

Tests parsing performance on a variety of sentences to measure
the improvement from GPU acceleration.
"""

import time
from src.tree.depccg_treeviz import CCGTreeVisualizer

def test_parsing_performance():
    """Test parsing performance on various sentence types."""
    
    test_sentences = [
        "The cat sits on the mat.",
        "John loves Mary deeply.",
        "The quick brown fox jumps over the lazy dog.",
        "She gave him a beautiful handwritten letter yesterday.",
        "Complex grammatical structures can challenge even sophisticated parsers.",
        "When the professor arrived at the university, students were already waiting.",
        "The company's quarterly earnings report exceeded all expectations this year.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest last month.",
        "Although the weather was terrible, the concert continued as planned.",
        "The artificial intelligence system learned to recognize patterns in human speech."
    ]
    
    print("🚀 CCG Parsing Performance Test")
    print("=" * 60)
    
    viz = CCGTreeVisualizer()
    total_time = 0
    successful_parses = 0
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 Test {i}: {sentence}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            viz.save_tree_image(sentence, f'/workspace/perf_test_{i}.png')
            successful_parses += 1
            
            end_time = time.time()
            parse_time = end_time - start_time
            total_time += parse_time
            
            print(f"✅ Parse completed in {parse_time:.3f} seconds")
            
        except Exception as e:
            print(f"❌ Parse failed: {e}")
            end_time = time.time()
            parse_time = end_time - start_time
            total_time += parse_time
            print(f"⏱️  Failed after {parse_time:.3f} seconds")
    
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total sentences tested: {len(test_sentences)}")
    print(f"Successful parses: {successful_parses}")
    print(f"Success rate: {successful_parses/len(test_sentences)*100:.1f}%")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Average time per sentence: {total_time/len(test_sentences):.3f} seconds")
    
    if successful_parses > 0:
        print(f"Average time per successful parse: {total_time/successful_parses:.3f} seconds")
    
    # Performance targets
    print("\n🎯 PERFORMANCE TARGETS:")
    avg_time = total_time/len(test_sentences)
    if avg_time < 3.0:
        print("🚀 EXCELLENT: GPU acceleration is working optimally!")
    elif avg_time < 6.0:
        print("✅ GOOD: Significant improvement from GPU acceleration")
    elif avg_time < 10.0:
        print("⚠️  MODERATE: Some improvement, but GPU may not be fully utilized")
    else:
        print("❌ SLOW: GPU acceleration may not be working properly")
    
    return {
        'total_time': total_time,
        'successful_parses': successful_parses,
        'average_time': avg_time
    }

if __name__ == "__main__":
    test_parsing_performance() 