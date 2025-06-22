#!/usr/bin/env python3
"""
create_sample_corpus.py
-----------------------
Creates sample corpus files for testing the corpus processor.
"""

import argparse
import random


def create_sample_corpus(output_file: str, num_sentences: int = 1000):
    """Create a sample corpus file with varied sentences.
    
    Args:
        output_file: Path to output corpus file
        num_sentences: Number of sentences to generate
    """
    
    # Sample sentence templates and words
    subjects = ["The cat", "She", "He", "The dog", "Students", "Teachers", "Scientists", 
               "Children", "Researchers", "Artists", "The bird", "They", "People", "Experts"]
    
    verbs = ["sleeps", "reads", "writes", "runs", "walks", "studies", "analyzes", "creates",
            "discovers", "explains", "understands", "develops", "investigates", "observes"]
    
    objects = ["books", "papers", "data", "code", "art", "music", "problems", "solutions",
              "theories", "experiments", "projects", "ideas", "patterns", "results"]
    
    adjectives = ["interesting", "complex", "beautiful", "difficult", "important", "new",
                 "advanced", "simple", "effective", "innovative", "creative", "detailed"]
    
    prepositions = ["in", "on", "with", "about", "through", "during", "for", "from"]
    
    locations = ["the lab", "the library", "the university", "the garden", "the office",
                "the classroom", "the field", "the studio", "the workshop", "home"]
    
    sentence_patterns = [
        "{subject} {verb}.",
        "{subject} {verb} {object}.",
        "{subject} {verb} {adj} {object}.",
        "{subject} {verb} {object} {prep} {location}.",
        "{subject} {verb} {adj} {object} {prep} {location}.",
        "The {adj} {subject} {verb} {object}.",
        "{subject} carefully {verb} {object}.",
        "{subject} {verb} and analyzes {object}.",
        "When {subject} {verb}, they discover {object}.",
        "{subject} {verb} {object} while studying {adj} theories."
    ]
    
    print(f"🔧 Creating sample corpus with {num_sentences} sentences...")
    
    with open(output_file, 'w') as f:
        for i in range(num_sentences):
            pattern = random.choice(sentence_patterns)
            
            sentence = pattern.format(
                subject=random.choice(subjects),
                verb=random.choice(verbs),
                object=random.choice(objects),
                adj=random.choice(adjectives),
                prep=random.choice(prepositions),
                location=random.choice(locations)
            )
            
            f.write(sentence + '\n')
            
            if (i + 1) % 100 == 0:
                print(f"   Generated {i + 1} sentences...")
    
    print(f"✅ Created corpus file: {output_file}")


def create_linguistic_test_corpus(output_file: str):
    """Create a corpus with linguistically interesting test cases."""
    
    test_sentences = [
        # Simple sentences
        "The cat sleeps.",
        "She reads books.",
        "Dogs bark loudly.",
        
        # Complex sentences
        "The quick brown fox jumps over the lazy dog.",
        "Scientists study complex linguistic phenomena in computational frameworks.",
        "When researchers analyze data, they discover interesting patterns.",
        
        # Different syntactic structures
        "Reading books helps students learn.",
        "To understand language requires deep analysis.",
        "The book that she read was fascinating.",
        "Students who study linguistics often become researchers.",
        
        # Coordination
        "She reads books and writes papers.",
        "The cat sleeps peacefully and the dog runs quickly.",
        "Both teachers and students enjoy learning new concepts.",
        
        # Embedded clauses
        "I think that computational linguistics is important.",
        "The fact that parsing is difficult surprises nobody.",
        "She believes the theory explains the phenomena.",
        
        # Questions and other sentence types
        "What does the parser analyze?",
        "How do linguists study syntax?",
        "Parse this sentence carefully!",
        
        # Edge cases
        "Colorless green ideas sleep furiously.",
        "The the the syntax error.",
        "Very very very long adjective phrase modification.",
        
        # Technical terms
        "Syntactic parsing involves constituency analysis.",
        "CCG categories encode subcategorization information.",
        "Benepar utilizes neural network architectures.",
        "DepCCG implements combinatory categorial grammar.",
        
        # Varying lengths
        "Short.",
        "Medium length sentence with some complexity.",
        "This is a significantly longer sentence that contains multiple clauses and should test the parser's ability to handle complex syntactic structures with embedded phrases and various grammatical constructions that appear in natural language text."
    ]
    
    print(f"🧪 Creating linguistic test corpus with {len(test_sentences)} sentences...")
    
    with open(output_file, 'w') as f:
        for sentence in test_sentences:
            f.write(sentence + '\n')
    
    print(f"✅ Created test corpus: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create sample corpus files")
    parser.add_argument("--output", default="sample_corpus.txt", 
                      help="Output corpus file (default: sample_corpus.txt)")
    parser.add_argument("--size", type=int, default=1000,
                      help="Number of sentences to generate (default: 1000)")
    parser.add_argument("--test-corpus", action="store_true",
                      help="Create linguistic test corpus instead of random corpus")
    
    args = parser.parse_args()
    
    if args.test_corpus:
        create_linguistic_test_corpus(args.output)
    else:
        create_sample_corpus(args.output, args.size)


if __name__ == "__main__":
    main() 