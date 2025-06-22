#!/bin/bash

# WordNet Lemmatizer Setup Script
# This script downloads WordNet data, builds marisa-trie, and compiles the lemmatizer

set -e  # Exit on any error

echo "Setting up WordNet lemmatizer..."

# Check if Python and NLTK are available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not installed."
    exit 1
fi

# 1) Create WordNet lemma lookup file using NLTK
echo "Creating WordNet lemma lookup file using NLTK..."
python3 -c "
import nltk
try:
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    import sys
    
    # Download wordnet if not available
    try:
        wn.synsets('test')
    except:
        print('Downloading WordNet data...')
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    # Create lemma lookup file
    print('Creating lemma lookup file...')
    lemmatizer = WordNetLemmatizer()
    lemma_pairs = set()
    
    # Get all lemma names from WordNet
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            word = lemma.name().replace('_', ' ')
            # Add lemmatization mappings for different POS tags
            for pos in ['n', 'v', 'a', 'r']:  # noun, verb, adjective, adverb
                try:
                    lemma_form = lemmatizer.lemmatize(word, pos=pos)
                    if lemma_form != word:
                        lemma_pairs.add(f'{word}\t{lemma_form}')
                except:
                    continue
    
    # Write to file
    with open('wordnet-lemma-lookup.txt', 'w', encoding='utf-8') as f:
        for pair in sorted(lemma_pairs):
            f.write(pair + '\n')
    
    print(f'Created {len(lemma_pairs)} lemma mappings')
    
except ImportError:
    print('Error: NLTK is required. Install it with: pip install nltk')
    sys.exit(1)
except Exception as e:
    print(f'Error creating lemma file: {e}')
    sys.exit(1)
"

# 2) Install marisa-trie
echo "Cloning and building marisa-trie..."
if [ ! -d "marisa-trie" ]; then
    git clone https://github.com/s-yata/marisa-trie.git
fi

cd marisa-trie
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake ..
make -j$(nproc)

# Go back to lemma directory
cd ../..

# 3) Build lookup table
echo "Building lookup table..."
./marisa-trie/build/marisa-build -o lemmas.marisa wordnet-lemma-lookup.txt

# 4) Compile the lemmatizer
echo "Compiling lemmatizer..."
g++ -O3 -std=c++11 lemmatizer.cpp -I./marisa-trie/include -L./marisa-trie/build/lib -lmarisa -o lemma-lookup

echo "Setup complete! You can now use ./lemma-lookup for lemmatization."
echo "Usage: echo 'words' | ./lemma-lookup"
echo ""
echo "Testing the lemmatizer:"
echo -e "running\nfeet\nwomen\nbetter" | ./lemma-lookup 