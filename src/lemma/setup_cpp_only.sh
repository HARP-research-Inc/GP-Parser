#!/bin/bash

# Pure C++ WordNet Lemmatizer Setup Script
# This script downloads WordNet data directly and compiles a simple lemmatizer

set -e  # Exit on any error

echo "Setting up WordNet lemmatizer (C++ only)..."

# 1) Download WordNet lemma list from alternative source
echo "Downloading WordNet word list..."
if [ ! -f "wordnet-words.txt" ]; then
    wget -O wordnet-words.txt.gz https://s3.amazonaws.com/wordsapi/wordnetWords.txt.gz
    gunzip -f wordnet-words.txt.gz
fi

# Create a simple lemma lookup file (word -> word for now, can be enhanced)
echo "Creating basic lemma lookup file..."
cat wordnet-words.txt | while read word; do
    # Simple rules for basic lemmatization
    if [[ $word =~ ing$ ]]; then
        base="${word%ing}"
        if [[ ${#base} -gt 2 ]]; then
            echo -e "$word\t$base"
        fi
    elif [[ $word =~ ed$ ]]; then
        base="${word%ed}"
        if [[ ${#base} -gt 2 ]]; then
            echo -e "$word\t$base"
        fi
    elif [[ $word =~ s$ ]] && [[ ! $word =~ ss$ ]]; then
        base="${word%s}"
        if [[ ${#base} -gt 2 ]]; then
            echo -e "$word\t$base"
        fi
    fi
done > wordnet-lemma-lookup.txt

echo "Created $(wc -l < wordnet-lemma-lookup.txt) lemma mappings"

# 2) Compile the lemmatizer (no longer needs marisa-trie)
echo "Compiling lemmatizer..."
g++ -O3 -std=c++17 lemmatizer.cpp -o lemma-lookup

echo "Setup complete! You can now use ./lemma-lookup for lemmatization."
echo "Usage: echo 'words' | ./lemma-lookup"
echo ""
echo "Testing the lemmatizer:"
echo -e "running\nfeet\nwomen\nbetter" | ./lemma-lookup

# Cleanup
rm -f wordnet-words.txt 