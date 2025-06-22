# WordNet Lemmatizer

This directory contains a fast C++ lemmatizer using WordNet data.

## Files

- `lemmatizer.cpp` - The main C++ lemmatizer program
- `setup.sh` - Shell script to create WordNet data using NLTK (requires Python)
- `setup_cpp_only.sh` - Shell script using only C++ and basic lemmatization rules
- `requirements.txt` - Python dependencies (for setup.sh only)
- `README.md` - This file

## Quick Start (C++ Only)

For a simple setup without Python dependencies:

```bash
chmod +x setup_cpp_only.sh
./setup_cpp_only.sh
```

This will:
- Download WordNet word list
- Create basic lemmatization rules (removes -ing, -ed, -s endings)
- Compile the C++ lemmatizer
- Test it with sample words

## Advanced Setup (with Python/NLTK)

For more accurate lemmatization using NLTK's WordNet data:

### Prerequisites

```bash
sudo apt-get update
sudo apt-get install build-essential python3 python3-nltk
# OR
pip3 install -r requirements.txt
```

### Setup

```bash
chmod +x setup.sh
./setup.sh
```

## Usage

After setup, you can use the lemmatizer:

```bash
# Lemmatize a single word
echo "running" | ./lemma-lookup

# Lemmatize multiple words
echo -e "running\nfeet\nwomen\nbetter" | ./lemma-lookup

# Lemmatize from a file
cat words.txt | ./lemma-lookup
```

## How it works

The lemmatizer uses a hash map to store word → lemma mappings:

1. **C++ Only**: Uses simple rules to remove common suffixes (-ing, -ed, -s)
2. **NLTK Version**: Uses NLTK to access comprehensive WordNet lemmatization data

The C++ program loads the mappings into memory and performs fast lookups.

## Performance

- Loads ~21,000 lemma mappings in milliseconds
- Hash map lookups are O(1) average case
- Memory usage: ~2-3MB for the hash map
- Much simpler than the original marisa-trie approach

## Troubleshooting

If you get compilation errors, make sure you have a C++17 compatible compiler:

```bash
g++ --version  # Should be GCC 7+ or Clang 5+
```

For NLTK issues:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```
