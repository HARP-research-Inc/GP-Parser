# Constituency Tree Depth Analysis

This module provides a high-accuracy constituency parser using the Berkeley Neural Parser (Benepar) to analyze the depth distribution of constituency trees in text datasets.

## Features

- **High-Accuracy Parsing**: Uses Benepar, one of the most accurate constituency parsers available
- **Tree Depth Analysis**: Calculates and analyzes the maximum depth of constituency trees
- **Histogram Generation**: Creates visual histograms of tree depth distributions
- **Large Dataset Support**: Efficiently processes large text datasets with progress tracking
- **Flexible Input**: Works with sample data or custom text files
- **Comprehensive Output**: Provides statistics, visualizations, and detailed JSON results

## Quick Start

### 1. Setup Environment

First, run the setup script to install dependencies and download required models:

```bash
python setup.py
```

This will:
- Install all required Python packages
- Download the spaCy English model
- Prepare the environment for Benepar

### 2. Run Analysis

#### Using Sample Dataset
```bash
python main.py
```

#### Using Your Own Text File
```bash
python main.py --input-file path/to/your/textfile.txt
```

#### Limiting Sample Size
```bash
python main.py --sample-size 500
```

#### Custom Output Directory
```bash
python main.py --output-dir my_results
```

## Command Line Options

- `--input-file`: Path to input text file (optional, uses sample data if not provided)
- `--sample-size`: Maximum number of sentences to analyze (default: 1000)
- `--output-dir`: Output directory for results (default: "output")
- `--model`: spaCy model to use (default: "en_core_web_sm")

## Output Files

The script generates several output files:

### 1. Histogram (`tree_depth_histogram.png`)
Visual histogram showing the distribution of constituency tree depths with:
- Frequency counts for each depth level
- Statistics overlay (total sentences, mean depth)
- Professional formatting

### 2. Detailed Results (`constituency_analysis_results.json`)
Comprehensive JSON file containing:
- Individual parse results for each sentence
- Depth count statistics
- Performance metrics
- Error information

## Example Output

```
==================================================
CONSTITUENCY TREE DEPTH ANALYSIS RESULTS
==================================================
Total sentences processed: 15
Successful parses: 15
Failed parses: 0
Success rate: 100.00%
Processing time: 12.34 seconds
Mean tree depth: 8.73
Median tree depth: 9.00
Depth range: 6 - 12

Depth distribution:
  Depth 6: 1 sentences (6.7%)
  Depth 7: 2 sentences (13.3%)
  Depth 8: 4 sentences (26.7%)
  Depth 9: 3 sentences (20.0%)
  Depth 10: 3 sentences (20.0%)
  Depth 11: 1 sentences (6.7%)
  Depth 12: 1 sentences (6.7%)
```

## Understanding Tree Depth

Constituency tree depth represents the maximum number of nested syntactic constituents in a sentence. For example:

- **Shallow trees** (depth 4-6): Simple sentences with basic structure
- **Medium trees** (depth 7-10): Complex sentences with embedded clauses
- **Deep trees** (depth 11+): Highly complex sentences with multiple levels of embedding

## Input File Format

The script accepts plain text files with sentences. It automatically:
- Splits text into sentences (using periods as delimiters)
- Filters out very short sentences (less than 3 words)
- Handles various text encodings (UTF-8)

## Performance

- **Processing Speed**: ~10-50 sentences per second (depending on hardware)
- **Memory Usage**: Moderate (loads spaCy and Benepar models)
- **Accuracy**: State-of-the-art constituency parsing accuracy

## Large Dataset Recommendations

For very large datasets:
1. Use the `--sample-size` parameter to limit analysis
2. Monitor progress with the built-in logging
3. Consider running in chunks for datasets > 10,000 sentences

## Troubleshooting

### Common Issues

1. **Model Download Fails**: Ensure internet connection and run setup again
2. **Memory Errors**: Reduce `--sample-size` for large datasets
3. **Parsing Failures**: Check input text encoding and sentence structure

### Requirements

- Python 3.7+
- Internet connection (for initial model downloads)
- ~2GB RAM for model loading
- ~1GB disk space for models

## Technical Details

### Parser Details
- **Model**: Benepar (Berkeley Neural Parser)
- **Backend**: spaCy pipeline integration
- **Accuracy**: ~95% F1 score on standard benchmarks

### Tree Depth Calculation
The script recursively traverses constituency trees to find the maximum depth from root to any leaf node, providing insights into syntactic complexity.

## Citation

If you use this tool in research, please cite:
- Benepar: Kitaev & Klein (2018)
- spaCy: Honnibal & Montani (2017) 