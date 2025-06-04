#!/usr/bin/env python3
"""
Upload DepCCG models to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo
import tempfile
import shutil

def upload_depccg_models():
    """Upload DepCCG models to Hugging Face."""
    
    # Initialize the API
    api = HfApi()
    
    # Repository name (change this to your username)
    repo_id = "mystichar/depccg-models"  # Change 'your-username'
    
    print("Creating repository...")
    try:
        create_repo(repo_id, exist_ok=True)
        print(f"✓ Repository created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Model files to upload
    models_dir = "/usr/local/lib/python3.8/dist-packages/depccg/models"
    
    models = {
        "en_hf_tri.tar.gz": {
            "description": "English Default Model (190MB) - Fast, reliable CCG parser",
            "language": "en"
        },
        "ja_headfinal.tar.gz": {
            "description": "Japanese Model (56MB) - Japanese CCG parser", 
            "language": "ja"
        },
        "lstm_parser_elmo.tar.gz": {
            "description": "English ELMo Model (649MB) - Higher accuracy with ELMo embeddings",
            "language": "en"
        }
    }
    
    # Create README content
    readme_content = """---
language:
- en
- ja
license: mit
library_name: depccg
tags:
- ccg
- parsing
- combinatory-categorial-grammar
- nlp
- syntax
---

# DepCCG Models

Pre-trained models for [DepCCG](https://github.com/masashi-y/depccg) - Combinatory Categorial Grammar Parser.

## Models Included

### English Default Model (`en_hf_tri.tar.gz`) - 190MB
- **Language**: English
- **Type**: Tri-training Chainer-based model
- **Performance**: Fast, reliable, excellent for compound sentences
- **Usage**: `python3 -m depccg en --input file.txt`

### Japanese Model (`ja_headfinal.tar.gz`) - 56MB  
- **Language**: Japanese
- **Type**: Head-final CCG parser
- **Usage**: `python3 -m depccg ja --input file.txt`

### English ELMo Model (`lstm_parser_elmo.tar.gz`) - 649MB
- **Language**: English
- **Type**: LSTM with ELMo embeddings
- **Performance**: Higher accuracy, slower processing
- **Usage**: `python3 -m depccg en --model elmo --input file.txt`

## Quick Start

```bash
# Download a model
wget https://huggingface.co/{}/resolve/main/en_hf_tri.tar.gz

# Extract to DepCCG models directory
tar -xzf en_hf_tri.tar.gz -C /path/to/depccg/models/

# Use with DepCCG
python3 -m depccg en --input your_text.txt --format json
```

## Output Formats

- `auto` - Human-readable parse trees
- `deriv` - Derivation trees with visual structure  
- `conll` - CoNLL-style dependency format
- `json` - Structured data with confidence scores

## License

MIT License - Same as original DepCCG project.

## Citation

If you use these models, please cite the original DepCCG paper:

```bibtex
@inproceedings{{yoshikawa-etal-2017-ccg,
    title = "A* CCG Parsing with a Supertag and Dependency Factored Model",
    author = "Yoshikawa, Masashi and Noji, Hiroshi and Matsumoto, Yuji",
    booktitle = "Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics",
    year = "2017"
}}
```

## Original Repository

https://github.com/masashi-y/depccg
""".format(repo_id)
    
    # Upload README
    print("\nUploading README...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(readme_content)
        readme_path = f.name
    
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            commit_message="Add model documentation"
        )
        print("✓ README uploaded")
    finally:
        os.unlink(readme_path)
    
    # Upload each model
    for filename, info in models.items():
        model_path = os.path.join(models_dir, filename)
        
        if os.path.exists(model_path):
            print(f"\nUploading {filename} ({info['description']})...")
            
            try:
                api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=filename,
                    repo_id=repo_id,
                    commit_message=f"Add {info['description']}"
                )
                print(f"✓ {filename} uploaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to upload {filename}: {e}")
        else:
            print(f"✗ {filename} not found at {model_path}")
    
    print(f"\n🎉 Upload complete!")
    print(f"🔗 Repository: https://huggingface.co/{repo_id}")
    print(f"\n📖 Now others can download with:")
    print(f"   wget https://huggingface.co/{repo_id}/resolve/main/en_hf_tri.tar.gz")

if __name__ == "__main__":
    print("DepCCG Models → Hugging Face Uploader")
    print("=" * 50)
    
    # Check if logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
        
        # Update the repo_id with actual username
        print("\n⚠️  Remember to update 'your-username' in the script!")
        print("Edit the repo_id = 'your-username/depccg-models' line")
        
    except Exception as e:
        print("❌ Please login first: huggingface-cli login")
        exit(1) 