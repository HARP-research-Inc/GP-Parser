#!/usr/bin/env python3
"""
Download DepCCG models from provided Google Drive links.
"""

import os
import tarfile
from googledrivedownloader import download_file_from_google_drive

def download_models():
    """Download all the provided models to the correct locations."""
    
    models_dir = "/usr/local/lib/python3.8/dist-packages/depccg/models"
    
    # Model definitions with Google Drive file IDs
    models = [
        {
            "name": "English Default Model (en_hf_tri.tar.gz)",
            "file_id": "1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv",
            "filename": "en_hf_tri.tar.gz",
            "extract": True,
            "extract_to": "tri_headfirst"
        },
        {
            "name": "English ELMo Model", 
            "file_id": "1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr",
            "filename": "lstm_parser_elmo.tar.gz",
            "extract": False  # Keep as tar.gz for AllennLP
        },
        {
            "name": "Japanese Default Model",
            "file_id": "1bblQ6FYugXtgNNKnbCYgNfnQRkBATSY3", 
            "filename": "ja_headfinal.tar.gz",
            "extract": True,
            "extract_to": "ja_headfinal"
        },
        {
            "name": "Tri-training Dataset",
            "file_id": "1rCJyb98AcNx5eBuC18-koCWJFfU4OV06",
            "filename": "tri_training_dataset.tar.gz", 
            "extract": False  # Optional dataset
        }
    ]
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Downloading: {model['name']}")
        print(f"{'='*60}")
        
        dest_path = os.path.join(models_dir, model['filename'])
        
        try:
            # Download the file
            print(f"Downloading to: {dest_path}")
            download_file_from_google_drive(
                file_id=model['file_id'],
                dest_path=dest_path,
                overwrite=True,
                showsize=True
            )
            
            # Extract if needed
            if model.get('extract', False):
                extract_to = model.get('extract_to')
                if extract_to:
                    extract_path = os.path.join(models_dir, extract_to)
                    print(f"Extracting to: {extract_path}")
                    
                    # Remove existing directory if it exists
                    if os.path.exists(extract_path):
                        import shutil
                        shutil.rmtree(extract_path)
                    
                    # Extract
                    with tarfile.open(dest_path, 'r:gz') as tar:
                        tar.extractall(models_dir)
                    
                    print(f"✓ Extracted successfully")
                else:
                    print(f"✓ Download completed (no extraction needed)")
            else:
                print(f"✓ Download completed")
                
        except Exception as e:
            print(f"✗ Failed to download {model['name']}: {e}")
    
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")
    
    # List what we have now
    print("\nAvailable model files:")
    for item in os.listdir(models_dir):
        if item.endswith(('.tar.gz', '.gz')) or os.path.isdir(os.path.join(models_dir, item)):
            if not item.startswith('__'):
                size_info = ""
                item_path = os.path.join(models_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    size_info = f" ({size:,} bytes)"
                elif os.path.isdir(item_path):
                    size_info = " (directory)"
                print(f"  ✓ {item}{size_info}")

if __name__ == "__main__":
    print("DepCCG Model Downloader")
    print("Downloading models from provided Google Drive links...")
    download_models() 