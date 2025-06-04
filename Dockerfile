# Use NVIDIA CUDA base image with runtime support
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# Core stuff
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
# Apparently installing tzdata here explicitly avoids an interactive dialog later
RUN apt-get install -y apt-utils tzdata
RUN ln -sf /usr/share/zoneinfo/US/Eastern /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata
#RUN ntpd -gq
#RUN service ntp start
RUN apt-get install -y wget curl gpg
RUN apt-get install -y aptitude
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    apt-utils \
    tzdata \
    wget \
    curl \
    gpg \
    aptitude \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set timezone
RUN ln -sf /usr/share/zoneinfo/US/Eastern /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Upgrade pip and install basic dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade wheel

# Install basic Python dependencies first
RUN pip3 install "cython<3.0" numpy

# Install heavy dependencies that depccg needs (most likely to be cached)
RUN pip3 install "pydantic<1.9" "typing-extensions<4.0"
RUN pip3 install "spacy<3.2"

# Install CUDA-enabled PyTorch that supports RTX 3050 Ti (sm_86)
RUN pip3 install torch==1.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torchaudio==0.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116

# Install CuPy for Chainer GPU support (required for depccg GPU acceleration)
RUN pip3 install cupy-cuda12x

# Install other ML dependencies
RUN pip3 install "transformers<4.21"
RUN pip3 install scipy scikit-learn
RUN pip3 install "allennlp<2.11"
RUN pip3 install "allennlp-models<2.11"
RUN pip3 install googledrivedownloader

# Install visualization dependencies
RUN pip3 install matplotlib holoviews networkx pandas pyvis

# Now install depccg (most likely to fail, so put it last)
RUN pip3 install depccg

# Fix the import name mismatch - create a proper module with the correct import
RUN mkdir -p /usr/local/lib/python3.8/dist-packages/google_drive_downloader && \
    echo "from googledrivedownloader import download_file_from_google_drive" > /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py && \
    echo "" >> /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py && \
    echo "class GoogleDriveDownloader:" >> /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py && \
    echo "    @staticmethod" >> /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py && \
    echo "    def download_file_from_google_drive(file_id, dest_path, unzip=False, overwrite=False, showsize=False):" >> /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py && \
    echo "        return download_file_from_google_drive(file_id, dest_path, unzip=unzip, overwrite=overwrite, showsize=showsize)" >> /usr/local/lib/python3.8/dist-packages/google_drive_downloader/__init__.py

# Download models using wget (Note: URLs may expire, use file IDs for reliability)
WORKDIR /usr/local/lib/python3.8/dist-packages/depccg/models

# Download English ELMo model (your provided URL)
RUN wget "https://drive.usercontent.google.com/download?id=1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr&export=download" \
    -O lstm_parser_elmo.tar.gz || echo "Direct download failed, URL may have expired"

# Download other models with file IDs (more reliable)
RUN wget "https://drive.usercontent.google.com/download?id=1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv&export=download" \
    -O en_hf_tri.tar.gz --no-check-certificate || echo "English default model download failed"

RUN wget "https://drive.usercontent.google.com/download?id=1bblQ6FYugXtgNNKnbCYgNfnQRkBATSY3&export=download" \
    -O ja_headfinal.tar.gz --no-check-certificate || echo "Japanese model download failed"

# Extract models
RUN tar -xzf en_hf_tri.tar.gz 2>/dev/null || echo "Failed to extract English model"
RUN tar -xzf ja_headfinal.tar.gz 2>/dev/null || echo "Failed to extract Japanese model"

# Download the default depccg model
RUN python3 -m depccg en download || echo "Model download via depccg failed"

# Set working directory back to root
WORKDIR /workspace

# Environment variables for GPU support
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_DEVICE_MAX_CONNECTIONS=2
RUN pip install holoviews networkx pandas pyvis
# Verify GPU support is working
RUN python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())" || echo "PyTorch GPU check failed"
RUN python3 -c "import cupy; print('CuPy version:', cupy.__version__)" || echo "CuPy check failed"

RUN pip install matplotlib
RUN pip3 install "cython<3.0" "numpy<2.0"
# Install CuPy for Chainer GPU support (required for depccg GPU acceleration)
RUN pip3 install cupy-cuda12x

# Install cuDNN for better GPU performance
RUN python3 -m cupyx.tools.install_library --library cudnn --cuda 12.x

# Fix distutils issue for chainer compatibility  
RUN pip3 install setuptools==59.5.0

RUN pip3 install chainer

# Create module alias for google_drive_downloader import issue
RUN mkdir -p /usr/local/lib/python3.10/dist-packages/google_drive_downloader && \
    echo "from googledrivedownloader import download_file_from_google_drive" > /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py && \
    echo "" >> /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py && \
    echo "class GoogleDriveDownloader:" >> /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py && \
    echo "    @staticmethod" >> /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py && \
    echo "    def download_file_from_google_drive(file_id, dest_path, unzip=False, overwrite=False, showsize=False):" >> /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py && \
    echo "        return download_file_from_google_drive(file_id, dest_path, unzip=unzip, overwrite=overwrite, showsize=showsize)" >> /usr/local/lib/python3.10/dist-packages/google_drive_downloader/__init__.py

RUN python3 -m depccg en download

# Health check to ensure everything is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import depccg, torch, cupy; print('All dependencies loaded successfully')" || exit 1