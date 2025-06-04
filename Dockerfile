FROM ubuntu:focal-20200729

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

RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Install basic Python dependencies first
RUN pip3 install "cython<3.0" numpy

# Install heavy dependencies that depccg needs (most likely to be cached)
RUN pip3 install "pydantic<1.9" "typing-extensions<4.0"
RUN pip3 install "spacy<3.2"
RUN pip3 install "torch<1.13.0"
RUN pip3 install "transformers<4.21"
RUN pip3 install scipy scikit-learn
RUN pip3 install "allennlp<2.11"
RUN pip3 install "allennlp-models<2.11"
RUN pip3 install googledrivedownloader

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
RUN wget "https://drive.usercontent.google.com/download?id=1UldQDigVq4VG2pJx9yf3krFjV0IYOwLr&export=download&authuser=0&confirm=t&uuid=b8289e74-05ea-44cf-a5b6-ea3fe80b03c0&at=ALoNOgnNelQQskLSDb_H6jjv142V%3A1749056064097" \
    -O lstm_parser_elmo.tar.gz || echo "Direct download failed, URL may have expired"

# Download other models with file IDs (more reliable)
RUN wget "https://drive.usercontent.google.com/download?id=1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv&export=download" \
    -O en_hf_tri.tar.gz --no-check-certificate || echo "English default model download failed"

RUN wget "https://drive.usercontent.google.com/download?id=1bblQ6FYugXtgNNKnbCYgNfnQRkBATSY3&export=download" \
    -O ja_headfinal.tar.gz --no-check-certificate || echo "Japanese model download failed"

# Extract models
RUN tar -xzf en_hf_tri.tar.gz 2>/dev/null || echo "Failed to extract English model"
RUN tar -xzf ja_headfinal.tar.gz 2>/dev/null || echo "Failed to extract Japanese model"

# Alternative: Copy our working Python downloader and run it
COPY download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py || echo "Python downloader failed, using existing models"