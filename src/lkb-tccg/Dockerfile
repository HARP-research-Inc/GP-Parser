# Dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install runtime deps: wget, git, X11 libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget ca-certificates git \
      libx11-6 libxpm4 libxaw7 libxt6 && \
    rm -rf /var/lib/apt/lists/*

# 2. Set up DELPHINHOME and temp directory
ENV DELPHINHOME=/delphin
ENV LD_LIBRARY_PATH=$DELPHINHOME/lkb_fos/lib/linux.x86.64:$LD_LIBRARY_PATH
RUN mkdir -p $DELPHINHOME /tmp

# 3. Download & unpack LKB-FOS (Linux binary)
RUN wget -O /tmp/lkb_fos.tgz https://users.sussex.ac.uk/~johnca/lkb_fos.tgz && \
    tar -xzf /tmp/lkb_fos.tgz -C $DELPHINHOME && \
    rm /tmp/lkb_fos.tgz
# :contentReference[oaicite:0]{index=0}

# 4. Clone the TCCG English fragment
RUN git clone https://github.com/delph-in/erg-tccg.git $DELPHINHOME/grammars/tccg

# 5. Add LKB-FOS to PATH
ENV PATH=$DELPHINHOME/lkb_fos/bin:$PATH

WORKDIR $DELPHINHOME
CMD ["/bin/bash"]
