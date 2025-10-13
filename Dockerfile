# ===============================
# Base: Ubuntu + CUDA 11.8
# ===============================
FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG ENV_NAME=mosca
ARG NUMPY_VERSION=1.26.4
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ===============================
# Install base system tools
# ===============================
RUN apt-get update && apt-get install -y \
    wget git curl bzip2 build-essential ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# ===============================
# Install full Anaconda
# ===============================
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p $CONDA_DIR && \
    rm /tmp/anaconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH


# # ===============================
# # Copy and run bash install script
# # ===============================
# # Copy your custom install script into the container
# COPY install.sh /tmp/install.sh

# # Make it executable
# RUN chmod +x /tmp/install.sh

# # Run it using bash
# RUN bash /tmp/install.sh
