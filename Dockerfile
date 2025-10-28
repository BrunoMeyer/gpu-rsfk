# Use NVIDIA CUDA base image with development tools
# FROM nvidia/cuda:13.0.1-devel-ubuntu22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace


# Build the CUDA library for multiple architectures
# Using archall target from Makefile which supports compute_30, compute_61, and compute_75
# RUN make python_archall

# Install Python package
# RUN cd python && pip3 install --no-cache-dir .

# Install common dependencies for examples
RUN pip3 install --no-cache-dir \
    numpy>=1.14.1 \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    scipy

# Set Python path to find the installed package
ENV PYTHONPATH=/workspace:$PYTHONPATH


# Copy project files
COPY . .

# Default command
CMD ["/bin/bash"]
