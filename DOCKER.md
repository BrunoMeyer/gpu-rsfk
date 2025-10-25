# Docker Usage Guide

This document describes how to use Docker to build and run GPU-RSFK.

## Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 1.29 or later)
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU with CUDA support

## Installing NVIDIA Docker Runtime

To use GPU inside Docker containers, you need to install the NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Verify the installation:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Building the Docker Image

Build the GPU-RSFK Docker image:

```bash
docker-compose build
```

Or build manually:

```bash
docker build -t gpu-rsfk:latest .
```

## Running Experiments

### Using Docker Compose

The Docker Compose configuration includes two services:
- `gpu-rsfk`: Interactive shell for development
- `experiments`: Pre-configured to run example scripts

Both services will automatically build the image (including compiling the CUDA library and installing the gpu_rsfk package) if it doesn't exist.

Run the default interactive shell:

```bash
docker-compose run --rm gpu-rsfk
```

Run a specific example:

```bash
docker-compose run --rm gpu-rsfk python3 examples/create_knn_graph.py
```

Run the experiments service:

```bash
docker-compose up experiments
```

### Using Docker Directly

Run interactively:

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/examples:/workspace/examples \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/results:/workspace/results \
  gpu-rsfk:latest
```

Run a specific Python script:

```bash
docker run --rm --gpus all \
  -v $(pwd)/examples:/workspace/examples \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/results:/workspace/results \
  gpu-rsfk:latest \
  python3 examples/create_knn_graph.py
```

## Volume Mounts

The Docker setup includes the following volume mounts:

- `./examples` → `/workspace/examples` - Example scripts
- `./datasets` → `/workspace/datasets` - Dataset files
- `./results` → `/workspace/results` - Output and results

Create the results directory if it doesn't exist:

```bash
mkdir -p results
```

## Troubleshooting

### GPU Not Detected

If the GPU is not detected inside the container:

1. Verify NVIDIA Docker runtime is installed correctly
2. Check that your GPU drivers are up to date
3. Ensure you're using the `--gpus all` flag or the deploy section in docker-compose

### Permission Issues

If you encounter permission issues with mounted volumes:

```bash
# Run container as current user
docker run --rm -it --gpus all --user $(id -u):$(id -g) \
  -v $(pwd)/examples:/workspace/examples \
  gpu-rsfk:latest
```

### Build Failures

If the build fails:

1. Ensure you have sufficient disk space
2. Check that CUDA 11.8 is compatible with your GPU
3. You may need to modify the Dockerfile to use a different CUDA version

## Custom CUDA Architecture

To build for a specific GPU architecture, modify the Dockerfile and change the make target:

```dockerfile
# For compute capability 8.6 (e.g., RTX 3090)
RUN make python_arch86

# For compute capability 7.5 (e.g., RTX 2080)
RUN make python_arch75
```

See the Makefile for available architecture targets.
