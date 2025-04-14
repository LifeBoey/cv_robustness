FROM python:3.12-slim

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    gnupg2 \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Add NVIDIA's CUDA repository
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update

# Install CUDA runtime libraries (12.2-compatible)
RUN apt-get install -y --no-install-recommends \
    cuda-cudart-12-2 \
    libcublas-12-2 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 (works with driver 535+)
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Test if CUDA is available
CMD ["python", "-c", "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"]

