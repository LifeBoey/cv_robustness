# Use an official Python runtime as a base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies needed for compiling packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    python3-dev \
    gcc \
    g++ \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888
