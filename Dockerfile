FROM python:3.11-slim

LABEL maintainer="Naga Venkata Kartheek Medathati <mnvhere@gmail.com>"
LABEL description="Wallach: Psychophysical Optical Flow Benchmark"

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output directories
RUN mkdir -p /data/output /data/samples

# Default: generate samples
CMD ["python", "generate_samples.py"]
