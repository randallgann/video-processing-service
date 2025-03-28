FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN pip3 install --no-cache-dir google-cloud-pubsub google-cloud-storage

# Copy application code
COPY transcribe-whisper-gpu.py /app/
COPY yt-dlp-aduio-processor-v1.py /app/
COPY k8s-processor.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PROGRESS_TOPIC_ID=""

# Run the processor script
CMD ["python3", "k8s-processor.py"]