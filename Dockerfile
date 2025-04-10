FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and dependencies
COPY requirements.txt replicate-requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt -r replicate-requirements.txt

# Copy application code
COPY cloud-vm-processor.py /app/
COPY yt-dlp-aduio-processor-v1.py /app/
COPY transcribe-whisper-replicate.py /app/
COPY debug-replicate.py /app/
COPY monitor-progress.py /app/

# Create credentials directory (will be mounted at runtime)
RUN mkdir -p /app/credentials

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_WAV_FORMAT=true
ENV CHUNK_LENGTH_SECONDS=300
ENV MAX_CONCURRENT_UPLOADS=5
ENV MAX_CONCURRENT_TRANSCRIPTIONS=5
ENV MAX_MESSAGES=1
ENV DEFAULT_MODEL_TYPE=openai

# Create directories for data persistence
RUN mkdir -p /data/inputs /data/outputs
VOLUME ["/data"]

# Run the processor script
ENTRYPOINT ["python3", "cloud-vm-processor.py"]
CMD []