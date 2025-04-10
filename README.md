# YouTube Video Transcription Service

This service processes YouTube videos by:
1. Downloading the video and extracting audio
2. Splitting audio into chunks
3. Uploading chunks to Google Cloud Storage
4. Using Replicate's Whisper API for transcription
5. Combining results and formatting output
6. Uploading final transcript to Google Cloud Storage

## Architecture

![Cloud VM Architecture]

The service is a cloud-based processor that listens for messages on a Google Cloud PubSub topic. Each message contains a YouTube URL to process. The service downloads the video, extracts audio, splits it into chunks, and uploads these chunks to Google Cloud Storage. It then uses Replicate's Whisper API to transcribe each chunk, combines the results, and uploads the final transcript.

## Setup

### Prerequisites

- Docker and Docker Compose
- Google Cloud project with:
  - PubSub topics and subscriptions
  - Cloud Storage bucket
  - Service account with appropriate permissions
- Replicate account with API token

### Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit the `.env` file with your specific configuration:

```
# Google Cloud Project settings
PROJECT_ID=your-project-id
SUBSCRIPTION_ID=video-processing-requests-sub
BUCKET_NAME=your-transcription-outputs-bucket
PROGRESS_TOPIC_ID=video-processing-progress

# No server configuration needed

# Replicate API settings
REPLICATE_API_TOKEN=your-replicate-api-token

# Model configurations
OPENAI_WHISPER=openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e
FAST_WHISPER=vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c

# Processing settings
DEFAULT_MODEL_TYPE=openai
USE_WAV_FORMAT=true
CHUNK_LENGTH_SECONDS=300
MAX_CONCURRENT_UPLOADS=5
MAX_CONCURRENT_TRANSCRIPTIONS=5
MAX_MESSAGES=1
```

### Running with Docker

Build and start the container:

```bash
docker-compose up -d
```

To run with a specific video for testing:

```bash
docker-compose run --rm transcriber --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

To run with the fast whisper model:

```bash
docker-compose run --rm transcriber --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID --model-type fast
```

### Running Locally (Development)

For local development without Docker:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r replicate-requirements.txt

# Run the service
python cloud-vm-processor.py

# Or for testing with a specific video:
python cloud-vm-processor.py --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

## Message Format

To process a video, publish a message to the configured PubSub topic with this format:

```json
{
  "video_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
  "model_type": "openai",  // Optional, "openai" or "fast"
  "use_wav": true          // Optional, default from env
}
```

## Progress Tracking

If `PROGRESS_TOPIC_ID` is configured, the service publishes progress updates with this format:

```json
{
  "message_id": "original-message-id",
  "video_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
  "video_id": "YOUR_VIDEO_ID",
  "status": "processing",
  "progress_percent": 45,
  "current_stage": "uploading",
  "stage_progress_percent": 100,
  "processing_time_seconds": 120,
  "estimated_time_remaining_seconds": 180,
  "timestamp": "2023-04-09T12:34:56.789Z",
  "error": null
}
```

## Output Format

The service produces a JSON file with this format:

```json
[
  {
    "text": "Spoken Words: The transcribed text for this segment...",
    "metadata": {
      "title": "Video Title",
      "video_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
      "video_id": "YOUR_VIDEO_ID",
      "timestamp_start": 0,
      "timestamp_end": 60,
      "upload_date": "04-09-2023"
    }
  },
  // Additional segments...
]
```

## Troubleshooting

### Common Issues

- **PubSub Connection Issues**: Verify your Google Cloud credentials are correctly set up
- **Replicate API Issues**: Check your API token and model configuration
- **Audio Processing Issues**: Ensure ffmpeg is installed correctly
- **Storage Issues**: Verify your bucket permissions and settings

### Logs

To view the container logs:

```bash
docker-compose logs -f transcriber
```

## License

[License information]