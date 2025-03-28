# Video Processing Progress Tracking Implementation

This document outlines the implementation plan for adding real-time progress tracking to the video transcription pipeline using Pub/Sub.

## Architecture Overview

```
                             ┌─── Progress Updates ────┐
                             │                         │
                             ▼                         │
Video Request → Processor → Pub/Sub → Cloud Functions → Database/Frontend
  (Pub/Sub)      (GKE)     (Topic)      (Optional)      (Status Display)
```

## 1. Create Progress Tracking Topic

Create a new Pub/Sub topic called `video-processing-progress` to publish progress updates.

## 2. Progress Message Structure

Define a standardized message format for progress updates:

```json
{
  "message_id": "original-pubsub-message-id",
  "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_id": "VIDEO_ID",
  "status": "downloading|processing|completed|failed",
  "progress_percent": 75,
  "current_stage": "download|transcription|upload",
  "stage_progress_percent": 50,
  "processing_time_seconds": 120,
  "estimated_time_remaining_seconds": 40,
  "timestamp": "2025-03-28T12:34:56Z",
  "error": null
}
```

## 3. Modify Processor Script

Update `k8s-processor.py` to add progress reporting at key stages:

### 3.1 Add Progress Publishing Function

```python
def publish_progress(message_id, video_url, status, progress_percent, 
                    current_stage, stage_progress_percent, 
                    start_time, error=None):
    """Publish progress update to Pub/Sub"""
    if not PROGRESS_TOPIC_ID:
        logger.debug("Progress reporting disabled (PROGRESS_TOPIC_ID not set)")
        return
        
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, PROGRESS_TOPIC_ID)
        
        # Extract video_id from URL
        video_id = None
        if video_url:
            # Extract video ID from YouTube URL
            url_parts = video_url.split("v=")
            if len(url_parts) > 1:
                video_id = url_parts[1].split("&")[0]
        
        processing_time_seconds = int(time.time() - start_time)
        
        # Calculate estimated time remaining
        estimated_time_remaining = 0
        if progress_percent > 0 and progress_percent < 100:
            estimated_time_remaining = int((processing_time_seconds / progress_percent) * 
                                        (100 - progress_percent))
        
        # Construct progress message
        progress_message = {
            "message_id": message_id,
            "video_url": video_url,
            "video_id": video_id,
            "status": status,
            "progress_percent": progress_percent,
            "current_stage": current_stage,
            "stage_progress_percent": stage_progress_percent,
            "processing_time_seconds": processing_time_seconds,
            "estimated_time_remaining_seconds": estimated_time_remaining,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "error": error
        }
        
        # Publish message
        message_json = json.dumps(progress_message)
        future = publisher.publish(topic_path, data=message_json.encode('utf-8'))
        message_id = future.result()
        
        logger.debug(f"Published progress update: {progress_message}")
        
    except Exception as e:
        logger.error(f"Error publishing progress update: {e}")
```

### 3.2 Add Progress Update Calls

Insert progress update calls at key points in the processor workflow:

```python
def process_message(message):
    """Process a single Pub/Sub message with progress reporting"""
    temp_dirs = []
    start_time = time.time()
    
    try:
        message_id = message.message_id
        logger.info(f"Processing message {message_id}")
        
        # Parse message data
        data = json.loads(message.data.decode('utf-8'))
        video_url = data.get('video_url')
        
        if not video_url:
            logger.error("Message missing video_url")
            message.ack()
            return
        
        # Initial progress update - Starting
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=0,
            current_stage="download",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        # Download video
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=10,
            current_stage="download",
            stage_progress_percent=50,
            start_time=start_time
        )
        
        audio_path, desc_path, download_dir = download_video(video_url)
        temp_dirs.append(download_dir)
        
        # Download complete progress update
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=20,
            current_stage="download",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # Start transcription progress update
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=20,
            current_stage="transcription",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        # For accurate transcription progress, we'll need to modify the transcription function
        output_dir = transcribe_video(
            audio_path, 
            desc_path, 
            message_id, 
            lambda progress: publish_progress(
                message_id=message_id,
                video_url=video_url,
                status="processing",
                progress_percent=int(20 + progress * 0.7),  # 20-90% of overall progress
                current_stage="transcription",
                stage_progress_percent=progress,
                start_time=start_time
            )
        )
        
        temp_dirs.append(output_dir)
        
        # Start upload progress update
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=90,
            current_stage="upload",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        upload_results(output_dir, message_id)
        
        # Final progress update - Completed
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="completed",
            progress_percent=100,
            current_stage="completed",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # Acknowledge the message
        message.ack()
        logger.info(f"Successfully processed message {message_id}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Report error progress
        publish_progress(
            message_id=message_id if 'message_id' in locals() else "unknown",
            video_url=video_url if 'video_url' in locals() else None,
            status="failed",
            progress_percent=0,
            current_stage="error",
            stage_progress_percent=0,
            start_time=start_time if 'start_time' in locals() else time.time(),
            error=str(e)
        )
        # Don't acknowledge to allow redelivery
    finally:
        # Cleanup
        for dir_path in temp_dirs:
            try:
                logger.info(f"Cleaning up directory: {dir_path}")
                subprocess.run(["rm", "-rf", dir_path], check=False)
            except Exception as e:
                logger.warning(f"Error cleaning up directory {dir_path}: {e}")
```

### 3.3 Modify Transcribe Function to Report Progress

```python
def transcribe_video(audio_path, desc_path, message_id, progress_callback=None):
    """
    Transcribe the video using whisper-gpu script with progress reporting
    
    Args:
        audio_path: Path to audio file
        desc_path: Path to description file
        message_id: Original message ID for tracking
        progress_callback: Callback function for progress updates
    """
    output_dir = tempfile.mkdtemp()
    logger.info(f"Transcribing audio: {audio_path} with model {MODEL_NAME}")
    
    # Get total audio duration for progress calculation
    try:
        total_duration = get_audio_duration(audio_path)
        logger.info(f"Audio duration: {total_duration:.2f} seconds")
    except Exception as e:
        logger.warning(f"Could not determine audio duration: {e}")
        total_duration = None
    
    # Extract chunks to process, so we can track progress
    command = [
        "python3", "transcribe-whisper-gpu.py",
        "--audio", audio_path,
        "--desc", desc_path,
        "--model", MODEL_NAME,
        "--gpus", str(GPU_COUNT),
        "--list-chunks-only"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error listing chunks: {result.stderr}")
            raise Exception(f"Failed to list chunks: {result.stderr}")
        
        # Parse chunk list
        chunks = result.stdout.strip().split('\n')
        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks")
        
        # Function to monitor progress file
        def monitor_progress():
            try:
                progress_file = f"{output_dir}/progress.txt"
                
                # Create progress file
                with open(progress_file, 'w') as f:
                    f.write("0")
                
                processed_chunks = 0
                while processed_chunks < total_chunks:
                    try:
                        with open(progress_file, 'r') as f:
                            processed_chunks = int(f.read().strip())
                    except:
                        processed_chunks = 0
                    
                    progress = min(100, int((processed_chunks / total_chunks) * 100))
                    logger.info(f"Transcription progress: {progress}%")
                    
                    if progress_callback:
                        progress_callback(progress)
                    
                    if processed_chunks >= total_chunks:
                        break
                        
                    time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error monitoring progress: {e}")
        
        # Start progress monitoring in a thread
        import threading
        monitor_thread = threading.Thread(target=monitor_progress)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run the actual transcription
        command = [
            "python3", "transcribe-whisper-gpu.py",
            "--audio", audio_path,
            "--desc", desc_path,
            "--model", MODEL_NAME,
            "--gpus", str(GPU_COUNT),
            "--progress-file", f"{output_dir}/progress.txt"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error transcribing video: {result.stderr}")
            raise Exception(f"Failed to transcribe video: {result.stderr}")
        
        # Find the output transcript JSON file
        transcript_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
        transcript_path = os.path.join(os.path.dirname(audio_path), transcript_file)
        
        if os.path.exists(transcript_path):
            # Copy the transcript to output dir
            os.system(f"cp {transcript_path} {output_dir}/")
        else:
            logger.error(f"Transcript file not found: {transcript_path}")
            raise Exception("Transcript file not found after processing")
        
        # Wait for monitor thread to complete
        monitor_thread.join(timeout=1)
        
        return output_dir
    except Exception as e:
        logger.error(f"Error in transcribe_video: {e}")
        raise

def get_audio_duration(audio_path):
    """Get the duration of an audio file in seconds"""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            audio_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return 0
```

## 4. Modify the Whisper GPU Script

Update the `transcribe-whisper-gpu.py` script to track and report progress:

```python
# Add these arguments
parser.add_argument("--progress-file", type=str, help="File to write progress updates to")
parser.add_argument("--list-chunks-only", action="store_true", help="Only list chunks without processing")

# In the code where it processes chunks, add:
if args.list_chunks_only:
    for chunk_path in chunk_paths:
        print(chunk_path)
    sys.exit(0)

# In the transcribe_chunks_parallel function, add progress tracking:
processed_chunks = 0
total_chunks = len(chunk_paths)

# After each chunk is processed
processed_chunks += 1
if args.progress_file:
    try:
        with open(args.progress_file, 'w') as f:
            f.write(str(processed_chunks))
    except Exception as e:
        print(f"Error writing to progress file: {e}")
```

## 5. Update Dockerfile

Update the Dockerfile to include the progress tracking environment variable:

```dockerfile
# Add environment variable for progress tracking
ENV PROGRESS_TOPIC_ID=""
```

## 6. Update Kubernetes Deployment

Add the progress topic ID environment variable to the Kubernetes deployment:

```yaml
env:
  - name: PROJECT_ID
    value: "your-project-id"
  - name: SUBSCRIPTION_ID
    value: "video-transcription-processor"
  - name: BUCKET_NAME
    value: "your-bucket-name"
  - name: PROGRESS_TOPIC_ID
    value: "video-processing-progress"
```

## 7. Create a Progress Consumer (Optional)

### Cloud Function to Process Progress Updates

```python
import base64
import json
import functions_framework
from google.cloud import firestore

@functions_framework.cloud_event
def process_progress_updates(cloud_event):
    """Process Pub/Sub progress updates and store in Firestore"""
    # Get Pub/Sub message
    pubsub_message = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    message_data = json.loads(pubsub_message)
    
    # Initialize Firestore client
    db = firestore.Client()
    
    # Extract message ID to use as document ID
    message_id = message_data.get("message_id", "unknown")
    
    # Store progress update in Firestore
    progress_ref = db.collection("video_progress").document(message_id)
    progress_ref.set(message_data, merge=True)
    
    # Add progress update to history collection
    history_ref = progress_ref.collection("history").document()
    history_ref.set({
        **message_data,
        "recorded_at": firestore.SERVER_TIMESTAMP
    })
    
    print(f"Stored progress update for message {message_id}: {message_data['status']} - {message_data['progress_percent']}%")
    return "OK"
```

## 8. Testing the Progress Implementation

### 8.1 Local Testing

Test the progress reporting locally:

```bash
# Run with progress reporting enabled
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=dummy-value \
  -e BUCKET_NAME=your-bucket-name \
  -e PROGRESS_TOPIC_ID=video-processing-progress \
  -e GPU_COUNT=1 \
  video-transcriber:latest \
  --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

### 8.2 Verify Progress Updates

Verify progress updates are published to the topic:

```bash
# Create a pull subscription to the progress topic
gcloud pubsub subscriptions create progress-test-sub \
  --topic=video-processing-progress

# Pull messages from the subscription
gcloud pubsub subscriptions pull progress-test-sub \
  --auto-ack \
  --limit=100
```

## 9. Implementation Timeline

1. **Day 1**: Create progress topic and update processor script
2. **Day 2**: Modify transcribe-whisper-gpu.py to report progress
3. **Day 3**: Test locally and fix any issues
4. **Day 4**: Update Kubernetes deployment and test in GKE
5. **Day 5**: (Optional) Create Cloud Function for progress consumer

## 10. Scaling Considerations

- Progress updates increase Pub/Sub traffic but are lightweight
- When processing large batches, consider reducing progress update frequency
- Filter unimportant updates on the consumer side
- For very high-volume processing, consider aggregating progress updates