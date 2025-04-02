import os
import time
import json
import subprocess
import tempfile
import datetime
from google.cloud import pubsub_v1
from google.cloud import storage
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
PROJECT_ID = os.environ.get('PROJECT_ID')
SUBSCRIPTION_ID = os.environ.get('SUBSCRIPTION_ID')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
PROGRESS_TOPIC_ID = os.environ.get('PROGRESS_TOPIC_ID', '')
MAX_MESSAGES = int(os.environ.get('MAX_MESSAGES', '1'))
GPU_COUNT = int(os.environ.get('GPU_COUNT', '1'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'medium')

def download_video(video_url):
    """Download video using yt-dlp and return the audio path"""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Downloading video: {video_url} to {temp_dir}")
    
    # Run the yt-dlp processor script with full path
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt-dlp-aduio-processor-v1.py")
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    command = [
        python_executable, script_path,
        video_url
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode != 0:
            logger.error(f"Error downloading video: {result.stderr}")
            raise Exception(f"Failed to download video: {result.stderr}")
        
        # Find the downloaded audio file
        audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
        if not audio_files:
            raise Exception("No audio file found after download")
        
        audio_path = os.path.join(temp_dir, audio_files[0])
        desc_path = os.path.join(temp_dir, os.path.splitext(audio_files[0])[0] + '.txt')
        
        if not os.path.exists(desc_path):
            # Create empty description file if not exists
            with open(desc_path, 'w') as f:
                f.write(f"Title: {os.path.splitext(audio_files[0])[0]}\n\n")
            logger.warning(f"No description file found, created an empty one: {desc_path}")
        
        return audio_path, desc_path, temp_dir
    except Exception as e:
        logger.error(f"Error in download_video: {e}")
        raise

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
    transcribe_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcribe-whisper-gpu.py")
    # Use the same Python interpreter that's running this script
    python_executable = sys.executable
    command = [
        python_executable, transcribe_script_path,
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
        # Use the same Python interpreter that's running this script
        command = [
            python_executable, transcribe_script_path,
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
        
        # Find the output transcript JSON file - it could be in the current directory or in the audio directory
        transcript_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
        possible_paths = [
            os.path.join(os.path.dirname(audio_path), transcript_file),  # In the audio directory
            os.path.join(os.getcwd(), transcript_file),                  # In the current working directory
            transcript_file                                              # Direct in current dir
        ]
        
        transcript_found = False
        for transcript_path in possible_paths:
            if os.path.exists(transcript_path):
                # Copy the transcript to output dir
                logger.info(f"Found transcript at: {transcript_path}")
                os.system(f"cp {transcript_path} {output_dir}/")
                transcript_found = True
                break
        
        if not transcript_found:
            # List files in audio directory to debug
            audio_dir = os.path.dirname(audio_path)
            logger.error(f"Transcript file not found in any of the expected locations: {possible_paths}")
            logger.error(f"Files in audio directory: {os.listdir(audio_dir)}")
            logger.error(f"Files in current directory: {os.listdir(os.getcwd())}")
            raise Exception("Transcript file not found after processing")
        
        # Wait for monitor thread to complete
        monitor_thread.join(timeout=1)
        
        return output_dir
    except Exception as e:
        logger.error(f"Error in transcribe_video: {e}")
        raise

def publish_progress(message_id, video_url, status, progress_percent, 
                    current_stage, stage_progress_percent, 
                    start_time, error=None):
    """Publish progress update to Pub/Sub"""
    if not PROGRESS_TOPIC_ID:
        logger.debug("Progress reporting disabled (PROGRESS_TOPIC_ID not set)")
        return
        
    try:
        # Use the publisher credentials
        publisher_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "rag-widget-pubsub-publisher-key.json")
        publisher = pubsub_v1.PublisherClient.from_service_account_json(publisher_credentials_path)
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

def upload_results(output_dir, message_id):
    """Upload transcription results to GCS"""
    # Use publisher credentials for storage access
    storage_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     "rag-widget-pubsub-publisher-key.json")
    storage_client = storage.Client.from_service_account_json(storage_credentials_path)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    results_uploaded = 0
    logger.info(f"Uploading results from {output_dir} to gs://{BUCKET_NAME}/{message_id}/")
    
    try:
        for root, _, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, output_dir)
                blob_path = f"{message_id}/{relative_path}"
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                results_uploaded += 1
        
        logger.info(f"Uploaded {results_uploaded} files to GCS")
        return results_uploaded
    except Exception as e:
        logger.error(f"Error uploading results: {e}")
        raise

def process_message(message):
    """Process a single Pub/Sub message with progress reporting"""
    temp_dirs = []
    start_time = time.time()
    message_id = None
    video_url = None
    
    try:
        message_id = message.message_id
        logger.info(f"Processing message {message_id}")
        
        # Parse message data
        data = json.loads(message.data.decode('utf-8'))
        
        # Handle both direct video_url and nested structure
        video_url = data.get('video_url')
        
        # If video_url is not directly in the data, check if it's in a nested 'video' object
        if not video_url and 'video' in data and isinstance(data['video'], dict):
            video_url = data['video'].get('url')
            
        if not video_url:
            logger.error("Message missing video_url or video.url")
            logger.error(f"Message data: {data}")
            message.ack()  # Acknowledge to avoid reprocessing the same invalid message
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
        
        # For accurate transcription progress, we'll use the modified transcription function
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
        
        # Check if we have a transcript file before trying to upload
        transcript_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
        transcript_path = os.path.join(output_dir, transcript_file)
        
        if not os.path.exists(transcript_path):
            logger.warning(f"No transcript file found in output directory: {output_dir}")
            # Search for it in other possible locations
            possible_locations = [
                os.path.join(os.getcwd(), transcript_file),
                os.path.join(os.path.dirname(audio_path), transcript_file)
            ]
            for possible_path in possible_locations:
                if os.path.exists(possible_path):
                    logger.info(f"Found transcript at {possible_path}, copying to output directory")
                    os.system(f"cp {possible_path} {output_dir}/")
                    break
        
        # Make sure we have files to upload
        if len(os.listdir(output_dir)) == 0:
            raise Exception("Output directory is empty, no files to upload")
            
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
        if video_url and message_id:
            publish_progress(
                message_id=message_id,
                video_url=video_url,
                status="failed",
                progress_percent=0,
                current_stage="error",
                stage_progress_percent=0,
                start_time=start_time,
                error=str(e)
            )
            
            # Don't retry if we made significant progress (got past download and into transcription)
            # Just check where we were in the process
            if 'output_dir' in locals():  # We got as far as creating the output directory for transcription
                logger.warning("Error occurred after significant processing, acknowledging message to avoid retry")
                message.ack()
            else:
                logger.info("Not acknowledging message to allow redelivery")
        else:
            # If we can't even extract the message ID and URL, just ack the message
            logger.warning("Couldn't extract message_id or video_url, acknowledging to prevent infinite retry")
            message.ack()
    finally:
        # Cleanup
        for dir_path in temp_dirs:
            try:
                if dir_path and os.path.exists(dir_path):
                    logger.info(f"Cleaning up directory: {dir_path}")
                    subprocess.run(["rm", "-rf", dir_path], check=False)
            except Exception as e:
                logger.warning(f"Error cleaning up directory {dir_path}: {e}")

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = ['PROJECT_ID', 'SUBSCRIPTION_ID', 'BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables and restart the container")
        return False
    
    # Check for credential files
    publisher_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "rag-widget-pubsub-publisher-key.json")
    subscriber_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "rag-widget-pubsub-subscriber-key.json")
    
    if not os.path.exists(publisher_credentials_path):
        logger.error(f"Publisher credentials file not found at: {publisher_credentials_path}")
        return False
    
    if not os.path.exists(subscriber_credentials_path):
        logger.error(f"Subscriber credentials file not found at: {subscriber_credentials_path}")
        return False
    
    logger.info(f"Environment check passed")
    logger.info(f"PROJECT_ID: {PROJECT_ID}")
    logger.info(f"SUBSCRIPTION_ID: {SUBSCRIPTION_ID}")
    logger.info(f"BUCKET_NAME: {BUCKET_NAME}")
    logger.info(f"MAX_MESSAGES: {MAX_MESSAGES}")
    logger.info(f"GPU_COUNT: {GPU_COUNT}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    
    if PROGRESS_TOPIC_ID:
        logger.info(f"Progress reporting enabled to topic: {PROGRESS_TOPIC_ID}")
    else:
        logger.warning("Progress reporting disabled (PROGRESS_TOPIC_ID not set)")
    
    logger.info(f"Publisher credentials: {publisher_credentials_path}")
    logger.info(f"Subscriber credentials: {subscriber_credentials_path}")
    
    return True

def main():
    """Main processing loop"""
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    logger.info("Starting video transcription service")
    
    # Use the subscriber credentials
    subscriber_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                        "rag-widget-pubsub-subscriber-key.json")
    subscriber = pubsub_v1.SubscriberClient.from_service_account_json(subscriber_credentials_path)
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
    
    # Configure flow control
    flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES)
    
    def callback(message):
        process_message(message)
    
    # Subscribe to the subscription
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control
    )
    
    logger.info(f"Listening for messages on {subscription_path}")
    logger.info(f"Using GPU count: {GPU_COUNT}")
    logger.info(f"Using model: {MODEL_NAME}")
    
    try:
        # Keep the main thread from exiting
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        logger.info("Subscription canceled")
    except Exception as e:
        logger.error(f"Streaming pull error: {e}")
        streaming_pull_future.cancel()
        raise

if __name__ == "__main__":
    # Add command-line arguments for local testing
    parser = argparse.ArgumentParser(description='Video Transcription Service')
    parser.add_argument('--test', action='store_true', help='Process a test video instead of listening for Pub/Sub messages')
    parser.add_argument('--video-url', type=str, help='YouTube URL to process in test mode')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in test mode")
        if not args.video_url:
            logger.error("No video URL provided for test mode. Use --video-url")
            sys.exit(1)
        
        try:
            # Configure credentials paths
            publisher_credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                             "rag-widget-pubsub-publisher-key.json")
            
            # Check if credentials files exist
            if os.path.exists(publisher_credentials_path):
                logger.info(f"Found publisher credentials at: {publisher_credentials_path}")
            else:
                logger.warning(f"Publisher credentials not found at: {publisher_credentials_path}")
            
            # Simulate a message
            class TestMessage:
                def __init__(self, url):
                    self.message_id = "test-message-id"
                    self.data = json.dumps({"video_url": url}).encode('utf-8')
                
                def ack(self):
                    logger.info("Test message acknowledged")
            
            # Process the test message
            process_message(TestMessage(args.video_url))
            logger.info("Test completed successfully")
        except Exception as e:
            logger.error(f"Test failed: {e}")
            sys.exit(1)
    else:
        main()