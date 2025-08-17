#!/usr/bin/env python3
"""
Cloud VM Processor for YouTube Video Transcription Service

This script runs on a cloud VM to:
1. Listen for messages on a Google Cloud PubSub topic
2. Download YouTube videos and extract audio
3. Split audio into chunks and upload to Google Cloud Storage
4. Send transcription requests to Replicate API
5. Track transcription progress and report status
6. Process transcription results and upload to GCS
"""

import os
import sys
import json
import time
import uuid
import tempfile
import datetime
import subprocess
import logging
import argparse
import threading
import replicate
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from google.cloud import pubsub_v1
from google.cloud import storage
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

# Load environment variables
load_dotenv()

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
AUDIO_CHUNKS_BUCKET = os.environ.get('AUDIO_CHUNKS_BUCKET', 'rag-widget-audio-chunks')
PROGRESS_TOPIC_ID = os.environ.get('PROGRESS_TOPIC_ID', '')
RESULTS_TOPIC_ID = os.environ.get('RESULTS_TOPIC_ID', '')
MAX_MESSAGES = int(os.environ.get('MAX_MESSAGES', '1'))
CHUNK_LENGTH_SECONDS = int(os.environ.get('CHUNK_LENGTH_SECONDS', '300'))  # 5 minutes by default
MAX_CONCURRENT_UPLOADS = int(os.environ.get('MAX_CONCURRENT_UPLOADS', '5'))
MAX_CONCURRENT_TRANSCRIPTIONS = int(os.environ.get('MAX_CONCURRENT_TRANSCRIPTIONS', '5'))
OPENAI_WHISPER = os.environ.get('OPENAI_WHISPER', '')
FAST_WHISPER = os.environ.get('FAST_WHISPER', '')
DEFAULT_MODEL_TYPE = os.environ.get('DEFAULT_MODEL_TYPE', 'openai')  # 'openai' or 'fast'
USE_WAV_FORMAT = os.environ.get('USE_WAV_FORMAT', '').lower() in ('true', '1', 'yes')
HTTP_PORT = int(os.environ.get('PORT', '8080'))
METRICS_PORT = int(os.environ.get('METRICS_PORT', '9090'))

# Global service status
service_ready = False
service_healthy = False

class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health and readiness endpoints"""
    
    def do_GET(self):
        global service_ready, service_healthy
        
        if self.path == '/health':
            # Health check - service is healthy if it can process requests
            if service_healthy:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "healthy"}).encode())
            else:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "unhealthy"}).encode())
                
        elif self.path == '/ready':
            # Readiness check - service is ready if it can accept new requests
            if service_ready:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ready"}).encode())
            else:
                self.send_response(503)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "not ready"}).encode())
                
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def log_message(self, format, *args):
        # Override to use our logger instead of printing to stderr
        logger.debug(f"Health endpoint: {format % args}")

def start_health_server():
    """Start the health check HTTP server in a separate thread"""
    global service_ready, service_healthy
    
    try:
        server = HTTPServer(('0.0.0.0', HTTP_PORT), HealthHandler)
        logger.info(f"Health server starting on port {HTTP_PORT}")
        
        # Mark service as healthy once server starts
        service_healthy = True
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        
        logger.info(f"Health server running on http://0.0.0.0:{HTTP_PORT}")
        return server
        
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")
        service_healthy = False
        raise

def get_model_info(model_type=None, model_version=None):
    """
    Get model information based on model_type or model_version
    
    Args:
        model_type: Type of model to use ("openai" or "fast")
        model_version: Specific model version (overrides model_type)
        
    Returns:
        Tuple of (model_id, model_version, is_fast_model)
    """
    is_fast_model = False
    
    if model_version:
        # If specific version is provided, use it directly
        model_id = "openai/whisper"  # Default to OpenAI model ID
        # Check if this is the fast model version
        if ":" in FAST_WHISPER and model_version == FAST_WHISPER.split(":", 1)[1]:
            is_fast_model = True
    elif model_type == "fast":
        # Use fast whisper model from env var
        if not FAST_WHISPER or ":" not in FAST_WHISPER:
            raise ValueError("FAST_WHISPER environment variable not properly set")
        model_id, model_version = FAST_WHISPER.split(":", 1)
        is_fast_model = True
    else:
        # Default to OpenAI whisper model from env var
        if not OPENAI_WHISPER or ":" not in OPENAI_WHISPER:
            raise ValueError("OPENAI_WHISPER environment variable not properly set")
        model_id, model_version = OPENAI_WHISPER.split(":", 1)
    
    return model_id, model_version, is_fast_model

def download_video(video_url, output_dir):
    """
    Download video using yt-dlp and return the audio path
    
    Args:
        video_url: URL of the video to download
        output_dir: Directory to save audio files
        
    Returns:
        Tuple of (audio_path, desc_path)
    """
    logger.info(f"Downloading video: {video_url} to {output_dir}")
    
    # Use the Python executable running this script
    python_executable = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yt-dlp-aduio-processor-v1.py")
    
    command = [
        python_executable, script_path,
        video_url,
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error downloading video: {result.stderr}")
            raise Exception(f"Failed to download video: {result.stderr}")
        
        # Find the downloaded audio file
        audio_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
        if not audio_files:
            raise Exception("No audio file found after download")
        
        audio_path = os.path.join(output_dir, audio_files[0])
        desc_path = os.path.join(output_dir, os.path.splitext(audio_files[0])[0] + '.txt')
        
        if not os.path.exists(desc_path):
            # Create empty description file if not exists
            with open(desc_path, 'w') as f:
                f.write(f"Title: {os.path.splitext(audio_files[0])[0]}\n\n")
            logger.warning(f"No description file found, created an empty one: {desc_path}")
        
        return audio_path, desc_path
        
    except Exception as e:
        logger.error(f"Error in download_video: {e}")
        raise

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in seconds
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds (float)
    """
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

def split_audio(audio_path, output_dir, chunk_length=300, use_wav=False):
    """
    Split audio into chunks
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save chunks
        chunk_length: Length of each chunk in seconds
        use_wav: Whether to use WAV format (better for API compatibility)
        
    Returns:
        List of chunk paths
    """
    logger.info(f"Splitting audio {audio_path} into {chunk_length}s chunks")
    
    # Create a directory for chunks if it doesn't exist
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Get total audio duration
    total_length = get_audio_duration(audio_path)
    num_chunks = int((total_length + chunk_length - 1) / chunk_length)  # Ceiling division
    
    # Determine output format
    output_ext = "wav" if use_wav else "mp3"
    logger.info(f"Using {output_ext.upper()} format for audio chunks")
    
    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_length
        
        # For the last chunk, make sure we don't go past the end
        duration = min(chunk_length, total_length - start)
        
        # If duration <= 0, no more valid chunks left
        if duration <= 0:
            break
        
        chunk_output = os.path.join(chunks_dir, f"audio-chunk-{i:03d}.{output_ext}")
        
        # ffmpeg command to slice audio
        if use_wav:
            # For WAV, we need to decode and re-encode (can't use copy codec)
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", audio_path,
                "-ss", str(start),
                "-t", str(chunk_length),
                "-acodec", "pcm_s16le",  # Standard 16-bit PCM WAV format
                "-ar", "16000",          # 16kHz sample rate (good for speech)
                "-ac", "1",              # Mono channel
                chunk_output
            ]
        else:
            # For MP3, use re-encoding instead of copy to ensure proper chunking
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", audio_path,
                "-ss", str(start),
                "-t", str(chunk_length),
                "-acodec", "libmp3lame",  # Use MP3 encoding
                "-ar", "44100",           # Standard audio rate
                "-ab", "128k",            # Bitrate
                chunk_output
            ]
        
        # Run ffmpeg and capture output for error logging
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to create chunk: {chunk_output}")
            logger.error(f"FFmpeg error: {result.stderr}")
        
        if os.path.exists(chunk_output):
            chunk_info = {
                "path": chunk_output,
                "start_time": start,
                "duration": duration,
                "id": i
            }
            chunk_paths.append(chunk_info)
        else:
            logger.error(f"Failed to create chunk: {chunk_output}")
    
    logger.info(f"Created {len(chunk_paths)} audio chunks")
    return chunk_paths

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    """
    Upload a file to Google Cloud Storage
    
    Args:
        file_path: Path to the file to upload
        bucket_name: Name of the bucket
        destination_blob_name: Name of the destination blob
        
    Returns:
        Public URL of the uploaded file
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(file_path)
        
        # Make the file publicly accessible
        blob.make_public()
        
        logger.info(f"File {file_path} uploaded to gs://{bucket_name}/{destination_blob_name}")
        return blob.public_url
        
    except Exception as e:
        logger.error(f"Error uploading to GCS: {e}")
        raise

def upload_chunks(chunks, bucket_name, message_id):
    """
    Upload audio chunks to Google Cloud Storage
    
    Args:
        chunks: List of chunk info dictionaries
        bucket_name: Name of the GCS bucket
        message_id: ID of the message for creating folder structure
        
    Returns:
        List of updated chunk info dictionaries with GCS URLs
    """
    logger.info(f"Uploading {len(chunks)} chunks to GCS")
    
    # Create a ThreadPoolExecutor to upload chunks in parallel
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_UPLOADS) as executor:
        futures = []
        
        for chunk in chunks:
            chunk_path = chunk["path"]
            file_name = os.path.basename(chunk_path)
            destination_blob_name = f"{message_id}/chunks/{file_name}"
            
            # Submit upload task to executor
            future = executor.submit(
                upload_to_gcs, 
                chunk_path, 
                bucket_name,
                destination_blob_name
            )
            futures.append((future, chunk))
        
        # Collect results and update chunk info
        for future, chunk in futures:
            try:
                chunk["gcs_url"] = future.result()
            except Exception as e:
                logger.error(f"Error uploading chunk {chunk['path']}: {e}")
                # Mark chunk as failed
                chunk["upload_failed"] = True
    
    # Count successful uploads
    successful_uploads = sum(1 for chunk in chunks if "gcs_url" in chunk and not chunk.get("upload_failed", False))
    logger.info(f"Successfully uploaded {successful_uploads} of {len(chunks)} chunks")
    
    return chunks

def transcribe_chunk_replicate(chunk_info, model_type=None, model_version=None):
    """
    Transcribe a single audio chunk using Replicate API
    
    Args:
        chunk_info: Dictionary with chunk information
        model_type: Type of model to use ("openai" or "fast")
        model_version: Specific model version (overrides model_type)
        
    Returns:
        Transcription result dictionary
    """
    gcs_url = chunk_info.get("gcs_url")
    if not gcs_url:
        logger.error(f"Chunk {chunk_info['id']} has no GCS URL")
        return None
        
    # Get model info
    model_id, version_id, is_fast_model = get_model_info(model_type, model_version)
    
    # Set language based on model type
    language = "english" if is_fast_model else "auto"
    
    logger.info(f"Transcribing chunk {chunk_info['id']} with model {model_id}:{version_id}")
    logger.info(f"URL: {gcs_url}")
    
    # Maximum retries for API calls
    max_retries = 3
    retry_delay = 5  # seconds
    
    # Try to call Replicate API with retries
    for attempt in range(max_retries):
        try:
            output = replicate.run(
                f"{model_id}:{version_id}",
                input={
                    "audio": gcs_url,
                    "language": language,
                    "translate": False,
                    "temperature": 0,
                    "transcription": "srt"  # Use SRT format to get timestamps
                }
            )
            
            # Handle different response formats based on model type
            segments = []
            
            if is_fast_model:
                # Fast Whisper model returns different format with timestamp chunks
                logger.info(f"Processing Fast Whisper response format for chunk {chunk_info['id']}")
                logger.info(f"Response type: {type(output)}")
                
                # Log the raw response structure for diagnostics
                if isinstance(output, dict):
                    logger.info(f"Fast Whisper output keys: {list(output.keys())}")
                    # Output part of the text to help debug
                    if 'output' in output and isinstance(output['output'], dict) and 'text' in output['output']:
                        text_sample = output['output']['text'][:200] if output['output']['text'] else "EMPTY TEXT"
                        logger.info(f"Text sample: {text_sample}...")
                    # Additional diagnostic info
                    if 'logs' in output:
                        logger.info(f"Response logs: {output['logs'][:200]}...")
                
                try:
                    # DIRECT FORMAT FROM LOGS: Fast Whisper output keys: ['chunks', 'text']
                    # This is a special case for vaibhavs10/incredibly-fast-whisper model
                    if isinstance(output, dict) and 'chunks' in output and 'text' in output:
                        logger.info("Detected direct format from incredibly-fast-whisper model")
                        # Use the full text as fallback
                        full_text = output['text'].strip()
                        logger.info(f"Full text from output (sample): {full_text[:100]}...")
                        
                        chunks_data = output['chunks']
                        logger.info(f"Found {len(chunks_data)} chunks in direct Fast Whisper response")
                        
                        # Log sample chunks
                        for i, chunk in enumerate(chunks_data[:3]):
                            logger.info(f"Direct chunk {i} data: {chunk}")
                        
                        processed_chunks = 0
                        for chunk in chunks_data:
                            # Most probably the format is {'text': '...', 'timestamp': [start, end]}
                            if 'text' in chunk and 'timestamp' in chunk and len(chunk['timestamp']) == 2:
                                start_time = float(chunk['timestamp'][0]) + chunk_info["start_time"]
                                end_time = float(chunk['timestamp'][1]) + chunk_info["start_time"]
                                text = chunk['text'].strip()
                                
                                segments.append({
                                    "start": start_time,
                                    "end": end_time,
                                    "text": text
                                })
                                processed_chunks += 1
                        
                        logger.info(f"Successfully processed {processed_chunks} chunks from direct Fast Whisper")
                        
                        # If no chunks were processed but we have the full text, use it
                        if processed_chunks == 0 and full_text:
                            logger.info("Using full text as one segment since no chunks were processed")
                            segments.append({
                                "start": chunk_info["start_time"],
                                "end": chunk_info["start_time"] + chunk_info["duration"],
                                "text": full_text
                            })
                    # Standard nested output format
                    elif isinstance(output, dict) and 'output' in output:
                        output_content = output['output']
                        logger.info(f"Output content type: {type(output_content)}")
                        
                        if isinstance(output_content, dict):
                            # Check if we have chunks with timestamps
                            if 'chunks' in output_content:
                                chunks_data = output_content['chunks']
                                logger.info(f"Found {len(chunks_data)} chunks in Fast Whisper response")
                                
                                for i, chunk in enumerate(chunks_data[:3]):  # Log first 3 chunks for sample
                                    logger.info(f"Chunk {i} data: {chunk}")
                                
                                processed_chunks = 0
                                for chunk in chunks_data:
                                    try:
                                        if ('text' in chunk and 'timestamp' in chunk and 
                                            len(chunk['timestamp']) == 2 and 
                                            chunk['timestamp'][0] is not None and 
                                            chunk['timestamp'][1] is not None):
                                            
                                            start_time = float(chunk['timestamp'][0]) + chunk_info["start_time"]
                                            end_time = float(chunk['timestamp'][1]) + chunk_info["start_time"]
                                            text = chunk['text'].strip()
                                            
                                            segments.append({
                                                "start": start_time,
                                                "end": end_time,
                                                "text": text
                                            })
                                            processed_chunks += 1
                                        elif 'text' in chunk:
                                            # Handle chunks with missing/invalid timestamps
                                            # Use the chunk_info start_time and an estimated duration
                                            logger.warning(f"Chunk has invalid timestamp: {chunk.get('timestamp')}, using default timing")
                                            text = chunk['text'].strip()
                                            
                                            # Calculate approximate duration based on text length
                                            approx_duration = len(text.split()) * 0.5  # ~0.5 seconds per word
                                            
                                            segments.append({
                                                "start": chunk_info["start_time"],
                                                "end": chunk_info["start_time"] + approx_duration,
                                                "text": text
                                            })
                                            processed_chunks += 1
                                    except (ValueError, TypeError) as e:
                                        logger.warning(f"Error processing chunk: {str(e)}, skipping")
                                
                                logger.info(f"Successfully processed {processed_chunks} chunks from Fast Whisper")
                            else:
                                # No chunks, use full text as one segment
                                if 'text' in output_content:
                                    text = output_content['text'].strip()
                                    logger.info(f"No chunks found, using full text: {text[:50]}...")
                                    segments.append({
                                        "start": chunk_info["start_time"],
                                        "end": chunk_info["start_time"] + chunk_info["duration"],
                                        "text": text
                                    })
                                    logger.info("Created one segment from full text")
                                else:
                                    logger.error("No 'text' field found in output_content")
                        else:
                            # If output_content is a string, treat it as the text
                            if isinstance(output_content, str):
                                text = output_content.strip()
                                logger.info(f"Output is direct string: {text[:50]}...")
                                segments.append({
                                    "start": chunk_info["start_time"],
                                    "end": chunk_info["start_time"] + chunk_info["duration"],
                                    "text": text
                                })
                                logger.info("Created one segment from string output")
                    elif isinstance(output, str):
                        # Handle case where output is directly a string
                        text = output.strip()
                        logger.info(f"Output is directly a string: {text[:50]}...")
                        segments.append({
                            "start": chunk_info["start_time"],
                            "end": chunk_info["start_time"] + chunk_info["duration"],
                            "text": text
                        })
                        logger.info("Created one segment from direct string output")
                except Exception as parsing_error:
                    logger.error(f"Error parsing Fast Whisper output: {parsing_error}")
                    
                    # Log the structure of the output without the full content
                    if isinstance(output, dict):
                        keys = list(output.keys())
                        logger.error(f"Output keys: {keys}")
                        if 'chunks' in output:
                            chunk_count = len(output['chunks'])
                            logger.error(f"Found {chunk_count} chunks, first few have issues")
                    else:
                        logger.error(f"Output type: {type(output)}")
                    
                    # Make a best effort to extract text from whatever we have
                    try:
                        if isinstance(output, dict):
                            if 'output' in output and isinstance(output['output'], dict) and 'text' in output['output']:
                                text = output['output']['text'].strip()
                                logger.info(f"Recovered text from error: {text[:50]}...")
                                segments.append({
                                    "start": chunk_info["start_time"],
                                    "end": chunk_info["start_time"] + chunk_info["duration"],
                                    "text": text
                                })
                                logger.info("Created recovery segment")
                    except Exception as recovery_error:
                        logger.error(f"Failed to recover text: {recovery_error}")
                
                logger.info(f"Final segments count from Fast Whisper: {len(segments)}")
            else:
                # Standard OpenAI Whisper model returns SRT format
                if isinstance(output, dict):
                    srt_content = output.get("transcription", "")
                elif isinstance(output, str):
                    srt_content = output
                else:
                    logger.warning(f"Unexpected response type: {type(output)}")
                    srt_content = str(output) if output else ""
                
                # Parse SRT and adjust timestamps
                segments = parse_srt(srt_content, chunk_info["start_time"])
            
            # Create result object
            result = {
                "chunk_id": chunk_info["id"],
                "start_time": chunk_info["start_time"],
                "duration": chunk_info["duration"],
                "segments": segments,
                "transcript_text": " ".join([seg["text"] for seg in segments])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_info['id']} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed after {max_retries} attempts")
                return {
                    "chunk_id": chunk_info["id"],
                    "start_time": chunk_info["start_time"],
                    "duration": chunk_info["duration"],
                    "error": str(e),
                    "failed": True
                }

def parse_srt(srt_content, time_offset=0):
    """
    Parse SRT content and adjust timestamps
    
    Args:
        srt_content: SRT formatted string
        time_offset: Time offset in seconds to add to all timestamps
        
    Returns:
        List of segment dictionaries with start, end, and text
    """
    segments = []
    lines = srt_content.split('\n')
    i = 0
    
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        
        # Try to parse sequence number
        try:
            int(lines[i].strip())  # Just check if it's a number
            i += 1
            
            # Parse timestamp line
            if i < len(lines):
                timestamp_line = lines[i].strip()
                import re
                timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                
                if timestamp_match:
                    start_time_str, end_time_str = timestamp_match.groups()
                    
                    # Convert timestamp to seconds
                    def srt_time_to_seconds(time_str):
                        h, m, s = time_str.split(':')
                        s, ms = s.split(',')
                        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                    
                    start_time = srt_time_to_seconds(start_time_str) + time_offset
                    end_time = srt_time_to_seconds(end_time_str) + time_offset
                    
                    i += 1
                    
                    # Collect text lines
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = " ".join(text_lines)
                    
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "text": text
                    })
            else:
                i += 1
        except:
            # Skip this line if it's not a valid sequence number
            i += 1
    
    return segments

def transcribe_chunks_parallel(chunks, model_type=None, model_version=None):
    """
    Transcribe chunks in parallel using Replicate API
    
    Args:
        chunks: List of chunk info dictionaries
        model_type: Type of model to use ("openai" or "fast")
        model_version: Specific model version (overrides model_type)
        
    Returns:
        List of transcription results
    """
    logger.info(f"Transcribing {len(chunks)} chunks in parallel with max concurrency {MAX_CONCURRENT_TRANSCRIPTIONS}")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS) as executor:
        futures = []
        
        for chunk in chunks:
            # Skip chunks that failed to upload
            if chunk.get("upload_failed", False):
                logger.warning(f"Skipping chunk {chunk['id']} due to upload failure")
                continue
                
            future = executor.submit(
                transcribe_chunk_replicate,
                chunk,
                model_type,
                model_version
            )
            futures.append((future, chunk))
        
        # Track progress
        completed = 0
        total = len(futures)
        
        for future, chunk in futures:
            try:
                result = future.result()
                if result:
                    results.append(result)
                    
                # Update progress
                completed += 1
                progress = int((completed / total) * 100)
                logger.info(f"Transcription progress: {progress}% ({completed}/{total} chunks)")
                
            except Exception as e:
                logger.error(f"Error getting transcription result for chunk {chunk['id']}: {e}")
    
    # Sort results by chunk_id to maintain order
    results.sort(key=lambda x: x.get("chunk_id", 0))
    return results

def combine_transcription_results(results, video_info):
    """
    Combine transcription results into a single document
    
    Args:
        results: List of transcription results
        video_info: Dictionary with video metadata
        
    Returns:
        Combined transcript document
    """
    logger.info(f"Combining transcription results: {len(results)} chunks to process")
    logger.info(f"Video info: {video_info}")
    
    # Debug log of results structure
    for i, result in enumerate(results[:3]):  # Log first 3 results as sample
        logger.info(f"Result {i} structure: {list(result.keys())}")
        if 'segments' in result:
            logger.info(f"Result {i} has {len(result['segments'])} segments")
            if result['segments']:
                logger.info(f"First segment sample: {result['segments'][0]}")
        else:
            logger.info(f"Result {i} has NO segments!")
    
    # First, sort results by start_time to ensure proper order
    results.sort(key=lambda x: x.get("start_time", 0))
    
    # Combine all segments
    all_segments = []
    for result in results:
        if not result.get("failed", False) and "segments" in result:
            all_segments.extend(result["segments"])
    
    logger.info(f"Combined {len(all_segments)} segments from all results")
    
    # If no segments were found, let's attempt to create segments from transcript_text
    if not all_segments:
        logger.warning("No segments found in results, attempting to create segments from transcript_text")
        for result in results:
            if not result.get("failed", False) and "transcript_text" in result:
                # Create a single segment for this chunk
                all_segments.append({
                    "start": result.get("start_time", 0),
                    "end": result.get("start_time", 0) + result.get("duration", 300),
                    "text": result["transcript_text"]
                })
        logger.info(f"Created {len(all_segments)} fallback segments from transcript_text")
    
    # Still no segments? Log the entire results for debugging
    if not all_segments:
        logger.error(f"Still no segments after fallback, dumping results:")
        for i, result in enumerate(results):
            logger.error(f"Full Result {i}: {result}")
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x["start"])
    
    # Create 1-minute chunks for better context
    segments_for_vector = []
    current_chunk = []
    current_start = None
    current_end = None
    chunk_duration = 60.0  # 1 minute chunks
    
    for seg in all_segments:
        if not current_chunk:
            current_start = seg["start"]
        current_chunk.append(seg["text"].strip())
        current_end = seg["end"]
        
        if (current_end - current_start) >= chunk_duration:
            chunk_text = " ".join(current_chunk)
            segments_for_vector.append({
                "start_time": current_start,
                "end_time": current_end,
                "text": chunk_text
            })
            current_chunk = []
            current_start = None
    
    # Add leftover chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        segments_for_vector.append({
            "start_time": current_start,
            "end_time": current_end,
            "text": chunk_text
        })
    
    logger.info(f"Created {len(segments_for_vector)} vector-ready segments")
    
    # Prepare final output
    final_output = []
    
    for segment in segments_for_vector:
        final_output.append({
            "text": f"Spoken Words: {segment['text']}",
            "metadata": {
                "title": video_info.get("title", ""),
                "video_url": video_info.get("url", ""),
                "video_id": video_info.get("id", ""),
                "timestamp_start": segment["start_time"],
                "timestamp_end": segment["end_time"],
                "upload_date": video_info.get("upload_date", "")
            }
        })
    
    return final_output

def extract_video_id(video_url):
    """
    Extract video ID from YouTube URL
    
    Args:
        video_url: YouTube URL
        
    Returns:
        Video ID as string
    """
    video_id = None
    if video_url:
        # For youtube.com URLs
        if 'youtube.com' in video_url:
            if 'v=' in video_url:
                url_parts = video_url.split("v=")
                if len(url_parts) > 1:
                    video_id = url_parts[1].split("&")[0]
        # For youtu.be URLs
        elif 'youtu.be' in video_url:
            parsed_url = urlparse(video_url)
            video_id = parsed_url.path.lstrip('/')
    
    return video_id

def publish_progress(message_id, video_url, status, progress_percent, 
                    current_stage, stage_progress_percent, 
                    start_time, error=None):
    """
    Publish progress update to Pub/Sub
    
    Args:
        message_id: Original message ID
        video_url: YouTube video URL
        status: Status message
        progress_percent: Overall progress percentage
        current_stage: Current processing stage
        stage_progress_percent: Progress percentage of current stage
        start_time: Processing start time
        error: Error message, if any
    """
    if not PROGRESS_TOPIC_ID:
        logger.info("Progress reporting disabled (PROGRESS_TOPIC_ID not set)")
        return
    
    logger.info(f"Preparing progress update: {current_stage} at {progress_percent}%")
        
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, PROGRESS_TOPIC_ID)
        
        # Extract video_id from URL
        video_id = extract_video_id(video_url)
        
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
        pub_message_id = future.result()
        
        # Log at INFO level so it appears in standard logs
        logger.info(f"Published progress update to {PROGRESS_TOPIC_ID}: status={status}, stage={current_stage}, progress={progress_percent}%")
        # Keep the detailed debug log for troubleshooting if needed
        logger.debug(f"Full progress message: {progress_message}")
        
    except Exception as e:
        logger.error(f"Error publishing progress update: {e}")

def publish_completion_result(message_id, video_url, status, transcript_location=None, error=None):
    """
    Publish completion result to the results topic for memory import processing
    
    Args:
        message_id: Original message ID
        video_url: YouTube video URL
        status: 'completed' or 'failed'
        transcript_location: GCS path to transcript JSON file (for successful completions)
        error: Error message (for failed completions)
    """
    if not RESULTS_TOPIC_ID:
        logger.info("Results publishing disabled (RESULTS_TOPIC_ID not set)")
        return
    
    logger.info(f"Publishing completion result: status={status}, location={transcript_location}")
        
    try:
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(PROJECT_ID, RESULTS_TOPIC_ID)
        
        # Extract video_id from URL
        video_id = extract_video_id(video_url)
        
        # Construct completion message
        completion_message = {
            "message_id": message_id,
            "video_id": video_id,
            "video_url": video_url,
            "status": status,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        # Add transcript location for successful completions
        if status == "completed" and transcript_location:
            completion_message["transcript_location"] = transcript_location
        
        # Add error for failed completions
        if status == "failed" and error:
            completion_message["error"] = error
        
        # Publish message
        message_json = json.dumps(completion_message)
        future = publisher.publish(topic_path, data=message_json.encode('utf-8'))
        pub_message_id = future.result()
        
        logger.info(f"Published completion result to {RESULTS_TOPIC_ID}: status={status}, video_id={video_id}")
        logger.debug(f"Full completion message: {completion_message}")
        
    except Exception as e:
        logger.error(f"Error publishing completion result: {e}")

def delete_gcs_chunks(bucket_name, message_id):
    """
    Delete audio chunks from GCS after successful transcription
    
    Args:
        bucket_name: Name of the GCS bucket
        message_id: ID of the message for identifying folder structure
        
    Returns:
        Number of deleted blobs
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # List all blobs in the chunks directory
        prefix = f"{message_id}/chunks/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            logger.warning(f"No audio chunks found to delete in gs://{bucket_name}/{prefix}")
            return 0
            
        logger.info(f"Deleting {len(blobs)} audio chunks from gs://{bucket_name}/{prefix}")
        
        # Delete blobs in batches
        deleted_count = 0
        for blob in blobs:
            try:
                blob.delete()
                deleted_count += 1
            except Exception as blob_error:
                logger.warning(f"Failed to delete blob {blob.name}: {blob_error}")
        
        logger.info(f"Successfully deleted {deleted_count} audio chunks")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting audio chunks: {e}")
        # Don't raise the exception to avoid failing the whole process
        return 0

def upload_results(result_file_path, bucket_name, message_id):
    """
    Upload final results to GCS
    
    Args:
        result_file_path: Path to the result file
        bucket_name: Name of the GCS bucket
        message_id: ID of the message for creating folder structure
        
    Returns:
        Public URL of the uploaded file
    """
    try:
        file_name = os.path.basename(result_file_path)
        destination_blob_name = f"{message_id}/results/{file_name}"
        
        # Upload to GCS
        result_url = upload_to_gcs(result_file_path, bucket_name, destination_blob_name)
        logger.info(f"Uploaded results to {result_url}")
        
        return result_url
        
    except Exception as e:
        logger.error(f"Error uploading results: {e}")
        raise

def process_message(message):
    """
    Process a single Pub/Sub message
    
    Args:
        message: Pub/Sub message
    """
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
        
        # Get model type from message data or use default
        model_type = data.get('model_type', DEFAULT_MODEL_TYPE)
        model_version = data.get('model_version')
        
        # Extract custom parameters
        use_wav = data.get('use_wav', USE_WAV_FORMAT)
        
        # Create a unique ID for this processing run
        run_id = str(uuid.uuid4())
        
        # Video metadata
        video_info = {
            "url": video_url,
            "id": extract_video_id(video_url),
            "title": "",
            "upload_date": ""
        }
        
        # Create a temporary directory for processing
        work_dir = tempfile.mkdtemp()
        temp_dirs.append(work_dir)
        
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
        
        # 1. Download video
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=5,
            current_stage="download",
            stage_progress_percent=20,
            start_time=start_time
        )
        
        audio_path, desc_path = download_video(video_url, work_dir)
        
        # Extract video title from description file
        try:
            with open(desc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.lower().startswith('title:'):
                        video_info["title"] = line.split(":", 1)[1].strip()
                        break
                    if line.lower().startswith('upload date:'):
                        video_info["upload_date"] = line.split(":", 1)[1].strip()
        except Exception as e:
            logger.warning(f"Error reading description file: {e}")
        
        # Download complete progress update
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=15,
            current_stage="download",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # 2. Split audio into chunks
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=20,
            current_stage="chunking",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        chunks = split_audio(audio_path, work_dir, chunk_length=CHUNK_LENGTH_SECONDS, use_wav=use_wav)
        
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=30,
            current_stage="chunking",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # 3. Upload chunks to GCS
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=35,
            current_stage="uploading",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        chunks = upload_chunks(chunks, AUDIO_CHUNKS_BUCKET, message_id)
        
        # Check if all chunks were uploaded successfully
        failed_uploads = sum(1 for chunk in chunks if chunk.get("upload_failed", False))
        if failed_uploads > 0:
            logger.warning(f"{failed_uploads} of {len(chunks)} chunks failed to upload")
            
            if failed_uploads == len(chunks):
                raise Exception("All chunks failed to upload")
        
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=45,
            current_stage="uploading",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # 4. Transcribe chunks
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=50,
            current_stage="transcription",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        # Create a progress tracking callback
        def transcription_progress_callback(progress):
            publish_progress(
                message_id=message_id,
                video_url=video_url,
                status="processing",
                progress_percent=int(50 + progress * 0.4),  # 50-90% of overall progress
                current_stage="transcription",
                stage_progress_percent=progress,
                start_time=start_time
            )
        
        # Start a thread to periodically update progress during transcription
        stop_progress_monitor = threading.Event()
        
        def monitor_progress():
            progress = 0
            while not stop_progress_monitor.is_set() and progress < 100:
                transcription_progress_callback(progress)
                logger.info(f"Transcription background progress update: {progress}% complete")
                progress += 5  # Increment by 5% each time
                progress = min(progress, 95)  # Never reach 100% in monitoring thread
                stop_progress_monitor.wait(30)  # Update every 30 seconds
        
        progress_thread = threading.Thread(target=monitor_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Transcribe chunks
            transcription_results = transcribe_chunks_parallel(
                chunks, 
                model_type=model_type, 
                model_version=model_version
            )
            
            # Signal progress monitor to stop
            stop_progress_monitor.set()
            
            # Wait for thread to terminate
            progress_thread.join(timeout=5)
            
            # Check if we have results
            if not transcription_results:
                raise Exception("No transcription results were returned")
            
            # Count failed transcriptions
            failed_transcriptions = sum(1 for result in transcription_results if result.get("failed", False))
            if failed_transcriptions > 0:
                logger.warning(f"{failed_transcriptions} of {len(transcription_results)} transcription tasks failed")
            
        except Exception as e:
            # Make sure to signal the progress monitor to stop
            stop_progress_monitor.set()
            if progress_thread.is_alive():
                progress_thread.join(timeout=5)
            raise
        
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=90,
            current_stage="transcription",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # 5. Combine results and format output
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=92,
            current_stage="processing",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        # Log information about the transcription results
        logger.info(f"Model type used: {model_type}")
        logger.info(f"Transcription results count: {len(transcription_results)}")
        
        final_transcript = combine_transcription_results(transcription_results, video_info)
        
        # Log results of the combination
        logger.info(f"Final transcript entries count: {len(final_transcript)}")
        if not final_transcript:
            logger.error("CRITICAL: Final transcript is empty!")
        else:
            logger.info(f"First transcript entry sample: {final_transcript[0]}")
        
        # Save combined results
        result_file_path = os.path.join(work_dir, f"transcript_{video_info['id'] or 'result'}.json")
        with open(result_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_transcript, f, indent=2)
        
        logger.info(f"Saved results to {result_file_path}")
        
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=95,
            current_stage="processing",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # 6. Upload results to GCS
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=97,
            current_stage="finalizing",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        result_url = upload_results(result_file_path, BUCKET_NAME, message_id)
        
        # 7. Clean up audio chunks from GCS
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=88,
            current_stage="cleanup",
            stage_progress_percent=0,
            start_time=start_time
        )
        
        # Delete audio chunks now that we have the final transcript
        deleted_count = delete_gcs_chunks(AUDIO_CHUNKS_BUCKET, message_id)
        logger.info(f"Cleanup completed: {deleted_count} audio chunks deleted from GCS")
        
        # Final progress update - Transcription completed (90%)
        # Memory import will handle 90% → 100%
        publish_progress(
            message_id=message_id,
            video_url=video_url,
            status="processing",
            progress_percent=90,
            current_stage="transcription_completed",
            stage_progress_percent=100,
            start_time=start_time
        )
        
        # Publish completion result for memory import processing
        transcript_gcs_path = f"gs://{BUCKET_NAME}/{message_id}/results/{os.path.basename(result_file_path)}"
        publish_completion_result(
            message_id=message_id,
            video_url=video_url,
            status="completed",
            transcript_location=transcript_gcs_path
        )
        
        # Acknowledge the message
        message.ack()
        logger.info(f"Successfully processed message {message_id}")
        logger.info(f"Results available at: {result_url}")
        logger.info(f"Audio chunks cleanup: {deleted_count} files removed from GCS")
        
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
            
            # Publish failure result for memory import processing
            publish_completion_result(
                message_id=message_id,
                video_url=video_url,
                status="failed",
                error=str(e)
            )
            
            # Don't retry if we made significant progress
            if 'transcription_results' in locals():
                logger.warning("Error occurred after significant processing, acknowledging message to avoid retry")
                message.ack()
            else:
                logger.info("Not acknowledging message to allow redelivery")
        else:
            # If we can't even extract the message ID and URL, just ack the message
            logger.warning("Couldn't extract message_id or video_url, acknowledging to prevent infinite retry")
            message.ack()
    finally:
        # Cleanup temp directories
        for dir_path in temp_dirs:
            try:
                if dir_path and os.path.exists(dir_path):
                    logger.info(f"Cleaning up directory: {dir_path}")
                    import shutil
                    shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Error cleaning up directory {dir_path}: {e}")

def check_environment():
    """
    Check if all required environment variables are set
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = ['PROJECT_ID', 'SUBSCRIPTION_ID', 'BUCKET_NAME', 'REPLICATE_API_TOKEN']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables and restart the service")
        return False
    
    # Check model environment variables
    if not OPENAI_WHISPER:
        logger.warning("OPENAI_WHISPER environment variable not set")
    
    if not FAST_WHISPER:
        logger.warning("FAST_WHISPER environment variable not set")
    
    if not OPENAI_WHISPER and not FAST_WHISPER:
        logger.error("No model environment variables set")
        return False
    
    logger.info(f"Environment check passed")
    logger.info(f"PROJECT_ID: {PROJECT_ID}")
    logger.info(f"SUBSCRIPTION_ID: {SUBSCRIPTION_ID}")
    logger.info(f"BUCKET_NAME: {BUCKET_NAME}")
    logger.info(f"AUDIO_CHUNKS_BUCKET: {AUDIO_CHUNKS_BUCKET}")
    logger.info(f"CHUNK_LENGTH_SECONDS: {CHUNK_LENGTH_SECONDS}")
    logger.info(f"MAX_CONCURRENT_UPLOADS: {MAX_CONCURRENT_UPLOADS}")
    logger.info(f"MAX_CONCURRENT_TRANSCRIPTIONS: {MAX_CONCURRENT_TRANSCRIPTIONS}")
    logger.info(f"DEFAULT_MODEL_TYPE: {DEFAULT_MODEL_TYPE}")
    logger.info(f"USE_WAV_FORMAT: {USE_WAV_FORMAT}")
    
    # Check model configurations
    try:
        openai_model_id, openai_version, _ = get_model_info("openai")
        logger.info(f"OpenAI Whisper model: {openai_model_id}:{openai_version}")
    except:
        logger.warning("OpenAI Whisper model not properly configured")
    
    try:
        fast_model_id, fast_version, _ = get_model_info("fast")
        logger.info(f"Fast Whisper model: {fast_model_id}:{fast_version}")
    except:
        logger.warning("Fast Whisper model not properly configured")
    
    # Check progress topic configuration
    if PROGRESS_TOPIC_ID:
        logger.info(f"Progress reporting enabled to topic: {PROGRESS_TOPIC_ID}")
        # Test if we can create a publisher client for the topic
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(PROJECT_ID, PROGRESS_TOPIC_ID)
            logger.info(f"Progress updates will be published to: {topic_path}")
        except Exception as e:
            logger.error(f"Error setting up progress publisher: {e}")
    else:
        logger.warning("Progress reporting disabled (PROGRESS_TOPIC_ID not set)")
    
    # Check results topic configuration
    if RESULTS_TOPIC_ID:
        logger.info(f"Results publishing enabled to topic: {RESULTS_TOPIC_ID}")
        # Test if we can create a publisher client for the topic
        try:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(PROJECT_ID, RESULTS_TOPIC_ID)
            logger.info(f"Completion results will be published to: {topic_path}")
        except Exception as e:
            logger.error(f"Error setting up results publisher: {e}")
    else:
        logger.warning("Results publishing disabled (RESULTS_TOPIC_ID not set)")
    
    return True

def main():
    """Main processing loop"""
    global service_ready, service_healthy
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    logger.info("Starting video transcription service")
    
    # Start health server first
    try:
        health_server = start_health_server()
        logger.info("Health endpoints available at /health and /ready")
    except Exception as e:
        logger.error(f"Failed to start health server: {e}")
        sys.exit(1)
    
    # Create Pub/Sub subscriber client
    try:
        subscriber = pubsub_v1.SubscriberClient()
        subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        
        # Configure flow control to limit concurrent messages
        flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES)
        
        def callback(message):
            process_message(message)
        
        # Subscribe to the subscription
        streaming_pull_future = subscriber.subscribe(
            subscription_path, callback=callback, flow_control=flow_control
        )
        
        # Mark service as ready once Pub/Sub subscription is active
        service_ready = True
        logger.info(f"Service ready - listening for messages on {subscription_path}")
        
    except Exception as e:
        logger.error(f"Failed to set up Pub/Sub subscription: {e}")
        service_healthy = False
        sys.exit(1)
    
    try:
        # Keep the main thread from exiting
        streaming_pull_future.result()
    except KeyboardInterrupt:
        service_ready = False
        service_healthy = False
        streaming_pull_future.cancel()
        logger.info("Subscription canceled")
    except Exception as e:
        logger.error(f"Streaming pull error: {e}")
        service_ready = False
        service_healthy = False
        streaming_pull_future.cancel()
        raise

if __name__ == "__main__":
    # Parse command line arguments for local testing
    parser = argparse.ArgumentParser(description='Video Transcription Service')
    parser.add_argument('--test', action='store_true', help='Process a test video instead of listening for Pub/Sub messages')
    parser.add_argument('--video-url', type=str, help='YouTube URL to process in test mode')
    parser.add_argument('--model-type', choices=['openai', 'fast'], help='Model type to use (openai or fast)')
    
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in test mode")
        if not args.video_url:
            logger.error("No video URL provided for test mode. Use --video-url")
            sys.exit(1)
        
        # Check if REPLICATE_API_TOKEN is set
        if not os.environ.get("REPLICATE_API_TOKEN"):
            logger.error("REPLICATE_API_TOKEN environment variable not set")
            sys.exit(1)
        
        try:
            # Simulate a message
            class TestMessage:
                def __init__(self, url):
                    self.message_id = f"test-{int(time.time())}"
                    message_data = {
                        "video_url": url
                    }
                    if args.model_type:
                        message_data["model_type"] = args.model_type
                    self.data = json.dumps(message_data).encode('utf-8')
                
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