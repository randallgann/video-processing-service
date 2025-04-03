import os
import subprocess
import json
import time
import math
import re
import argparse
import sys
import tempfile
import requests
import replicate
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_audio_length(audio_file_path):
    """
    Uses ffprobe to get the duration of the audio file in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    duration_str = result.stdout.strip()
    return float(duration_str)

def split_audio(audio_file_path, chunk_length=300, max_duration=None, use_wav=False):
    """
    Splits the audio into smaller chunks of `chunk_length` seconds each.
    Returns a list of chunk file paths and the total length of the audio.
    
    Args:
        audio_file_path: Path to audio file
        chunk_length: Length of each chunk in seconds
        max_duration: Maximum duration to process
        use_wav: Whether to output WAV format (better compatibility with APIs)
    """
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_dir = f"{base_name}_chunks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_length = get_audio_length(audio_file_path)

    # If max_duration is specified and less than total_length, use it.
    if max_duration is not None and max_duration < total_length:
        total_length = max_duration

    num_chunks = math.ceil(total_length / chunk_length)

    # Determine output format
    output_ext = "wav" if use_wav else "mp3"
    print(f"Using {output_ext.upper()} format for audio chunks")

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_length

        duration = min(chunk_length, total_length - start)
        
        # If duration <= 0, no more valid chunks left
        if duration <= 0:
            break

        chunk_output = os.path.join(output_dir, f"chunk_{i:03d}.{output_ext}")
        
        # ffmpeg command to slice audio
        if use_wav:
            # For WAV, we need to decode and re-encode (can't use copy codec)
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", audio_file_path,
                "-ss", str(start),
                "-t", str(chunk_length),
                "-acodec", "pcm_s16le",  # Standard 16-bit PCM WAV format
                "-ar", "16000",          # 16kHz sample rate (good for speech)
                "-ac", "1",              # Mono channel
                chunk_output
            ]
        else:
            # For MP3, we can try to copy codec for speed
            cmd = [
                "ffmpeg",
                "-y",  # overwrite
                "-i", audio_file_path,
                "-ss", str(start),
                "-t", str(chunk_length),
                "-c", "copy",
                chunk_output
            ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(chunk_output):
            chunk_paths.append(chunk_output)
        else:
            break

    return chunk_paths, total_length

def load_episode_description(desc_file_path):
    """
    Load episode metadata and timestamps from a file.
    Expected top lines:
    Upload Date: MM-DD-YYYY
    Episode_number: ###
    Title: Some Title

    Then timestamp lines, blank line, and rest of description.

    Returns:
    episode_description (str), episode_topics (list), date_str (str), ep_num (int), ep_title (str)
    """
    with open(desc_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # First, parse the metadata lines (Upload Date, Episode_number, Title)
    date_str = None
    ep_num = None
    ep_title = None

    # We'll read lines until we find a blank line or timestamps start
    metadata_lines = []
    idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            idx = i + 1
            break
        metadata_lines.append(line.strip())

    # Extract known fields from metadata_lines
    for m_line in metadata_lines:
        if m_line.lower().startswith("upload date:"):
            # Format: Upload Date: MM-DD-YYYY
            # Normalize or just store as is
            date_str = m_line.split(":", 1)[1].strip()
        elif m_line.lower().startswith("episode_number:"):
            # Format: Episode_number: ###
            ep_num_str = m_line.split(":", 1)[1].strip()
            try:
                ep_num = int(ep_num_str)
            except ValueError:
                ep_num = ep_num_str
        elif m_line.lower().startswith("title:"):
            # Format: Title: ...
            ep_title = m_line.split(":", 1)[1].strip()

    # Now parse timestamps and the episode description
    timestamps = []
    description_lines = []

    for line in lines[idx:]:
        stripped = line.strip()
        if re.match(r'^\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?(?:\s*-\s*\d{2}:\d{2}:\d{2})?', stripped):
            # Assume this is a timestamp if it contains a time marker
            parts = stripped.split(" ", 1)
            time_marker = parts[0].split("-")[0].strip()  # Take first part if there's a range
            topic = parts[1] if len(parts) > 1 else ""
            timestamps.append({"time": time_marker, "topic": topic})
        elif stripped:  # Non-blank line
            # Add this to the description
            description_lines.append(stripped)

    episode_description = "\n".join(description_lines)

    return episode_description, timestamps, date_str, ep_num, ep_title

def time_str_to_seconds(t_str):
    """
    Converts a time string like 'MM:SS' or 'HH:MM:SS' to total seconds.
    """
    # Check if the string matches the pattern of a timestamp
    if not re.match(r'^\d+(?::\d+)*$', t_str):
        return 0
    parts = t_str.split(':')
    try:
        parts = list(map(int, parts))
        while len(parts) < 3:
            parts.insert(0, 0)  # Pad to hh:mm:ss if needed
        h, m, s = parts
        return h*3600 + m*60 + s
    except ValueError:
        return 0

def assign_topics_to_chunks(chunks, episode_topics):
    """
    Assign topics to each chunk based on the episode_topics time markers.
    A topic is assigned to all chunks starting from its timestamp
    until the next timestamp is reached.
    """
    # Convert episode topics to a sorted list of (seconds, topic)
    topic_map = [(time_str_to_seconds(et["time"]), et["topic"]) for et in episode_topics]
    topic_map.sort(key=lambda x: x[0])
    print("Topic Map:", topic_map)

    # Initialize variables to track the active topic
    current_topic = "No specific topic"
    topic_index = 0

    for chunk in chunks:
        chunk_start = chunk["start_time"]

        # Advance the topic_index if we've reached the next topic's time
        while topic_index < len(topic_map) and chunk_start >= topic_map[topic_index][0]:
            current_topic = topic_map[topic_index][1]
            topic_index += 1

        # Assign the current topic to the chunk
        chunk["topic"] = current_topic

    return chunks

def get_file_url(file_path):
    """
    Get a URL for the file that can be accessed by the Replicate API.
    This will attempt to use the mini_audio_server to serve the file.
    """
    try:
        # Import the mini_audio_server module
        import mini_audio_server
        
        # Check if server is already running, if not start it
        try:
            # Try to get a URL - this will fail if server is not running
            url = mini_audio_server.get_url_for_file(file_path)
        except RuntimeError:
            # Server not running, start it in the parent directory of the audio file
            parent_dir = os.path.dirname(os.path.abspath(file_path))
            server = mini_audio_server.start_server(directory=parent_dir, port=8000)
            url = mini_audio_server.get_url_for_file(file_path)
            
        print(f"Serving audio file at: {url}")
        return url
    
    except ImportError:
        print("mini_audio_server module not found. Please ensure it's in the same directory.")
        return None
    except Exception as e:
        print(f"Error setting up audio server: {e}")
        return None

def process_chunk_replicate(chunk_info, model_version=None, language="en"):
    """
    Process a single audio chunk using Replicate's Whisper API.
    
    Args:
        chunk_info: Dictionary with keys: 'chunk_id', 'chunk_path', 'chunk_length'
        model_version: Replicate model version to use
        language: Language code for transcription
    
    Returns:
        Processed segments with adjusted timestamps
    """
    chunk_id = chunk_info['chunk_id']
    chunk_path = chunk_info['chunk_path']
    chunk_length = chunk_info['chunk_length']
    
    print(f"Processing chunk {chunk_id+1}: {chunk_path}")
    
    # Create a temporary directory for uploads
    temp_dir = tempfile.mkdtemp()
    
    # Maximum number of retries for API calls
    max_retries = 3
    retry_delay = 5  # seconds
    
    # Check if REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN environment variable not set")
        
    print(f"Using Replicate API with model version: {model_version}")
    
    # Default model version if not provided
    if not model_version:
        model_version = "8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
    
    # Get a URL for the audio file using the mini_audio_server
    audio_url = get_file_url(chunk_path)
    
    if not audio_url:
        raise ValueError("Failed to create audio URL. Make sure mini_audio_server.py is accessible.")
    
    # Prepare transcription segments
    all_segments = []
    
    # Try to call Replicate API with retries
    for attempt in range(max_retries):
        try:
            # Call Replicate API with more detailed debug information
            print(f"Calling Replicate API with URL: {audio_url}")
            
            output = replicate.run(
                f"openai/whisper:{model_version}",
                input={
                    "audio": audio_url,
                    "language": language,
                    "translate": False,
                    "temperature": 0,
                    "transcription": "srt",  # Get SRT format for timestamp information
                    "suppress_tokens": "-1",
                    "condition_on_previous_text": True
                }
            )
            
            # Debug the output
            print(f"Replicate API response type: {type(output)}")
            if output is None:
                print("WARNING: Received None response from Replicate API")
                raise ValueError("Empty response from Replicate API")
                
            if isinstance(output, dict):
                print(f"Response keys: {list(output.keys())}")
            elif isinstance(output, str):
                print(f"Received string response (first 100 chars): {output[:100]}...")
                # If it's a string, it might be the transcription directly
                srt_content = output
            else:
                print(f"Unexpected response type: {type(output)}")
                
            # Handle different response formats
            if isinstance(output, dict):
                srt_content = output.get("transcription", "")
                if not srt_content and "srt_file" in output:
                    # Some versions might return a URL to the SRT file
                    srt_url = output.get("srt_file")
                    if srt_url:
                        print(f"Downloading SRT from URL: {srt_url}")
                        try:
                            srt_response = requests.get(srt_url)
                            srt_content = srt_response.text
                        except Exception as e:
                            print(f"Error downloading SRT file: {e}")
            elif isinstance(output, str):
                # If it's a string, it might be the transcription directly
                srt_content = output
            
            # Simple SRT parser
            segments = []
            lines = srt_content.split('\n')
            i = 0
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue
                
                # Try to parse sequence number
                try:
                    seq_num = int(lines[i].strip())
                    i += 1
                    
                    # Parse timestamp line
                    if i < len(lines):
                        timestamp_line = lines[i].strip()
                        timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                        
                        if timestamp_match:
                            start_time_str, end_time_str = timestamp_match.groups()
                            
                            # Convert timestamp to seconds
                            def srt_time_to_seconds(time_str):
                                h, m, s = time_str.split(':')
                                s, ms = s.split(',')
                                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                            
                            start_time = srt_time_to_seconds(start_time_str)
                            end_time = srt_time_to_seconds(end_time_str)
                            
                            i += 1
                            
                            # Collect text lines
                            text_lines = []
                            while i < len(lines) and lines[i].strip():
                                text_lines.append(lines[i].strip())
                                i += 1
                            
                            text = " ".join(text_lines)
                            
                            # Adjust times for chunk position
                            offset = chunk_id * chunk_length
                            segments.append({
                                "start": start_time + offset,
                                "end": end_time + offset,
                                "text": text
                            })
                    else:
                        i += 1
                except:
                    # Skip this line if it's not a valid sequence number
                    i += 1
            
            all_segments = segments
            break  # Success, exit retry loop
            
        except Exception as e:
            print(f"Error calling Replicate API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts")
                raise
    
    return all_segments

def transcribe_chunks_replicate(chunk_paths, total_length, chunk_length=300, model_version=None, 
                               max_concurrent=5, progress_file=None):
    """
    Transcribe chunks in parallel using Replicate API.
    
    Args:
        chunk_paths: List of paths to audio chunk files
        total_length: Total audio length in seconds
        chunk_length: Length of each chunk in seconds
        model_version: Replicate model version
        max_concurrent: Maximum number of concurrent API calls
        progress_file: Optional path to a file for writing progress updates
    
    Returns:
        List of transcribed segments with adjusted timestamps
    """
    # Prepare chunk info for parallel processing
    chunk_infos = []
    for i, chunk_path in enumerate(chunk_paths):
        chunk_infos.append({
            'chunk_id': i,
            'chunk_path': chunk_path,
            'chunk_length': chunk_length
        })
    
    # If progress file is specified, initialize it
    if progress_file:
        try:
            with open(progress_file, 'w') as f:
                f.write("0")
        except Exception as e:
            print(f"Error initializing progress file: {e}")
    
    # Process chunks in parallel with limited concurrency
    all_segments = []
    processed_chunks = 0
    total_chunks = len(chunk_paths)
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Process chunks in batches
        results = []
        for chunk_info in chunk_infos:
            future = executor.submit(process_chunk_replicate, chunk_info, model_version)
            results.append(future)
        
        # As results complete, collect them
        for i, future in enumerate(tqdm(results, desc="Processing chunks")):
            try:
                segments = future.result()
                if segments:
                    all_segments.extend(segments)
                
                # Update progress counter
                processed_chunks += 1
                
                # Update progress file if specified
                if progress_file:
                    try:
                        with open(progress_file, 'w') as f:
                            f.write(str(processed_chunks))
                    except Exception as e:
                        print(f"Error writing to progress file: {e}")
                
                # Calculate and print progress
                progress_percent = (processed_chunks / total_chunks) * 100
                print(f"Processed chunk {processed_chunks}/{total_chunks}. Progress: {progress_percent:.2f}%")
                
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def chunk_segments_for_vector(all_segments, chunk_duration=60.0):
    """
    Combine raw segments into larger ~1-minute chunks for vector store ingestion.
    """
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for seg in all_segments:
        if not current_chunk:
            current_start = seg["start"]
        current_chunk.append(seg["text"].strip())
        current_end = seg["end"]

        if (current_end - current_start) >= chunk_duration:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "start_time": current_start,
                "end_time": current_end,
                "text": chunk_text
            })
            current_chunk = []
            current_start = None

    # Add leftover chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "start_time": current_start,
            "end_time": current_end,
            "text": chunk_text
        })

    return chunks

def main(audio_path, desc_path, max_duration=None, model_version=None, max_concurrent=5, use_wav=False):
    start_time = time.time()
    
    # 1. Load episode description and topics
    print("Loading episode metadata...")
    episode_description, episode_topics, date_str, ep_num, ep_title = load_episode_description(desc_path)
    print(f"Episode Title: {ep_title}")
    print(f"Episode Topics found: {len(episode_topics)}")

    raw_words = ep_title.split()
    cleaned_words = []
    for word in raw_words:
        cleaned_word = re.sub(r'[^a-zA-Z0-9]+', '', word)
        if cleaned_word:
            cleaned_words.append(cleaned_word)
    title_topics = ",".join(cleaned_words)

    # 2. Split audio into chunks and get total_length
    print("Splitting audio into chunks...")
    # Check for environment variable override
    env_use_wav = os.environ.get('USE_WAV_FORMAT', '').lower() in ('true', '1', 'yes')
    use_wav = use_wav or env_use_wav
    
    chunk_paths, total_length = split_audio(audio_path, chunk_length=300, max_duration=max_duration, use_wav=use_wav)
    
    # If list-chunks-only is specified, just print the chunks and exit
    if args.list_chunks_only:
        for chunk_path in chunk_paths:
            print(chunk_path)
        sys.exit(0)
    
    # 3. Transcribe chunks using Replicate API
    print(f"Starting transcription using Replicate API...")
    print(f"Number of chunks: {len(chunk_paths)}")
    
    all_segments = transcribe_chunks_replicate(
        chunk_paths, 
        total_length, 
        chunk_length=300, 
        model_version=model_version, 
        max_concurrent=max_concurrent,
        progress_file=args.progress_file
    )

    # 4. Combine segments into ~1-minute chunks for better vector context
    print("Combining segments into chunks...")
    chunks = chunk_segments_for_vector(all_segments, chunk_duration=60.0)

    # 5. Assign topics to each chunk
    for ch in chunks:
        print(f"Chunk start: {ch['start_time']} end: {ch['end_time']}")
    chunks = assign_topics_to_chunks(chunks, episode_topics)

    # Prepare final output in the requested format:
    # A list of objects each with "text" and "metadata"
    output_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
    final_output = []
    for ch in chunks:
        final_output.append({
            "text": f"Topic: {ch['topic']} Spoken Words: {ch['text']}",
            "metadata": {
                "date": date_str,
                "episode_number": ep_num,
                "episode_title": ep_title,
                "timestamp_start": ch["start_time"],
                "timestamp_end": ch["end_time"],
                "chunk_topic": ch["topic"],
                "topics": title_topics
            }
        })

    # 6. Save final outputs
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)

    end_time = time.time()
    processing_time = end_time - start_time
    audio_duration = total_length
    speedup = audio_duration / processing_time
    
    print("\n====== Transcription Summary ======")
    print(f"Transcription complete in {processing_time:.2f} seconds!")
    print(f"Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.2f} minutes)")
    print(f"Processing speedup: {speedup:.2f}x realtime")
    print(f"Output file: {output_file}")
    print("==================================")

if __name__ == "__main__":
    # Parse command line arguments if any
    parser = argparse.ArgumentParser(description="Transcribe audio using Replicate's Whisper API")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--desc", type=str, help="Path to description file")
    parser.add_argument("--model-version", type=str, 
                        default="8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e", 
                        help="Replicate model version to use")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent API calls (default: 5)")
    parser.add_argument("--max-duration", type=float, default=None, 
                        help="Maximum duration to transcribe in seconds (default: full audio)")
    parser.add_argument("--progress-file", type=str, help="File to write progress updates to")
    parser.add_argument("--list-chunks-only", action="store_true", help="Only list chunks without processing")
    parser.add_argument("--use-wav", action="store_true", help="Use WAV format for audio chunks (better API compatibility)")
    
    args = parser.parse_args()
    
    # Check if Replicate API token is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        print("Please set it with: export REPLICATE_API_TOKEN=your_token_here")
        sys.exit(1)
    
    # If no arguments provided, use default paths
    if args.audio is None or args.desc is None:
        print("No audio or description file provided, using default paths")
        audio_path = "./episode-audios/sample.mp3"
        desc_path = "./episode-descriptions/sample.txt"
    else:
        audio_path = args.audio
        desc_path = args.desc
    
    main(audio_path, desc_path, max_duration=args.max_duration, 
         model_version=args.model_version, max_concurrent=args.max_concurrent,
         use_wav=args.use_wav)