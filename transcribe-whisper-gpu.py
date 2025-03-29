import os
import subprocess
import json
import time
import math
import whisper
import re
import torch
import torch.multiprocessing as mp
from functools import partial

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

def split_audio(audio_file_path, chunk_length=300, max_duration=None):
    """
    Splits the audio into smaller chunks of `chunk_length` seconds each.
    Returns a list of chunk file paths and the total length of the audio.
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

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_length

        duration = min(chunk_length, total_length - start)
        
        # If duration <= 0, no more valid chunks left
        if duration <= 0:
            break

        chunk_output = os.path.join(output_dir, f"chunk_{i:03d}.mp3")
        # ffmpeg command to slice audio
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

def process_chunk_gpu(chunk_info, model_name="base", language="en", fp16=True):
    """
    Process a single audio chunk using a specific GPU.
    
    Args:
        chunk_info: Dictionary with keys: 'chunk_id', 'chunk_path', 'chunk_length', 'gpu_id'
        model_name: Whisper model to use ('base', 'small', 'medium', 'large')
        language: Language code for transcription
        fp16: Whether to use half-precision (faster but slightly less accurate)
    
    Returns:
        Processed segments with adjusted timestamps
    """
    chunk_id = chunk_info['chunk_id']
    chunk_path = chunk_info['chunk_path']
    chunk_length = chunk_info['chunk_length']
    gpu_id = chunk_info['gpu_id']
    
    # Set the device
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # Load the model on the specified GPU
    model = whisper.load_model(model_name).to(device)
    
    # Transcribe the chunk
    result = model.transcribe(
        chunk_path, 
        language=language,
        fp16=fp16
    )
    
    # Adjust segment times based on chunk position
    offset = chunk_id * chunk_length
    for seg in result["segments"]:
        seg["start"] += offset
        seg["end"] += offset
    
    return result["segments"]

def transcribe_chunks_parallel(chunk_paths, total_length, chunk_length=300, model_name="medium", num_gpus=2, progress_file=None):
    """
    Transcribe chunks in parallel using multiple GPUs.
    
    Args:
        chunk_paths: List of paths to audio chunk files
        total_length: Total audio length in seconds
        chunk_length: Length of each chunk in seconds
        model_name: Whisper model name ('base', 'small', 'medium', 'large')
        num_gpus: Number of GPUs to use
        progress_file: Optional path to a file for writing progress updates
    
    Returns:
        List of transcribed segments with adjusted timestamps
    """
    global args
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No GPUs available. Falling back to CPU.")
        return transcribe_chunks_cpu(chunk_paths, total_length, chunk_length, model_name)
    
    num_gpus = min(num_gpus, available_gpus)
    print(f"Using {num_gpus} GPUs for transcription")
    
    # Prepare chunk info for parallel processing
    chunk_infos = []
    for i, chunk_path in enumerate(chunk_paths):
        gpu_id = i % num_gpus
        chunk_infos.append({
            'chunk_id': i,
            'chunk_path': chunk_path,
            'chunk_length': chunk_length,
            'gpu_id': gpu_id
        })
    
    all_segments = []
    processed_time = 0.0
    
    # Use multiprocessing for parallel processing
    # Keep track of the number of processed chunks
    processed_chunks = 0
    total_chunks = len(chunk_paths)
    
    # If progress file is specified, initialize it
    if args.progress_file:
        try:
            with open(args.progress_file, 'w') as f:
                f.write("0")
        except Exception as e:
            print(f"Error initializing progress file: {e}")
    
    with mp.Pool(processes=num_gpus) as pool:
        # Create a partial function with fixed parameters
        process_fn = partial(process_chunk_gpu, model_name=model_name)
        
        # Process chunks in parallel
        for i, segments in enumerate(pool.imap(process_fn, chunk_infos)):
            if segments:
                all_segments.extend(segments)
                
                # Calculate progress
                if segments:
                    chunk_duration = segments[-1]["end"] - segments[0]["start"]
                else:
                    chunk_duration = chunk_length
                    
                processed_time += chunk_duration
                if processed_time > total_length:
                    processed_time = total_length
                
                # Increment processed chunks counter
                processed_chunks += 1
                
                # Update progress file if specified
                if args.progress_file:
                    try:
                        with open(args.progress_file, 'w') as f:
                            f.write(str(processed_chunks))
                    except Exception as e:
                        print(f"Error writing to progress file: {e}")
                
                progress = min((processed_time / total_length) * 100, 100)
                print(f"Processed chunk {i+1}/{len(chunk_paths)}. Progress: {progress:.2f}% ({processed_chunks}/{total_chunks} chunks)")
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x["start"])
    return all_segments

def transcribe_chunks_cpu(chunk_paths, total_length, chunk_length=300, model_name="base", progress_file=None):
    """
    Fallback function to transcribe each chunk using CPU if no GPUs are available.
    """
    global args
    model = whisper.load_model(model_name)
    all_segments = []
    processed_time = 0.0
    
    # Keep track of the number of processed chunks
    processed_chunks = 0
    total_chunks = len(chunk_paths)
    
    # If progress file is specified, initialize it
    if args.progress_file:
        try:
            with open(args.progress_file, 'w') as f:
                f.write("0")
        except Exception as e:
            print(f"Error initializing progress file: {e}")

    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
        result = model.transcribe(chunk_path, language="en")

        offset = i * chunk_length
        if result["segments"]:
            for seg in result["segments"]:
                seg["start"] += offset
                seg["end"] += offset
                all_segments.append(seg)

            chunk_duration = result["segments"][-1]["end"] - result["segments"][0]["start"]
        else:
            # No segments? Assume full chunk length or skip.
            chunk_duration = chunk_length
            
        # Increment processed chunks counter
        processed_chunks += 1
        
        # Update progress file if specified
        if args.progress_file:
            try:
                with open(args.progress_file, 'w') as f:
                    f.write(str(processed_chunks))
            except Exception as e:
                print(f"Error writing to progress file: {e}")
            
        processed_time += chunk_duration
        if processed_time > total_length:
            processed_time = total_length
            
        progress = min((processed_time / total_length) * 100, 100)
        print(f"Processed {processed_time:.2f} sec of {total_length:.2f} sec. Progress: {progress:.2f}% ({processed_chunks}/{total_chunks} chunks)")

        if processed_time >= total_length:
            break
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

def main(audio_path, desc_path, max_duration=None, model_name="medium", num_gpus=2):
    start_time = time.time()
    
    # 1. Load episode description and topics
    print("Loading episode metadata...")
    episode_description, episode_topics, date_str, ep_num, ep_title = load_episode_description(desc_path)
    print("Episode Topics:", episode_topics)

    raw_words = ep_title.split()
    cleaned_words = []
    for word in raw_words:
        cleaned_word = re.sub(r'[^a-zA-Z0-9]+', '', word)
        if cleaned_word:
            cleaned_words.append(cleaned_word)
    title_topics = ",".join(cleaned_words)

    # 2. Split audio into chunks and get total_length
    print("Splitting audio into chunks...")
    chunk_paths, total_length = split_audio(audio_path, chunk_length=300, max_duration=max_duration)
    
    # If list-chunks-only is specified, just print the chunks and exit
    if args.list_chunks_only:
        for chunk_path in chunk_paths:
            print(chunk_path)
        sys.exit(0)
    
    # 3. Transcribe chunks using parallel GPU processing
    print(f"Starting transcription using {model_name} model...")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        all_segments = transcribe_chunks_parallel(chunk_paths, total_length, chunk_length=300, 
                                                model_name=model_name, num_gpus=num_gpus, 
                                                progress_file=args.progress_file)
    else:
        print("No GPUs detected, using CPU (will be much slower)")
        all_segments = transcribe_chunks_cpu(chunk_paths, total_length, chunk_length=300, 
                                          model_name=model_name, progress_file=args.progress_file)

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
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn')
    
    # Parse command line arguments if any
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Transcribe audio using GPU-accelerated Whisper")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    parser.add_argument("--desc", type=str, help="Path to description file")
    parser.add_argument("--model", type=str, default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"], 
                        help="Whisper model to use (default: medium)")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs to use (default: 2)")
    parser.add_argument("--max-duration", type=float, default=None, 
                        help="Maximum duration to transcribe in seconds (default: full audio)")
    parser.add_argument("--progress-file", type=str, help="File to write progress updates to")
    parser.add_argument("--list-chunks-only", action="store_true", help="Only list chunks without processing")
    
    args = parser.parse_args()
    
    # If no arguments provided, use default paths
    if args.audio is None or args.desc is None:
        print("No audio or description file provided, using default paths")
        audio_path = "./episode-audios/704_-_The_2024_MAME_Movie_Awards_Stumbling_Giant_Theory_Luka_Doncic_Tragedy_movies_nba_lukadoncic.mp3"
        desc_path = "./episode-descriptions/704_-_The_2024_MAME_Movie_Awards_Stumbling_Giant_Theory_Luka_Doncic_Tragedy_movies_nba_lukadoncic.txt"
    else:
        audio_path = args.audio
        desc_path = args.desc
    
    main(audio_path, desc_path, max_duration=args.max_duration, 
         model_name=args.model, num_gpus=args.gpus)