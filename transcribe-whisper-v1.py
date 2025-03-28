import os
import subprocess
import json
import time
import math
import whisper
import re

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
    # structured_timestamps = []
    # for t in timestamps:
    #     # Format: "MM:SS Title" or "HH:MM:SS Title"
    #     parts = t.split(" ", 1)
    #     time_marker = parts[0]
    #     topic = parts[1] if len(parts) > 1 else ""
    #     structured_timestamps.append({"time": time_marker, "topic": topic})

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

def transcribe_chunks(chunk_paths, total_length, chunk_length=300):
    """
    Transcribe each chunk using Whisper, showing progress after each chunk.
    Return a list of segments with adjusted times.
    """
    model = whisper.load_model("base")
    all_segments = []
    processed_time = 0.0

    for i, chunk_path in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
        result = model.transcribe(chunk_path, language="en", word_timestamps=False)

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
            

        processed_time += chunk_duration
        if processed_time > total_length:
            processed_time = total_length
        progress = min((processed_time / total_length) * 100, 100)
        print(f"Processed {processed_time:.2f} sec of {total_length:.2f} sec. Progress: {progress:.2f}%")

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

def main(audio_path, desc_path, max_duration=None):
    # 1. Load episode description and topics
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

    # 3. Transcribe chunks and show progress
    all_segments = transcribe_chunks(chunk_paths, total_length, chunk_length=300)

    # 4. Combine segments into ~1-minute chunks for better vector context
    chunks = chunk_segments_for_vector(all_segments, chunk_duration=60.0)

    # 5. Assign topics to each chunk (optional, if you still want topics)
    for ch in chunks:
        print(f"Chunk start: {ch['start_time']} end: {ch['end_time']}")
    chunks = assign_topics_to_chunks(chunks, episode_topics)

    # Prepare final output in the requested format:
    # A list of objects each with "text" and "metadata"
    final_output = []
    for ch in chunks:
        final_output.append({
            "text": f"Topic: {ch["topic"]} Spoken Words: {ch["text"]}",
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
    with open('final_output.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)

    print("Transcription complete!")
    print("File generated:\n- final_output.json")

if __name__ == "__main__":
    audio_path = "./episode-audios/704_-_The_2024_MAME_Movie_Awards_Stumbling_Giant_Theory_Luka_Doncic_Tragedy_movies_nba_lukadoncic.mp3"
    desc_path = "./episode-descriptions/704_-_The_2024_MAME_Movie_Awards_Stumbling_Giant_Theory_Luka_Doncic_Tragedy_movies_nba_lukadoncic.txt"
    main(audio_path, desc_path, max_duration=None)
