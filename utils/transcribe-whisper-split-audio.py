import os
import subprocess
import json
import time
import math
import whisper

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

def split_audio(audio_file_path, chunk_length=300):
    """
    Splits the audio into smaller chunks of `chunk_length` seconds each.
    Returns a list of chunk file paths and the total length of the audio.
    """
    base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
    output_dir = f"{base_name}_chunks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_length = get_audio_length(audio_file_path)
    num_chunks = math.ceil(total_length / chunk_length)

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_length
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
    Load episode description and timestamps from a file.
    Format:
    03:30 Under the Umbrella
    09:30 1991 - Our favorite movies
    ...
    <blank line>
    Greetings Travelers,
    <rest of description>
    """
    with open(desc_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    timestamps = []
    description_lines = []
    parsing_timestamps = True
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            parsing_timestamps = False
            continue
        if parsing_timestamps and ":" in stripped:
            timestamps.append(stripped)
        else:
            description_lines.append(stripped)

    episode_description = "\n".join(description_lines)
    structured_timestamps = []
    for t in timestamps:
        # Format: "MM:SS Title" or "HH:MM:SS Title"
        parts = t.split(" ", 1)
        time_marker = parts[0]
        topic = parts[1] if len(parts) > 1 else ""
        structured_timestamps.append({"time": time_marker, "topic": topic})

    return episode_description, structured_timestamps

def time_str_to_seconds(t_str):
    """
    Converts a time string like 'MM:SS' or 'HH:MM:SS' to total seconds.
    """
    parts = t_str.split(':')
    parts = list(map(int, parts))
    while len(parts) < 3:
        parts.insert(0, 0)  # Pad to hh:mm:ss if needed
    h, m, s = parts
    return h*3600 + m*60 + s

def assign_topics_to_chunks(chunks, episode_topics):
    """
    Assign topics to each chunk based on the episode_topics time markers.
    We find the latest topic time <= chunk start_time.
    """
    topic_map = []
    for et in episode_topics:
        seconds = time_str_to_seconds(et["time"])
        topic_map.append((seconds, et["topic"]))

    topic_map.sort(key=lambda x: x[0])

    for chunk in chunks:
        chunk_start = chunk["start_time"]
        assigned_topic = None
        for t_time, t_topic in topic_map:
            if t_time <= chunk_start:
                assigned_topic = t_topic
            else:
                break
        chunk["topic"] = assigned_topic if assigned_topic else "No specific topic"

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
        progress = min((processed_time / total_length) * 100, 100)
        print(f"Processed {processed_time:.2f} sec of {total_length:.2f} sec. Progress: {progress:.2f}%")

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

def main(audio_path, desc_path):
    # 1. Load episode description and topics
    episode_description, episode_topics = load_episode_description(desc_path)

    # 2. Split audio into chunks and get total_length
    print("Splitting audio into chunks...")
    chunk_paths, total_length = split_audio(audio_path, chunk_length=300)

    # 3. Transcribe chunks and show progress
    all_segments = transcribe_chunks(chunk_paths, total_length, chunk_length=300)

    # 4. Combine segments into ~1-minute chunks for better vector context
    chunks = chunk_segments_for_vector(all_segments, chunk_duration=60.0)

    # 5. Assign topics to each chunk
    chunks = assign_topics_to_chunks(chunks, episode_topics)

    # Prepare final output
    output = {
        "episode_description": episode_description,
        "transcript_chunks": chunks
    }

    # 6. Save final outputs
    with open('final_output.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    with open('final_output.txt', 'w', encoding='utf-8') as f:
        f.write(f"Episode Description:\n{episode_description}\n\n")
        for ch in chunks:
            f.write(f"[{ch['start_time']:.2f} - {ch['end_time']:.2f}] Topic: {ch['topic']}\n{ch['text']}\n\n")

    print("Transcription complete!")
    print("Files generated:\n- final_output.json\n- final_output.txt")

if __name__ == "__main__":
    audio_path = "./episode-audios/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3"
    desc_path = "./episode-descriptions/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.txt"
    main(audio_path, desc_path)
