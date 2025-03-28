import whisper
import json
import time

def transcribe_with_whisper(audio_file_path):
    print(f"Starting Whisper transcription for {audio_file_path}")
    model = whisper.load_model("base")
    print("Transcribing...")
    start_time = time.time()

    # Basic transcription (no word timestamps)
    result = model.transcribe(
        audio_file_path,
        language="en",
        word_timestamps=False
    )

    end_time = time.time()
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")

    # Return segments as is, we'll chunk them later
    return result["segments"]

def load_episode_description(desc_file_path):
    # Assuming the description file has a known format.
    # For example, a structured format or just raw text.
    # Let's say the description file is structured like your example:
    # 03:30 Under the Umbrella 
    # 09:30 1991 - Our favorite movies
    # ...
    # Followed by a greeting and a paragraph description.

    with open(desc_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Separate line-based timestamps from the greeting/description
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
    # Convert timestamps into a structured form: [{"time":"03:30", "topic":"Under the Umbrella"}, ...]
    structured_timestamps = []
    for t in timestamps:
        # Format: "MM:SS Title"
        parts = t.split(" ", 1)
        time_marker = parts[0]
        topic = parts[1] if len(parts) > 1 else ""
        structured_timestamps.append({"time": time_marker, "topic": topic})

    return episode_description, structured_timestamps

def time_str_to_seconds(t_str):
    # Convert "MM:SS" or "HH:MM:SS" to seconds
    parts = t_str.split(':')
    parts = list(map(int, parts))
    while len(parts) < 3:
        parts.insert(0, 0)  # Pad to hh:mm:ss
    h, m, s = parts
    return h*3600 + m*60 + s

def chunk_segments(segments, chunk_duration=60.0):
    """Combine transcript segments into larger chunks for better context."""
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None

    for seg in segments:
        if current_chunk == []:
            current_start = seg["start"]
        current_chunk.append(seg["text"].strip())
        current_end = seg["end"]

        # If the current chunk exceeds the desired duration, finalize it
        if (current_end - current_start) >= chunk_duration:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "start_time": current_start,
                "end_time": current_end,
                "text": chunk_text
            })
            current_chunk = []
            current_start = None

    # Add any leftover segment chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "start_time": current_start,
            "end_time": current_end,
            "text": chunk_text
        })

    return chunks

def assign_topics_to_chunks(chunks, episode_topics):
    # Convert times to seconds for easy comparison
    # episode_topics is a list of dicts like [{"time":"03:30","topic":"Under the Umbrella"}, ...]
    # We assume they're in chronological order.
    topic_map = []
    for et in episode_topics:
        seconds = time_str_to_seconds(et["time"])
        topic_map.append((seconds, et["topic"]))

    # topic_map: [(210, "Under the Umbrella"), (570, "1991 - Our favorite movies"), ...]
    # We'll assign the most recent topic that starts before the chunk start_time.
    # Or you can map each chunk to all topics it overlaps. For simplicity, let's pick the latest relevant topic.
    
    # Sort by time just in case
    topic_map.sort(key=lambda x: x[0])

    for chunk in chunks:
        # Find the topic that best matches the chunk start
        chunk_start = chunk["start_time"]
        # Find the latest topic time <= chunk_start
        assigned_topic = None
        for t_time, t_topic in topic_map:
            if t_time <= chunk_start:
                assigned_topic = t_topic
            else:
                break
        chunk["topic"] = assigned_topic if assigned_topic else "No specific topic"

    return chunks

def main(audio_path, desc_path):
    try:
        # Transcription
        segments = transcribe_with_whisper(audio_path)

        # Load episode description and topics
        episode_description, episode_topics = load_episode_description(desc_path)

        # Chunk segments into larger pieces
        chunks = chunk_segments(segments, chunk_duration=60.0)

        # Assign topics to each chunk if possible
        chunks = assign_topics_to_chunks(chunks, episode_topics)

        # Prepare final output structure
        output = {
            "episode_description": episode_description,
            "transcript_chunks": chunks
        }

        # Save as JSON
        with open('final_output.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        # Save a more readable text version with metadata
        with open('final_output.txt', 'w', encoding='utf-8') as f:
            f.write(f"Episode Description:\n{episode_description}\n\n")
            for ch in chunks:
                f.write(f"[{ch['start_time']:.2f} - {ch['end_time']:.2f}] Topic: {ch['topic']}\n{ch['text']}\n\n")

        print("Files generated:\n- final_output.json\n- final_output.txt")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audio_path = "./audios/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3"
    desc_path = "./episode-descriptions/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.txt"
    main(audio_path, desc_path)
