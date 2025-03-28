import json
from pydub import AudioSegment
import os

def extract_speaker_audio(diarization_json_path, audio_path, output_dir):
    # Load the diarization data
    with open(diarization_json_path, 'r') as f:
        data = json.load(f)

    # Extract diarization entries
    diarization_entries = data['output']['diarization']

    # Group segments by speaker
    speaker_segments = {}
    for entry in diarization_entries:
        speaker = entry['speaker']
        start = float(entry['start'])
        end = float(entry['end'])

        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((start, end))
    
    # Load the original audio
    audio = AudioSegment.from_file(audio_path)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # For each speaker, concatenate all their audio segments and export to MP3
    for speaker, segments in speaker_segments.items():
        # Sort segments by start time to keep them in chronological order
        segments.sort(key=lambda x: x[0])

        speaker_audio = AudioSegment.empty()
        for (start, end) in segments:
            # pydub works in milliseconds
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            # Extract segment
            segment_audio = audio[start_ms:end_ms]

            # Append to speaker's audio
            speaker_audio += segment_audio

        # Export
        output_path = os.path.join(output_dir, f"{speaker}.mp3")
        speaker_audio.export(output_path, format="mp3")
        print(f"Exported audio for {speaker} to {output_path}")


if __name__ == "__main__":
    # Example usage
    # Replace these with the correct paths
    diarization_json_path = "diarizations/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.json"
    audio_path = "episode-audios/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3"
    output_dir = "speaker-audios"

    extract_speaker_audio(diarization_json_path, audio_path, output_dir)
