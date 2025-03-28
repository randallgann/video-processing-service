from pydub import AudioSegment
import sys
import os

def extract_first_10_minutes(input_file_path, output_file_path):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_file_path)
    
    # Calculate 10 minutes in milliseconds
    ten_minutes_ms = 5 * 60 * 1000
    
    # Slice the first 10 minutes
    first_10 = audio[:ten_minutes_ms]
    
    # Export the sliced audio to a new mp3 file
    first_10.export(output_file_path, format="mp3")
    print(f"Successfully created {output_file_path}")

if __name__ == "__main__":
    extract_first_10_minutes("./episode-audios/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3","606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3")