from pydub import AudioSegment

audio = AudioSegment.from_mp3("Under_The_Umbrella_ep1._-_The_.mp3")
print(f"Duration: {len(audio)/1000} seconds")
print(f"Channels: {audio.channels}")
print(f"Sample width: {audio.sample_width} bytes")
print(f"Frame rate: {audio.frame_rate} Hz")