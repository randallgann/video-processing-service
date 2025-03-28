from google.cloud import speech_v1
import json

def test_transcribe_formats():
    client = speech_v1.SpeechClient()
    
    # Test the gs:// format
    gcs_uri = "gs://mm-audio-files/Under_The_Umbrella_ep1._-_The_.mp3"
    
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.MP3,
        language_code="en-US",
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True
    )
    
    try:
        print(f"\nTesting with URI: {gcs_uri}")
        audio = speech_v1.RecognitionAudio(uri=gcs_uri)
        # Just start the operation to test if the URI is valid
        operation = client.long_running_recognize(config=config, audio=audio)
        print("URI is valid! Operation started successfully.")
        # Cancel the operation since this is just a test
        operation.cancel()
    except Exception as e:
        print(f"Error with gs:// format: {str(e)}")

if __name__ == "__main__":
    test_transcribe_formats()