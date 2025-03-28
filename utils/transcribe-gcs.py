from google.cloud import speech_v1
import json

def transcribe_gcs_audio(gcs_uri):
    """Transcribes audio file from Google Cloud Storage."""
    client = speech_v1.SpeechClient()

    audio = speech_v1.RecognitionAudio(uri=gcs_uri)

    # Enhanced configuration for better results
    config = speech_v1.RecognitionConfig(
        encoding=speech_v1.RecognitionConfig.AudioEncoding.MP3,
        language_code="en-US",
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        model="default",  # Using default model first
        audio_channel_count=2,  # Stereo audio
        sample_rate_hertz=44100,  # 44.1kHz
        use_enhanced=True,
        profanity_filter=False,  # Disable profanity filter to get all words
        enable_spoken_punctuation=True
    )

    print("Starting transcription...")
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for transcription operation to complete...")

    # Add timeout for long-running operation
    response = operation.result(timeout=500)

    # Print some metadata about the response
    print(f"\nNumber of results: {len(response.results)}")

     # Let's log the raw response first
    # print("\nRaw response results:")
    # for result in response.results:
    #     print(f"\nTranscript: {result.alternatives[0].transcript}")
    #     print(f"Confidence: {result.alternatives[0].confidence}")

    # Extract words with timestamps
    transcript_data = []

    for i, result in enumerate(response.results):
        alternative = result.alternatives[0]
        
        print(f"\nSegment {i+1}/{len(response.results)}")
        print(f"Transcript: {alternative.transcript}")
        print(f"Confidence: {alternative.confidence}")
        
        for word in alternative.words:
            word_info = {
                'word': word.word,
                'start_time': float(word.start_time.total_seconds()),
                'end_time': float(word.end_time.total_seconds())
            }
            transcript_data.append(word_info)
    
    print(f"\nTotal words processed: {len(transcript_data)}")
    
    if transcript_data:
        print(f"\nFirst word: {transcript_data[0]}")
        print(f"Last word: {transcript_data[-1]}")


    # for result in response.results:
    #     alternative = result.alternatives[0]
    #     print(f"\nProcessing segment: {alternative.transcript}")
    #     for word in result.alternatives[0].words:
    #         word_info = {
    #             'word': word.word,
    #             'start_time': float(word.start_time.total_seconds()),
    #             'end_time': float(word.end_time.total_seconds())
    #         }
    #         print(f"Word timing: {word_info}")
    #         transcript_data.append({
    #             'word': word.word,
    #             'start_time': float(word.start_time.total_seconds()),
    #             'end_time': float(word.end_time.total_seconds())
    #         })
    
    # print(f"\nTotal words processed: {len(transcript_data)}")
    return transcript_data

def combine_diarization_and_transcription(diarization, transcript_data):
    """Combines diarization and transcription data."""
    print(f"Starting combination process:")
    print(f"- Number of diarization segments: {len(diarization)}")
    print(f"- Number of transcribed words: {len(transcript_data)}")

    combined_data = []
    unmatched_words = 0

    for word_data in transcript_data:
        word_start = word_data['start_time']
        word_end = word_data['end_time']
        speaker = None
        
        # Find matching speaker segment
        for segment in diarization:
            if (word_start >= segment['start'] and 
                word_start <= segment['end']):
                speaker = segment['speaker']
                break
        
        if not speaker:
            unmatched_words += 1
            
        combined_data.append({
            'word': word_data['word'],
            'start_time': word_start,
            'end_time': word_end,
            'speaker': speaker
        })
    
    print(f"\nCombination results:")
    print(f"- Words matched with speakers: {len(transcript_data) - unmatched_words}")
    print(f"- Words without speaker match: {unmatched_words}")
    
    return combined_data

    # combined_output = {
    #     'diarization': diarization,
    #     'transcription': transcript_data,
    #     'combined': []
    # }
    
    # # Match words with speakers based on timestamps
    # for word_data in transcript_data:
    #     word_start = word_data['start_time']
    #     word_end = word_data['end_time']
        
    #     # Find the speaker for this timestamp
    #     speaker = None
    #     for segment in diarization:
    #         if (word_start >= segment['start'] and 
    #             word_start <= segment['end']):
    #             speaker = segment['speaker']
    #             break
        
    #     combined_output['combined'].append({
    #         'word': word_data['word'],
    #         'start_time': word_start,
    #         'end_time': word_end,
    #         'speaker': speaker
    #     })
    
    # return combined_output

def main():
    try:
         # Load diarization and transcript data from the combined_output.json
        with open('combined_output.json', 'r') as f:
            data = json.load(f)
        
        # Get the separate components
        diarization = data['diarization']
        transcription = data['transcription']
        
        # Combine the data
        combined_data = combine_diarization_and_transcription(diarization, transcription)
        
        # Create final output
        output = {
            'diarization': diarization,
            'transcription': transcription,
            'combined': combined_data
        }
        
        # Save a readable version
        with open('readable_output.txt', 'w') as f:
            current_speaker = None
            current_text = []
            
            for item in combined_data:
                if item['speaker'] != current_speaker:
                    if current_text:
                        f.write(' '.join(current_text) + '\n\n')
                        current_text = []
                    if item['speaker']:
                        f.write(f"{item['speaker']}: ")
                    current_speaker = item['speaker']
                current_text.append(item['word'])
            
            if current_text:
                f.write(' '.join(current_text))
        
        print("\nFiles generated:")
        print("- readable_output.txt (formatted conversation)")
        
    except Exception as e:
        print(f"Error combining data: {str(e)}")
        import traceback
        traceback.print_exc()


    #     # Load diarization and transcript data from the combined_output.json
    #     with open('combined_output.json', 'r') as f:
    #         data = json.load(f)
        
    #     # Get the separate components
    #     diarization = data['diarization']
    #     transcription = data['transcription']

    #     # Combine the data
    #     combined_data = combine_diarization_and_transcription(diarization, transcription)

    #     # Create final output
    #     output = {
    #         'diarization': diarization,
    #         'transcription': transcription,
    #         'combined': combined_data
    #     }

    #     # Load diarization data
    #     with open('diarization_output.json', 'r') as f:
    #         diarization_data = json.load(f)
        
    #     # Load transcription data
    #     with open('transcription_only.json', 'r') as f:
    #         transcript_data = json.load(f)
        
    #     # Combine the data
    #     combined_data = combine_diarization_and_transcription(
    #         diarization_data['output']['diarization'],
    #         transcript_data
    #     )
        
    #     # Create final output
    #     output = {
    #         'diarization': diarization_data['output']['diarization'],
    #         'transcription': transcript_data,
    #         'combined': combined_data
    #     }
        
    #     # Save combined output
    #     with open('combined_output.json', 'w') as f:
    #         json.dump(output, f, indent=2)
        
    #     # Save a more readable version
    #     with open('readable_output.txt', 'w') as f:
    #         current_speaker = None
    #         current_text = []
            
    #         for item in combined_data:
    #             if item['speaker'] != current_speaker:
    #                 if current_text:
    #                     f.write(' '.join(current_text) + '\n\n')
    #                     current_text = []
    #                 if item['speaker']:
    #                     f.write(f"{item['speaker']}: ")
    #                 current_speaker = item['speaker']
    #             current_text.append(item['word'])
            
    #         if current_text:
    #             f.write(' '.join(current_text))
        
    #     print("\nFiles generated:")
    #     print("- combined_output.json (full detailed output)")
    #     print("- readable_output.txt (formatted conversation)")
        
    # except Exception as e:
    #     print(f"Error combining data: {str(e)}")
    #     import traceback
    #     traceback.print_exc()




    # # Load diarization data from local file
    # with open('diarization_output.json', 'r') as f:
    #     data = json.load(f)
    
    # diarization = data['output']['diarization']
    
    # # GCS URI of your audio file
    # gcs_uri = "gs://mm-audio-files/Under_The_Umbrella_ep1._-_The_.mp3"
    
    # try:
    #     # Get transcription
    #     transcript_data = transcribe_gcs_audio(gcs_uri)
        
    #     # Combine the data
    #     combined_output = combine_diarization_and_transcription(diarization, transcript_data)
        
    #     # Save results to a file
    #     with open('combined_output.json', 'w') as f:
    #         json.dump(combined_output, f, indent=2)
        
    #     print("Processing complete! Results saved to combined_output.json")
        
    # except Exception as e:
    #     print(f"Error processing transcription: {str(e)}")

if __name__ == "__main__":
    main()