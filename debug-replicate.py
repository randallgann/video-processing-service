import os
import sys
import replicate
import argparse

def test_replicate_api(audio_url, model_version=None):
    """
    Test the Replicate API with a given audio URL.
    
    Args:
        audio_url: URL of the audio file to transcribe
        model_version: Replicate model version to use
    """
    # Check if REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        print("Please set it with: export REPLICATE_API_TOKEN=your_token_here")
        sys.exit(1)
    
    # Default model version if not provided
    if not model_version:
        model_version = "8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
    
    print(f"Testing Replicate API with:")
    print(f"- Audio URL: {audio_url}")
    print(f"- Model version: {model_version}")
    print(f"- API token: {'*' * 8}{os.environ.get('REPLICATE_API_TOKEN')[-4:]}")
    
    try:
        # Call Replicate API
        print("\nCalling Replicate API...")
        output = replicate.run(
            f"openai/whisper:{model_version}",
            input={
                "audio": audio_url,
                "language": "auto",
                "translate": False,
                "temperature": 0,
                "transcription": "plain text"  # Using plain text for simpler testing
            }
        )
        
        # Print output information
        print("\nAPI call successful!")
        print(f"Output type: {type(output)}")
        
        if output is None:
            print("Warning: Received None response")
        elif isinstance(output, dict):
            print(f"Response keys: {list(output.keys())}")
            for key, value in output.items():
                if isinstance(value, str):
                    # Truncate long strings
                    if len(value) > 100:
                        print(f"- {key}: {value[:100]}...")
                    else:
                        print(f"- {key}: {value}")
                else:
                    print(f"- {key}: {value}")
        elif isinstance(output, str):
            if len(output) > 500:
                print(f"Received text (first 500 chars):\n{output[:500]}...")
            else:
                print(f"Received text:\n{output}")
        else:
            print(f"Unexpected output type: {type(output)}")
            print(f"Output: {output}")
            
    except Exception as e:
        print(f"\nError calling Replicate API: {e}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting tips:")
        print("1. Verify your API token is correct")
        print("2. Make sure the audio URL is publicly accessible")
        print("3. Try a different model version")
        print("4. Check your network connection and firewall settings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Replicate's Whisper API with an audio URL")
    parser.add_argument("audio_url", help="URL of the audio file to transcribe")
    parser.add_argument("--model-version", 
                        default="8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e",
                        help="Replicate model version to use")
    
    args = parser.parse_args()
    test_replicate_api(args.audio_url, args.model_version)