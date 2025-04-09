import os
import sys
import replicate
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_replicate_api(audio_url, model_type=None, model_version=None):
    """
    Test the Replicate API with a given audio URL.
    
    Args:
        audio_url: URL of the audio file to transcribe
        model_type: Type of model to use ("openai" or "fast")
        model_version: Replicate model version to use (overrides model_type if provided)
    """
    # Check if REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        print("Please set it with: export REPLICATE_API_TOKEN=your_token_here")
        sys.exit(1)
    
    # Get model information based on model_type or model_version
    if model_version:
        # If specific version is provided, use it directly
        model_id = "openai/whisper"  # Default to OpenAI model ID
    elif model_type == "fast":
        # Use fast whisper model from env var
        fast_whisper = os.environ.get("FAST_WHISPER")
        if not fast_whisper or ":" not in fast_whisper:
            print("Error: FAST_WHISPER environment variable not properly set")
            sys.exit(1)
        model_id, model_version = fast_whisper.split(":", 1)
    else:
        # Default to OpenAI whisper model from env var
        openai_whisper = os.environ.get("OPENAI_WHISPER")
        if not openai_whisper or ":" not in openai_whisper:
            print("Error: OPENAI_WHISPER environment variable not properly set")
            sys.exit(1)
        model_id, model_version = openai_whisper.split(":", 1)
    
    print(f"Testing Replicate API with:")
    print(f"- Audio URL: {audio_url}")
    print(f"- Model ID: {model_id}")
    print(f"- Model version: {model_version}")
    print(f"- API token: {'*' * 8}{os.environ.get('REPLICATE_API_TOKEN')[-4:]}")
    
    try:
        # Call Replicate API
        print("\nCalling Replicate API...")
        output = replicate.run(
            f"{model_id}:{model_version}",
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
    parser.add_argument("--model-type", 
                        choices=["openai", "fast"],
                        help="Type of model to use ('openai' for standard Whisper, 'fast' for fast Whisper)")
    parser.add_argument("--model-version", 
                        help="Specific Replicate model version to use (overrides model-type)")
    
    args = parser.parse_args()
    test_replicate_api(args.audio_url, args.model_type, args.model_version)