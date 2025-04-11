# Changelog

All notable changes to the YouTube Video Transcription Service will be documented in this file.

## [1.0.1] - 2025-04-11

### Fixed
- Fixed audio chunking issue where only the first chunk was being processed correctly
- Modified FFmpeg command to re-encode MP3 chunks instead of using copy codec, ensuring proper chunk generation
- Added improved error detection and logging for FFmpeg processes
- Added comprehensive support for Fast Whisper model response format
- Enhanced logging throughout the processing pipeline to better diagnose issues
- Added fallback mechanisms to extract transcript text when segments aren't available
- Improved error recovery when processing Fast Whisper model responses

### Modified
- Enhanced the transcription processing code to handle both OpenAI and Fast Whisper model formats
- Added detailed diagnostic logs for debugging API responses
- Improved robustness of segment processing to prevent empty final outputs
- Added more verbose logging to track data flow through the entire pipeline

## [1.0.0] - 2025-04-09

### Architecture Change
- Major architecture change from GPU-based local processing to cloud VM with Replicate API integration
- Switched from direct Whisper model usage to Replicate API for transcription
- Added support for both standard Whisper and fast Whisper models

### Added
- New `cloud-vm-processor.py` service for PubSub-triggered video processing
- Added model selection capability via environment variables and message parameters
- Created comprehensive Docker configuration for containerized deployment
- Added direct integration with Google Cloud Storage for audio chunks and results
- Implemented robust progress tracking and reporting system
- Created detailed deployment documentation for cloud VMs
- Added model type parameter for fast or standard transcription options
- Added chunk-based audio processing with GCS storage
- Implemented parallel processing for audio chunks
- Created `.env.example` file for simplified configuration

### Modified
- Updated `debug-replicate.py` to support model type selection via env vars
- Updated `Dockerfile` to use Ubuntu 22.04 base image without CUDA dependencies
- Simplified deployment with docker-compose configuration
- Removed dependency on mini_audio_server for audio hosting
- Updated `requirements.txt` and `replicate-requirements.txt` with python-dotenv

### Fixed
- Fixed compatibility issues with fast Whisper model by setting language to "english"
- Improved error handling for Replicate API calls
- Improved cleanup of temporary files

### Documentation
- Created new `README.md` with updated architecture and usage instructions
- Added `DEPLOYMENT.md` with step-by-step deployment guide
- Added `CHANGELOG.md` to track project evolution

## [0.3.0] - Prior History

### Added
- Initial implementation of GKE-based processing
- Added NVIDIA GPU support for local transcription
- Created Kubernetes deployment configurations

### Modified
- Updated audio processing to support various formats
- Enhanced progress tracking and reporting