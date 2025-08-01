# Changelog

All notable changes to the YouTube Video Transcription Service will be documented in this file.

## [1.1.0] - 2025-07-08

### Added
- **Vector Integration Support**: Added publishing to `video-processing-results` topic for transcript-to-vector pipeline
- **New Function**: `publish_completion_result()` publishes completion messages with transcript GCS location
- **Environment Variable**: `RESULTS_TOPIC_ID` for configuring results topic
- **Kubernetes Configuration**: Added RESULTS_TOPIC_ID mapping in deployment YAML

### Changed
- **Progress Tracking**: Transcription now completes at 90% instead of 100% to allow vector processing
- **Final Stage**: Changed from "completed" to "transcription_completed" at 90%
- **Error Handling**: Failed transcriptions now publish to results topic for proper handling

### Architecture
- Transcription service now triggers downstream vector processing via Pub/Sub
- Enables automatic embedding generation and Qdrant storage through API service integration

## [1.0.5] - 2025-04-11

### Added
- Added automatic cleanup of audio chunks from GCS after successful transcription
- Added new progress stage for cleanup operations
- Implemented GCS directory listing and safe blob deletion

## [1.0.4] - 2025-04-11

### Fixed
- Fixed error handling for Fast Whisper chunks with missing or invalid timestamps
- Added fallback timing calculation for chunks with invalid timestamps
- Prevented dumping of large raw data to logs on error
- Added graceful handling of malformed chunk data

## [1.0.3] - 2025-04-11

### Enhanced
- Added comprehensive INFO-level logging for PubSub progress updates
- Improved visibility of progress reporting in logs
- Added startup verification of PubSub topic availability
- Added detailed transcription progress monitoring logs

## [1.0.2] - 2025-04-11

### Fixed
- Added direct support for the "incredibly-fast-whisper" model output format
- Fixed issue with Fast Whisper model response parsing where text was not being extracted
- Now properly handles top-level 'chunks' and 'text' fields in the API response
- Added fallback to full transcript text when chunk processing fails

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