# Cloud VM Architecture for YouTube Video Transcription Service

## Overview

This document outlines the architecture for a cloud-based video transcription service deployed on a standard Ubuntu VM. The service processes YouTube videos by extracting audio, chunking it, and utilizing Replicate's Whisper API for transcription.

## System Architecture

![Architecture Diagram]

The system operates as follows:

1. **Message Reception**: VM listens to a Google Cloud PubSub topic for new video transcription requests
2. **Video Processing**: Service downloads video and extracts audio
3. **Audio Chunking**: Audio is split into manageable chunks
4. **GCS Upload**: Audio chunks are uploaded to Google Cloud Storage
5. **Transcription Request**: Service sends requests to Replicate API with GCS URLs
6. **Progress Tracking**: System monitors transcription progress and publishes updates
7. **Result Processing**: Transcribed text is processed and formatted as required
8. **Result Storage**: Final outputs are uploaded to GCS

## Components and Implementation

### 1. Message Processor

**Primary Implementation**: Modified version of `k8s-processor.py`

- Listens to PubSub topic for incoming transcription requests
- Extracts video metadata (URL, ID, requested format)
- Manages overall processing workflow
- Handles error reporting and retries

### 2. Video and Audio Processor

**Primary Implementation**: Adapted from `yt-dlp-aduio-processor-v1.py`

- Downloads videos using yt-dlp
- Extracts audio in specified format
- Applies any audio preprocessing required

### 3. Audio Chunking System 

**Primary Implementation**: Adapted from `split_audio()` in `transcribe-whisper-replicate.py`

- Splits large audio files into processable chunks
- Handles different audio formats (MP3 and WAV)
- Manages chunk naming and metadata

### 4. Cloud Storage Manager

**Primary Implementation**: Adapted from `k8s-processor.py`'s `upload_results()`

- Uploads audio chunks to GCS with appropriate metadata
- Generates signed URLs if needed
- Verifies successful uploads

### 5. Replicate API Client

**Primary Implementation**: Modified version of `process_chunk_replicate()` from `transcribe-whisper-replicate.py`

- Sends transcription requests to Replicate API
- Selects appropriate model based on configuration (fast vs. standard)
- Tracks request status and retrieves results

### 6. Progress Tracker

**Primary Implementation**: Using `publish_progress()` from `k8s-processor.py`

- Tracks overall transcription progress
- Publishes status updates to PubSub topic
- Reports completion status and any errors

### 7. Result Processor

**Primary Implementation**: Based on post-processing functions in `transcribe-whisper-replicate.py`

- Processes raw transcription results
- Combines results from multiple chunks
- Formats output according to requirements
- Uploads final JSON results to GCS

## Implementation Plan

### Phase 1: Core Service Setup

1. Set up VM with required dependencies
2. Implement PubSub listener using code from `k8s-processor.py`
3. Adapt video downloading from `yt-dlp-aduio-processor-v1.py`
4. Implement audio chunking from `transcribe-whisper-replicate.py`

### Phase 2: Replicate Integration

1. Implement Replicate API client based on `debug-replicate.py` and `transcribe-whisper-replicate.py`
2. Set up model selection mechanism (fast vs. standard)
3. Add progress tracking for Replicate jobs

### Phase 3: Result Processing

1. Implement post-processing for transcriptions
2. Add GCS upload functionality for results
3. Set up final status reporting

### Phase 4: Testing and Optimization

1. End-to-end testing with various video types
2. Performance optimization
3. Error handling improvements

## Key Files for Reuse

1. **Message Processing**: `k8s-processor.py` (PubSub handling)
2. **Video Processing**: `yt-dlp-aduio-processor-v1.py` (video download and audio extraction)
3. **Audio Chunking**: `transcribe-whisper-replicate.py` (audio splitting)
4. **Replicate API**: `debug-replicate.py` (API testing) and `transcribe-whisper-replicate.py` (API interaction)
5. **Progress Tracking**: `k8s-processor.py` and `monitor-progress.py`
6. **Result Processing**: `transcribe-whisper-replicate.py` (transcription formatting)

## Environment Requirements

- Python 3.8+
- ffmpeg for audio processing
- Required Python packages (from `requirements.txt` and `replicate-requirements.txt`)
- Google Cloud SDK
- Replicate API token

## Configuration

The system will use environment variables for configuration:
- `REPLICATE_API_TOKEN`: For Replicate API access
- `OPENAI_WHISPER`: Standard Whisper model ID
- `FAST_WHISPER`: Fast Whisper model ID
- `PROJECT_ID`: Google Cloud project ID
- `SUBSCRIPTION_ID`: PubSub subscription ID
- `BUCKET_NAME`: GCS bucket for storing results
- `PROGRESS_TOPIC_ID`: Topic for progress updates

## Next Steps

1. Create a consolidated service script that integrates components from existing files
2. Adapt the configuration system for VM environment
3. Implement logging and monitoring
4. Develop deployment automation