import os
import json
import argparse
import logging
import boto3
import time
import glob
from pathlib import Path

from src.audio_analyzer import AudioAnalyzer
from src.utils.s3_utils import download_from_s3, upload_to_s3, get_s3_bucket_and_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Audio Analysis with Essentia")
    parser.add_argument("--s3-input-uri", type=str, help="S3 URI for input audio files")
    parser.add_argument("--s3-output-uri", type=str, help="S3 URI for output analysis files")
    parser.add_argument("--use-essentia", type=bool, default=True, help="Whether to use Essentia for analysis")
    return parser.parse_args()

def process_audio_files(input_path, output_path, use_essentia=True):
    """Process all audio files in the input path and save results to the output path"""
    analyzer = AudioAnalyzer(use_essentia=use_essentia)
    results = []
    
    # Find all audio files
    audio_files = []
    for ext in ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a", "*.aac"]:
        audio_files.extend(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        try:
            relative_path = os.path.relpath(audio_file, input_path)
            logger.info(f"Processing {relative_path}")
            
            # Extract track_id from file path if available, or use filename
            file_parts = relative_path.split("/")
            track_id = None
            for i, part in enumerate(file_parts):
                if part == "tracks" and i < len(file_parts) - 1:
                    track_id = file_parts[i + 1]
                    break
            
            if not track_id:
                track_id = os.path.splitext(os.path.basename(audio_file))[0]
            
            # Analyze audio file
            start_time = time.time()
            analysis_result = analyzer.analyze(audio_file)
            processing_time = time.time() - start_time
            
            # Add metadata
            result = {
                "track_id": track_id,
                "file_path": relative_path,
                "file_name": os.path.basename(audio_file),
                "analysis": analysis_result,
                "processing_time": processing_time
            }
            
            # Save individual result to JSON
            output_file = os.path.join(output_path, f"{track_id}.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            logger.info(f"Processed {relative_path} in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {str(e)}")
            # Continue with next file
    
    # Save summary of all results
    summary_file = os.path.join(output_path, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "total_files": len(audio_files),
            "processed_files": len(results),
            "processing_time": sum(r["processing_time"] for r in results),
            "timestamp": time.time()
        }, f, indent=2)
    
    return results

def main():
    args = parse_args()
    
    # Get paths from environment if not provided as arguments (SageMaker sets these)
    s3_input_uri = args.s3_input_uri or os.environ.get("SM_CHANNEL_INPUT")
    s3_output_uri = args.s3_output_uri or os.environ.get("SM_OUTPUT_DATA_DIR")
    use_essentia = args.use_essentia
    
    # Set local paths
    input_path = "/opt/ml/processing/input/data"
    output_path = "/opt/ml/processing/output/analysis"
    
    # Create directories
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Input URI: {s3_input_uri}")
    logger.info(f"Output URI: {s3_output_uri}")
    logger.info(f"Using Essentia: {use_essentia}")
    
    # Download from S3 if s3_input_uri is provided
    if s3_input_uri and s3_input_uri.startswith("s3://"):
        logger.info(f"Downloading from {s3_input_uri} to {input_path}")
        bucket, key = get_s3_bucket_and_key(s3_input_uri)
        
        if key.endswith(("/", "")):
            # It's a directory, download all files
            s3_client = boto3.client("s3")
            paginator = s3_client.get_paginator("list_objects_v2")
            
            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        if not obj["Key"].endswith("/"):  # Skip directories
                            relative_key = obj["Key"][len(key):]
                            target_file = os.path.join(input_path, relative_key)
                            os.makedirs(os.path.dirname(target_file), exist_ok=True)
                            download_from_s3(bucket, obj["Key"], target_file)
        else:
            # It's a single file
            download_from_s3(bucket, key, os.path.join(input_path, os.path.basename(key)))
    
    # Process audio files
    logger.info("Starting audio processing")
    process_audio_files(input_path, output_path, use_essentia)
    logger.info("Audio processing complete")
    
    # Upload results to S3 if s3_output_uri is provided
    if s3_output_uri and s3_output_uri.startswith("s3://"):
        logger.info(f"Uploading results to {s3_output_uri}")
        bucket, key = get_s3_bucket_and_key(s3_output_uri)
        
        for root, _, files in os.walk(output_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, output_path)
                s3_key = os.path.join(key, relative_path).replace("\\", "/")
                upload_to_s3(bucket, s3_key, local_path)
    
    logger.info("Processing job complete")

if __name__ == "__main__":
    main()