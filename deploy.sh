#!/bin/bash

# Set variables
AWS_ACCOUNT_ID=XXXXXXXXXXXX  # Replace with your AWS account ID
AWS_REGION=us-west-2
S3_BUCKET_NAME=your-bucket-name  # Replace with your S3 bucket name
IMAGE_TAG=latest  # or the specific tag from your build

# Use the correctly formatted role ARN
ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerAudioAnalysisRole"
INPUT_URI="s3://${S3_BUCKET_NAME}/audio-analysis-input/"
OUTPUT_URI="s3://${S3_BUCKET_NAME}/audio-analysis-output/"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/track-audio-analysis:${IMAGE_TAG}"

echo "===== Starting SageMaker Processing Job ====="
echo "Input URI: ${INPUT_URI}"
echo "Output URI: ${OUTPUT_URI}"
echo "Image URI: ${IMAGE_URI}"
echo "Role ARN: ${ROLE_ARN}"
echo "========================================"

# Make sure audio-analysis-input and audio-analysis-output directories exist
echo "Ensuring S3 directories exist..."
aws s3 ls s3://${S3_BUCKET_NAME}/audio-analysis-input/ || aws s3 mb s3://${S3_BUCKET_NAME}/audio-analysis-input/
aws s3 ls s3://${S3_BUCKET_NAME}/audio-analysis-output/ || aws s3 mb s3://${S3_BUCKET_NAME}/audio-analysis-output/

# Check if we have any audio files in the input directory
echo "Checking for audio files in input directory..."
INPUT_FILES=$(aws s3 ls ${INPUT_URI} | grep -E '\.mp3$|\.wav$|\.ogg$|\.flac$|\.m4a$' | wc -l)

if [ "$INPUT_FILES" -eq 0 ]; then
    echo "WARNING: No audio files found in input directory. Please upload audio files to ${INPUT_URI}"
    echo "Would you like to continue anyway? (y/n)"
    read -r CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        echo "Aborting deployment"
        exit 1
    fi
fi

# Run a SageMaker Processing Job
python - << PYTHON_SCRIPT
import boto3
import time
from datetime import datetime

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Generate a unique job name
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
job_name = f"audio-analysis-{timestamp}"

try:
    # Create processing job
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.xlarge',
                'VolumeSizeInGB': 30
            }
        },
        AppSpecification={
            'ImageUri': "${IMAGE_URI}",
            'ContainerArguments': [
                '--s3-input-uri', "${INPUT_URI}",
                '--s3-output-uri', "${OUTPUT_URI}",
                '--use-essentia', 'true'
            ]
        },
        ProcessingInputs=[
            {
                'InputName': 'input-data',
                'S3Input': {
                    'S3Uri': "${INPUT_URI}",
                    'LocalPath': '/opt/ml/processing/input/data',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        ],
        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'analysis-output',
                    'S3Output': {
                        'S3Uri': "${OUTPUT_URI}",
                        'LocalPath': '/opt/ml/processing/output/analysis',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },
        RoleArn="${ROLE_ARN}",
        Environment={
            'ESSENTIA_TENSORFLOW_MODEL_PATH': '/opt/ml/processing/model/',
            'ESSENTIA_TENSORFLOW_MODELS_DIR': '/opt/ml/processing/model/',
            'ESSENTIA_EXTRACTORS_PATH': '/opt/ml/processing/model/discogs-effnet-bs64-1.pb',
            'ESSENTIA_CLASSIFIER_GENRE_PATH': '/opt/ml/processing/model/genre_discogs400-discogs-effnet-1.pb',
            'ESSENTIA_CLASSIFIER_MOOD_PATH': '/opt/ml/processing/model/mood_acoustic-discogs-effnet-1.pb'
        }
    )

    print(f"Started processing job: {job_name}")
    print("Waiting for job to complete...")

    # Wait for job to complete
    while True:
        status = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)['ProcessingJobStatus']
        print(f"Job status: {status}")
        if status in ['Completed', 'Failed', 'Stopped']:
            break
        time.sleep(30)

    if status == 'Completed':
        print("Job completed successfully")
        print(f"Results available at: {OUTPUT_URI}")
        
        # List the output files
        s3_client = boto3.client('s3')
        response = s3_client.list_objects_v2(
            Bucket="${S3_BUCKET_NAME}",
            Prefix="audio-analysis-output/"
        )
        
        if 'Contents' in response:
            print("Output files:")
            for obj in response['Contents'][:10]:  # Show first 10 files
                print(f" - {obj['Key']}")
            if len(response['Contents']) > 10:
                print(f"... and {len(response['Contents']) - 10} more files")
        else:
            print("No output files found")
    else:
        print(f"Job ended with status: {status}")
        
        # Get the failure reason
        job_details = sagemaker_client.describe_processing_job(ProcessingJobName=job_name)
        if 'FailureReason' in job_details:
            print(f"Failure reason: {job_details['FailureReason']}")

except Exception as e:
    print(f"Error creating or monitoring processing job: {str(e)}")
PYTHON_SCRIPT