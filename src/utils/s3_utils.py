import os
import boto3
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize S3 client
s3_client = boto3.client('s3')

def get_s3_bucket_and_key(s3_uri):
    """Extract bucket and key from S3 URI
    
    Args:
        s3_uri (str): S3 URI in the format s3://bucket/key
        
    Returns:
        tuple: (bucket, key)
    """
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI: {s3_uri}. Must start with 's3://'")
    
    path_parts = s3_uri[5:].split('/', 1)
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ''
    
    return bucket, key

def download_from_s3(bucket, key, local_path):
    """Download a file from S3
    
    This function downloads an object from an S3 bucket to a local file path.
    It automatically creates any necessary directories in the local path.
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key (path within the bucket, e.g., "folder/filename.wav")
        local_path (str): Local path to save the file
        
    Returns:
        str: Local file path where the file was saved
        
    Raises:
        ClientError: If there's an error accessing S3 (permissions, no such object, etc.)
        
    Example:
        >>> download_from_s3("my-bucket", "audio/track1.mp3", "/tmp/downloads/track1.mp3")
        '/tmp/downloads/track1.mp3'
    """
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except ClientError as e:
        logger.error(f"Error downloading from S3: {str(e)}")
        raise

def upload_to_s3(bucket, key, local_path):
    """Upload a file to S3
    
    This function uploads a local file to an S3 bucket with appropriate content types.
    It automatically detects and sets the content type based on file extension:
    - .json files: application/json
    - .csv files: text/csv
    - Other files: Default content type (application/octet-stream)
    
    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key (path within the bucket, e.g., "folder/filename.json")
        local_path (str): Local path of the file to upload
        
    Returns:
        bool: True if successful, False otherwise
        
    Example:
        >>> upload_to_s3("my-bucket", "results/analysis.json", "/tmp/analysis.json")
        True
        
    Note:
        This function requires appropriate AWS credentials with write access to the bucket.
        Errors are logged but exceptions are caught to prevent program termination.
    """
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{key}")
        
        # Set content type based on file extension
        content_type = None
        if key.endswith('.json'):
            content_type = 'application/json'
        elif key.endswith('.csv'):
            content_type = 'text/csv'
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        s3_client.upload_file(local_path, bucket, key, ExtraArgs=extra_args)
        return True
    except ClientError as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        return False