FROM public.ecr.aws/docker/library/python:3.9-slim

# Install system dependencies for audio processing and Essentia
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories for code and models
WORKDIR /opt/ml/processing

# Create directories for input and output
RUN mkdir -p /opt/ml/processing/input \
    /opt/ml/processing/input/data \
    /opt/ml/processing/output \
    /opt/ml/processing/output/analysis \
    /opt/ml/processing/model \
    && chmod -R 755 /opt/ml/processing

# Copy requirements and install dependencies
COPY requirements.txt .
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Install specific versions known to work with Essentia
RUN pip install --upgrade pip && \
    # Install packages in the right order
    pip install --no-cache-dir numpy==1.22.4 && \
    # Install protobuf first with a compatible version for TensorFlow 2.8.0
    pip install --no-cache-dir protobuf==3.20.3 && \
    pip install --no-cache-dir tensorflow-cpu==2.8.0 && \
    # Force reinstall scipy and numba to ensure compatibility
    pip install --no-cache-dir --force-reinstall scipy==1.7.3 numba==0.55.2 && \
    # Install Essentia version compatible with the models
    pip install --no-cache-dir essentia-tensorflow==2.1b6.dev1110

# Download Essentia TensorFlow models
RUN mkdir -p /opt/ml/processing/model && \
    wget -q https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb -P /opt/ml/processing/model/ && \
    wget -q https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb -P /opt/ml/processing/model/ && \
    wget -q https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb -P /opt/ml/processing/model/ && \
    chmod -R 755 /opt/ml/processing/model

# Create a Python script to fix random device issues
RUN echo 'import os\nimport random\nimport numpy as np\n\n# Patch for random_device issues\ntry:\n    # Set fixed random seeds\n    random.seed(42)\n    np.random.seed(42)\n    # Try to patch tensorflow random seed\n    import tensorflow as tf\n    tf.random.set_seed(42)\n    # Override TensorFlow random source\n    import tensorflow.python.framework.random_seed as tfrandom\n    def _truncated_normal(mean=0.0, stddev=1.0, shape=None, dtype=None, seed=None):\n        """Generate a non-random truncated normal distribution using numpy instead."""\n        rng = np.random.RandomState(42 if seed is None else seed)\n        return tf.convert_to_tensor(rng.normal(loc=mean, scale=stddev, size=shape), dtype=dtype)\n    tfrandom.get_truncated_normal = _truncated_normal\nexcept Exception as e:\n    print(f"Warning: Could not fully patch random generation: {e}")\n\n# Set environment variables\nos.environ["TF_DETERMINISTIC_OPS"] = "1"\nos.environ["TF_DISABLE_SECURE_RANDOM"] = "1"\nos.environ["PYTHONHASHSEED"] = "0"\n\n# Add environment variable to make protobuf use pure Python implementation\nos.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"\n\n# Import and run the real script\nimport sys\nimport runpy\nsys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name="__main__")' > /opt/ml/processing/run_with_random_fix.py && \
    chmod +x /opt/ml/processing/run_with_random_fix.py

# Set environment variables for Essentia
ENV ESSENTIA_TENSORFLOW_MODEL_PATH=/opt/ml/processing/model/
ENV ESSENTIA_TENSORFLOW_MODELS_DIR=/opt/ml/processing/model/
ENV ESSENTIA_EXTRACTORS_PATH=/opt/ml/processing/model/discogs-effnet-bs64-1.pb
ENV ESSENTIA_CLASSIFIER_GENRE_PATH=/opt/ml/processing/model/genre_discogs400-discogs-effnet-1.pb
ENV ESSENTIA_CLASSIFIER_MOOD_PATH=/opt/ml/processing/model/mood_acoustic-discogs-effnet-1.pb
# Set TensorFlow environment variables to fix random_device issues
ENV TF_DETERMINISTIC_OPS=1
ENV TF_DISABLE_SECURE_RANDOM=1
ENV PYTHONHASHSEED=0
ENV TF_CPP_MIN_LOG_LEVEL=2
# Disable graph validation
ENV VALIDATE_TF_GRAPHS=false
# Set protobuf implementation to Python to avoid C++ issues
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Copy source code
COPY src /opt/ml/processing/src
COPY processing_script.py /opt/ml/processing/

# Use our Python wrapper that fixes random issues before running the script
ENTRYPOINT ["python", "/opt/ml/processing/run_with_random_fix.py", "processing_script.py"]