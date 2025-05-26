# Audio Analysis Project

This project provides audio analysis capabilities using Essentia and TensorFlow models for genre and mood classification. It's designed to analyze audio files and extract high-level features such as music genre and emotional mood.

## Features

- Audio genre classification using Discogs-trained models
- Mood detection (aggressive, relaxed, happy, sad, party, emotional)
- High-level feature extraction
- Support for various audio formats
- Automatic resampling to required sample rates
- Robust error handling and logging

## Prerequisites

- Python 3.7+
- TensorFlow 2.x
- Essentia
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the environment variables:
```bash
# Set the path to your TensorFlow models
export ESSENTIA_TENSORFLOW_MODEL_PATH=/path/to/your/models

# Optional: Enable TensorFlow graph validation
export VALIDATE_TF_GRAPHS=true
```

## Required Models

The project requires the following TensorFlow models (`.pb` files):

1. Feature Extractor Model:
   - `discogs-effnet-bs64-1.pb`

2. Genre Classification Model:
   - A model file containing 'genre_discogs' in its name

3. Mood Classification Model:
   - A model file containing 'mood_' in its name

Place these models in the directory specified by `ESSENTIA_TENSORFLOW_MODEL_PATH`.

## Usage

```python
from src.models.essentia_models import EssentiaModels

# Initialize the analyzer
analyzer = EssentiaModels()

# Load the models
analyzer.load_models()

# Analyze an audio file
import numpy as np
import librosa

# Load audio file
audio_signal, sample_rate = librosa.load('path/to/your/audio/file.mp3', sr=None)

# Analyze the audio
results = analyzer.analyze_audio(audio_signal, sample_rate)

# Print results
print("Genre:", results.get('genre', {}).get('value'))
print("Genre Confidence:", results.get('genre', {}).get('confidence'))
print("Mood:", results.get('mood', {}).get('value'))
print("Mood Confidence:", results.get('mood', {}).get('confidence'))
```

## Model Details

### Genre Classification
The genre model supports the following genres:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock
- Electronic
- Folk
- Soul
- Funk
- Alternative
- Indie
- Ambient
- Experimental

### Mood Classification
The mood model detects the following moods:
- Aggressive
- Relaxed
- Happy
- Sad
- Party
- Emotional

## Error Handling

The system includes comprehensive error handling and logging. All operations are logged with appropriate detail levels. Common error scenarios include:

- Missing model files
- Invalid audio input
- Model loading failures
- Feature extraction errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Acknowledgments

- Essentia for audio analysis capabilities
- TensorFlow for model support
- Discogs for genre classification training data 