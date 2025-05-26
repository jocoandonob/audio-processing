import os
import json
import logging
import numpy as np
import time
import soundfile as sf
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Audio analyzer class that uses Essentia and Librosa for audio analysis"""
    
    def __init__(self, use_essentia=True):
        """Initialize the audio analyzer
        
        Args:
            use_essentia (bool): Whether to use Essentia for analysis
        """
        self.use_essentia = use_essentia
        
        # Import these here to avoid slow startup
        import librosa
        self.librosa = librosa
        
        # Import Essentia if enabled
        if self.use_essentia:
            try:
                # Import the EssentiaModels class from the models directory
                from src.models.essentia_models import EssentiaModels
                self.essentia_models = EssentiaModels()
                logger.info("Essentia initialized successfully")
                
                # Check if models are available
                model_path = os.environ.get('ESSENTIA_TENSORFLOW_MODEL_PATH', '/opt/ml/processing/model/')
                if os.path.isdir(model_path):
                    models = [os.path.basename(m) for m in os.listdir(model_path) if m.endswith('.pb')]
                    logger.info(f"Found Essentia models: {models}")
                else:
                    logger.warning(f"Essentia models not found in {model_path}")
            except ImportError as e:
                logger.warning(f"Essentia import failed: {str(e)}")
                logger.warning("Falling back to Librosa only")
                self.use_essentia = False
    
    def analyze(self, file_path):
        """Analyze audio file and extract features using Librosa and optionally Essentia
        
        This is the main entry point for audio analysis. It performs the following steps:
        1. Loads the audio file using Librosa
        2. Performs basic analysis with Librosa (always)
        3. If enabled, performs advanced analysis with Essentia models
        4. Combines and structures the results
        
        Args:
            file_path (str): Path to audio file (mp3, wav, ogg, flac, m4a, etc.)
            
        Returns:
            dict: A structured dictionary with analysis results with the following format:
                {
                    "version": "1.0.0",
                    "analyses": [
                        {
                            "processor": "librosa",
                            "tempo": {"value": 120.0, "confidence": 0.8},
                            "key": {"value": "C major", "confidence": 0.6},
                            "energy": {"value": 0.75, "variance": 0.05, "confidence": 0.9},
                            "beat_consistency": {"value": 0.8, "confidence": 0.8},
                            "duration": 180.5,
                            "spectral": {...},
                            "zero_crossing_rate": 0.05,
                            "spectral_contrast": 0.3,
                            "mfcc": {"mean": [...], "variance": [...], "confidence": 0.9},
                            "processing_time": 1.23
                        },
                        {
                            "processor": "essentia",
                            "classification": {
                                "genre": {"value": "rock", "confidence": 0.85},
                                "mood": {"value": "energetic", "confidence": 0.7},
                                "instruments": [],
                                "confidence": 0.85
                            },
                            "duration": 180.5,
                            "essentia_processing_time": 2.45
                        }
                    ]
                }
                
                If an error occurs during analysis, the structure will be:
                {
                    "version": "1.0.0",
                    "error": "Error message",
                    "analyses": []
                }
        """
        try:
            # Load audio file with librosa
            y, sr = self.librosa.load(file_path, sr=None)
            
            # Initialize results dictionary with metadata
            results = {
                "version": "1.0.0",
                "analyses": []
            }
            
            # Always run librosa analysis
            librosa_results = self._analyze_with_librosa(y, sr)
            librosa_results["processor"] = "librosa"
            results["analyses"].append(librosa_results)
            
            # Run Essentia analysis if enabled
            if self.use_essentia:
                essentia_results = self._analyze_with_essentia(y, sr, file_path)
                
                # Only add essentia results if there was no error
                if "error" not in essentia_results:
                    essentia_results["processor"] = "essentia"
                    
                    # Structure classification data
                    essentia_results["classification"] = {
                        "genre": essentia_results.pop("genre", {"value": "unknown", "confidence": 0.0}),
                        "mood": essentia_results.pop("mood", {"value": "neutral", "confidence": 0.0}),
                        "instruments": [],  # No instrument detection yet
                        "confidence": 0.0   # Will be updated below
                    }
                    
                    # Calculate overall confidence as max of individual confidences
                    genre_confidence = essentia_results["classification"]["genre"].get("confidence", 0.0)
                    mood_confidence = essentia_results["classification"]["mood"].get("confidence", 0.0)
                    essentia_results["classification"]["confidence"] = max(genre_confidence, mood_confidence)
                    
                    results["analyses"].append(essentia_results)
                else:
                    logger.warning(f"Essentia analysis error: {essentia_results.get('error')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return {
                "version": "1.0.0",
                "error": str(e),
                "analyses": []
            }
    
    def _analyze_with_librosa(self, y, sr):
        """Analyze audio with Librosa
        
        Args:
            y (numpy.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Analysis results
        """
        start_time = time.time()
        
        # Calculate duration
        duration = float(len(y) / sr)
        
        try:
            # Extract tempo and beats
            tempo, beats = self.librosa.beat.beat_track(y=y, sr=sr)
            tempo_result = {"value": float(tempo), "confidence": 0.8}
            
            # Extract key
            chroma = self.librosa.feature.chroma_cqt(y=y, sr=sr)
            key_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            chroma_sum = np.sum(chroma, axis=1)
            key_index = np.argmax(chroma_sum)
            
            # Simple major/minor detection
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            
            major_profile = major_profile / np.sum(major_profile)
            minor_profile = minor_profile / np.sum(minor_profile)
            
            major_correlation = np.corrcoef(chroma_sum, major_profile)[0, 1]
            minor_correlation = np.corrcoef(chroma_sum, minor_profile)[0, 1]
            
            key_type = "major" if major_correlation > minor_correlation else "minor"
            key_confidence = max(major_correlation, minor_correlation)
            
            key_result = {"value": f"{key_map[key_index]} {key_type}", "confidence": float(key_confidence)}
            
            # Calculate energy
            rms = self.librosa.feature.rms(y=y)[0]
            energy_value = float(np.mean(rms))
            energy_variance = float(np.var(rms))
            energy_result = {
                "value": min(1.0, energy_value * 10),  # Scale to 0-1
                "variance": energy_variance,
                "confidence": 0.9
            }
            
            # Calculate beat consistency
            beat_consistency_result = {"value": 0.5, "confidence": 0.5}
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                if len(beat_intervals) > 0 and np.mean(beat_intervals) > 0:
                    cv = np.std(beat_intervals) / np.mean(beat_intervals)
                    consistency = max(0.0, min(1.0, 1.0 - cv))
                    beat_consistency_result = {"value": float(consistency), "confidence": 0.8}
            
            # Extract spectral features
            spectral_centroid = self.librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_bandwidth = self.librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_rolloff = self.librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            spectral_result = {
                "centroid": float(np.mean(spectral_centroid) / (sr/2)),  # Normalize to 0-1
                "bandwidth": float(np.mean(spectral_bandwidth) / (sr/2)),
                "rolloff": float(np.mean(spectral_rolloff) / (sr/2)),
                "confidence": 0.9
            }
            
            # Extract MFCC features
            mfccs = self.librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_result = {
                "mean": [float(x) for x in np.mean(mfccs, axis=1)],
                "variance": [float(x) for x in np.var(mfccs, axis=1)],
                "confidence": 0.9
            }
            
            # Extract zero crossing rate
            zcr = self.librosa.feature.zero_crossing_rate(y)[0]
            zcr_value = float(np.mean(zcr))
            
            # Extract spectral contrast
            contrast = self.librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_value = float(np.mean(contrast))
            
            return {
                "processor": "librosa",
                "tempo": tempo_result,
                "key": key_result,
                "energy": energy_result,
                "beat_consistency": beat_consistency_result,
                "duration": duration,
                "spectral": spectral_result,
                "zero_crossing_rate": zcr_value,
                "spectral_contrast": contrast_value,
                "mfcc": mfcc_result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error in Librosa analysis: {str(e)}")
            
            # Return minimal valid structure with default values
            return {
                "processor": "librosa",
                "tempo": {"value": 120, "confidence": 0},
                "key": {"value": "C major", "confidence": 0},
                "energy": {"value": 0.5, "variance": 0, "confidence": 0},
                "beat_consistency": {"value": 0.5, "confidence": 0},
                "duration": duration,
                "spectral": {"centroid": 0, "bandwidth": 0, "rolloff": 0, "confidence": 0},
                "zero_crossing_rate": 0,
                "spectral_contrast": 0,
                "mfcc": {"mean": [0] * 13, "variance": [0] * 13, "confidence": 0},
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _analyze_with_essentia(self, y, sr, file_path):
        """Analyze audio with Essentia TensorFlow models for high-level features
        
        This method delegates the advanced audio analysis to the EssentiaModels class.
        It provides high-level semantic features like genre and mood classification
        that complement the low-level acoustic features from Librosa.
        
        The analysis pipeline:
        1. Sends the audio signal to EssentiaModels.analyze_audio()
        2. Gets back classification results (genre, mood)
        3. Adds metadata like processing time and duration
        
        Args:
            y (numpy.ndarray): Audio time series as a 1D numpy array with values in [-1, 1]
            sr (int): Sample rate of the audio signal
            file_path (str): Path to original audio file (for logging/debugging)
            
        Returns:
            dict: Analysis results containing:
                - genre: Genre classification with confidence
                - mood: Mood classification with confidence
                - duration: Length of the audio in seconds
                - essentia_processing_time: Time taken for Essentia analysis
                - error: Present only if an error occurred
                
        Note:
            If any error occurs during Essentia analysis, an error message will be
            included in the results dictionary, and the Librosa results will still
            be available to the caller.
        """
        start_time = time.time()
        
        try:
            # Initialize result dictionary with duration
            result = {
                "duration": float(len(y) / sr)
            }
            
            # Use the EssentiaModels class to analyze audio directly
            essentia_results = self.essentia_models.analyze_audio(y, sr)
            
            # Add Essentia results to the output
            for key, value in essentia_results.items():
                result[key] = value
                
            # Add processing time
            processing_time = time.time() - start_time
            result["essentia_processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Essentia analysis: {e}")
            return {"error": str(e)}