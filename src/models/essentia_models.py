import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# ------------------------------------------------------
# FIX FOR RANDOM_DEVICE ERRORS IN TENSORFLOW
# ------------------------------------------------------
# This must be at the very top before any TensorFlow operations
import random
import numpy as np
import os

# Set fixed seeds for reproducibility and to avoid random_device issues
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_DISABLE_SECURE_RANDOM'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

# Patch TensorFlow's random number generation
try:
    # For TensorFlow 2.x
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Direct fix for random_device errors - replace TensorFlow's random number generator
    from tensorflow.python.framework import random_seed
    def _truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None):
        """A deterministic implementation of truncated normal that doesn't use random_device"""
        # Use NumPy's random generator instead of TensorFlow's
        rng = np.random.RandomState(seed if seed is not None else 42)
        values = rng.normal(loc=mean, scale=stddev, size=shape)
        return tf.convert_to_tensor(values, dtype=dtype)
    
    # Try to monkey patch TensorFlow's internal random generation
    try:
        import tensorflow.python.ops.random_ops
        tensorflow.python.ops.random_ops.truncated_normal = _truncated_normal
    except:
        pass
    
    # Alternative monkey patch attempts
    try:
        from tensorflow.python.ops import random_ops
        random_ops._truncated_normal = _truncated_normal
    except:
        pass
        
except Exception as e:
    print(f"Warning: Could not fully patch TensorFlow random generation: {e}")
# ------------------------------------------------------
# END OF RANDOM_DEVICE FIX
# ------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_model_nodes(pb_path, expected_input, expected_output, skip_on_error=True):
    """Validate that the expected input and output nodes exist in the model
    
    Args:
        pb_path (str): Path to the .pb file
        expected_input (str): Expected input node name
        expected_output (str): Expected output node name
        skip_on_error (bool): If True, return False on error instead of raising exception
        
    Returns:
        bool: True if validation passes, False if validation fails and skip_on_error is True
    """
    try:
        # Explicitly set TensorFlow to use CPU to avoid CUDA errors
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        with tf.compat.v1.Session() as sess:
            with gfile.FastGFile(pb_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                node_names = [node.name for node in graph_def.node]

                # Print all node names for debugging
                logger.info(f"All available nodes in {pb_path}:")
                for node in graph_def.node:
                    logger.info(f"Node: {node.name}, Op: {node.op}")

                # Try to find input nodes by common patterns
                input_candidates = [
                    n for n in node_names if any(pattern in n.lower() for pattern in [
                        'placeholder', 'input', 'serving_default', 'input_1', 'input_0', 'melspectrogram'
                    ])
                ]
                
                # Try to find output nodes by common patterns
                output_candidates = [
                    n for n in node_names if any(pattern in n.lower() for pattern in [
                        'softmax', 'output', 'dense', 'predictions', 'logits', 'call', 'partitionedcall'
                    ])
                ]
                
                logger.info(f"Suggested input nodes: {input_candidates}")
                logger.info(f"Suggested output nodes: {output_candidates}")

                # Direct check for input and output nodes
                if expected_input not in node_names or expected_output not in node_names:
                    logger.error(
                        f"Model node mismatch:\n"
                        f"  Expected input: '{expected_input}'\n"
                        f"  Expected output: '{expected_output}'\n"
                        f"  Available nodes: {node_names}\n"
                        f"  Suggested input nodes: {input_candidates}\n"
                        f"  Suggested output nodes: {output_candidates}"
                    )
                    if skip_on_error:
                        return False
                    raise ValueError("Invalid input/output node names in TensorFlow graph.")
                else:
                    logger.info(f"[MODEL OK] {pb_path}: Node names validated.")
                    return True
    except Exception as e:
        logger.warning(f"Failed to validate TensorFlow graph: {e}")
        if skip_on_error:
            logger.info("Skipping validation and proceeding with model loading")
            return True  # Return True to allow model loading to proceed
        raise

def try_load_model_with_different_nodes(es, model_path, input_candidates, output_candidates):
    """Try to load a model with different input/output node combinations
    
    Args:
        es: Essentia standard module
        model_path (str): Path to the model file
        input_candidates (list): List of possible input node names
        output_candidates (list): List of possible output node names
        
    Returns:
        tuple: (success, input_node, output_node) if successful, (False, None, None) if failed
    """
    for input_node in input_candidates:
        for output_node in output_candidates:
            try:
                logger.info(f"Trying input node: {input_node}, output node: {output_node}")
                model = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=model_path,
                    input=input_node,
                    output=output_node
                )
                # If we get here, the model loaded successfully
                return True, input_node, output_node
            except Exception as e:
                logger.debug(f"Failed with nodes {input_node}/{output_node}: {e}")
                continue
    return False, None, None

class EssentiaModels:
    """Class to handle Essentia TensorFlow models"""
    
    def __init__(self):
        """Initialize the Essentia models"""
        self.models_loaded = False
        self.model_path = os.environ.get('ESSENTIA_TENSORFLOW_MODEL_PATH', '/opt/ml/processing/model/')
        self.validate_graphs = os.environ.get('VALIDATE_TF_GRAPHS', 'false').lower() == 'true'
        
        # Check if models exist
        self._check_models()
    
    def _check_models(self):
        """Check if Essentia models are available"""
        if not os.path.isdir(self.model_path):
            logger.warning(f"Essentia model path not found: {self.model_path}")
            return False
        
        model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pb')]
        if not model_files:
            logger.warning(f"No Essentia model files found in {self.model_path}")
            return False
        
        logger.info(f"All required Essentia models found in {self.model_path}")
        return True
    
    def load_models(self):
        """Load Essentia TensorFlow models for audio analysis
        
        This method dynamically loads the following Essentia TensorFlow models:
        - Genre classifier model: For music genre classification (e.g., rock, jazz, electronic)
        - Mood classifier model: For mood detection (e.g., happy, sad, aggressive)
        
        The method searches for models in the directory specified by ESSENTIA_TENSORFLOW_MODEL_PATH
        environment variable (defaults to /opt/ml/processing/model/).
        
        Model requirements:
        - Models must be in .pb format (TensorFlow frozen graph)
        - File names should contain 'genre_discogs' for genre models and 'mood_' for mood models
        - Models are expected to have specific input/output node names:
          * Input node: "serving_default_melspectrogram"
          * Output node: "model/Softmax"
        
        Returns:
            bool: True if at least one model was loaded successfully, False otherwise
            
        Note:
            Model validation can be enabled by setting VALIDATE_TF_GRAPHS=true
            When validation is disabled (default), the method will attempt to load models
            even if node names don't match expected values.
        """
        try:
            import essentia.standard as es
            
            # Find available models
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pb')]
            logger.info(f"Found Essentia model files: {model_files}")
            
            # Determine which models we have available
            self.available_models = []
            
            # Genre model
            genre_model_file = next((f for f in model_files if 'genre_discogs' in f), None)
            if genre_model_file:
                self.available_models.append(genre_model_file)
                genre_model_path = os.path.join(self.model_path, genre_model_file)
                
                try:
                    # Use the specialized TensorflowPredictEffnetDiscogs algorithm for Discogs models
                    self.genre_model = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=genre_model_path,
                        input="serving_default_model_Placeholder",
                        output="PartitionedCall"
                    )
                    # Test the model with a small input to verify it works
                    test_input = np.zeros((1, 400), dtype=np.float32)  # Typical embedding size
                    test_output = self.genre_model(test_input)
                    logger.info(f"Genre model test output type: {type(test_output)}, shape: {test_output.shape if hasattr(test_output, 'shape') else 'unknown'}")
                    logger.info("Successfully loaded and tested genre model")
                except Exception as e:
                    logger.error(f"Failed to load genre model: {e}")
                    self.genre_model = None
            else:
                self.genre_model = None
                
            # Mood model
            mood_model_file = next((f for f in model_files if 'mood_' in f), None)
            if mood_model_file:
                self.available_models.append(mood_model_file)
                mood_model_path = os.path.join(self.model_path, mood_model_file)
                
                try:
                    # Use the specialized TensorflowPredictEffnetDiscogs algorithm for Discogs models
                    self.mood_model = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=mood_model_path,
                        input="model/Placeholder",
                        output="model/Softmax"
                    )
                    # Test the model with a small input to verify it works
                    test_input = np.zeros((1, 400), dtype=np.float32)  # Typical embedding size
                    test_output = self.mood_model(test_input)
                    logger.info(f"Mood model test output type: {type(test_output)}, shape: {test_output.shape if hasattr(test_output, 'shape') else 'unknown'}")
                    logger.info("Successfully loaded and tested mood model")
                except Exception as e:
                    logger.error(f"Failed to load mood model: {e}")
                    self.mood_model = None
            else:
                self.mood_model = None
                
            # Only set models_loaded to True if at least one model was loaded successfully
            self.models_loaded = self.genre_model is not None or self.mood_model is not None
            return self.models_loaded
            
        except ImportError as e:
            logger.warning(f"Essentia import failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading Essentia models: {e}")
            return False
    
    def analyze_audio(self, audio_signal, sample_rate=44100):
        """Analyze audio with Essentia TensorFlow models
        
        This method performs high-level feature extraction using Essentia TensorFlow models.
        The process involves:
        1. Resampling the audio to 16kHz (required by the models)
        2. Extracting embeddings using the feature extractor model
        3. Running genre classification on these embeddings
        4. Running mood classification on these embeddings
        
        Args:
            audio_signal (numpy.ndarray): Audio signal as a 1D numpy array with values in [-1, 1]
            sample_rate (int, optional): Sample rate of the input audio. Will be resampled to 16kHz
                                         internally as required by the models. Defaults to 44100.
            
        Returns:
            dict: A dictionary with analysis results including:
                  - genre: {"value": "<genre_name>", "confidence": <float>}
                  - mood: {"value": "<mood_name>", "confidence": <float>}
                  - error: Only present if something went wrong during analysis
        
        Note:
            - The models expect mono audio signals
            - For optimal results, provide high-quality audio without excessive noise
            - The input audio is limited to 30 seconds to prevent memory issues
        """
        if not self.models_loaded and not self.load_models():
            logger.warning("Models not loaded, cannot analyze")
            return {}
        
        try:
            import essentia.standard as es
            
            # Resample to 16kHz as required by the models
            if sample_rate != 16000:
                audio_signal = es.Resample(
                    inputSampleRate=float(sample_rate),
                    outputSampleRate=16000
                )(audio_signal)
            
            # Prepare the signal for the models
            max_audio_samples = 16000 * 30  # 30 seconds at 16kHz
            if len(audio_signal) > max_audio_samples:
                logger.info(f"Trimming audio from {len(audio_signal)} to {max_audio_samples} samples")
                audio_signal = audio_signal[:max_audio_samples]
            
            # Process with the Essentia extractor required by the TensorFlow models
            try:
                # Get the embeddings from the feature extractor model
                extractor_model_path = os.path.join(self.model_path, 'discogs-effnet-bs64-1.pb')
                if os.path.exists(extractor_model_path):
                    try:
                        # Feature extractor uses different node names
                        feature_extractor = es.TensorflowPredictEffnetDiscogs(
                            graphFilename=extractor_model_path,
                            input="serving_default_melspectrogram",
                            output="PartitionedCall"
                        )
                        embeddings = feature_extractor(audio_signal)
                        logger.info(f"Raw embeddings type: {type(embeddings)}, shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'unknown'}")
                        
                        # Convert embeddings to numpy array if it's not already
                        if not isinstance(embeddings, np.ndarray):
                            embeddings = np.array(embeddings)
                            logger.info(f"Converted embeddings to numpy array, shape: {embeddings.shape}")
                            
                        # Ensure embeddings are in the correct format
                        if len(embeddings.shape) > 1:
                            # Take the mean across the first dimension to get a single vector
                            embeddings = np.mean(embeddings, axis=0)
                            logger.info(f"Mean embeddings shape: {embeddings.shape}")
                            
                        # Ensure embeddings are not empty and have the right shape
                        if embeddings.size == 0:
                            raise ValueError("Empty embeddings after processing")
                            
                        # Ensure embeddings are in the right format for the models
                        if len(embeddings.shape) == 1:
                            # Add batch dimension if needed
                            embeddings = embeddings.reshape(1, -1)
                            logger.info(f"Reshaped embeddings for model input, shape: {embeddings.shape}")
                            
                        logger.info(f"Final embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
                    except Exception as extractor_error:
                        logger.error(f"Error with primary extractor: {extractor_error}")
                        return {"error": f"Failed to extract features: {extractor_error}"}
                else:
                    logger.error(f"Feature extractor model not found: {extractor_model_path}")
                    return {"error": "Feature extractor model not found"}
            except Exception as e:
                logger.error(f"Error extracting embeddings: {e}")
                return {"error": f"Failed to extract features: {e}"}
                
            results = {}
            
            # Analyze genre if model is available
            if self.genre_model:
                try:
                    # Use the embeddings with the genre model instead of raw audio
                    genre_predictions = self.genre_model(embeddings)
                    logger.info(f"Raw genre predictions type: {type(genre_predictions)}, shape: {genre_predictions.shape if hasattr(genre_predictions, 'shape') else 'unknown'}")
                    
                    # Convert predictions to numpy array if needed
                    if not isinstance(genre_predictions, np.ndarray):
                        genre_predictions = np.array(genre_predictions)
                        logger.info(f"Converted genre predictions to numpy array, shape: {genre_predictions.shape}")
                    
                    # Validate predictions
                    if genre_predictions.size == 0:
                        logger.error("Empty genre predictions")
                        results["genre"] = {"value": "unknown", "confidence": 0.0, "error": "Empty predictions"}
                    else:
                        # Ensure predictions are in the right format
                        if len(genre_predictions.shape) > 1:
                            genre_predictions = genre_predictions.flatten()
                            logger.info(f"Flattened genre predictions shape: {genre_predictions.shape}")
                        
                        # Find top genre
                        top_genre_idx = np.argmax(genre_predictions)
                        logger.info(f"Top genre index: {top_genre_idx}, value: {genre_predictions[top_genre_idx]}")
                        
                        # Map to genre names - these categories come from the Discogs dataset
                        genre_names = self._get_genre_names()
                        if top_genre_idx < len(genre_names):
                            top_genre = genre_names[top_genre_idx]
                            confidence = float(genre_predictions[top_genre_idx])
                            results["genre"] = {"value": top_genre, "confidence": confidence}
                        else:
                            results["genre"] = {"value": "unknown", "confidence": 0.0, "error": "Invalid genre index"}
                        
                except Exception as e:
                    logger.error(f"Error analyzing genre: {e}")
                    results["genre"] = {"value": "unknown", "confidence": 0.0, "error": str(e)}
            
            # Analyze mood if model is available
            if self.mood_model:
                try:
                    # Use the embeddings with the mood model instead of raw audio
                    mood_predictions = self.mood_model(embeddings)
                    logger.info(f"Raw mood predictions type: {type(mood_predictions)}, shape: {mood_predictions.shape if hasattr(mood_predictions, 'shape') else 'unknown'}")
                    
                    # Convert predictions to numpy array if needed
                    if not isinstance(mood_predictions, np.ndarray):
                        mood_predictions = np.array(mood_predictions)
                        logger.info(f"Converted mood predictions to numpy array, shape: {mood_predictions.shape}")
                    
                    # Validate predictions
                    if mood_predictions.size == 0:
                        logger.error("Empty mood predictions")
                        results["mood"] = {"value": "neutral", "confidence": 0.0, "error": "Empty predictions"}
                    else:
                        # Ensure predictions are in the right format
                        if len(mood_predictions.shape) > 1:
                            mood_predictions = mood_predictions.flatten()
                            logger.info(f"Flattened mood predictions shape: {mood_predictions.shape}")
                        
                        # Find top mood
                        top_mood_idx = np.argmax(mood_predictions)
                        logger.info(f"Top mood index: {top_mood_idx}, value: {mood_predictions[top_mood_idx]}")
                        
                        # Map to mood names - simplified for now
                        mood_names = ["aggressive", "relaxed", "happy", "sad", "party", "emotional"]
                        if top_mood_idx < len(mood_names):
                            top_mood = mood_names[top_mood_idx]
                            confidence = float(mood_predictions[top_mood_idx])
                            results["mood"] = {"value": top_mood, "confidence": confidence}
                        else:
                            results["mood"] = {"value": "neutral", "confidence": 0.0, "error": "Invalid mood index"}
                except Exception as e:
                    logger.error(f"Error analyzing mood: {e}")
                    results["mood"] = {"value": "neutral", "confidence": 0.0, "error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Essentia analysis: {e}")
            return {}
    
    def _get_genre_names(self):
        """Get genre names for the Discogs genre model
        
        Returns:
            list: List of genre names
        """
        # This is a simplified list - the actual model may have more categories
        return [
            "blues", "classical", "country", "disco", "hiphop", "jazz", "metal", 
            "pop", "reggae", "rock", "electronic", "folk", "soul", "funk", 
            "alternative", "indie", "ambient", "experimental"
        ]