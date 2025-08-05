# backend/src/prediction.py

#!/usr/bin/env python3
"""
Enhanced Flower Predictor with optimized real image processing
"""

import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Union, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import image processing
try:
    from PIL import Image, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image processing will be limited")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

class FlowerPredictor:
    """Enhanced flower image predictor for real images"""
    
    def __init__(self, model_manager=None, input_size=(128, 128)):
        self.model_manager = model_manager
        self.input_size = input_size
        self.classes = ['rose', 'tulip', 'sunflower']
        self.upload_dir = os.path.join('data', 'uploads')
        
        # Create uploads directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        for class_name in self.classes:
            os.makedirs(os.path.join(self.upload_dir, class_name), exist_ok=True)

        # Class colors for UI (more vibrant and distinct)
        self.class_colors = {
            'rose': '#e91e63',      # Pink/Red
            'tulip': '#9c27b0',     # Purple
            'sunflower': '#ffc107'  # Yellow/Gold
        }
        
        # Supported image formats
        self.supported_formats = self._get_supported_formats()
        
        logger.info("✅ Enhanced FlowerPredictor initialized")
        logger.info(f"Supported formats: {', '.join(self.supported_formats)}")
    
    def _get_supported_formats(self) -> List[str]:
        """Get list of supported image formats based on available libraries"""
        formats = []
        
        if PIL_AVAILABLE:
            formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
        elif CV2_AVAILABLE:
            formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        else:
            logger.warning("No image processing library available")
            formats = ['jpg', 'jpeg', 'png']  # Basic assumption
        
        return formats
    
    def save_uploaded_image(self, image_input: Union[np.ndarray, Image.Image, str], class_name: str = None) -> Optional[str]:
        """Save uploaded image to the appropriate directory"""
        try:
            if class_name and class_name not in self.classes:
                logger.error(f"Invalid class name: {class_name}")
                return None
            
            # Preprocess the image first
            image_array = self.preprocess_image(image_input)
            if image_array is None:
                logger.error("Failed to preprocess image for saving")
                return None
            
            # Convert back to uint8 for saving
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"upload_{timestamp}.jpg"
            
            # Determine save path
            if class_name:
                save_dir = os.path.join(self.upload_dir, class_name)
            else:
                save_dir = os.path.join(self.upload_dir, 'unclassified')
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            
            # Save the image
            if PIL_AVAILABLE:
                Image.fromarray(image_array).save(save_path)
            elif CV2_AVAILABLE:
                # Convert RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_image)
            else:
                logger.error("No image saving library available")
                return None
            
            logger.info(f"✅ Saved uploaded image to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save uploaded image: {e}")
            return None
    
    def preprocess_image(self, image_input: Union[np.ndarray, Image.Image, str]) -> Optional[np.ndarray]:
        """Enhanced image preprocessing with better error handling"""
        try:
            image_array = None
            
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return None
                
                # Validate file extension
                ext = os.path.splitext(image_input)[1].lower().lstrip('.')
                if ext not in self.supported_formats:
                    logger.error(f"Unsupported format: {ext}")
                    return None
                
                if PIL_AVAILABLE:
                    try:
                        image = Image.open(image_input)
                        # Handle EXIF orientation
                        image = ImageOps.exif_transpose(image)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image_array = np.array(image)
                    except Exception as e:
                        logger.error(f"PIL failed to load image: {e}")
                        return None
                elif CV2_AVAILABLE:
                    try:
                        image_array = cv2.imread(image_input)
                        if image_array is None:
                            logger.error("OpenCV failed to load image")
                            return None
                        # Convert BGR to RGB
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        logger.error(f"OpenCV failed to load image: {e}")
                        return None
                else:
                    logger.error("No image loading library available")
                    return None
            
            elif isinstance(image_input, Image.Image):
                # PIL Image
                try:
                    # Handle EXIF orientation
                    image_input = ImageOps.exif_transpose(image_input)
                    if image_input.mode != 'RGB':
                        image_input = image_input.convert('RGB')
                    image_array = np.array(image_input)
                except Exception as e:
                    logger.error(f"Failed to process PIL image: {e}")
                    return None
            
            elif isinstance(image_input, np.ndarray):
                # NumPy array
                image_array = image_input.copy()
                
                # Validate array
                if len(image_array.shape) not in [2, 3]:
                    logger.error(f"Invalid image array shape: {image_array.shape}")
                    return None
            
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return None
            
            # Validate image array
            if image_array is None or image_array.size == 0:
                logger.error("Empty or invalid image array")
                return None
            
            # Handle grayscale images (convert to RGB)
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            elif len(image_array.shape) == 3:
                if image_array.shape[-1] == 4:
                    # Remove alpha channel
                    image_array = image_array[:, :, :3]
                elif image_array.shape[-1] == 1:
                    # Convert single channel to RGB
                    image_array = np.repeat(image_array, 3, axis=-1)
                elif image_array.shape[-1] != 3:
                    logger.error(f"Unsupported number of channels: {image_array.shape[-1]}")
                    return None
            
            # Resize image to input size
            if image_array.shape[:2] != self.input_size:
                if PIL_AVAILABLE:
                    # Use PIL for high-quality resizing
                    pil_image = Image.fromarray(image_array.astype('uint8'))
                    pil_image = pil_image.resize(self.input_size, Image.Resampling.LANCZOS)
                    image_array = np.array(pil_image)
                elif CV2_AVAILABLE:
                    # Use OpenCV for resizing
                    image_array = cv2.resize(image_array, self.input_size, interpolation=cv2.INTER_LANCZOS4)
                else:
                    logger.warning("No resize library available, using original size")
                    # Basic nearest neighbor resizing as fallback
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        self.input_size[0] / image_array.shape[0],
                        self.input_size[1] / image_array.shape[1],
                        1
                    )
                    image_array = zoom(image_array, zoom_factors, order=1)
            
            # Convert to float32 and normalize
            image_array = image_array.astype(np.float32)
            
            # Normalize to [0, 1] range
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            # Clip values to ensure they're in valid range
            image_array = np.clip(image_array, 0.0, 1.0)
            
            # Final validation
            if image_array.shape != (*self.input_size, 3):
                logger.error(f"Final shape mismatch: {image_array.shape} vs {(*self.input_size, 3)}")
                return None
            
            return image_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def predict(self, image_input: Union[np.ndarray, Image.Image, str]) -> Dict:
        """Predict flower class for image with enhanced error handling"""
        try:
            start_time = datetime.now()
            
            # Preprocess image
            image_array = self.preprocess_image(image_input)
            
            if image_array is None:
                logger.error("Image preprocessing failed")
                return self._create_error_result("Image preprocessing failed")
            
            # Use model if available
            if self.model_manager and hasattr(self.model_manager, 'predict'):
                try:
                    # Add batch dimension for model
                    batch_image = np.expand_dims(image_array, axis=0)
                    result = self.model_manager.predict(batch_image)
                    
                    # Ensure result has required fields
                    if not isinstance(result, dict):
                        logger.error("Model returned invalid result format")
                        return self._mock_prediction()
                    
                    # Add colors to probabilities
                    if 'probabilities' in result:
                        for prob in result['probabilities']:
                            if 'name' in prob:
                                prob['color'] = self.class_colors.get(prob['name'], '#666666')
                    
                    # Ensure confidence is percentage
                    if 'confidence' in result and result['confidence'] <= 1.0:
                        result['confidence'] *= 100
                    
                    # Add processing time
                    processing_time = (datetime.now() - start_time).total_seconds()
                    result['processing_time_ms'] = round(processing_time * 1000, 2)
                    result['timestamp'] = datetime.now().isoformat()
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Model prediction failed: {e}")
                    return self._mock_prediction()
            
            else:
                logger.warning("No model available, using mock prediction")
                return self._mock_prediction()
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._create_error_result(f"Prediction failed: {str(e)}")
    
    def _mock_prediction(self) -> Dict:
        """Generate realistic mock prediction for testing"""
        try:
            # Generate realistic mock probabilities with some randomness
            base_probs = np.random.dirichlet([3, 2, 1])  # Bias toward rose
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.05, len(base_probs))
            mock_probs = base_probs + noise
            mock_probs = np.clip(mock_probs, 0.01, 0.99)  # Keep reasonable bounds
            mock_probs = mock_probs / mock_probs.sum()  # Renormalize
            
            # Choose predicted class
            predicted_idx = np.argmax(mock_probs)
            predicted_class = self.classes[predicted_idx]
            confidence = float(mock_probs[predicted_idx])
            
            # Format probabilities
            probabilities = []
            for i, cls in enumerate(self.classes):
                probabilities.append({
                    'class_name': cls,
                    'probability': float(mock_probs[i]),
                    'percentage': round(float(mock_probs[i]) * 100, 1),
                    'color': self.class_colors[cls]
                })
            
            # Sort by probability (highest first)
            probabilities.sort(key=lambda x: x['probability'], reverse=True)
            
            return {
                'class_name': predicted_class,
                'confidence': round(confidence * 100, 1),
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': np.random.randint(50, 200),  # Mock processing time
                'mock': True,
                'model_available': False
            }
            
        except Exception as e:
            logger.error(f"Mock prediction failed: {e}")
            return self._create_error_result("Mock prediction failed")
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create standardized error result"""
        return {
            'error': True,
            'message': error_message,
            'class': None,
            'confidence': 0,
            'probabilities': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, image_list: list, max_batch_size: int = 32) -> List[Dict]:
        """Predict multiple images with batch processing"""
        results = []
        
        # Process in chunks to avoid memory issues
        for i in range(0, len(image_list), max_batch_size):
            batch = image_list[i:i + max_batch_size]
            
            for j, image_input in enumerate(batch):
                try:
                    result = self.predict(image_input)
                    result['index'] = i + j
                    result['batch_index'] = j
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to predict image {i + j}: {e}")
                    results.append({
                        'index': i + j,
                        'batch_index': j,
                        'error': True,
                        'message': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        return results
    
    def validate_image(self, image_path: str) -> Dict:
        """Enhanced image validation with detailed feedback"""
        result = {
            'valid': False,
            'exists': False,
            'supported_format': False,
            'loadable': False,
            'details': {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                result['details']['error'] = f"File not found: {image_path}"
                return result
            
            result['exists'] = True
            
            # Check file size
            file_size = os.path.getsize(image_path)
            result['details']['file_size_bytes'] = file_size
            result['details']['file_size_mb'] = round(file_size / (1024 * 1024), 2)
            
            if file_size == 0:
                result['details']['error'] = "File is empty"
                return result
            
            # Check file extension
            ext = os.path.splitext(image_path)[1].lower().lstrip('.')
            result['details']['extension'] = ext
            
            if ext not in self.supported_formats:
                result['details']['error'] = f"Unsupported format: {ext}"
                return result
            
            result['supported_format'] = True
            
            # Try to load and get image info
            if PIL_AVAILABLE:
                try:
                    with Image.open(image_path) as img:
                        result['details']['original_size'] = img.size
                        result['details']['mode'] = img.mode
                        result['details']['format'] = img.format
                        
                        # Check if image has EXIF data
                        if hasattr(img, '_getexif') and img._getexif():
                            result['details']['has_exif'] = True
                        
                        result['loadable'] = True
                        
                except Exception as e:
                    result['details']['error'] = f"PIL load error: {str(e)}"
                    return result
            
            # Try preprocessing
            preprocessed = self.preprocess_image(image_path)
            if preprocessed is not None:
                result['valid'] = True
                result['details']['preprocessed_shape'] = preprocessed.shape
                result['details']['data_range'] = f"[{preprocessed.min():.3f}, {preprocessed.max():.3f}]"
            else:
                result['details']['error'] = "Preprocessing failed"
            
        except Exception as e:
            result['details']['error'] = f"Validation error: {str(e)}"
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        info = {
            'model_available': self.model_manager is not None,
            'input_size': self.input_size,
            'classes': self.classes,
            'supported_formats': self.supported_formats,
            'pil_available': PIL_AVAILABLE,
            'cv2_available': CV2_AVAILABLE
        }
        
        if self.model_manager and hasattr(self.model_manager, 'get_model_info'):
            try:
                model_details = self.model_manager.get_model_info()
                info.update(model_details)
            except Exception as e:
                info['model_error'] = str(e)
        
        return info

# Test the predictor if run directly
if __name__ == '__main__':
    print("🧪 Testing Enhanced FlowerPredictor...")
    
    predictor = FlowerPredictor()
    
    # Show model info
    print("\n📋 Model Information:")
    model_info = predictor.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test with mock image
    print("\n🔮 Testing prediction with mock image...")
    mock_image = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    result = predictor.predict(mock_image)
    
    if 'error' in result and result['error']:
        print(f"❌ Prediction failed: {result['message']}")
    else:
        print(f"✅ Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print(f"Processing time: {result.get('processing_time_ms', 'N/A')} ms")
        print("Probabilities:")
        for prob in result['probabilities']:
            print(f"  {prob['name']}: {prob['percentage']:.1f}%")
    
    # Test saving uploaded image
    print("\n💾 Testing saving uploaded image...")
    saved_path = predictor.save_uploaded_image(mock_image, 'rose')
    if saved_path:
        print(f"✅ Image saved to: {saved_path}")
    else:
        print("❌ Failed to save image")
    
    print("\n✅ Enhanced FlowerPredictor test completed")