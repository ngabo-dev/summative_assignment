import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)

class FlowerPredictor:
    """Handles flower image prediction"""
    
    def __init__(self, model_manager=None):
        from .model import ModelManager
        self.model_manager = model_manager or ModelManager()
        self.class_names = ['rose', 'tulip', 'sunflower']
        self.input_size = (224, 224)
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Preprocess image for model prediction"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize image
            image = cv2.resize(image, self.input_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> Dict:
        """Predict flower type from image"""
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Make prediction using model manager
            result = self.model_manager.predict(processed_image)
            
            # Add color information for frontend charts
            color_map = {
                'rose': 'hsl(var(--chart-1))',
                'tulip': 'hsl(var(--chart-2))',
                'sunflower': 'hsl(var(--chart-3))'
            }
            
            # Add colors to probabilities
            for prob in result['probabilities']:
                prob['color'] = color_map.get(prob['name'], 'hsl(var(--chart-1))')
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def predict_batch(self, images: list) -> list:
        """Predict flower types for multiple images"""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.predict(image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting image {i}: {str(e)}")
                results.append({
                    'image_index': i,
                    'error': str(e),
                    'class': None,
                    'confidence': 0,
                    'probabilities': []
                })
        
        return results
    
    def validate_prediction(self, image: Union[Image.Image, np.ndarray], 
                          expected_class: str) -> Dict:
        """Validate prediction against expected class"""
        try:
            result = self.predict(image)
            
            is_correct = result['class'].lower() == expected_class.lower()
            
            return {
                **result,
                'expected_class': expected_class,
                'is_correct': is_correct,
                'validation_accuracy': 1.0 if is_correct else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise
    
    def get_prediction_confidence_stats(self, predictions: list) -> Dict:
        """Calculate confidence statistics from multiple predictions"""
        if not predictions:
            return {}
        
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]
        
        if not confidences:
            return {}
        
        return {
            'mean_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'total_predictions': len(confidences)
        }
    
    def analyze_prediction_distribution(self, predictions: list) -> Dict:
        """Analyze class distribution in predictions"""
        if not predictions:
            return {}
        
        class_counts = {}
        confidence_by_class = {}
        
        for pred in predictions:
            if 'class' in pred and pred['class']:
                class_name = pred['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                if class_name not in confidence_by_class:
                    confidence_by_class[class_name] = []
                confidence_by_class[class_name].append(pred.get('confidence', 0))
        
        # Calculate percentages and average confidence by class
        total_predictions = sum(class_counts.values())
        
        distribution = []
        for class_name, count in class_counts.items():
            avg_confidence = np.mean(confidence_by_class[class_name])
            distribution.append({
                'class': class_name,
                'count': count,
                'percentage': (count / total_predictions) * 100,
                'avg_confidence': avg_confidence
            })
        
        return {
            'distribution': distribution,
            'total_predictions': total_predictions,
            'unique_classes': len(class_counts)
        }