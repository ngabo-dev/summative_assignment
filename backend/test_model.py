#!/usr/bin/env python3
"""
Test script to verify model initialization works correctly
Run this to debug model loading issues
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_initialization():
    """Test that the model can be initialized and make predictions"""
    try:
        logger.info("🧪 Testing model initialization...")
        
        # Import the ML components
        from src.model import ModelManager
        from src.prediction import FlowerPredictor
        
        logger.info("✅ Successfully imported ML modules")
        
        # Initialize ModelManager
        logger.info("🏗️ Initializing ModelManager...")
        model_manager = ModelManager()
        
        if model_manager.current_model is None:
            logger.error("❌ ModelManager failed to create a model")
            return False
        
        logger.info("✅ ModelManager initialized successfully")
        
        # Initialize FlowerPredictor
        logger.info("🔮 Initializing FlowerPredictor...")
        predictor = FlowerPredictor(model_manager)
        logger.info("✅ FlowerPredictor initialized successfully")
        
        # Test prediction
        logger.info("🎯 Testing prediction with dummy image...")
        import numpy as np
        from PIL import Image
        
        # Create a dummy image
        dummy_image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_image_array)
        
        # Make prediction
        result = predictor.predict(dummy_image)
        
        logger.info("✅ Prediction successful!")
        logger.info(f"   Predicted class: {result['class']}")
        logger.info(f"   Confidence: {result['confidence']:.1f}%")
        logger.info(f"   Probabilities: {[f\"{p['name']}: {p['probability']:.3f}\" for p in result['probabilities']]}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        logger.error("   Make sure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        return False

def test_tensorflow():
    """Test that TensorFlow is properly installed"""
    try:
        logger.info("🔧 Testing TensorFlow installation...")
        import tensorflow as tf
        logger.info(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Test basic TensorFlow operations
        logger.info("🧮 Testing TensorFlow operations...")
        x = tf.constant([[1, 2], [3, 4]])
        y = tf.constant([[5, 6], [7, 8]])
        z = tf.matmul(x, y)
        logger.info(f"✅ TensorFlow operations working: {z.numpy()}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ TensorFlow import error: {str(e)}")
        logger.error("   Install TensorFlow: pip install tensorflow==2.13.0")
        return False
    except Exception as e:
        logger.error(f"❌ TensorFlow error: {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("🌸 Flower Classification Model Test Suite")
    logger.info("=" * 50)
    
    success = True
    
    # Test TensorFlow
    if not test_tensorflow():
        success = False
    
    logger.info("-" * 50)
    
    # Test model initialization
    if not test_model_initialization():
        success = False
    
    logger.info("=" * 50)
    
    if success:
        logger.info("🎉 All tests passed! The model should work correctly.")
        logger.info("💡 You can now start the Flask server: python app.py")
    else:
        logger.error("💥 Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()