#!/usr/bin/env python3
"""
Optimized Flower Classification Model Training Script
Fast training and prediction with retraining capability
"""

import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ML components
try:
    from src.model import ModelManager
    from src.preprocessing import DataPreprocessor
    from src.prediction import FlowerPredictor
    import tensorflow as tf
    
    # Optimize TensorFlow for speed
    tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

class OptimizedFlowerTrainer:
    """Optimized trainer for fast flower classification"""
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Initialize components
        logger.info("🚀 Initializing optimized flower trainer...")
        
        # Data preprocessor
        self.preprocessor = DataPreprocessor(data_dir=data_dir, target_size=(96, 96))  # Smaller size for speed
        
        # Model manager with optimized settings
        self.model_manager = ModelManager(
            models_dir=models_dir, 
            data_dir=data_dir, 
            input_size=(96, 96)  # Smaller input for faster training/prediction
        )
        
        # Connect preprocessor to model
        self.model_manager.set_data_preprocessor(self.preprocessor)
        
        # Predictor
        self.predictor = FlowerPredictor(self.model_manager)
        
        logger.info("✅ Trainer initialized successfully")
    
    def optimize_model_for_speed(self):
        """Create an optimized model architecture for fast training and inference"""
        if not self.model_manager.model:
            logger.error("No model available to optimize")
            return False
        
        try:
            from tensorflow.keras import layers, Model
            
            # Create a more efficient model architecture
            input_layer = layers.Input(shape=(96, 96, 3))
            
            # Efficient feature extraction with depthwise separable convolutions
            x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_layer)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.SeparableConv2D(128, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.3)(x)
            
            # Classifier
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            output = layers.Dense(3, activation='softmax', name='flower_prediction')(x)
            
            # Create optimized model
            optimized_model = Model(inputs=input_layer, outputs=output, name='fast_flower_classifier')
            
            # Compile with optimized settings
            optimized_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),  # Higher learning rate for faster convergence
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Replace the model
            self.model_manager.model = optimized_model
            self.model_manager.model_built = True
            
            logger.info("✅ Model optimized for speed and efficiency")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize model: {e}")
            return False
    
    def check_dataset_balance(self):
        """Check and report dataset balance"""
        stats = self.preprocessor.get_dataset_statistics()
        
        logger.info("📊 Dataset Statistics:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Training images: {stats['train_images']}")
        logger.info(f"  Dataset health: {stats['dataset_health']}")
        logger.info(f"  Balance ratio: {stats['balance_ratio']:.2f}")
        
        for class_name, counts in stats['class_distribution'].items():
            total = counts['total']
            train = counts['train']
            logger.info(f"  {class_name}: {total} total ({train} train)")
        
        return stats['total_images'] > 100  # Need at least 100 images
    
    def train_fast_model(self, epochs=15, batch_size=32, validation_split=0.2):
        """Train the model with optimized settings for speed"""
        
        start_time = time.time()
        logger.info("🏋️ Starting optimized training...")
        
        # Check dataset
        if not self.check_dataset_balance():
            logger.error("Insufficient training data")
            return False
        
        # Optimize model architecture
        if not self.optimize_model_for_speed():
            logger.error("Failed to optimize model")
            return False
        
        # Custom progress callback
        class FastProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                logger.info(f"📈 Epoch {epoch + 1}/{epochs} starting...")
            
            def on_epoch_end(self, epoch, logs=None):
                acc = logs.get('accuracy', 0)
                val_acc = logs.get('val_accuracy', 0)
                loss = logs.get('loss', 0)
                val_loss = logs.get('val_loss', 0)
                logger.info(f"✅ Epoch {epoch + 1} complete - Acc: {acc:.3f}, Val Acc: {val_acc:.3f}, Loss: {loss:.3f}, Val Loss: {val_loss:.3f}")
        
        # Training callbacks for efficiency
        callbacks = [
            FastProgressCallback(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,  # Reduced patience for faster training
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        try:
            # Load training data directly
            x_data, y_data = self.preprocessor.load_training_data()
            if x_data is None:
                logger.error("Failed to load training data")
                return False
            
            # Convert labels to categorical
            from tensorflow.keras.utils import to_categorical
            y_categorical = to_categorical(y_data, 3)
            
            # Split data
            from sklearn.model_selection import train_test_split
            x_train, x_val, y_train, y_val = train_test_split(
                x_data, y_categorical, test_size=validation_split, random_state=42
            )
            
            logger.info(f"Training with {len(x_train)} train samples, {len(x_val)} validation samples")
            
            # Train the model
            history = self.model_manager.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            logger.info(f"🎉 Training completed in {training_time:.1f} seconds!")
            
            # Test the model
            self.test_model_performance()
            
            return True
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def test_model_performance(self):
        """Test model performance and speed"""
        logger.info("🧪 Testing model performance...")
        
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            
            # Test prediction speed
            start_time = time.time()
            result = self.predictor.predict(test_image)
            prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if not result.get('error'):
                logger.info(f"✅ Prediction successful!")
                logger.info(f"🚀 Prediction time: {prediction_time:.1f}ms")
                logger.info(f"🎯 Predicted class: {result['class']}")
                logger.info(f"💪 Confidence: {result['confidence']:.1f}%")
                
                # Show all probabilities
                logger.info("📊 Class probabilities:")
                for prob in result['probabilities']:
                    logger.info(f"  {prob['name']}: {prob['percentage']:.1f}%")
            else:
                logger.error(f"❌ Prediction failed: {result['message']}")
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
    
    def save_optimized_model(self):
        """Save the trained model with metadata"""
        try:
            model_path = os.path.join(self.models_dir, 'fast_flower_model.keras')
            self.model_manager.model.save(model_path)
            
            # Save training metadata
            metadata = {
                'model_type': 'fast_flower_classifier',
                'classes': ['rose', 'tulip', 'sunflower'],
                'input_size': [96, 96, 3],
                'trained_date': datetime.now().isoformat(),
                'training_images': self.preprocessor.get_dataset_statistics()['total_images'],
                'optimized_for_speed': True
            }
            
            import json
            metadata_path = os.path.join(self.models_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved to {model_path}")
            logger.info(f"📋 Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def retrain_with_new_data(self, epochs=10):
        """Retrain the model when new data is added"""
        logger.info("🔄 Starting retraining with updated data...")
        
        # Refresh dataset statistics
        self.preprocessor._update_statistics()
        
        # Quick retrain with fewer epochs
        return self.train_fast_model(epochs=epochs, batch_size=32)

def main():
    """Main training function"""
    logger.info("🌸 Fast Flower Classification Training")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = OptimizedFlowerTrainer()
    
    # Start training
    logger.info("🚀 Starting optimized training process...")
    success = trainer.train_fast_model(epochs=20, batch_size=32)
    
    if success:
        # Save the model
        trainer.save_optimized_model()
        
        logger.info("=" * 50)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info("✅ Model is ready for fast predictions")
        logger.info("🔄 Model supports retraining with new data")
        logger.info("💡 Use the Flask API for predictions")
        logger.info("=" * 50)
    else:
        logger.error("❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
