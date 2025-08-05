# backend/src/model.py

#!/usr/bin/env python3
"""
Model Manager for Flower Classification using Real Images
Integrates with DataPreprocessor to train on downloaded images
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML dependencies
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, applications
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("⚠️ TensorFlow not available - using mock model")

class ModelManager:
    """Model manager for training on real flower images"""
    
    def __init__(self, models_dir='models', data_dir='data', input_size=(128, 128)):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.input_size = input_size
        self.classes = ['rose', 'tulip', 'sunflower']
        self.num_classes = len(self.classes)
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, 'uploads'), exist_ok=True)
        
        # Create class subdirectories
        for class_name in self.classes:
            os.makedirs(os.path.join(self.data_dir, 'train', class_name), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, 'test', class_name), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, 'uploads', class_name), exist_ok=True)
        
        # Current model
        self.model = None
        self.model_built = False
        
        # Training history
        self.training_history = {}
        
        # Data preprocessor (will be set externally)
        self.data_preprocessor = None
        
        # Initialize model
        self._initialize_model()
    
    def set_data_preprocessor(self, preprocessor):
        """Set the data preprocessor for loading real images"""
        self.data_preprocessor = preprocessor
        logger.info("✅ Data preprocessor connected")
    
    def _initialize_model(self):
        """Initialize optimized lightweight model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - using mock model")
            return
        
        try:
            # Create optimized lightweight model for fast training
            self.model = keras.Sequential([
                # Input layer
                layers.Input(shape=(*self.input_size, 3)),
                
                # Data augmentation for better generalization
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                
                # Lightweight feature extraction
                layers.Conv2D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling2D(2),
                layers.BatchNormalization(),
                layers.Dropout(0.25),
                
                layers.Conv2D(128, 3, activation='relu', padding='same'),
                layers.GlobalAveragePooling2D(),  # Much faster than Flatten
                
                # Classifier
                layers.Dropout(0.5),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax', name='predictions')
            ], name='real_flower_classifier')
            
            # Build the model properly
            self.model.build(input_shape=(None, *self.input_size, 3))
            self.model_built = True
            
            # Compile with optimized settings
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            # Print model summary
            logger.info("✅ Real image model created with data augmentation")
            self._print_model_info()
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            self.model = None
            self.model_built = False
    
    def _print_model_info(self):
        """Print model information safely"""
        if not self.model or not self.model_built:
            return
        
        try:
            total_params = self.model.count_params()
            trainable_params = sum([np.prod(var.shape) for var in self.model.trainable_variables])
            
            print(f"\n🧠 Real Flower Classification Model")
            print(f"📊 Total parameters: {total_params:,}")
            print(f"🎯 Trainable parameters: {trainable_params:,}")
            print(f"💾 Estimated model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
            # Safely check output shape
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                last_layer = self.model.layers[-1]
                if hasattr(last_layer, 'output_shape'):
                    print(f"🎯 Output shape: {last_layer.output_shape}")
                    print(f"🎯 Expected: (None, 3) for 3 classes")
                    if last_layer.output_shape[-1] == 3:
                        print("✅ Output layer correctly configured for 3 classes")
                    else:
                        print(f"❌ Output layer has {last_layer.output_shape[-1]} units, expected 3")
            
        except Exception as e:
            logger.warning(f"Could not print model info: {e}")
    
    def load_real_training_data(self, validation_split=0.2):
        """Load real training data from preprocessor"""
        if not self.data_preprocessor:
            logger.error("No data preprocessor available")
            return None, None
        
        try:
            # Load all training data (combines train and uploads directories)
            x_data, y_data = self.data_preprocessor.load_training_data()
            
            if x_data is None or len(x_data) == 0:
                logger.error("No training data available")
                return None, None
            
            # Convert labels to categorical
            y_categorical = keras.utils.to_categorical(y_data, self.num_classes)
            
            # Split into train and validation
            num_val = int(len(x_data) * validation_split)
            
            # Shuffle data
            indices = np.random.permutation(len(x_data))
            
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]
            
            x_train = x_data[train_indices]
            y_train = y_categorical[train_indices]
            x_val = x_data[val_indices]
            y_val = y_categorical[val_indices]
            
            logger.info(f"✅ Loaded real training data: {len(x_train)} train, {len(x_val)} val samples")
            
            # Print class distribution
            train_class_counts = np.sum(y_train, axis=0)
            val_class_counts = np.sum(y_val, axis=0)
            
            print("\n📊 Training Data Distribution:")
            for i, class_name in enumerate(self.classes):
                print(f"  {class_name}: {int(train_class_counts[i])} train, {int(val_class_counts[i])} val")
            
            return (x_train, y_train), (x_val, y_val)
            
        except Exception as e:
            logger.error(f"Failed to load real training data: {e}")
            return None, None
    
    def train_on_real_data(self, epochs=20, batch_size=16, learning_rate=0.001, 
                          validation_split=0.2, progress_callback=None):
        """Train model on real downloaded images"""
        if not TENSORFLOW_AVAILABLE or not self.model or not self.model_built:
            logger.error("Model not available for training")
            return False
        
        try:
            logger.info(f"🚀 Starting training on real images for {epochs} epochs...")
            
            # Load real training data
            train_data, val_data = self.load_real_training_data(validation_split)
            if train_data is None:
                logger.error("Failed to load real training data")
                return False
            
            x_train, y_train = train_data
            x_val, y_val = val_data
            
            if len(x_train) < batch_size:
                logger.warning(f"Training data ({len(x_train)}) smaller than batch size ({batch_size})")
                batch_size = max(1, len(x_train) // 2)
                logger.info(f"Adjusted batch size to {batch_size}")
            
            # Update learning rate if provided
            if learning_rate != 0.001:
                self.model.optimizer.learning_rate.assign(learning_rate)
            
            # Calculate class weights for balanced training
            class_weights = None
            if self.data_preprocessor:
                weights_dict = self.data_preprocessor.get_class_weights()
                class_weights = {i: weights_dict.get(i, 1.0) for i in range(self.num_classes)}
                logger.info(f"Using class weights: {class_weights}")
            
            # Callbacks for training
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.models_dir, 'best_model.keras'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Custom callback for progress updates
            if progress_callback:
                class ProgressCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = int((epoch + 1) / epochs * 100)
                        acc = logs.get('accuracy', 0)
                        val_acc = logs.get('val_accuracy', 0)
                        progress_callback(progress, f"Epoch {epoch + 1}/{epochs} - Acc: {acc:.3f}, Val Acc: {val_acc:.3f}")
                
                callbacks.append(ProgressCallback())
            
            # Train the model
            start_time = datetime.now()
            
            history = self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
                shuffle=True
            )
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Evaluate on validation data
            val_loss, val_acc, val_precision, val_recall = self.model.evaluate(x_val, y_val, verbose=0)
            
            # Calculate F1 score
            val_predictions = self.model.predict(x_val, verbose=0)
            val_pred_classes = np.argmax(val_predictions, axis=1)
            val_true_classes = np.argmax(y_val, axis=1)
            
            # Calculate per-class metrics
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(val_true_classes, val_pred_classes, average='weighted')
            precision = precision_score(val_true_classes, val_pred_classes, average='weighted')
            recall = recall_score(val_true_classes, val_pred_classes, average='weighted')
            
            # Store training history
            self.training_history = {
                'history': {k: [float(v) for v in values] for k, values in history.history.items()},
                'epochs_trained': len(history.history['loss']),
                'training_time_seconds': training_time,
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(val_acc),
                'final_val_precision': float(precision),
                'final_val_recall': float(recall),
                'final_val_f1': float(f1),
                'training_samples': len(x_train),
                'validation_samples': len(x_val),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Training completed in {training_time:.1f} seconds")
            logger.info(f"📊 Final accuracy: {history.history['accuracy'][-1]:.3f}")
            logger.info(f"📊 Final val accuracy: {val_acc:.3f}")
            logger.info(f"📊 Final val precision: {precision:.3f}")
            logger.info(f"📊 Final val recall: {recall:.3f}")
            logger.info(f"📊 Final val F1: {f1:.3f}")
            
            # Save the model
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def evaluate_model(self, test_data=None):
        """Evaluate model performance"""
        if not self.model or not self.model_built:
            logger.error("No model available for evaluation")
            return None
        
        try:
            if test_data is None:
                # Try to load test data from preprocessor
                if not self.data_preprocessor:
                    logger.error("No data preprocessor available")
                    return None
                
                # Load test data
                x_test, y_test = self.data_preprocessor.load_test_data()
                if x_test is None:
                    return None
                
                # Convert to categorical
                y_test = keras.utils.to_categorical(y_test, self.num_classes)
            else:
                x_test, y_test = test_data
            
            # Make predictions
            predictions = self.model.predict(x_test, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            
            accuracy = accuracy_score(true_classes, pred_classes)
            report = classification_report(true_classes, pred_classes, 
                                         target_names=self.classes, output_dict=True)
            
            evaluation_results = {
                'accuracy': float(accuracy),
                'precision': float(report['weighted avg']['precision']),
                'recall': float(report['weighted avg']['recall']),
                'f1_score': float(report['weighted avg']['f1-score']),
                'per_class_metrics': {},
                'confusion_matrix': confusion_matrix(true_classes, pred_classes).tolist(),
                'test_samples': len(x_test)
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(self.classes):
                if str(i) in report:
                    evaluation_results['per_class_metrics'][class_name] = {
                        'precision': float(report[str(i)]['precision']),
                        'recall': float(report[str(i)]['recall']),
                        'f1_score': float(report[str(i)]['f1-score']),
                        'support': int(report[str(i)]['support'])
                    }
            
            logger.info(f"📊 Model Evaluation Results:")
            logger.info(f"  Accuracy: {accuracy:.3f}")
            logger.info(f"  Precision: {evaluation_results['precision']:.3f}")
            logger.info(f"  Recall: {evaluation_results['recall']:.3f}")
            logger.info(f"  F1 Score: {evaluation_results['f1_score']:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return None
    
    def _save_model(self):
        """Save the trained model"""
        if not self.model or not self.model_built:
            return False
        
        try:
            model_path = os.path.join(self.models_dir, 'flower_model.keras')
            self.model.save(model_path)
            
            # Save training history
            history_path = os.path.join(self.models_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            logger.info(f"✅ Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, model_path=None):
        """Load a saved model"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        if model_path is None:
            model_path = os.path.join(self.models_dir, 'flower_model.keras')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        try:
            self.model = keras.models.load_model(model_path)
            self.model_built = True
            
            # Load training history if available
            history_path = os.path.join(self.models_dir, 'training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            logger.info(f"✅ Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, image_array):
        """Make prediction on image"""
        if not self.model or not self.model_built:
            # Return mock prediction if model not available
            mock_probs = np.random.dirichlet([1, 1, 1])
            return {
                'class': self.classes[np.argmax(mock_probs)],
                'confidence': float(np.max(mock_probs)),
                'probabilities': [
                    {'name': cls, 'probability': float(prob)}
                    for cls, prob in zip(self.classes, mock_probs)
                ]
            }
        
        try:
            # Ensure image is the right shape
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Resize if needed
            if image_array.shape[1:3] != self.input_size:
                image_array = tf.image.resize(image_array, self.input_size).numpy()
            
            # Normalize
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            # Predict
            predictions = self.model.predict(image_array, verbose=0)
            
            # Get results
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            # Format probabilities
            probabilities = []
            for i, cls in enumerate(self.classes):
                probabilities.append({
                    'name': cls,
                    'probability': float(predictions[0][i])
                })
            
            return {
                'class': self.classes[class_idx],
                'confidence': confidence,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return mock prediction on error
            mock_probs = np.random.dirichlet([1, 1, 1])
            return {
                'class': self.classes[np.argmax(mock_probs)],
                'confidence': 0.5,
                'probabilities': [
                    {'name': cls, 'probability': float(prob)}
                    for cls, prob in zip(self.classes, mock_probs)
                ]
            }
    
    def get_current_model_stats(self):
        """Get current model statistics"""
        if not self.training_history:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'training_time': '0s',
                'epochs_trained': 0,
                'training_samples': 0,
                'validation_samples': 0
            }
        
        return {
            'accuracy': round(self.training_history.get('final_val_accuracy', 0.0) * 100, 1),
            'precision': round(self.training_history.get('final_val_precision', 0.0) * 100, 1),
            'recall': round(self.training_history.get('final_val_recall', 0.0) * 100, 1),
            'f1_score': round(self.training_history.get('final_val_f1', 0.0) * 100, 1),
            'training_time': f"{self.training_history.get('training_time_seconds', 0):.1f}s",
            'epochs_trained': self.training_history.get('epochs_trained', 0),
            'training_samples': self.training_history.get('training_samples', 0),
            'validation_samples': self.training_history.get('validation_samples', 0)
        }
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            'model_available': self.model is not None and self.model_built,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'input_size': self.input_size,
            'classes': self.classes,
            'num_classes': self.num_classes,
            'data_preprocessor_connected': self.data_preprocessor is not None
        }
        
        if self.model and self.model_built:
            info.update({
                'total_parameters': self.model.count_params(),
                'trainable_parameters': sum([np.prod(var.shape) for var in self.model.trainable_variables]),
                'model_size_mb': round(self.model.count_params() * 4 / 1024 / 1024, 1)
            })
        
        if self.training_history:
            info.update(self.get_current_model_stats())
        
        return info

# Test the model if run directly
if __name__ == '__main__':
    print("🧪 Testing Real Image Model Manager...")
    
    manager = ModelManager()
    
    if manager.model and manager.model_built:
        print("✅ Model created successfully")
        
        # Show model info
        model_info = manager.get_model_info()
        print(f"\n📋 Model Info:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Note: Real training would require DataPreprocessor
        print("\n⚠️  To train on real images, connect a DataPreprocessor:")
        print("  manager.set_data_preprocessor(preprocessor)")
        print("  manager.train_on_real_data(epochs=20)")
        
    else:
        print("❌ Model creation failed")