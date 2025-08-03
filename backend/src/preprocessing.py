import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import json
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing and augmentation for flower classification"""
    
    def __init__(self, data_dir='data', target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.class_names = ['rose', 'tulip', 'sunflower']
        
        # Create directory structure
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.uploads_dir = os.path.join(data_dir, 'uploads')
        
        for dir_path in [self.train_dir, self.test_dir, self.uploads_dir]:
            os.makedirs(dir_path, exist_ok=True)
            for class_name in self.class_names:
                os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
        
        # Data augmentation parameters
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        self.test_datagen = ImageDataGenerator(rescale=1./255)
    
    def add_training_image(self, image_path: str, class_name: str) -> bool:
        """Add an image to the training dataset"""
        try:
            if class_name.lower() not in self.class_names:
                raise ValueError(f"Invalid class name: {class_name}")
            
            # Load and preprocess image
            image = self.load_and_preprocess_image(image_path)
            
            if image is None:
                return False
            
            # Generate filename
            class_dir = os.path.join(self.train_dir, class_name.lower())
            existing_files = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            filename = f"{class_name.lower()}_{existing_files + 1:04d}.jpg"
            
            # Save processed image
            output_path = os.path.join(class_dir, filename)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            logger.info(f"Added training image: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding training image: {str(e)}")
            return False
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL for other formats
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            image = cv2.resize(image, self.target_size)
            
            # Basic image enhancement
            image = self.enhance_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply basic image enhancement"""
        try:
            # Convert to LAB color space for better contrast adjustment
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Error enhancing image: {str(e)}")
            return image
    
    def create_data_generators(self, batch_size: int = 32) -> Tuple:
        """Create data generators for training and validation"""
        try:
            # Training data generator
            train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
            )
            
            # Validation data generator
            validation_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=False
            )
            
            # Test data generator
            test_generator = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=self.target_size,
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            ) if os.listdir(self.test_dir) else None
            
            return train_generator, validation_generator, test_generator
            
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            return None, None, None
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the current dataset"""
        try:
            stats = {
                'total_images': 0,
                'train_images': 0,
                'test_images': 0,
                'class_distribution': {},
                'dataset_health': 'healthy'
            }
            
            # Count training images
            for class_name in self.class_names:
                train_class_dir = os.path.join(self.train_dir, class_name)
                test_class_dir = os.path.join(self.test_dir, class_name)
                
                train_count = len([f for f in os.listdir(train_class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                test_count = len([f for f in os.listdir(test_class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                
                stats['class_distribution'][class_name] = {
                    'train': train_count,
                    'test': test_count,
                    'total': train_count + test_count
                }
                
                stats['train_images'] += train_count
                stats['test_images'] += test_count
                stats['total_images'] += train_count + test_count
            
            # Check dataset health
            min_images_per_class = min([cls['total'] for cls in stats['class_distribution'].values()])
            max_images_per_class = max([cls['total'] for cls in stats['class_distribution'].values()])
            
            if min_images_per_class == 0:
                stats['dataset_health'] = 'critical'
            elif min_images_per_class < 50:
                stats['dataset_health'] = 'warning'
            elif max_images_per_class / min_images_per_class > 3:
                stats['dataset_health'] = 'imbalanced'
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return {}
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets"""
        try:
            for class_name in self.class_names:
                class_dir = os.path.join(self.uploads_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                # Get all images in class directory
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                
                if len(images) < 2:
                    continue
                
                # Split into train and test
                train_images, test_images = train_test_split(
                    images, test_size=test_size, random_state=random_state
                )
                
                # Move images to appropriate directories
                train_class_dir = os.path.join(self.train_dir, class_name)
                test_class_dir = os.path.join(self.test_dir, class_name)
                
                for img in train_images:
                    src = os.path.join(class_dir, img)
                    dst = os.path.join(train_class_dir, img)
                    shutil.copy2(src, dst)
                
                for img in test_images:
                    src = os.path.join(class_dir, img)
                    dst = os.path.join(test_class_dir, img)
                    shutil.copy2(src, dst)
                
                logger.info(f"Split {class_name}: {len(train_images)} train, {len(test_images)} test")
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
    
    def augment_dataset(self, target_images_per_class: int = 1000):
        """Augment dataset to reach target number of images per class"""
        try:
            for class_name in self.class_names:
                class_dir = os.path.join(self.train_dir, class_name)
                existing_images = [f for f in os.listdir(class_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
                
                current_count = len(existing_images)
                if current_count >= target_images_per_class:
                    continue
                
                images_needed = target_images_per_class - current_count
                logger.info(f"Augmenting {class_name}: need {images_needed} more images")
                
                # Create augmented images
                for i, img_file in enumerate(existing_images):
                    if images_needed <= 0:
                        break
                    
                    img_path = os.path.join(class_dir, img_file)
                    image = self.load_and_preprocess_image(img_path)
                    
                    if image is None:
                        continue
                    
                    # Generate augmented versions
                    augmentations_per_image = min(images_needed, 5)
                    for j in range(augmentations_per_image):
                        augmented = self._apply_augmentation(image)
                        
                        aug_filename = f"{class_name}_aug_{current_count + i * 5 + j + 1:04d}.jpg"
                        aug_path = os.path.join(class_dir, aug_filename)
                        
                        cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                        images_needed -= 1
                
                logger.info(f"Augmentation completed for {class_name}")
            
        except Exception as e:
            logger.error(f"Error augmenting dataset: {str(e)}")
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to an image"""
        try:
            # Random rotation
            angle = np.random.uniform(-20, 20)
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h))
            
            # Random brightness adjustment
            brightness = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness, 0, 255).astype(np.uint8)
            
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = cv2.flip(image, 1)
            
            # Random zoom
            if np.random.random() > 0.5:
                zoom_factor = np.random.uniform(0.8, 1.2)
                h, w = image.shape[:2]
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                if zoom_factor > 1:
                    # Crop center
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    resized = cv2.resize(image, (new_w, new_h))
                    image = resized[start_h:start_h + h, start_w:start_w + w]
                else:
                    # Pad with reflection
                    resized = cv2.resize(image, (new_w, new_h))
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    image = cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h, 
                                             pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
            
            return image
            
        except Exception as e:
            logger.warning(f"Error applying augmentation: {str(e)}")
            return image
    
    def cleanup_dataset(self):
        """Remove corrupted or invalid images from dataset"""
        try:
            removed_count = 0
            
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            # Try to load image
                            image = cv2.imread(file_path)
                            if image is None:
                                pil_image = Image.open(file_path)
                                pil_image.verify()
                            
                            # Check image size
                            if image is not None and (image.shape[0] < 32 or image.shape[1] < 32):
                                raise ValueError("Image too small")
                                
                        except Exception:
                            # Remove corrupted image
                            os.remove(file_path)
                            removed_count += 1
                            logger.warning(f"Removed corrupted image: {file_path}")
            
            logger.info(f"Dataset cleanup completed. Removed {removed_count} corrupted images.")
            
        except Exception as e:
            logger.error(f"Error during dataset cleanup: {str(e)}")