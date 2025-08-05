# backend/src/preprocessing.py

#!/usr/bin/env python3
"""
Optimized Data Preprocessor for loading real images
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

# Try to import image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available")

class DataPreprocessor:
    """Data preprocessor for flower classification using real images"""
    
    def __init__(self, data_dir='data', target_size=(128, 128)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.classes = ['rose', 'tulip', 'sunflower']
        
        # Verify directory structure
        self._verify_directory_structure()
        
        # Dataset statistics
        self.dataset_stats = {
            'total_images': 0,
            'class_distribution': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Update statistics
        self._update_statistics()
        
        logger.info("✅ DataPreprocessor initialized for real images")
    
    def _verify_directory_structure(self):
        """Verify that the required directory structure exists"""
        required_dirs = [
            os.path.join(self.data_dir, 'train'),
            os.path.join(self.data_dir, 'test'),
            os.path.join(self.data_dir, 'uploads')
        ]
        
        # Verify class subdirectories
        for base_dir in ['train', 'test', 'uploads']:
            for class_name in self.classes:
                dir_path = os.path.join(self.data_dir, base_dir, class_name)
                if not os.path.exists(dir_path):
                    raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    def scan_and_organize_images(self, source_dir: str = None) -> bool:
        """Scan for existing images and organize them"""
        try:
            if source_dir is None:
                source_dir = self.data_dir
            
            logger.info(f"Scanning for images in {source_dir}...")
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            found_images = 0
            
            # Walk through all directories
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        found_images += 1
                        logger.debug(f"Found image: {os.path.join(root, file)}")
            
            logger.info(f"✅ Found {found_images} images")
            self._update_statistics()
            return True
            
        except Exception as e:
            logger.error(f"Failed to scan images: {e}")
            return False
    
    def add_training_image(self, image_path: str, class_name: str) -> bool:
        """Add image to training dataset"""
        try:
            if class_name not in self.classes:
                logger.error(f"Invalid class name: {class_name}")
                return False
            
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return False
            
            # Copy image to appropriate directory
            class_dir = os.path.join(self.data_dir, 'uploads', class_name)
            filename = os.path.basename(image_path)
            dest_path = os.path.join(class_dir, filename)
            
            # Process image if PIL available
            if PIL_AVAILABLE:
                try:
                    # Load, resize, and save
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize to target size
                    image = image.resize(self.target_size, Image.Resampling.LANCZOS)
                    image.save(dest_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process image: {e}")
                    return False
            else:
                # Simple file copy
                import shutil
                shutil.copy2(image_path, dest_path)
            
            # Update statistics
            self._update_statistics()
            
            logger.info(f"✅ Added image to {class_name} class")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add training image: {e}")
            return False
    
    def _update_statistics(self):
        """Update dataset statistics"""
        try:
            total_images = 0
            class_distribution = {}
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            
            for class_name in self.classes:
                class_count = 0
                
                # Count images in train directory
                train_dir = os.path.join(self.data_dir, 'train', class_name)
                if os.path.exists(train_dir):
                    train_files = [f for f in os.listdir(train_dir) 
                                 if f.lower().endswith(image_extensions)]
                    train_count = len(train_files)
                else:
                    train_count = 0
                
                # Count images in test directory
                test_dir = os.path.join(self.data_dir, 'test', class_name)
                if os.path.exists(test_dir):
                    test_files = [f for f in os.listdir(test_dir) 
                                 if f.lower().endswith(image_extensions)]
                    test_count = len(test_files)
                else:
                    test_count = 0
                
                # Count images in uploads directory
                upload_dir = os.path.join(self.data_dir, 'uploads', class_name)
                if os.path.exists(upload_dir):
                    upload_files = [f for f in os.listdir(upload_dir) 
                                  if f.lower().endswith(image_extensions)]
                    upload_count = len(upload_files)
                else:
                    upload_count = 0
                
                class_count = train_count + test_count + upload_count
                class_distribution[class_name] = {
                    'total': class_count,
                    'train': train_count,
                    'test': test_count,
                    'uploads': upload_count
                }
                
                total_images += class_count
            
            # Calculate balance ratio
            if total_images > 0:
                counts = [class_distribution[cls]['total'] for cls in self.classes]
                min_count = min(counts) if counts else 0
                max_count = max(counts) if counts else 1
                balance_ratio = min_count / max_count if max_count > 0 else 1.0
            else:
                balance_ratio = 1.0
            
            # Determine dataset health
            if total_images == 0:
                health = 'empty'
            elif balance_ratio >= 0.8:
                health = 'excellent'
            elif balance_ratio >= 0.6:
                health = 'good'
            elif balance_ratio >= 0.4:
                health = 'fair'
            else:
                health = 'poor'
            
            self.dataset_stats = {
                'total_images': total_images,
                'train_images': sum(class_distribution[cls]['train'] for cls in self.classes),
                'test_images': sum(class_distribution[cls]['test'] for cls in self.classes),
                'upload_images': sum(class_distribution[cls]['uploads'] for cls in self.classes),
                'class_distribution': class_distribution,
                'balance_ratio': balance_ratio,
                'dataset_health': health,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to update statistics: {e}")
    
    def get_dataset_statistics(self) -> Dict:
        """Get current dataset statistics"""
        self._update_statistics()
        return self.dataset_stats.copy()
    
    def load_training_data(self, batch_size=32, split_ratio=0.8):
        """Load real training data from images"""
        try:
            x_data = []
            y_data = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            
            for class_idx, class_name in enumerate(self.classes):
                # Load from train directory
                train_dir = os.path.join(self.data_dir, 'train', class_name)
                if os.path.exists(train_dir):
                    for filename in os.listdir(train_dir):
                        if filename.lower().endswith(image_extensions):
                            img_path = os.path.join(train_dir, filename)
                            img_array = self._load_and_preprocess_image(img_path)
                            if img_array is not None:
                                x_data.append(img_array)
                                y_data.append(class_idx)
                
                # Load from uploads directory
                upload_dir = os.path.join(self.data_dir, 'uploads', class_name)
                if os.path.exists(upload_dir):
                    for filename in os.listdir(upload_dir):
                        if filename.lower().endswith(image_extensions):
                            img_path = os.path.join(upload_dir, filename)
                            img_array = self._load_and_preprocess_image(img_path)
                            if img_array is not None:
                                x_data.append(img_array)
                                y_data.append(class_idx)
            
            if not x_data:
                logger.warning("No images found for training")
                return None, None
            
            # Convert to numpy arrays
            x_data = np.array(x_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.int32)
            
            logger.info(f"Loaded {len(x_data)} images for training")
            return x_data, y_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None, None
    
    def load_test_data(self):
        """Load test data from images"""
        try:
            x_data = []
            y_data = []
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            
            for class_idx, class_name in enumerate(self.classes):
                # Load from test directory
                test_dir = os.path.join(self.data_dir, 'test', class_name)
                if os.path.exists(test_dir):
                    for filename in os.listdir(test_dir):
                        if filename.lower().endswith(image_extensions):
                            img_path = os.path.join(test_dir, filename)
                            img_array = self._load_and_preprocess_image(img_path)
                            if img_array is not None:
                                x_data.append(img_array)
                                y_data.append(class_idx)
            
            if not x_data:
                logger.warning("No images found for testing")
                return None, None
            
            # Convert to numpy arrays
            x_data = np.array(x_data, dtype=np.float32)
            y_data = np.array(y_data, dtype=np.int32)
            
            logger.info(f"Loaded {len(x_data)} images for testing")
            return x_data, y_data
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None, None
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image"""
        try:
            if not PIL_AVAILABLE:
                logger.error("PIL not available for image loading")
                return None
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def get_class_weights(self) -> Dict:
        """Calculate class weights for balanced training"""
        stats = self.get_dataset_statistics()
        
        if stats['total_images'] == 0:
            return {i: 1.0 for i in range(len(self.classes))}
        
        # Calculate weights inversely proportional to class frequency
        class_counts = [stats['class_distribution'][cls]['total'] for cls in self.classes]
        total_samples = sum(class_counts)
        
        if total_samples == 0:
            return {i: 1.0 for i in range(len(self.classes))}
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total_samples / (len(self.classes) * count)
            else:
                weights[i] = 1.0
        
        return weights
    
    def create_train_test_split(self, test_ratio=0.2):
        """Split existing images into train and test sets"""
        try:
            logger.info(f"Creating train/test split with {test_ratio:.1%} for testing...")
            
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
            moved_count = 0
            
            for class_name in self.classes:
                # Get all images from uploads directory
                upload_dir = os.path.join(self.data_dir, 'uploads', class_name)
                if not os.path.exists(upload_dir):
                    continue
                
                images = [f for f in os.listdir(upload_dir) 
                         if f.lower().endswith(image_extensions)]
                
                if not images:
                    continue
                
                # Shuffle and split
                np.random.shuffle(images)
                test_count = max(1, int(len(images) * test_ratio))
                
                train_dir = os.path.join(self.data_dir, 'train', class_name)
                test_dir = os.path.join(self.data_dir, 'test', class_name)
                
                # Move test images
                for i in range(test_count):
                    src = os.path.join(upload_dir, images[i])
                    dst = os.path.join(test_dir, images[i])
                    
                    import shutil
                    shutil.move(src, dst)
                    moved_count += 1
                
                # Move remaining to train
                for i in range(test_count, len(images)):
                    src = os.path.join(upload_dir, images[i])
                    dst = os.path.join(train_dir, images[i])
                    
                    import shutil
                    shutil.move(src, dst)
                    moved_count += 1
            
            self._update_statistics()
            logger.info(f"✅ Split complete. Moved {moved_count} images")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create train/test split: {e}")
            return False

# Test the preprocessor if run directly
if __name__ == '__main__':
    print("🧪 Testing DataPreprocessor with real images...")
    
    preprocessor = DataPreprocessor()
    
    # Scan for existing images
    print("\n🔍 Scanning for existing images...")
    preprocessor.scan_and_organize_images()
    
    # Get statistics
    stats = preprocessor.get_dataset_statistics()
    print(f"\n📈 Dataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    print(f"Train images: {stats['train_images']}")
    print(f"Test images: {stats['test_images']}")
    print(f"Upload images: {stats['upload_images']}")
    print(f"Dataset health: {stats['dataset_health']}")
    print("Class distribution:")
    for cls, counts in stats['class_distribution'].items():
        print(f"  {cls}: {counts['total']} images (train: {counts['train']}, test: {counts['test']}, uploads: {counts['uploads']})")
    
    if stats['total_images'] > 0:
        print("\n🔀 Creating train/test split...")
        preprocessor.create_train_test_split(test_ratio=0.2)
        
        # Load training data
        print("\n📊 Loading training data...")
        x_train, y_train = preprocessor.load_training_data()
        
        if x_train is not None:
            print(f"✅ Loaded {len(x_train)} training samples")
            print(f"Image shape: {x_train[0].shape}")
        else:
            print("❌ Failed to load training data")
            
        # Load test data
        print("\n📊 Loading test data...")
        x_test, y_test = preprocessor.load_test_data()
        
        if x_test is not None:
            print(f"✅ Loaded {len(x_test)} test samples")
        else:
            print("❌ Failed to load test data")
    else:
        print("ℹ️  No images found. Place images in data/uploads/[class_name]/ directories")
    
    print("\n✅ DataPreprocessor test completed")