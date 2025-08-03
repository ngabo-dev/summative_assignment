
# Configuration Template for Flower Classification
# Modify this file to match your setup

import os

class Config:
    """Configuration class for the flower classification pipeline"""
    
    # Data paths - MODIFY THESE TO MATCH YOUR SETUP
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Model configuration
    MODEL_PATH = os.path.join(MODEL_DIR, "flower_cnn_model.h5")
    INPUT_SHAPE = (150, 150, 3)
    NUM_CLASSES = 3
    CLASS_NAMES = ['roses', 'tulips', 'sunflowers']
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Image parameters
    IMAGE_SIZE = (150, 150)
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Ensure directories exist
def create_directories():
    """Create all necessary directories"""
    directories = [
        Config.DATA_DIR,
        Config.TRAIN_DIR,
        Config.TEST_DIR,
        Config.UPLOAD_DIR,
        Config.MODEL_DIR,
        Config.LOGS_DIR
    ]
    
    # Create class subdirectories
    for base_dir in [Config.TRAIN_DIR, Config.TEST_DIR]:
        for class_name in Config.CLASS_NAMES:
            directories.append(os.path.join(base_dir, class_name))
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create directories on import
create_directories()
