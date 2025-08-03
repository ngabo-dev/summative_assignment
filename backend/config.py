import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'data/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # ML Model Configuration
    MODEL_DIR = os.environ.get('MODEL_DIR') or 'models'
    DATA_DIR = os.environ.get('DATA_DIR') or 'data'
    TARGET_SIZE = (224, 224)
    CLASS_NAMES = ['rose', 'tulip', 'sunflower']
    
    # API Configuration
    API_VERSION = 'v1'
    API_PREFIX = '/api'
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'logs/app.log'
    
    # Database Configuration (if needed for future extensions)
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    UPLOAD_FOLDER = 'tests/data/uploads'
    MODEL_DIR = 'tests/models'
    DATA_DIR = 'tests/data'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}