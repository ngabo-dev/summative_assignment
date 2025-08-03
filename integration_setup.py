# Integration Setup for Flower Classification Frontend + Backend
# This script shows how to connect your existing pipeline with the web interface

import os
import sys
from pathlib import Path

def setup_project_structure():
    """Setup the project directory structure"""
    
    # Create main project directory
    project_structure = {
        'static/': ['index.html', 'app.js'],
        'templates/': [],
        'uploads/': [],
        'logs/': [],
        'models/': [],
        'data/': ['train/', 'test/', 'new_uploads/'],
        'data/train/': ['roses/', 'tulips/', 'sunflowers/'],
        'data/test/': ['roses/', 'tulips/', 'sunflowers/'],
        'data/new_uploads/': ['pending/', 'processed/']
    }
    
    print("📁 Creating project structure...")
    for directory, subdirs in project_structure.items():
        os.makedirs(directory, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(directory, subdir), exist_ok=True)
    
    print("✅ Project structure created!")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """
# Core ML packages
tensorflow>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
Pillow>=9.0.0
opencv-python>=4.7.0

# Web framework
flask>=2.3.0
flask-cors>=4.0.0
flask-socketio>=5.3.0  # Optional for WebSocket support

# Utilities
python-socketio>=5.8.0  # Optional for WebSocket support
python-engineio>=4.6.0  # Optional for WebSocket support
werkzeug>=2.3.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Optional packages for enhanced functionality
gunicorn>=20.1.0  # For production deployment
requests>=2.28.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements.strip())
    
    print("📋 requirements.txt created!")

def create_app_runner():
    """Create the main application runner"""
    
    app_code = '''
# Main Application Runner for Flower Classification Dashboard
# Run this file to start the web interface

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing flower classification modules
# YOU NEED TO MODIFY THESE IMPORTS BASED ON YOUR ACTUAL FILE STRUCTURE
try:
    # Option 1: If your code is in a separate file (e.g., flower_classification.py)
    from flower_classification import (
        Config, 
        preprocessor, 
        model_instance, 
        predictor, 
        retraining_manager,
        train_flower_model
    )
    print("✅ Successfully imported flower classification modules")
    
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("📝 Please ensure your flower classification code is available.")
    print("   You can either:")
    print("   1. Save your notebook code as 'flower_classification.py'")
    print("   2. Modify the imports above to match your file structure")
    print("   3. Copy-paste your classes directly into this file")
    sys.exit(1)

# Import the Flask backend
from flask_backend import app, socketio

if __name__ == '__main__':
    print("🌸 Starting Flower Classification Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    
    # Start the application
    if socketio:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    with open('run_app.py', 'w') as f:
        f.write(app_code)
    
    print("🚀 run_app.py created!")

def create_config_template():
    """Create a configuration template"""
    
    config_code = '''
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
'''
    
    with open('config.py', 'w') as f:
        f.write(config_code)
    
    print("⚙️ config.py created!")

def create_readme():
    """Create README with setup instructions"""
    
    readme_content = '''
# 🌸 Flower Classification Dashboard

A complete web-based interface for training and using a flower classification AI model.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install Python packages
pip install -r requirements.txt
```

### 2. Prepare Your Data
```
data/
├── train/
│   ├── roses/        (Add rose images here)
│   ├── tulips/       (Add tulip images here)
│   └── sunflowers/   (Add sunflower images here)
└── test/
    ├── roses/        (Add test rose images)
    ├── tulips/       (Add test tulip images)
    └── sunflowers/   (Add test sunflower images)
```

### 3. Integration with Your Code

#### Option A: Existing Notebook/Script
If you have your flower classification code in a notebook or script:

1. Save your code as `flower_classification.py`
2. Ensure it exports these components:
   - `Config`
   - `preprocessor`
   - `model_instance`
   - `predictor`
   - `retraining_manager`

#### Option B: Modify Imports
Update the imports in `run_app.py` to match your file structure.

### 4. Run the Application
```bash
python run_app.py
```

The dashboard will be available at: http://localhost:5000

## 📱 Features

### 🔮 **Image Prediction**
- Drag & drop or click to upload flower images
- Real-time prediction with confidence scores
- Support for multiple image formats

### 📤 **Bulk Data Upload**
- Upload multiple images for training
- Organize by flower class
- Progress tracking

### 🔄 **Model Retraining**
- Upload new data and retrain the model
- Real-time training progress
- Automatic model backup

### 📊 **Analytics Dashboard**
- Dataset statistics and visualization
- Model performance metrics
- Training history tracking

## 🔧 API Endpoints

The backend provides these REST API endpoints:

- `GET /api/status` - System status
- `POST /api/predict` - Predict flower type
- `POST /api/upload` - Upload training images
- `POST /api/retrain` - Trigger retraining
- `GET /api/dataset-stats` - Dataset statistics
- `POST /api/evaluate` - Evaluate model

## 📁 File Structure

```
flower-classification-dashboard/
├── static/
│   ├── index.html          # Frontend HTML
│   └── app.js              # Frontend JavaScript
├── data/                   # Training/test data
├── models/                 # Saved models
├── logs/                   # Application logs
├── uploads/                # Temporary uploads
├── flask_backend.py        # Backend API
├── run_app.py             # Main application runner
├── config.py              # Configuration
└── requirements.txt       # Python dependencies
```

## 🔄 Workflow

1. **Initial Setup**: Add training images to `data/train/` folders
2. **Train Model**: Use the retraining interface or train programmatically
3. **Upload New Data**: Use the web interface to upload new training images
4. **Retrain**: Click "Trigger Retraining" to improve the model
5. **Predict**: Upload images for classification

## 🛠️ Customization

### Adding New Flower Classes
1. Update `Config.CLASS_NAMES` in `config.py`
2. Create corresponding directories in `data/train/` and `data/test/`
3. Update the frontend class options in `index.html`

### Modifying Model Architecture
Edit the `_build_model()` method in your `FlowerClassificationModel` class.

### Changing UI Theme
Modify the CSS styles in `index.html` or add custom CSS files.

## 🐛 Troubleshooting

### "Model not found" Error
- Ensure you have trained a model first
- Check the `MODEL_PATH` in `config.py`

### Import Errors
- Verify your flower classification code is accessible
- Update import statements in `run_app.py`

### Upload Issues
- Check file permissions on upload directory
- Verify supported file formats

## 🚀 Production Deployment

For production deployment:

1. Use a production WSGI server like Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 run_app:app
```

2. Set up a reverse proxy with Nginx
3. Use environment variables for configuration
4. Enable proper logging and monitoring

## 📝 License

This project is open source and available under the MIT License.
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("📖 README.md created!")

def main():
    """Main setup function"""
    print("🌸 Setting up Flower Classification Dashboard...")
    print("=" * 50)
    
    setup_project_structure()
    create_requirements_file()
    create_config_template()
    create_app_runner()
    create_readme()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Add your flower images to data/train/ folders")
    print("3. Copy your flower classification code or modify imports in run_app.py")
    print("4. Run the application: python run_app.py")
    print("5. Open http://localhost:5000 in your browser")
    
    print("\n💡 Tips:")
    print("- Start with at least 10-20 images per class for basic training")
    print("- Use high-quality, diverse images for better results")
    print("- The web interface allows easy retraining with new data")

if __name__ == "__main__":
    main()