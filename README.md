
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
