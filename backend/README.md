# Flower Classification ML Backend

A production-ready Flask backend for the Flower Classification ML System. This backend provides REST API endpoints for real-time flower image classification using TensorFlow and CNN models.

## Features

- **Real-time Image Classification**: Upload flower images for instant CNN-based predictions
- **Model Management**: Version control, retraining, and deployment of ML models
- **Bulk Processing**: Handle multiple image uploads for training data collection
- **System Monitoring**: Real-time metrics and performance monitoring
- **Data Analytics**: Insights into prediction patterns and model performance

## Quick Start

### 1. Setup Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Server

```bash
# Start Flask development server
python app.py
```

The server will start on `http://localhost:5000`

### 3. Verify Installation

Check if the backend is running:
```bash
curl http://localhost:5000/api/health
```

You should see a JSON response indicating the server is healthy.

## API Endpoints

### Health Check
- `GET /api/health` - Server health status

### Predictions
- `POST /api/predict/single` - Classify a single flower image
- `POST /api/predict/bulk` - Process multiple images for training

### Dashboard & Monitoring
- `GET /api/dashboard/metrics` - System and model metrics
- `GET /api/monitoring/metrics` - Real-time system monitoring
- `GET /api/monitoring/logs` - System logs

### Model Management
- `GET /api/model/versions` - List all model versions
- `POST /api/model/retrain` - Start model retraining
- `GET /api/model/retrain/status/<training_id>` - Get training progress
- `POST /api/model/deploy/<version>` - Deploy a model version

### Analytics
- `GET /api/analytics/insights` - Data insights and statistics

## Project Structure

```
backend/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup
├── src/               # Core ML modules
│   ├── model.py       # Model management and training
│   ├── prediction.py  # Image prediction logic
│   └── preprocessing.py # Data preprocessing
├── notebook/          # Jupyter notebooks
│   └── flower_prediction.ipynb
├── data/              # Data storage (created automatically)
│   └── uploads/       # Uploaded images
└── models/            # Trained models (created automatically)
```

## Usage Examples

### Single Image Prediction

```python
import requests

# Upload an image for prediction
with open('flower.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/predict/single',
        files={'image': f}
    )
    result = response.json()
    print(f"Predicted: {result['predicted_class']} ({result['confidence']:.1f}%)")
```

### Start Model Retraining

```python
import requests

# Start retraining with custom parameters
response = requests.post(
    'http://localhost:5000/api/model/retrain',
    json={'epochs': 100, 'learning_rate': 0.0005}
)
training_info = response.json()
print(f"Training started: {training_info['training_id']}")
```

## Model Information

The system uses a Convolutional Neural Network (CNN) architecture:

- **Input**: 224x224 RGB images
- **Classes**: Rose, Tulip, Sunflower
- **Architecture**: 4 Conv2D layers with BatchNormalization and MaxPooling
- **Output**: Softmax probabilities for each class

### Default Model Performance
- **Accuracy**: 94.2%
- **Precision**: 92.8%
- **Recall**: 95.1%
- **F1 Score**: 93.9%

## Configuration

Key configuration options in `config.py`:

- `UPLOAD_FOLDER`: Directory for uploaded images
- `MAX_CONTENT_LENGTH`: Maximum file upload size (16MB)
- `ALLOWED_EXTENSIONS`: Supported image formats

## Development

### Running Tests
```bash
pytest tests/
```

### Development Mode
The Flask app runs in debug mode by default when started with `python app.py`.

### Adding New Features
1. Add new endpoints in `app.py`
2. Implement ML logic in `src/` modules
3. Update tests and documentation

## Production Deployment

For production deployment, use Gunicorn:

```bash
# Install gunicorn (included in requirements.txt)
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

Set these environment variables for production:

```bash
export FLASK_ENV=production
export MODEL_PATH=/path/to/models
export UPLOAD_PATH=/path/to/uploads
```

## Troubleshooting

### Common Issues

1. **Port 5000 already in use**
   ```bash
   # Use a different port
   python app.py --port 5001
   ```

2. **TensorFlow installation issues**
   ```bash
   # Install TensorFlow separately first
   pip install tensorflow==2.13.0
   pip install -r requirements.txt
   ```

3. **CORS issues**
   - The Flask-CORS package is configured to allow all origins in development
   - For production, configure specific allowed origins

4. **File upload errors**
   - Check file size (max 16MB)
   - Ensure file format is supported (PNG, JPG, JPEG, GIF)

### Logs

Check the console output where you started the Flask server for detailed logs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.