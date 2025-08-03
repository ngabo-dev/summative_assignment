# Flower Classification ML Platform

A production-ready full-stack application for flower image classification using deep learning. Built with React frontend, Flask backend, and TensorFlow CNN models.

![Platform Overview](https://via.placeholder.com/800x400/2563eb/ffffff?text=Flower+Classification+ML+Platform)

## 🌟 Features

- **Real-time Image Classification**: Upload flower images for instant CNN-based predictions
- **Interactive Dashboard**: Monitor ML system performance, metrics, and analytics
- **Model Management**: Version control, retraining, and deployment of ML models
- **Bulk Processing**: Handle multiple image uploads for training data collection
- **System Monitoring**: Real-time metrics, logs, and performance monitoring
- **Responsive UI**: Modern React interface with dark/light mode support

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git**

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd flower-classification-platform

# Setup backend
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

### 2. Start the Backend

```bash
# Navigate to backend and start Flask server
cd backend
python app.py
```

The backend will start on `http://localhost:5000`

### 3. Start the Frontend

Open a new terminal:

```bash
# Install frontend dependencies (if needed)
npm install

# Start the React development server
npm start
```

The frontend will start on `http://localhost:3000`

### 4. Access the Platform

Open your browser and navigate to `http://localhost:3000`

You should see the Flower Classification ML Platform with a green "Connected" status indicator.

## 📊 Platform Overview

### Dashboard Features

1. **Home Dashboard**
   - Real-time ML system metrics
   - Model performance statistics
   - System resource monitoring
   - Prediction analytics

2. **Single Image Prediction**
   - Drag-and-drop image upload
   - Instant flower classification
   - Confidence scores and probability distribution
   - Response time metrics

3. **Bulk Upload & Retraining**
   - Multiple image upload
   - Data labeling for training
   - Model retraining with progress tracking
   - Training parameter configuration

4. **Data Insights**
   - Prediction patterns and trends
   - Class distribution analytics
   - Confidence score analysis
   - Upload statistics

5. **Model Management**
   - Model version history
   - Performance comparison
   - Model deployment controls
   - Training status monitoring

6. **System Monitoring**
   - Real-time system metrics
   - Container health status
   - API performance logs
   - Resource utilization

## 🏗️ Architecture

### Frontend (React + TypeScript)
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: shadcn/ui component library
- **Charts**: Recharts for data visualization
- **State Management**: React hooks and context

### Backend (Python + Flask)
- **Framework**: Flask with CORS support
- **ML Library**: TensorFlow 2.13 for CNN models
- **Image Processing**: OpenCV and Pillow
- **API**: RESTful endpoints with JSON responses

### ML Model
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224x224 RGB images
- **Classes**: Rose, Tulip, Sunflower
- **Performance**: 94.2% accuracy, 92.8% precision

## 🔧 Development

### Project Structure

```
flower-classification-platform/
├── README.md                 # This file
├── App.tsx                   # Main React application
├── components/               # React components
│   ├── Dashboard.tsx         # Main dashboard
│   ├── SinglePrediction.tsx  # Image prediction interface
│   ├── BulkUpload.tsx        # Bulk upload & retraining
│   ├── DataInsights.tsx      # Analytics dashboard
│   ├── ModelManagement.tsx   # Model version control
│   ├── SystemMonitoring.tsx  # System metrics
│   ├── ui/                   # UI component library
│   └── utils/                # Utility functions
├── styles/                   # CSS and styling
├── backend/                  # Python Flask backend
│   ├── app.py               # Main Flask application
│   ├── src/                 # ML modules
│   │   ├── model.py         # Model management
│   │   ├── prediction.py    # Prediction logic
│   │   └── preprocessing.py # Data processing
│   ├── requirements.txt     # Python dependencies
│   └── README.md           # Backend documentation
└── guidelines/              # Development guidelines
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict/single` | POST | Single image prediction |
| `/api/predict/bulk` | POST | Bulk image processing |
| `/api/dashboard/metrics` | GET | Dashboard metrics |
| `/api/model/versions` | GET | Model versions |
| `/api/model/retrain` | POST | Start retraining |
| `/api/analytics/insights` | GET | Data insights |
| `/api/monitoring/metrics` | GET | System monitoring |

### Adding New Features

1. **Frontend Components**: Add new React components in `/components`
2. **Backend Endpoints**: Add new routes in `backend/app.py`
3. **ML Logic**: Implement in `backend/src/` modules
4. **Styling**: Use Tailwind classes and CSS custom properties

## 🧪 Testing

### Backend Testing
```bash
cd backend
pytest tests/
```

### Frontend Testing
```bash
npm test
```

## 🚀 Production Deployment

### Backend Production
```bash
cd backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend Production
```bash
npm run build
# Deploy build/ directory to your hosting platform
```

### Environment Variables

Set these for production:

```bash
# Backend
export FLASK_ENV=production
export MODEL_PATH=/path/to/models
export UPLOAD_PATH=/path/to/uploads

# Frontend
export REACT_APP_API_URL=https://your-api-domain.com
```

## 🔧 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure Flask server is running on port 5000
   - Check firewall settings
   - Verify CORS configuration

2. **TensorFlow Installation**
   ```bash
   pip install tensorflow==2.13.0 --upgrade
   ```

3. **Port Conflicts**
   ```bash
   # Use different ports
   # Backend: python app.py --port 5001
   # Frontend: PORT=3001 npm start
   ```

4. **Image Upload Issues**
   - Check file size (max 16MB)
   - Ensure supported formats (PNG, JPG, JPEG, GIF)
   - Verify upload directory permissions

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for training)
- **Storage**: 2GB free space for models and data
- **GPU**: Optional but recommended for model training

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:

1. Check the [Backend README](backend/README.md) for detailed setup instructions
2. Review the troubleshooting section above
3. Open an issue on GitHub
4. Check system logs for error details

---

**Made with ❤️ for the ML community**