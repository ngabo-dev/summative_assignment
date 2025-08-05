# 🌸 Flower Classification API Backend

Production-ready Flask API for flower image classification with real-time statistics and model training capabilities.

## 📁 Simplified Structure

```
backend/
├── app.py                        # Main Flask application ⭐
├── requirements.txt              # Dependencies ⭐
├── src/                          # ML modules
│   ├── model.py
│   ├── prediction.py
│   └── preprocessing.py
├── render.yaml                   # Render configuration
├── Procfile                      # Process configuration
├── runtime.txt                   # Python version
└── RENDER_DEPLOYMENT_GUIDE.md    # Deployment guide
```

**Key Files:**
- **`app.py`** - Main Flask application (all endpoints)
- **`requirements.txt`** - All dependencies
- **`src/`** - ML source code modules

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Production Deployment (Render.com)
```bash
# 1. Push to GitHub
git add app.py requirements.txt src/ render.yaml Procfile runtime.txt
git commit -m "Deploy to Render"
git push origin main

# 2. On Render.com:
#    - New Web Service
#    - Connect GitHub repo
#    - Build: pip install -r requirements.txt
#    - Start: python app.py
#    - Environment: PORT=10000, FLASK_ENV=production

# 3. Test deployment
curl https://your-app-name.onrender.com/
```

## 📋 Features

### ✅ Fixed Issues
- **Real-time statistics** - Uptime, last trained, and performance metrics update correctly
- **Dynamic data tracking** - No more static "2 hours ago" messages
- **All API endpoints working** - Complete functionality for production use
- **Model retraining support** - Background training with progress tracking
- **Balanced predictions** - Fixed single-class bias issue
- **Simplified structure** - Only app.py and requirements.txt needed

### 🛠️ API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/api/stats` | Real-time system statistics |
| POST | `/api/predict/single` | Single image prediction |
| POST | `/api/predict/batch` | Batch image prediction |
| POST | `/api/upload/training` | Upload training images |
| POST | `/api/train/start` | Start model training |
| GET | `/api/train/status/<job_id>` | Get training progress |
| GET | `/api/models` | List model versions |
| POST | `/api/models/<version>/deploy` | Deploy model version |
| GET | `/api/data/insights` | Dataset insights |
| GET | `/api/system/logs` | System logs |

## 📊 Real-time Statistics (FIXED!)

The API now provides **accurate, live-updating statistics**:

```json
{
  "system": {
    "uptime": "2h 45m",           // ✅ Real uptime
    "uptime_percentage": 99.86,    // ✅ Dynamic calculation
    "last_trained": "15 minutes ago", // ✅ Updates when training completes
    "version": "v2.0.0-production",
    "status": "operational",
    "prediction_count": 42,        // ✅ Increments with each prediction
    "error_count": 1               // ✅ Tracks real errors
  }
}
```

## 🔄 Model Training

### Start Training
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"epochs": 25, "learning_rate": 0.001}' \
  https://your-api.onrender.com/api/train/start
```

### Check Progress
```bash
curl https://your-api.onrender.com/api/train/status/JOB_ID
```

### Response
```json
{
  "status": "running",
  "progress": 65,
  "stage": "Training model...",
  "eta": "8m 32s",              // ✅ Real ETA calculation
  "start_time": "2024-01-20T10:30:00Z"
}
```

## 🔮 Image Prediction

### Single Image
```bash
curl -X POST \
  -F "image=@flower.jpg" \
  https://your-api.onrender.com/api/predict/single
```

### Response
```json
{
  "success": true,
  "predicted_class": "rose",
  "confidence": 94.2,
  "probabilities": [
    {"class": "rose", "probability": 94.2, "color": "hsl(350, 70%, 60%)"},
    {"class": "tulip", "probability": 3.8, "color": "hsl(270, 70%, 60%)"},
    {"class": "sunflower", "probability": 2.0, "color": "hsl(50, 80%, 60%)"}
  ],
  "timestamp": "2024-01-20T10:35:22Z"
}
```

## 🛡️ Production Features

### Security
- File size limits (16MB max)
- Secure filename handling
- Input validation
- Error handling
- CORS configuration

### Monitoring
- Real-time uptime tracking ✅
- Error counting and logging ✅
- Performance metrics ✅
- Health check endpoints ✅
- Dynamic statistics updates ✅

### Reliability
- Graceful error handling
- Thread-safe operations
- Automatic service recovery
- Background job processing

## 🔧 Environment Variables

Required for production deployment:

```bash
PORT=10000                    # Server port
FLASK_ENV=production         # Environment
PYTHON_VERSION=3.9.18        # Python version
PYTHONPATH=/opt/render/project/src  # Module path
```

## 📈 Monitoring

### Health Check
```bash
# Simple health check
curl https://your-api.onrender.com/

# Test real-time stats
curl https://your-api.onrender.com/api/stats
```

### Real-time Monitoring
```bash
# Monitor continuously
python health_monitor.py https://your-api.onrender.com
```

## 🚀 Render.com Deployment

### Quick Deploy Commands
```bash
# 1. Prepare files
git add app.py requirements.txt src/ render.yaml Procfile runtime.txt
git commit -m "Production deployment"
git push origin main

# 2. Configure Render service:
#    Build Command: pip install -r requirements.txt
#    Start Command: python app.py
#    Environment Variables: PORT=10000, FLASK_ENV=production

# 3. Deploy and verify
curl https://your-app-name.onrender.com/api/stats
```

### Complete Guide
See [RENDER_DEPLOYMENT_GUIDE.md](RENDER_DEPLOYMENT_GUIDE.md) for detailed step-by-step instructions.

## 🧪 Testing

### Local Testing
```bash
# Start server
python app.py

# Test endpoints
curl http://localhost:5000/api/stats
curl -X POST -F "image=@test.jpg" http://localhost:5000/api/predict/single
```

### Production Testing
```bash
# Health monitoring
python health_monitor.py https://your-api.onrender.com --once

# Verify real-time stats update
curl https://your-api.onrender.com/api/stats
# Wait 1 minute
curl https://your-api.onrender.com/api/stats
# Uptime should have increased!
```

## 📚 Frontend Integration

```typescript
const API_BASE_URL = 'https://your-api.onrender.com';

// Get real-time stats (now updates correctly!)
const getStats = async () => {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  return response.json();
};

// Upload image for prediction
const predictImage = async (imageFile: File) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`${API_BASE_URL}/api/predict/single`, {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};
```

## 🧹 Cleanup (Optional)

If you have old duplicate files, run:
```bash
python cleanup_old_files.py
```

This removes:
- `production_app.py` (consolidated into `app.py`)
- `simplified_app.py` (consolidated into `app.py`)
- `requirements_production.txt` (consolidated into `requirements.txt`)
- `requirements_minimal.txt` (consolidated into `requirements.txt`)

## 🆘 Troubleshooting

### Common Issues

**Service won't start:**
- Check `python app.py` works locally
- Verify `requirements.txt` installs correctly
- Check environment variables on Render

**Statistics not updating:**
- ✅ **FIXED** - Now updates in real-time
- Uptime calculates from actual start time
- "Last trained" updates when training completes

**Predictions fail:**
- Ensure ML dependencies installed
- Check image format (PNG, JPG, JPEG, GIF)
- Verify file size (max 16MB)

### Debug Commands
```bash
# Check service health
curl https://your-api.onrender.com/

# View detailed stats (should update every refresh)
curl https://your-api.onrender.com/api/stats

# Test prediction
curl -X POST -F "image=@flower.jpg" https://your-api.onrender.com/api/predict/single
```

## 🎉 Ready for Production!

Your simplified API structure:
- ✅ **`app.py`** - Single main Flask file
- ✅ **`requirements.txt`** - Single requirements file  
- ✅ Real-time statistics that actually update
- ✅ All endpoints working correctly
- ✅ Model training and prediction
- ✅ Ready for Render.com deployment

Deploy now and your flower classification API will work perfectly! 🌸🌷🌻

**No more confusion with multiple files - just `app.py` and `requirements.txt`!**