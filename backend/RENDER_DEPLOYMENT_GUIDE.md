# 🚀 Complete Render.com Deployment Guide

## Step-by-Step Deployment Instructions

### Prerequisites
- GitHub account
- Render.com account (free tier works)
- Your flower classification backend code

---

## Part 1: Prepare Your Repository

### 1.1 Create GitHub Repository
```bash
# Create new repository on GitHub named 'flower-classification-api'
# Clone it locally
git clone https://github.com/YOUR_USERNAME/flower-classification-api.git
cd flower-classification-api
```

### 1.2 Copy Backend Files
Copy these files from your project to the new repository root:

**Required Files:**
```
flower-classification-api/
├── app.py                     # Main Flask application
├── requirements.txt           # Dependencies
├── runtime.txt                # Python version
├── Procfile                   # Process configuration
├── render.yaml                # Render configuration
├── src/                       # ML source code
│   ├── model.py
│   ├── prediction.py
│   └── preprocessing.py
└── README.md                  # Documentation
```

### 1.3 Push to GitHub
```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment setup for Render"

# Push to GitHub
git push origin main
```

---

## Part 2: Deploy on Render

### 2.1 Create New Web Service

1. **Go to Render Dashboard**
   - Visit [render.com](https://render.com)
   - Sign in with GitHub account
   - Click "New +"
   - Select "Web Service"

2. **Connect Repository**
   - Select "Build and deploy from a Git repository"
   - Choose your `flower-classification-api` repository
   - Click "Connect"

### 2.2 Configure Service Settings

**Basic Settings:**
```
Name: flower-classification-api
Environment: Python 3
Region: Oregon (US West) or closest to you
Branch: main
```

**Build & Deploy Settings:**
```
Build Command: 
pip install -r requirements.txt

Start Command:
python app.py
```

**Advanced Settings:**
```
Auto-Deploy: Yes (recommended)
Health Check Path: /
```

### 2.3 Environment Variables
Add these environment variables in Render dashboard:

```
PORT = 10000
FLASK_ENV = production  
PYTHON_VERSION = 3.9.18
PYTHONPATH = /opt/render/project/src
```

### 2.4 Deploy
- Click "Create Web Service"
- Wait for deployment (usually 5-10 minutes)
- Monitor build logs for any issues

---

## Part 3: Verify Deployment

### 3.1 Test Basic Endpoints

**Health Check:**
```bash
curl https://your-app-name.onrender.com/
```

**Expected Response:**
```json
{
  "status": "healthy",
  "service": "Flower Classification API", 
  "version": "2.0.0",
  "ml_initialized": true,
  "uptime": "2m"
}
```

### 3.2 Test Statistics Endpoint

```bash
curl https://your-app-name.onrender.com/api/stats
```

**Expected Response:**
```json
{
  "system": {
    "uptime": "5m",
    "uptime_percentage": 99.5,
    "last_trained": "Never",
    "version": "v2.0.0-production",
    "status": "operational"
  },
  "model": {...},
  "dataset": {...}
}
```

### 3.3 Test Image Prediction

**Using cURL:**
```bash
curl -X POST \
  -F "image=@path/to/flower.jpg" \
  https://your-app-name.onrender.com/api/predict/single
```

### 3.4 Test Training Endpoint

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"epochs": 10, "learning_rate": 0.001}' \
  https://your-app-name.onrender.com/api/train/start
```

---

## Part 4: Frontend Integration

### 4.1 Update Frontend API URL

In your frontend project, update the API base URL:

```typescript
// Replace localhost with your Render URL
const API_BASE_URL = 'https://your-app-name.onrender.com';

// Example API calls
const uploadImage = async (formData: FormData) => {
  const response = await fetch(`${API_BASE_URL}/api/predict/single`, {
    method: 'POST',
    body: formData
  });
  return response.json();
};

const getStats = async () => {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  return response.json();
};
```

### 4.2 Handle CORS (Already Configured)
The app includes CORS headers for all origins, so your frontend should work without additional configuration.

---

## Part 5: Monitoring & Maintenance

### 5.1 Monitor Service Health

**Render Dashboard:**
- Check service status
- View logs
- Monitor resource usage
- Track deployment history

**Log Access:**
```bash
# View recent logs in Render dashboard
# Or use Render CLI (optional)
render logs -s your-service-name
```

### 5.2 Update and Redeploy

```bash
# Make changes to your code
git add .
git commit -m "Update API functionality"
git push origin main

# Render will auto-deploy if auto-deploy is enabled
```

---

## Part 6: Troubleshooting

### 6.1 Common Issues

**Build Fails:**
```bash
# Check requirements.txt
# Review build logs in Render dashboard
```

**App Won't Start:**
```bash
# Verify start command: python app.py
# Check environment variables
# Review application logs
```

**Predictions Fail:**
```bash
# ML components may not initialize
# App will return error messages
# Check logs for specific errors
```

### 6.2 Debugging Commands

**Check Service Status:**
- Visit: `https://your-app-name.onrender.com/`
- Should return healthy status

**View Real-time Stats:**
- Visit: `https://your-app-name.onrender.com/api/stats`
- Shows uptime, model status, performance

**Test Endpoint:**
```bash
# Basic test
curl https://your-app-name.onrender.com/api/stats

# Upload test
curl -X POST -F "image=@test.jpg" \
  https://your-app-name.onrender.com/api/predict/single
```

---

## Quick Deployment Checklist

- [ ] Repository created and pushed to GitHub
- [ ] Render account set up
- [ ] Web service created and configured
- [ ] Environment variables set
- [ ] Service deployed successfully
- [ ] Health check endpoint working
- [ ] API endpoints tested
- [ ] Frontend integration updated
- [ ] Monitoring set up

---

## Your API Endpoints

```
GET  /                           # Health check
GET  /api/stats                  # System statistics  
POST /api/predict/single         # Single image prediction
POST /api/predict/batch          # Batch predictions
POST /api/upload/training        # Upload training data
POST /api/train/start            # Start training
GET  /api/train/status/<job_id>  # Training status
GET  /api/models                 # List models
POST /api/models/<version>/deploy # Deploy model
GET  /api/data/insights          # Data insights
GET  /api/system/logs            # System logs
```

🎉 **Your API is now live and ready for production use!**

## Quick Commands for Immediate Deployment

```bash
# 1. Create repository and add files
git init
git add app.py requirements.txt runtime.txt Procfile render.yaml src/
git commit -m "Production-ready Flask API"
git remote add origin https://github.com/YOUR_USERNAME/flower-classification-api.git
git push -u origin main

# 2. Go to render.com
#    - New Web Service
#    - Connect GitHub repo
#    - Use these settings:
#      Build: pip install -r requirements.txt
#      Start: python app.py
#      Environment: Add PORT=10000, FLASK_ENV=production

# 3. Deploy and test
curl https://your-app-name.onrender.com/
curl https://your-app-name.onrender.com/api/stats
```

That's it! Your flower classification API will be live on Render in under 10 minutes! 🌸