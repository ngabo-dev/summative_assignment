# 🌸 Flower Classification ML Platform

> **Now powered by Vite + FastAPI for lightning-fast performance!** ⚡🚀

A production-ready full-stack application for flower image classification using deep learning. Built with React + Vite frontend, FastAPI backend, and TensorFlow CNN models.

## 📺 Video Tutorial

🎥 **Watch the complete setup and demo:** [Flower Classification ML Platform Tutorial](https://www.youtube.com/watch?v=E4B0uK4fOJ4)

![Platform Overview](https://via.placeholder.com/800x400/2563eb/ffffff?text=Flower+Classification+ML+Platform)

## ✨ What's New - Modern Stack

This project features a cutting-edge tech stack:

### Frontend Upgrades
- ⚡ **Lightning-fast Vite** (under 1 second startup)
- 🔄 **Instant hot module replacement** (HMR)
- 📦 **Optimized build times** (3x faster than CRA)
- 🎯 **Modern tooling** with native ES modules
- 🧩 **Better TypeScript integration**

### Backend Upgrades
- 🚀 **FastAPI** for high-performance async API
- 📖 **Automatic API documentation** at `/docs`
- ✨ **Type hints and validation** with Pydantic
- ⚡ **Async/await** for better concurrent performance
- 🛡️ **Built-in security features**

## 🌟 Features

- **Real-time Image Classification**: Upload flower images for instant CNN-based predictions
- **Interactive Dashboard**: Monitor ML system performance, metrics, and analytics
- **Model Management**: Version control, retraining, and deployment of ML models
- **Bulk Processing**: Handle multiple image uploads for training data collection
- **System Monitoring**: Real-time metrics, logs, and performance monitoring
- **Responsive UI**: Modern React interface with dark/light mode support
- **Production Ready**: Built with production-grade architecture and error handling
- **API Documentation**: Interactive docs with FastAPI's automatic OpenAPI generation

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** with npm
- **Python 3.8+** with pip
- **Git**

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for model training)
- **Storage**: At least 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+

### Automated Setup (Recommended)

1. **Extract the project:**
   ```bash
   unzip flower-classification-platform.zip
   cd flower-classification-platform
   ```

2. **Run the setup script:**
   
   **Windows:**
   ```cmd
   setup.bat
   ```
   
   **macOS/Linux:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start the application:**
   ```bash
   npm run start-all
   ```

4. **Open your browser:** 
   - Frontend: http://localhost:3000
   - API Docs: http://localhost:8000/docs

You should see a green "Connected" status and fully functional ML dashboard!

### Manual Setup Guide

<details>
<summary>Click here for detailed manual setup instructions</summary>

#### Step 1: Extract and Navigate

```bash
# Extract the project
unzip flower-classification-platform.zip
cd flower-classification-platform
```

#### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print('TensorFlow version:', tensorflow.__version__)"
python -c "import fastapi; print('FastAPI installed successfully')"

# Return to project root
cd ..
```

#### Step 3: Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Verify installation
npm run type-check
```

#### Step 4: Environment Configuration

```bash
# Copy environment template (if available)
cp .env.example .env.local

# Edit .env.local if needed (optional)
# The default settings should work for local development
```

</details>

## 📊 Platform Architecture

### Frontend Stack (Vite + React)
- **Build Tool**: Vite for lightning-fast development
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS v4 with custom design system
- **UI Components**: shadcn/ui component library
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React

### Backend Stack (FastAPI + Python)
- **Framework**: FastAPI with async support
- **ML Library**: TensorFlow 2.13 for CNN models
- **Image Processing**: OpenCV and Pillow
- **API**: RESTful endpoints with automatic OpenAPI docs
- **Model Management**: Version control and deployment system
- **Validation**: Pydantic models for request/response validation

### ML Model Specifications
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Classes**: Rose, Tulip, Sunflower
- **Performance**: 94.2% accuracy, 92.8% precision
- **Framework**: TensorFlow/Keras with transfer learning

## 🎯 Application Features

### 1. **Dashboard Overview**
- Real-time ML system metrics
- Model performance statistics
- System resource monitoring
- Prediction analytics and trends

### 2. **Single Image Prediction**
- Drag-and-drop image upload
- Instant flower classification
- Confidence scores with probability distribution
- Response time metrics

### 3. **Bulk Upload & Retraining**
- Multiple image upload capability
- Data labeling for training
- Model retraining with progress tracking
- Training parameter configuration

### 4. **Data Insights**
- Prediction patterns and trends
- Class distribution analytics
- Confidence score analysis
- Upload statistics and metrics

### 5. **Model Management**
- Model version history
- Performance comparison tools
- Model deployment controls
- Training status monitoring

### 6. **System Monitoring**
- Real-time system metrics
- Container health status
- API performance logs
- Resource utilization tracking

## 🔧 Development

### Project Structure

```
flower-classification-platform/
├── src/                      # Vite React frontend
│   ├── App.tsx              # Main application component
│   ├── main.tsx             # Vite entry point
│   └── styles/              # Tailwind CSS styling
├── components/              # React components
│   ├── Dashboard.tsx        # Main dashboard
│   ├── SinglePrediction.tsx # Image prediction
│   ├── ui/                  # UI component library
│   └── utils/               # Utility functions
├── backend/                 # Python FastAPI backend
│   ├── app.py              # FastAPI application
│   ├── src/                # ML modules
│   └── requirements.txt    # Python dependencies
├── public/                 # Static assets
├── vite.config.ts          # Vite configuration
├── package.json            # Node.js dependencies
└── tailwind.config.js      # Tailwind configuration
```

### Available Scripts

```bash
# Development
npm run dev              # Start Vite dev server (fast!)
npm run start-all        # Start both frontend and backend
npm run start-backend    # Start only FastAPI backend

# Building
npm run build           # Build for production
npm run preview         # Preview production build

# Code Quality
npm run type-check      # TypeScript type checking
npm run lint           # ESLint code linting

# Utilities
npm run clean          # Clean node_modules and build files
npm run setup          # Manual setup (alternative to scripts)
```

### Development Workflow

1. **Start development:**
   ```bash
   npm run start-all
   ```

2. **Frontend changes:**
   - Edit files in `src/` or `components/`
   - Vite provides instant hot reload
   - TypeScript errors show in real-time

3. **Backend changes:**
   - Edit files in `backend/`
   - FastAPI auto-reloads with `--reload` flag
   - API changes reflect immediately
   - Check interactive docs at http://localhost:8000/docs

4. **Styling:**
   - Use Tailwind CSS classes
   - Custom properties in `src/styles/globals.css`
   - Dark/light mode support built-in

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/api/predict/single` | POST | Single image prediction |
| `/api/predict/bulk` | POST | Bulk image processing |
| `/api/dashboard/metrics` | GET | Dashboard metrics |
| `/api/model/versions` | GET | Model versions |
| `/api/model/retrain` | POST | Start retraining |
| `/api/analytics/insights` | GET | Data insights |
| `/api/monitoring/metrics` | GET | System monitoring |

## 🚀 Production Deployment

### Frontend (Vite Build)
```bash
npm run build
# Deploy dist/ directory to your hosting platform
# Files are optimized and compressed automatically
```

### Backend (FastAPI)
```bash
cd backend
pip install uvicorn gunicorn
# Development
uvicorn app:app --reload --host 0.0.0.0 --port 8000
# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

### Environment Variables

```bash
# Frontend (.env.local)
VITE_API_BASE_URL=https://your-api-domain.com

# Backend
FASTAPI_ENV=production
MODEL_PATH=/path/to/models
UPLOAD_PATH=/path/to/uploads
PORT=8000
```

## 🐛 Troubleshooting

### Common Issues

<details>
<summary>1. Backend Connection Failed</summary>

**Symptoms:** Red "Disconnected" status, connection errors

**Solutions:**
```bash
# Check if FastAPI is running
ps aux | grep python  # macOS/Linux
tasklist | findstr python  # Windows

# Check port 8000
netstat -an | grep 8000  # macOS/Linux
netstat -an | findstr 8000  # Windows

# Restart backend
cd backend
python app.py
# Or with uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
</details>

<details>
<summary>2. "No model available for prediction" Error</summary>

This usually means TensorFlow isn't properly installed:
- Check backend logs for model initialization errors
- Try reinstalling TensorFlow: `pip install --upgrade tensorflow==2.13.0`
- Ensure you have enough RAM (minimum 4GB for model creation)
- Check Python version compatibility
</details>

<details>
<summary>3. TensorFlow Installation Issues</summary>

**Solutions:**
```bash
# Standard installation
pip uninstall tensorflow
pip install tensorflow==2.13.0

# For Apple Silicon Macs:
pip install tensorflow-macos tensorflow-metal

# For CUDA GPU support (Windows/Linux):
pip install tensorflow-gpu==2.13.0
```
</details>

<details>
<summary>4. Node.js/Frontend Issues</summary>

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Check Node.js version
node --version  # Should be 18+
```
</details>

<details>
<summary>5. Port Conflicts</summary>

```bash
# Kill processes on port 3000 (frontend)
npx kill-port 3000

# Kill processes on port 8000 (backend)
npx kill-port 8000

# Or use different ports
PORT=3001 npm run dev  # Frontend
# For backend, edit app.py: uvicorn.run(..., port=8001)
```
</details>

### Performance Tips

- **Development**: Use `npm run dev` for fastest reload
- **Production**: Use `npm run build && npm run preview` to test
- **Memory**: Close unused browser tabs and applications
- **GPU**: Enable hardware acceleration for better ML performance
- **API Docs**: Visit http://localhost:8000/docs for interactive testing

### Getting Help

1. Check browser console and terminal logs for errors
2. Visit the interactive API docs at http://localhost:8000/docs
3. Ensure all prerequisites are correctly installed
4. Watch the video tutorial linked above
5. Try the automated setup scripts first

## ✅ Verification Steps

After successful setup, you should have:

- ✅ Frontend running on http://localhost:3000
- ✅ Backend running on http://localhost:8000
- ✅ Interactive API docs at http://localhost:8000/docs
- ✅ Green "Connected" status in the header
- ✅ Working image classification
- ✅ Real-time dashboard metrics
- ✅ All 6 dashboard sections functional

### Test Commands

```bash
# Test backend health
curl http://localhost:8000/

# View API documentation
open http://localhost:8000/docs

# Test prediction endpoint
curl -X POST -F "image=@flower.jpg" http://localhost:8000/api/predict/single

# Monitor real-time stats
curl http://localhost:8000/api/stats
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Made with ❤️, ⚡ Vite, and 🚀 FastAPI for the ML community**

🎯 **Enjoy the lightning-fast development experience with modern async performance!**
