# 🌸 Flower Classification ML Platform

> **Now powered by Vite for lightning-fast development!** ⚡

A production-ready full-stack application for flower image classification using deep learning. Built with React + Vite frontend, Flask backend, and TensorFlow CNN models.

![Platform Overview](https://via.placeholder.com/800x400/2563eb/ffffff?text=Flower+Classification+ML+Platform)

## ✨ What's New - Vite Migration

This project has been upgraded from Create React App to **Vite** for:

- ⚡ **Lightning-fast startup** (under 1 second)
- 🔄 **Instant hot module replacement** (HMR)
- 📦 **Optimized build times** (3x faster than CRA)
- 🎯 **Modern tooling** with native ES modules
- 🧩 **Better TypeScript integration**

## 🌟 Features

- **Real-time Image Classification**: Upload flower images for instant CNN-based predictions
- **Interactive Dashboard**: Monitor ML system performance, metrics, and analytics
- **Model Management**: Version control, retraining, and deployment of ML models
- **Bulk Processing**: Handle multiple image uploads for training data collection
- **System Monitoring**: Real-time metrics, logs, and performance monitoring
- **Responsive UI**: Modern React interface with dark/light mode support
- **Production Ready**: Built with production-grade architecture and error handling

## 🚀 Quick Start

### Prerequisites

- **Node.js 18+** with npm
- **Python 3.8+** with pip
- **Git**

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

4. **Open your browser:** http://localhost:3000

You should see a green "Connected" status and fully functional ML dashboard!

### Manual Setup

If you prefer manual setup, see [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

## 📊 Platform Architecture

### Frontend Stack (Vite + React)
- **Build Tool**: Vite for lightning-fast development
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS v4 with custom design system
- **UI Components**: shadcn/ui component library
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React

### Backend Stack (Python + Flask)
- **Framework**: Flask with CORS support
- **ML Library**: TensorFlow 2.13 for CNN models
- **Image Processing**: OpenCV and Pillow
- **API**: RESTful endpoints with JSON responses
- **Model Management**: Version control and deployment system

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
├── backend/                 # Python Flask backend
│   ├── app.py              # Flask application
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
npm run start-backend    # Start only Flask backend

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
   - Flask auto-reloads in debug mode
   - API changes reflect immediately

4. **Styling:**
   - Use Tailwind CSS classes
   - Custom properties in `src/styles/globals.css`
   - Dark/light mode support built-in

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

## 🚀 Production Deployment

### Frontend (Vite Build)
```bash
npm run build
# Deploy dist/ directory to your hosting platform
# Files are optimized and compressed automatically
```

### Backend (Flask)
```bash
cd backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Environment Variables

```bash
# Frontend (.env.local)
VITE_API_BASE_URL=https://your-api-domain.com

# Backend
FLASK_ENV=production
MODEL_PATH=/path/to/models
UPLOAD_PATH=/path/to/uploads
```

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Check if Flask server is running: `python backend/app.py`
   - Verify port 5000 is available
   - Look for CORS configuration issues

2. **"No model available for prediction" Error**
   - This usually means TensorFlow isn't properly installed
   - Check backend logs for model initialization errors
   - Try reinstalling TensorFlow: `pip install --upgrade tensorflow==2.13.0`
   - Ensure you have enough RAM (minimum 4GB for model creation)

2. **Vite/Frontend Issues**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **TensorFlow Installation**
   ```bash
   cd backend
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install tensorflow==2.13.0 --upgrade
   ```

4. **Port Conflicts**
   ```bash
   # Use different ports
   PORT=3001 npm run dev  # Frontend
   # Edit backend/app.py for backend port
   ```

### Performance Tips

- **Development**: Use `npm run dev` for fastest reload
- **Production**: Use `npm run build && npm run preview` to test
- **Memory**: Close unused browser tabs and applications
- **GPU**: Enable hardware acceleration for better ML performance

### Getting Help

1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed setup instructions
2. Review browser console and terminal logs for errors
3. Ensure all prerequisites are correctly installed
4. Try the automated setup scripts first

## 🎉 Success Criteria

After successful setup, you should have:

- ✅ Frontend running on http://localhost:3000
- ✅ Backend running on http://localhost:5000
- ✅ Green "Connected" status in the header
- ✅ Working image classification
- ✅ Real-time dashboard metrics
- ✅ All 6 dashboard sections functional

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Made with ❤️ and ⚡ Vite for the ML community**

🚀 **Enjoy the lightning-fast development experience!**