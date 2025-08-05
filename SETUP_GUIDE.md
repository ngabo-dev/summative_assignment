# 🌸 Flower Classification ML Platform - Setup Guide

A complete guide to set up and run the Flower Classification ML Platform locally.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Setup](#detailed-setup)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)
6. [Development](#development)

## ✅ Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software

- **Node.js** (version 18 or higher)
  - [Download from nodejs.org](https://nodejs.org/)
  - Verify installation: `node --version` and `npm --version`

- **Python** (version 3.8 or higher)
  - [Download from python.org](https://www.python.org/)
  - Verify installation: `python --version` or `python3 --version`

- **Git**
  - [Download from git-scm.com](https://git-scm.com/)
  - Verify installation: `git --version`

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended for model training)
- **Storage**: At least 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+

## 🚀 Quick Start

### Option A: Automated Setup (Recommended)

1. **Extract the Project**
   ```bash
   # Extract the downloaded zip file
   unzip flower-classification-platform.zip
   cd flower-classification-platform
   ```

2. **Run Setup Script**
   
   **For Windows:**
   ```cmd
   setup.bat
   ```
   
   **For macOS/Linux:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Start the Application**
   ```bash
   npm run start-all
   ```

### Option B: Manual Setup

Follow the [Detailed Setup](#detailed-setup) section below.

## 🔧 Detailed Setup

### Step 1: Extract and Navigate

```bash
# Extract the project
unzip flower-classification-platform.zip
cd flower-classification-platform
```

### Step 2: Backend Setup

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

# Return to project root
cd ..
```

### Step 3: Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Verify installation
npm run type-check
```

### Step 4: Environment Configuration

```bash
# Copy environment template
cp .env.example .env.local

# Edit .env.local if needed (optional)
# The default settings should work for local development
```

## ▶️ Running the Application

### Option 1: Start Both Services Together

```bash
# This will start both frontend and backend simultaneously
npm run start-all
```

### Option 2: Start Services Separately

**Terminal 1 - Backend:**
```bash
cd backend
# Activate virtual environment if not already active
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

### Access the Application

1. **Frontend**: Open your browser to `http://localhost:3000`
2. **Backend API**: Available at `http://localhost:5000`

You should see:
- ✅ Green "Connected" status in the top header
- Fully functional ML dashboard
- Real-time system metrics

## 🎯 Verifying the Setup

### 1. Check Backend Health
```bash
curl http://localhost:5000/api/health
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_status": "online",
  "version": "v2.1.3"
}
```

### 2. Test Image Prediction
1. Go to "Single Prediction" page
2. Upload a flower image (JPG, PNG, GIF)
3. Click "Predict Flower Type"
4. You should see classification results with confidence scores

### 3. Check All Dashboard Features
- ✅ Dashboard Overview - Shows system metrics
- ✅ Single Prediction - Image classification works
- ✅ Bulk Upload - File upload interface loads
- ✅ Data Insights - Charts and analytics display
- ✅ Model Management - Version control interface
- ✅ System Monitoring - Real-time metrics

## 🐛 Troubleshooting

### Common Issues

#### 1. Backend Connection Failed

**Symptoms:** Red "Disconnected" status, connection errors

**Solutions:**
```bash
# Check if Flask is running
ps aux | grep python  # macOS/Linux
tasklist | findstr python  # Windows

# Check port 5000
netstat -an | grep 5000  # macOS/Linux
netstat -an | findstr 5000  # Windows

# Restart backend
cd backend
python app.py
```

#### 2. TensorFlow Installation Issues

**Symptoms:** Import errors when starting backend

**Solutions:**
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.13.0

# For Apple Silicon Macs:
pip install tensorflow-macos tensorflow-metal
```

#### 3. Node.js Dependencies Issues

**Symptoms:** Frontend build failures, module not found errors

**Solutions:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# Check Node.js version
node --version  # Should be 18+
```

#### 4. Port Conflicts

**Symptoms:** "Port already in use" errors

**Solutions:**
```bash
# Kill processes on port 3000 (frontend)
npx kill-port 3000

# Kill processes on port 5000 (backend)
npx kill-port 5000

# Or use different ports
PORT=3001 npm run dev  # Frontend
# For backend, edit app.py line 424: app.run(debug=True, host='0.0.0.0', port=5001)
```

#### 5. Python Virtual Environment Issues

**Symptoms:** Module not found errors, permission issues

**Solutions:**
```bash
# Recreate virtual environment
rm -rf backend/venv
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### Performance Optimization

#### For Better Performance:

1. **Close unnecessary applications** to free up RAM
2. **Use Chrome/Edge** for best frontend performance
3. **Enable hardware acceleration** in your browser
4. **For training models**: Consider using a machine with GPU support

#### Resource Monitoring:

```bash
# Monitor system resources
# Windows: Task Manager (Ctrl+Shift+Esc)
# macOS: Activity Monitor
# Linux: htop or top
```

### Logs and Debugging

#### Frontend Logs:
- Open browser Developer Tools (F12)
- Check Console tab for errors
- Network tab shows API requests

#### Backend Logs:
- Check terminal where you ran `python app.py`
- Logs show API requests, errors, and system status

### Getting Help

If you're still having issues:

1. **Check logs** in both frontend (browser console) and backend (terminal)
2. **Verify all prerequisites** are properly installed
3. **Try the troubleshooting steps** relevant to your issue
4. **Check system resources** - ensure you have enough RAM/storage
5. **Restart your computer** and try again

## 🚀 Development

### Project Structure

```
flower-classification-platform/
├── src/                      # React frontend source
│   ├── App.tsx              # Main React application
│   ├── main.tsx             # Vite entry point
│   └── styles/              # CSS and styling
├── components/              # React components
├── backend/                 # Python Flask backend
│   ├── app.py              # Main Flask application
│   ├── src/                # ML modules
│   └── requirements.txt    # Python dependencies
├── package.json            # Node.js dependencies
├── vite.config.ts          # Vite configuration
└── README.md               # Project documentation
```

### Available Scripts

```bash
# Frontend development
npm run dev              # Start Vite dev server
npm run build           # Build for production
npm run preview         # Preview production build
npm run type-check      # TypeScript type checking

# Development utilities
npm run start-all       # Start both frontend and backend
npm run lint           # Run ESLint
```

### Adding Features

1. **Frontend**: Add React components in `/components`
2. **Backend**: Add API endpoints in `backend/app.py`
3. **ML Logic**: Implement in `backend/src/` modules
4. **Styling**: Use Tailwind CSS classes

---

## 🎉 Success!

If you've followed this guide successfully, you should now have:

- ✅ A fully functional ML platform running locally
- ✅ Real-time flower classification capabilities
- ✅ Complete dashboard with analytics and monitoring
- ✅ Development environment ready for customization

**Next Steps:**
- Try uploading different flower images
- Explore the model management features
- Check out the system monitoring dashboard
- Review the code to understand how it works

Happy coding! 🌸🤖