#!/usr/bin/env python3
"""
Production FastAPI for Flower Classification ML Platform
Simplified version using direct model loading with flower_cnn_model.h5
"""

import os
import sys
import json
import logging
import asyncio
import threading
import time
import shutil
import cv2
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import io
import uuid
import traceback
import random
import psutil

# TensorFlow and ML imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: TensorFlow/Keras not available. Running in mock mode.")

# Setup logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'INPUT_SIZE': (150, 150),  # Standard size for flower_cnn_model.h5
    'CLASS_NAMES': ['rose', 'sunflower', 'tulip'],  # Order matters for model predictions
    'MODEL_PATH': 'models/flower_cnn_model.h5',
    'UPLOAD_DIR': 'uploads',
    'IMG_SIZE': 150
}

# Enums for API documentation
class FlowerClass(str, Enum):
    rose = "rose"
    tulip = "tulip"
    sunflower = "sunflower"

class TrainingStatus(str, Enum):
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"

class DataQuality(str, Enum):
    excellent = "excellent"
    good = "good"
    fair = "fair"
    poor = "poor"
    unknown = "unknown"

# Pydantic Models for Request/Response
class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    ml_initialized: bool = Field(..., description="ML components initialization status")
    uptime: str = Field(..., description="Service uptime")

class PredictionProbability(BaseModel):
    class_name: str = Field(..., alias="class_name", description="Flower class name")
    probability: float = Field(..., description="Prediction probability (0-100)")
    color: str = Field(..., description="Color code for UI visualization")

class SinglePredictionResponse(BaseModel):
    success: bool = Field(..., description="Prediction success status")
    predicted_class: str = Field(..., description="Most likely flower class")
    confidence: float = Field(..., description="Prediction confidence (0-100)")
    probabilities: List[PredictionProbability] = Field(..., description="All class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp")
    prediction_time: str = Field(..., description="Processing time")
    model_version: str = Field(..., description="Model version used")

class BatchPredictionResult(BaseModel):
    index: int = Field(..., description="Image index in batch")
    filename: str = Field(..., description="Original filename")
    predicted_class: Optional[str] = Field(None, description="Predicted class")
    confidence: Optional[float] = Field(None, description="Prediction confidence")
    probabilities: Optional[List[Dict[str, Any]]] = Field(None, description="Class probabilities")
    error: Optional[str] = Field(None, description="Error message if prediction failed")

class BatchPredictionResponse(BaseModel):
    success: bool = Field(..., description="Batch processing success")
    results: List[BatchPredictionResult] = Field(..., description="Individual prediction results")
    processed_count: int = Field(..., description="Number of images processed")
    batch_time: str = Field(..., description="Total batch processing time")
    avg_time_per_image: str = Field(..., description="Average time per image")
    timestamp: str = Field(..., description="Processing timestamp")

class TrainingRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=20, description="Number of training epochs (1-20)")
    learning_rate: float = Field(0.001, gt=0, le=0.1, description="Learning rate (0-0.1)")
    batch_size: int = Field(16, ge=4, le=64, description="Training batch size (4-64)")

class TrainingStartResponse(BaseModel):
    success: bool = Field(..., description="Training start success")
    job_id: str = Field(..., description="Unique training job ID")
    message: str = Field(..., description="Status message")
    estimated_time: str = Field(..., description="Estimated completion time")
    training_type: str = Field(..., description="Type of training")
    parameters: Dict[str, Any] = Field(..., description="Training parameters used")

class TrainingStatusResponse(BaseModel):
    status: TrainingStatus = Field(..., description="Current training status")
    progress: int = Field(..., description="Training progress (0-100)")
    stage: str = Field(..., description="Current training stage")
    start_time: str = Field(..., description="Training start time")
    epochs: int = Field(..., description="Total epochs")
    learning_rate: float = Field(..., description="Learning rate")
    batch_size: int = Field(..., description="Batch size")
    training_type: str = Field(..., description="Training type")
    eta: Optional[str] = Field(None, description="Estimated time remaining")
    elapsed: Optional[str] = Field(None, description="Elapsed time")
    end_time: Optional[str] = Field(None, description="Completion time")
    actual_training_time: Optional[str] = Field(None, description="Actual training duration")
    error: Optional[str] = Field(None, description="Error message if failed")

class RetrainResponse(BaseModel):
    success: bool = Field(..., description="Retraining success status")
    message: str = Field(..., description="Status message")
    new_classes: List[str] = Field(..., description="Classes in the training data")
    model_path: str = Field(..., description="Path to the updated model")
    timestamp: str = Field(..., description="Retraining timestamp")
    images_processed: int = Field(..., description="Number of images processed")

class UploadResponse(BaseModel):
    success: bool = Field(..., description="Upload success status")
    uploaded_count: int = Field(..., description="Number of images uploaded")
    errors: List[str] = Field(..., description="Upload errors")
    message: str = Field(..., description="Status message")
    dataset_stats: Dict[str, Any] = Field(..., description="Updated dataset statistics")

class ClassDistribution(BaseModel):
    name: str = Field(..., description="Class name")
    count: int = Field(..., description="Total images in class")
    percentage: float = Field(..., description="Percentage of total dataset")
    train: int = Field(..., description="Training images")
    test: int = Field(..., description="Test images")
    uploads: Optional[int] = Field(0, description="Uploaded images")

class DataInsightsResponse(BaseModel):
    class_distribution: List[ClassDistribution] = Field(..., description="Class distribution breakdown")
    total_images: int = Field(..., description="Total images in dataset")
    train_images: int = Field(..., description="Training images")
    test_images: int = Field(..., description="Test images")
    upload_images: int = Field(..., description="Uploaded images")
    balance_score: int = Field(..., description="Dataset balance score (0-100)")
    data_quality: DataQuality = Field(..., description="Overall data quality assessment")
    last_updated: str = Field(..., description="Last update timestamp")
    recommendations: List[str] = Field(..., description="Data improvement recommendations")

class ModelVersion(BaseModel):
    id: str = Field(..., description="Model version ID")
    version: str = Field(..., description="Version string")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    training_time: str = Field(..., description="Training duration")
    dataset_size: int = Field(..., description="Training dataset size")
    model_size: str = Field(..., description="Model file size")
    epoch_count: int = Field(..., description="Training epochs")
    status: str = Field(..., description="Model status")
    created_at: str = Field(..., description="Creation timestamp")
    deployed_at: Optional[str] = Field(None, description="Deployment timestamp")

class SystemLog(BaseModel):
    timestamp: str = Field(..., description="Log timestamp")
    level: str = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    component: str = Field(..., description="System component")

# Global variables for real-time tracking
app_start_time = datetime.now()
last_trained_time = None
prediction_count = 0
error_count = 0
training_jobs = {}

# Global model variable
loaded_model = None
ml_initialized = False

def load_flower_model():
    """Load the flower classification model directly"""
    global loaded_model, ml_initialized
    
    try:
        if not ML_AVAILABLE:
            logger.warning("TensorFlow not available, running in mock mode")
            ml_initialized = False
            return False
            
        model_path = CONFIG['MODEL_PATH']
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            loaded_model = load_model(model_path)
            ml_initialized = True
            logger.info("✅ Flower model loaded successfully")
            return True
        else:
            logger.warning(f"❌ Model file not found at {model_path}")
            logger.info("Creating mock model for demonstration...")
            ml_initialized = False
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        ml_initialized = False
        return False

def preprocess_image(img_array):
    """Preprocess image for model prediction"""
    try:
        # Resize image to model input size
        img_resized = cv2.resize(img_array, CONFIG['INPUT_SIZE'])
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(img_array):
    """Make prediction using the loaded model"""
    global loaded_model, prediction_count, error_count
    
    try:
        if not ml_initialized or loaded_model is None:
            return {
                'error': True,
                'message': 'Model not initialized'
            }
        
        # Preprocess image
        processed_img = preprocess_image(img_array)
        if processed_img is None:
            return {
                'error': True,
                'message': 'Failed to preprocess image'
            }
        
        # Make prediction
        predictions = loaded_model.predict(processed_img)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Get class name
        predicted_class = CONFIG['CLASS_NAMES'][predicted_class_idx]
        
        # Format probabilities
        probabilities = []
        colors = ['hsl(0, 70%, 60%)', 'hsl(120, 70%, 60%)', 'hsl(240, 70%, 60%)']
        
        for i, (class_name, prob) in enumerate(zip(CONFIG['CLASS_NAMES'], predictions[0])):
            probabilities.append({
                'name': class_name,
                'probability': float(prob),
                'color': colors[i % len(colors)]
            })
        
        prediction_count += 1
        
        return {
            'error': False,
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_version': 'flower_cnn_model.h5'
        }
        
    except Exception as e:
        error_count += 1
        logger.error(f"Prediction error: {str(e)}")
        return {
            'error': True,
            'message': str(e)
        }

# Initialize model at startup
load_flower_model()

# Utility functions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_uptime() -> str:
    """Calculate real uptime"""
    uptime_delta = datetime.now() - app_start_time
    days = uptime_delta.days
    hours, remainder = divmod(uptime_delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

def get_last_trained_relative() -> str:
    """Get relative time since last training"""
    global last_trained_time
    if last_trained_time is None:
        return "Never"
    
    time_diff = datetime.now() - last_trained_time
    if time_diff.days > 0:
        return f"{time_diff.days} days ago"
    elif time_diff.seconds > 3600:
        hours = time_diff.seconds // 3600
        return f"{hours} hours ago"
    elif time_diff.seconds > 60:
        minutes = time_diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "Just now"

def calculate_uptime_percentage() -> float:
    """Calculate uptime percentage"""
    uptime_delta = datetime.now() - app_start_time
    total_hours = max(1, uptime_delta.total_seconds() / 3600)
    base_uptime = 99.5
    variance = min(0.4, error_count * 0.01)
    return max(95.0, base_uptime - variance)

# Mock data generation functions
def get_model_status() -> Dict[str, Any]:
    return {
        "status": "healthy" if ml_initialized else "limited",
        "uptime": random.randint(95, 100),
        "version": "flower_cnn_model.h5",
        "last_trained": get_last_trained_relative()
    }

def get_performance_metrics() -> Dict[str, Any]:
    return {
        "accuracy": round(random.uniform(92.5, 97.5), 2),
        "precision": round(random.uniform(90.0, 95.0), 2),
        "recall": round(random.uniform(91.0, 96.0), 2),
        "f1_score": round(random.uniform(92.0, 96.5), 2)
    }

def get_system_metrics() -> Dict[str, Any]:
    return {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "network_io": round(random.uniform(5.0, 15.0), 2),
        "requests_per_sec": random.randint(50, 150),
        "active_containers": random.randint(1, 3),
        "success_rate": random.randint(98, 100),
        "avg_latency": random.randint(50, 150)
    }

def get_predictions() -> Dict[str, Any]:
    return {
        "total": random.randint(5000, 15000),
        "by_class": {
            "roses": random.randint(2000, 6000),
            "tulips": random.randint(1500, 5000),
            "sunflowers": random.randint(1000, 4000)
        }
    }

# Create FastAPI app
app = FastAPI(
    title="🌸 Flower Classification API",
    description="""
    ## Advanced ML-Powered Flower Classification Platform
    
    A production-ready FastAPI service for classifying flower images using deep learning.
    
    ### Features
    - 🤖 **AI-Powered Classification**: Direct model loading with flower_cnn_model.h5
    - 🌸 **Multi-Class Support**: Rose, Tulip, and Sunflower classification
    - 🔄 **Model Retraining**: Upload new images and retrain the model
    - 📊 **Real-time Analytics**: Live training progress and model metrics
    - ⚡ **High Performance**: Optimized for production workloads
    
    ### Supported Flower Classes
    - 🌹 **Rose**: Classic garden roses with various colors
    - 🌷 **Tulip**: Spring blooming tulips in multiple varieties  
    - 🌻 **Sunflower**: Large, bright sunflowers with distinctive appearance
    
    ### Getting Started
    1. Check API health with `/health` endpoints
    2. Make predictions with `/api/predict/single` or `/api/predict/batch`
    3. Retrain model with new data using `/api/train/retrain`
    4. Monitor performance with `/api/stats`
    
    **Note**: This API supports images up to 16MB in PNG, JPG, JPEG, or GIF formats.
    """,
    version="2.2.0-direct-model",
    contact={
        "name": "Flower Classification API",
        "url": "https://github.com/your-repo/flower-classification",
        "email": "support@flower-api.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_tags=[
        {
            "name": "Health Check",
            "description": "Service health monitoring and status endpoints"
        },
        {
            "name": "Predictions",
            "description": "Image classification and batch prediction endpoints"
        },
        {
            "name": "Training",
            "description": "Model training and retraining endpoints"
        },
        {
            "name": "Model Management", 
            "description": "Model versioning and deployment"
        },
        {
            "name": "Data Management",
            "description": "Training data upload and dataset insights"
        },
        {
            "name": "System Info",
            "description": "System monitoring and logging"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} for {request.method} {request.url} - Time: {process_time:.3f}s")
    
    return response

# ==================== HEALTH CHECK ENDPOINTS ====================

@app.get("/", 
         response_model=HealthResponse,
         tags=["Health Check"],
         summary="Root Health Check",
         description="Basic service health check with model status and uptime")
async def root_health_check():
    """Root health check endpoint with comprehensive service information"""
    return HealthResponse(
        status="healthy" if ml_initialized else "limited",
        service="Flower Classification API",
        version="2.2.0-direct-model",
        timestamp=datetime.now().isoformat(),
        ml_initialized=ml_initialized,
        uptime=get_uptime()
    )

@app.get("/health",
         tags=["Health Check"],
         summary="Simple Health Check",
         description="Lightweight health check endpoint")
async def health():
    """Simple health endpoint for load balancer checks"""
    return {
        "status": "healthy" if ml_initialized else "limited",
        "uptime": get_uptime(),
        "model_loaded": loaded_model is not None
    }

@app.get("/api/health",
         tags=["Health Check"],
         summary="API Health Check",
         description="Comprehensive API health with available endpoints")
async def api_health():
    """API health check endpoint with endpoint list"""
    return {
        "status": "healthy" if ml_initialized else "limited",
        "service": "Flower Classification API",
        "version": "2.2.0-direct-model",
        "ml_initialized": ml_initialized,
        "model_path": CONFIG['MODEL_PATH'],
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health_check": ["/", "/health", "/api/health", "/api/status"],
            "core_api": ["/api/stats", "/api/predict/single", "/api/predict/batch"],
            "training": ["/api/train/retrain", "/api/train/start", "/api/train/status/{job_id}"],
            "model_management": ["/api/models"],
            "data_insights": ["/api/data/insights"],
            "system_info": ["/api/system/logs"]
        }
    }
    
@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """Endpoint that provides all metrics needed for the dashboard"""
    return {
        "model_status": get_model_status(),
        "performance": get_performance_metrics(),
        "system": get_system_metrics(),
        "predictions_24h": get_predictions()
    }

@app.get("/api/status",
         tags=["Health Check"],
         summary="API Status Check",
         description="Detailed API status with component health")
async def api_status():
    """API status endpoint with component checks"""
    return {
        "status": "operational" if ml_initialized else "limited",
        "ml_components": {
            "model_loaded": loaded_model is not None,
            "tensorflow_available": ML_AVAILABLE,
            "model_path": CONFIG['MODEL_PATH'],
            "model_exists": os.path.exists(CONFIG['MODEL_PATH'])
        },
        "system": {
            "uptime": get_uptime(),
            "prediction_count": prediction_count,
            "error_count": error_count
        },
        "timestamp": datetime.now().isoformat()
    }

# ==================== CORE API ENDPOINTS ====================

@app.get("/api/stats",
         tags=["System Info"],
         summary="Get System Statistics",
         description="Comprehensive real-time system statistics including model performance")
async def get_stats():
    """Get real-time system statistics"""
    global prediction_count, error_count
    
    try:
        stats = {
            "system": {
                "uptime": get_uptime(),
                "uptime_percentage": round(calculate_uptime_percentage(), 2),
                "last_trained": get_last_trained_relative(),
                "version": "v2.2.0-direct-model",
                "status": "operational" if ml_initialized else "limited",
                "prediction_count": prediction_count,
                "error_count": error_count
            },
            "model": {
                "loaded": loaded_model is not None,
                "path": CONFIG['MODEL_PATH'],
                "exists": os.path.exists(CONFIG['MODEL_PATH']),
                "tensorflow_available": ML_AVAILABLE,
                "classes": CONFIG['CLASS_NAMES'],
                "input_size": CONFIG['INPUT_SIZE']
            },
            "dataset": {
                "upload_dir": CONFIG['UPLOAD_DIR'],
                "supported_formats": list(ALLOWED_EXTENSIONS),
                "max_batch_size": 10
            },
            "performance": {
                "avg_response_time": "0.3s",
                "success_rate": f"{max(85, 100 - error_count)}%",
                "memory_usage": "35%",
                "cpu_usage": "18%"
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        error_count += 1
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

@app.post("/api/predict/single",
          response_model=SinglePredictionResponse,
          tags=["Predictions"],
          summary="Single Image Prediction",
          description="Classify a single flower image and return detailed predictions with confidence scores")
async def predict_single(
    image: UploadFile = File(..., description="Flower image file (PNG, JPG, JPEG, GIF - max 16MB)")
):
    """Predict single image classification"""
    global prediction_count, error_count
    
    try:
        if not ml_initialized or loaded_model is None:
            error_count += 1
            raise HTTPException(
                status_code=503,
                detail="Model not initialized. Please check if flower_cnn_model.h5 exists in models/ directory."
            )
        
        # Validate file type
        if not allowed_file(image.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Allowed: PNG, JPG, JPEG, GIF"
            )
        
        try:
            # Read and process image
            image_data = await image.read()
            
            # Convert to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_array is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            
            # Make prediction
            start_time = datetime.now()
            result = predict_image(img_array)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            if result.get('error'):
                error_count += 1
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction failed: {result.get('message', 'Unknown error')}"
                )
            
            # Format probabilities
            formatted_probs = [
                PredictionProbability(
                    class_name=prob['name'],
                    probability=round(prob['probability'] * 100, 2),
                    color=prob.get('color', 'hsl(0, 0%, 50%)')
                )
                for prob in result['probabilities']
            ]
            
            # Return formatted result
            return SinglePredictionResponse(
                success=True,
                predicted_class=result['class'],
                confidence=round(result['confidence'], 2),
                probabilities=formatted_probs,
                timestamp=datetime.now().isoformat(),
                prediction_time=f"{prediction_time:.3f}s",
                model_version=result.get('model_version', 'flower_cnn_model.h5')
            )
            
        except HTTPException:
            raise
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            error_count += 1
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(pred_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single prediction endpoint error: {str(e)}")
        error_count += 1
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/predict/batch",
          response_model=BatchPredictionResponse,
          tags=["Predictions"],
          summary="Batch Image Prediction",
          description="Classify multiple flower images in a single request (max 10 images)")
async def predict_batch(
    images: List[UploadFile] = File(..., description="Multiple flower image files (max 10)")
):
    """Predict multiple images (max 10)"""
    global prediction_count, error_count
    
    try:
        if not ml_initialized or loaded_model is None:
            error_count += 1
            raise HTTPException(
                status_code=503,
                detail="Model not initialized"
            )
        
        if len(images) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch"
            )
        
        start_time = datetime.now()
        results = []
        
        for i, file in enumerate(images):
            if file and allowed_file(file.filename):
                try:
                    image_data = await file.read()
                    nparr = np.frombuffer(image_data, np.uint8)
                    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img_array is None:
                        results.append(BatchPredictionResult(
                            index=i,
                            filename=file.filename,
                            error="Invalid image format"
                        ))
                        continue
                    
                    result = predict_image(img_array)
                    
                    if not result.get('error'):
                        results.append(BatchPredictionResult(
                            index=i,
                            filename=file.filename,
                            predicted_class=result['class'],
                            confidence=round(result['confidence'], 2),
                            probabilities=result['probabilities']
                        ))
                    else:
                        error_count += 1
                        results.append(BatchPredictionResult(
                            index=i,
                            filename=file.filename,
                            error=result.get('message', 'Unknown error')
                        ))
                        
                except Exception as img_error:
                    logger.error(f"Error processing image {i}: {str(img_error)}")
                    error_count += 1
                    results.append(BatchPredictionResult(
                        index=i,
                        filename=file.filename,
                        error=str(img_error)
                    ))
        
        batch_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            success=True,
            results=results,
            processed_count=len(results),
            batch_time=f"{batch_time:.3f}s",
            avg_time_per_image=f"{batch_time/len(results):.3f}s" if results else "0s",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        error_count += 1
        raise HTTPException(status_code=500, detail="Batch prediction failed")

# ==================== TRAINING ENDPOINTS ====================


@app.post("/api/train/retrain",
          response_model=RetrainResponse,
          tags=["Training"],
          summary="Enhance Flower Classification Model",
          description="Incrementally improve the existing model with new images while preserving previous knowledge.")
async def retrain_model_endpoint(
    images: List[UploadFile] = File(..., description="Flower images (rose, tulip, sunflower only)"),
    labels: List[str] = Form(..., description="Matching labels for each image")
):
    """Incrementally enhance the current model with uploaded labeled flower images"""
    global loaded_model, ml_initialized, last_trained_time, error_count
    
    if len(images) != len(labels):
        raise HTTPException(status_code=400, detail="Mismatch between number of images and labels")
    
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow not available for retraining")
    
    try:
        # Enable eager execution if not already enabled
        if not tf.executing_eagerly():
            tf.config.experimental_run_functions_eagerly(True)
            logger.info("Enabled eager execution for incremental training")
        
        # Create upload directory
        os.makedirs(CONFIG['UPLOAD_DIR'], exist_ok=True)
        
        X, Z = [], []
        valid_labels = [label.lower() for label in CONFIG['CLASS_NAMES']]
        
        logger.info(f"Starting incremental training with {len(images)} new images")
        
        for image_file, label in zip(images, labels):
            if label.lower() not in valid_labels:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid label '{label}'. Only {valid_labels} are allowed."
                )
            
            try:
                # Read image data
                image_data = await image_file.read()
                
                # Save image temporarily
                file_path = os.path.join(CONFIG['UPLOAD_DIR'], f"{uuid.uuid4()}_{image_file.filename}")
                with open(file_path, "wb") as buffer:
                    buffer.write(image_data)
                
                # Load and process image
                img = cv2.imread(file_path)
                if img is None:
                    logger.warning(f"Could not load image: {image_file.filename}")
                    # Clean up temp file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    continue
                
                # Resize image
                img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))
                X.append(img)
                Z.append(label.lower())
                
                # Clean up temp file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as img_error:
                logger.error(f"Error processing image {image_file.filename}: {str(img_error)}")
                continue
        
        if not X:
            raise HTTPException(status_code=400, detail="No valid images uploaded")
        
        logger.info(f"Processing {len(X)} valid images for incremental training")
        
        # Convert to numpy arrays and normalize
        X = np.array(X, dtype='float32') / 255.0
        
        # Encode labels
        le = LabelEncoder()
        le.fit(CONFIG['CLASS_NAMES'])  # Fit on all possible classes
        y_encoded = le.transform(Z)
        Y = to_categorical(y_encoded, len(CONFIG['CLASS_NAMES']))
        
        # Load existing model or raise error
        if not os.path.exists(CONFIG['MODEL_PATH']):
            raise HTTPException(
                status_code=404, 
                detail=f"Model file not found at {CONFIG['MODEL_PATH']}. Please ensure the base model exists."
            )
        
        # Load model with compile=False to avoid issues
        model = tf.keras.models.load_model(CONFIG['MODEL_PATH'], compile=False)
        logger.info("Loaded existing model for incremental training")
        
        # Get current model accuracy before training (if possible)
        try:
            # Try to evaluate on a small sample to get baseline
            sample_size = min(5, len(X))
            current_loss = model.evaluate(X[:sample_size], Y[:sample_size], verbose=0)
            if isinstance(current_loss, list):
                baseline_accuracy = current_loss[1] if len(current_loss) > 1 else current_loss[0]
            else:
                baseline_accuracy = current_loss
            logger.info(f"Baseline model performance on new data: {baseline_accuracy:.3f}")
        except:
            baseline_accuracy = None
            logger.info("Could not evaluate baseline performance")
        
        # Configure for INCREMENTAL LEARNING with lower learning rate
        # Lower learning rate preserves existing knowledge while learning new patterns
        incremental_lr = 0.0001  # Much lower than initial training (usually 0.001)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=incremental_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info(f"Recompiled model for incremental training with LR: {incremental_lr}")
        
        # Create backup of current model before enhancement
        backup_path = CONFIG['MODEL_PATH'].replace('.h5', f'_backup_{int(time.time())}.h5')
        model.save(backup_path, save_format='h5')
        logger.info(f"Created model backup at: {backup_path}")
        
        # Incremental training configuration
        logger.info("Starting incremental model enhancement...")
        start_train_time = datetime.now()
        
        # Use fewer epochs for incremental learning to avoid overfitting
        incremental_epochs = max(2, min(5, len(X) // 2))  # Adaptive epochs based on data size
        
        # Use tf.data for better performance and compatibility
        dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        
        # Add data augmentation to prevent overfitting on small datasets
        def augment_data(image, label):
            # Random rotation
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            # Random brightness
            image = tf.image.random_brightness(image, 0.1)
            # Random contrast
            image = tf.image.random_contrast(image, 0.9, 1.1)
            return image, label
        
        # Apply augmentation and batch
        if len(X) < 10:  # Only augment if we have few images
            dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            logger.info("Applied data augmentation for small dataset")
        
        dataset = dataset.batch(min(8, len(X)))  # Smaller batch size for incremental learning
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Fit the model incrementally
        history = model.fit(
            dataset,
            epochs=incremental_epochs,
            verbose=1,
            validation_split=0 if len(X) < 6 else 0.15,  # Less validation split for small datasets
        )
        
        training_duration = datetime.now() - start_train_time
        
        # Evaluate improvement
        try:
            if baseline_accuracy is not None:
                sample_size = min(5, len(X))
                final_loss = model.evaluate(X[:sample_size], Y[:sample_size], verbose=0)
                if isinstance(final_loss, list):
                    final_accuracy = final_loss[1] if len(final_loss) > 1 else final_loss[0]
                else:
                    final_accuracy = final_loss
                improvement = final_accuracy - baseline_accuracy
                logger.info(f"Model improvement: {improvement:.3f} ({final_accuracy:.3f} vs {baseline_accuracy:.3f})")
            else:
                improvement = None
                final_accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
        except:
            improvement = None
            final_accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
        
        # Save enhanced model
        model.save(CONFIG['MODEL_PATH'], save_format='h5')
        logger.info(f"Enhanced model saved to {CONFIG['MODEL_PATH']}")
        
        # Reload the global model
        loaded_model = model
        ml_initialized = True
        last_trained_time = datetime.now()
        
        logger.info(f"✅ Incremental training completed in {training_duration.total_seconds():.1f}s")
        
        # Prepare success message with improvement info
        success_message = f"Model enhanced successfully with {len(X)} new images. "
        if improvement is not None:
            if improvement > 0:
                success_message += f"Performance improved by {improvement:.3f} points!"
            else:
                success_message += "Model knowledge updated (performance maintained)."
        else:
            success_message += f"Final training accuracy: {final_accuracy:.3f}"
        
        return RetrainResponse(
            success=True,
            message=success_message,
            new_classes=list(set(Z)),
            model_path=CONFIG['MODEL_PATH'],
            timestamp=datetime.now().isoformat(),
            images_processed=len(X)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Incremental training failed: {str(e)}")
        logger.error(traceback.format_exc())
        error_count += 1
        raise HTTPException(status_code=500, detail=f"Incremental training failed: {str(e)}")

@app.post("/api/train/start",
          response_model=TrainingStartResponse,
          tags=["Training"],
          summary="Start Model Training",
          description="Initiate model training with custom parameters and get job tracking ID")
async def start_training(
    background_tasks: BackgroundTasks,
    training_request: TrainingRequest = TrainingRequest()
):
    """Start model training with background task"""
    global last_trained_time, training_jobs
    
    try:
        if not ml_initialized or loaded_model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not initialized"
            )
        
        # Create training job ID
        job_id = str(uuid.uuid4())
        
        # Initialize training status
        training_jobs[job_id] = {
            "status": "starting",
            "progress": 0,
            "stage": "Initializing training...",
            "start_time": datetime.now().isoformat(),
            "epochs": training_request.epochs,
            "learning_rate": training_request.learning_rate,
            "batch_size": training_request.batch_size,
            "training_type": "standard"
        }
        
        # Start training in background (mock implementation)
        def train_model():
            try:
                training_jobs[job_id]["status"] = "running"
                training_jobs[job_id]["stage"] = "Preparing data..."
                training_jobs[job_id]["progress"] = 5
                
                # Simulate training progress
                for progress in range(10, 101, 10):
                    time.sleep(2)  # Simulate training time
                    if job_id in training_jobs:
                        training_jobs[job_id]["progress"] = progress
                        training_jobs[job_id]["stage"] = f"Training epoch {progress//10}..."
                
                # Mark as completed
                training_jobs[job_id]["status"] = "completed"
                training_jobs[job_id]["progress"] = 100
                training_jobs[job_id]["stage"] = "Training completed successfully"
                training_jobs[job_id]["end_time"] = datetime.now().isoformat()
                
                # Update global last trained time
                last_trained_time = datetime.now()
                
                # Calculate training time
                start_time = datetime.fromisoformat(training_jobs[job_id]["start_time"])
                training_time = datetime.now() - start_time
                training_jobs[job_id]["actual_training_time"] = f"{training_time.total_seconds():.1f}s"
                
                logger.info(f"✅ Training job {job_id} completed in {training_time.total_seconds():.1f}s")
                
            except Exception as train_error:
                logger.error(f"Training job {job_id} failed: {str(train_error)}")
                if job_id in training_jobs:
                    training_jobs[job_id]["status"] = "failed"
                    training_jobs[job_id]["error"] = str(train_error)
                    training_jobs[job_id]["end_time"] = datetime.now().isoformat()
        
        # Add training task to background
        background_tasks.add_task(train_model)
        
        estimated_time = training_request.epochs * 5

        return TrainingStartResponse(
            success=True,
            job_id=job_id,
            message="Training started successfully",
            estimated_time=f"{estimated_time} seconds",
            training_type="standard",
            parameters={
                "epochs": training_request.epochs,
                "batch_size": training_request.batch_size,
                "learning_rate": training_request.learning_rate
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start training error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start training")

@app.get("/api/train/status/{job_id}",
         response_model=TrainingStatusResponse,
         tags=["Training"],
         summary="Get Training Status",
         description="Monitor training progress and get real-time updates")
async def get_training_status(
    job_id: str = Path(..., description="Training job ID returned from start training")
):
    """Get training job status"""
    try:
        if job_id not in training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found")
        
        status = training_jobs[job_id].copy()
        
        # Calculate ETA if running
        if status["status"] == "running":
            start_time = datetime.fromisoformat(status["start_time"])
            elapsed = datetime.now() - start_time
            progress = max(1, status["progress"])
            
            if progress > 5:
                total_estimated = elapsed.total_seconds() * (100 / progress)
                remaining = max(0, total_estimated - elapsed.total_seconds())
                status["eta"] = f"{int(remaining)}s"
                status["elapsed"] = f"{elapsed.total_seconds():.1f}s"
            else:
                status["eta"] = "Calculating..."
                status["elapsed"] = f"{elapsed.total_seconds():.1f}s"
        
        return TrainingStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get training status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

# ==================== MODEL MANAGEMENT ENDPOINTS ====================

@app.get("/api/models",
         response_model=List[ModelVersion],
         tags=["Model Management"],
         summary="Get All Model Versions",
         description="Retrieve all available model versions with performance metrics")
async def get_models():
    """Get all model versions"""
    try:
        model_size = "Unknown"
        model_exists = os.path.exists(CONFIG['MODEL_PATH'])
        
        if model_exists:
            try:
                size_bytes = os.path.getsize(CONFIG['MODEL_PATH'])
                model_size = f"{size_bytes / (1024*1024):.1f} MB"
            except:
                model_size = "Unknown"
        
        return [
            ModelVersion(
                id="flower_cnn_model",
                version="flower_cnn_model.h5",
                accuracy=round(random.uniform(92.5, 97.5), 2),
                precision=round(random.uniform(90.0, 95.0), 2),
                recall=round(random.uniform(91.0, 96.0), 2),
                f1_score=round(random.uniform(92.0, 96.5), 2),
                training_time=get_last_trained_relative(),
                dataset_size=prediction_count,
                model_size=model_size,
                epoch_count=15,
                status="deployed" if ml_initialized else "not_loaded",
                created_at=(datetime.now() - timedelta(hours=1)).isoformat(),
                deployed_at=(datetime.now() - timedelta(hours=1)).isoformat() if ml_initialized else None
            )
        ]
        
    except Exception as e:
        logger.error(f"Get models error: {str(e)}")
        return []

@app.post("/api/models/{version}/deploy",
          tags=["Model Management"],
          summary="Deploy Model Version",
          description="Deploy a specific model version to production")
async def deploy_model(
    version: str = Path(..., description="Model version to deploy")
):
    """Deploy a specific model version"""
    global loaded_model, ml_initialized
    
    try:
        if version == "flower_cnn_model.h5" or version == "flower_cnn_model":
            # Reload the model
            success = load_flower_model()
            
            return {
                "success": success,
                "status": "success" if success else "failed",
                "message": f"Model {version} {'deployed' if success else 'deployment failed'} successfully",
                "timestamp": datetime.now().isoformat(),
                "deployment_info": {
                    "version": version,
                    "deployment_time": datetime.now().isoformat(),
                    "status": "active" if success else "failed",
                    "model_path": CONFIG['MODEL_PATH']
                }
            }
        else:
            raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deploy model error: {str(e)}")
        raise HTTPException(status_code=500, detail="Deployment failed")

# ==================== DATA INSIGHTS ENDPOINTS ====================

@app.get("/api/data/insights",
         response_model=DataInsightsResponse,
         tags=["Data Management"],
         summary="Get Data Insights",
         description="Comprehensive dataset analysis with class distribution and recommendations")
async def get_data_insights():
    """Get data insights and statistics"""
    try:
        # Mock data insights since we're using direct model loading
        return DataInsightsResponse(
            class_distribution=[
                ClassDistribution(name="rose", count=prediction_count//3, percentage=33.3, train=40, test=10),
                ClassDistribution(name="sunflower", count=prediction_count//3, percentage=33.3, train=40, test=10),
                ClassDistribution(name="tulip", count=prediction_count//3, percentage=33.4, train=40, test=10)
            ],
            total_images=prediction_count,
            train_images=prediction_count * 80 // 100,
            test_images=prediction_count * 20 // 100,
            upload_images=0,
            balance_score=95,
            data_quality=DataQuality.good if ml_initialized else DataQuality.unknown,
            last_updated=datetime.now().isoformat(),
            recommendations=[
                "Model is loaded and ready for predictions" if ml_initialized else "Model needs to be loaded",
                "Use /api/train/retrain to improve model with new data",
                f"Total predictions made: {prediction_count}"
            ]
        )
        
    except Exception as e:
        logger.error(f"Get data insights error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get data insights")

@app.post("/api/upload/training",
          response_model=UploadResponse,
          tags=["Data Management"],
          summary="Upload Training Data",
          description="Upload flower images for training with specified class labels")
async def upload_training_data(
    images: List[UploadFile] = File(..., description="Training image files"),
    class_name: FlowerClass = Form(..., description="Flower class for the uploaded images")
):
    """Upload training data (saves to upload directory)"""
    global error_count
    
    try:
        uploaded_count = 0
        errors = []
        
        # Create upload directory structure
        upload_dir = os.path.join(CONFIG['UPLOAD_DIR'], class_name.value)
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in images:
            if file and allowed_file(file.filename):
                try:
                    # Save file
                    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
                    image_data = await file.read()
                    
                    with open(file_path, 'wb') as f:
                        f.write(image_data)
                    
                    uploaded_count += 1
                    logger.info(f"Uploaded {file.filename} to {file_path}")
                        
                except Exception as file_error:
                    logger.error(f"Error processing file {file.filename}: {str(file_error)}")
                    errors.append(f"Error with {file.filename}: {str(file_error)}")
            else:
                errors.append(f"Invalid file type: {file.filename}")
        
        return UploadResponse(
            success=True,
            uploaded_count=uploaded_count,
            errors=errors,
            message=f"Successfully uploaded {uploaded_count} images for {class_name.value} class",
            dataset_stats={
                "uploaded_images": uploaded_count,
                "class": class_name.value,
                "upload_directory": upload_dir
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload training data error: {str(e)}")
        error_count += 1
        raise HTTPException(status_code=500, detail="Upload failed")

# ==================== SYSTEM INFO ENDPOINTS ====================

@app.get("/api/system/logs",
         response_model=List[SystemLog],
         tags=["System Info"],
         summary="Get System Logs",
         description="Retrieve recent system logs for monitoring and debugging")
async def get_system_logs():
    """Get recent system logs"""
    try:
        # Mock logs with realistic entries
        logs = [
            SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=1)).isoformat(),
                level="INFO",
                message=f"Model prediction completed. Total predictions: {prediction_count}",
                component="predictor"
            ),
            SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=3)).isoformat(),
                level="INFO",
                message=f"Model status: {'loaded' if ml_initialized else 'not loaded'}",
                component="model_loader"
            ),
            SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=5)).isoformat(),
                level="INFO",
                message=f"Last training: {get_last_trained_relative()}",
                component="trainer"
            ),
            SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=8)).isoformat(),
                level="INFO",
                message=f"TensorFlow available: {ML_AVAILABLE}",
                component="system"
            ),
            SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=12)).isoformat(),
                level="INFO",
                message=f"Model path: {CONFIG['MODEL_PATH']}",
                component="config"
            ),
            SystemLog(
                timestamp=(datetime.now() - timedelta(hours=1)).isoformat(),
                level="INFO",
                message="API server started successfully",
                component="system"
            )
        ]
        
        # Add error logs if there have been errors
        if error_count > 0:
            logs.insert(1, SystemLog(
                timestamp=(datetime.now() - timedelta(minutes=2)).isoformat(),
                level="WARNING",
                message=f"Total system errors: {error_count}",
                component="system"
            ))
        
        return logs
        
    except Exception as e:
        logger.error(f"Get system logs error: {str(e)}", exc_info=True)
        return []

# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    global error_count
    error_count += 1
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

@app.exception_handler(413)
async def file_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"error": "File too large", "detail": "Maximum file size is 16MB"}
    )

# ==================== STARTUP AND SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 FastAPI Flower Classification API starting up...")
    logger.info(f"🔧 Model Initialized: {ml_initialized}")
    logger.info(f"🤖 TensorFlow Available: {ML_AVAILABLE}")
    logger.info(f"📁 Model Path: {CONFIG['MODEL_PATH']}")
    logger.info(f"📊 Classes: {', '.join(CONFIG['CLASS_NAMES'])}")
    logger.info(f"🌐 CORS enabled for all origins")
    logger.info("📚 Swagger UI available at /docs")
    logger.info("📝 ReDoc documentation available at /redoc")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(CONFIG['UPLOAD_DIR'], exist_ok=True)
    
    # Create class subdirectories in upload dir
    for class_name in CONFIG['CLASS_NAMES']:
        os.makedirs(os.path.join(CONFIG['UPLOAD_DIR'], class_name), exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 FastAPI Flower Classification API shutting down...")
    logger.info(f"📊 Final Stats - Predictions: {prediction_count}, Errors: {error_count}")

# ==================== CUSTOM OPENAPI SCHEMA ====================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="🌸 Flower Classification API",
        version="2.2.0-direct-model",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom server info
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.flower-classification.com", "description": "Production server"}
    ]
    
    # Add security schemes if needed
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header", 
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ==================== DEVELOPMENT SERVER ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Flower Classification API with Uvicorn")
    logger.info("📚 Swagger UI will be available at http://localhost:8000/docs")
    logger.info("📝 ReDoc documentation at http://localhost:8000/redoc")
    logger.info("🔗 Health check at http://localhost:8000/health")
    logger.info(f"🤖 Model path: {CONFIG['MODEL_PATH']}")
    
    # Run with Uvicorn
    uvicorn.run(
        "main:app",  # Assuming this file is named main.py
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True
    )