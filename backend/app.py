from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from datetime import datetime
import json
import numpy as np
from werkzeug.utils import secure_filename
from src.prediction import FlowerPredictor
from src.model import ModelManager
from src.preprocessing import DataPreprocessor
import io
import base64
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize ML components
predictor = FlowerPredictor()
model_manager = ModelManager()
preprocessor = DataPreprocessor()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Mock system metrics (in production, these would come from monitoring tools)
system_metrics = {
    'cpu_usage': 67,
    'memory_usage': 45,
    'network_io': 23,
    'requests_per_sec': 23.4,
    'active_containers': 3,
    'uptime': 99.97,
    'success_rate': 98.6,
    'avg_latency': 145
}

# Mock model versions data
model_versions = [
    {
        'id': '1',
        'version': 'v2.1.3',
        'accuracy': 94.2,
        'precision': 92.8,
        'recall': 95.1,
        'f1_score': 93.9,
        'training_time': '2.3 hours',
        'dataset_size': 2847,
        'model_size': '45.2 MB',
        'epoch_count': 50,
        'status': 'deployed',
        'created_at': '2024-01-03 14:30',
        'deployed_at': '2024-01-03 16:45'
    },
    {
        'id': '2',
        'version': 'v2.1.2',
        'accuracy': 92.1,
        'precision': 90.5,
        'recall': 93.2,
        'f1_score': 91.8,
        'training_time': '2.1 hours',
        'dataset_size': 2672,
        'model_size': '44.8 MB',
        'epoch_count': 45,
        'status': 'archived',
        'created_at': '2024-01-01 09:15',
        'deployed_at': '2024-01-01 11:30'
    }
]

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': 'online',
        'version': 'v2.1.3'
    })

@app.route('/api/dashboard/metrics', methods=['GET'])
def get_dashboard_metrics():
    """Get dashboard metrics and statistics"""
    try:
        # Get real-time metrics from model manager
        model_stats = model_manager.get_current_model_stats()
        
        metrics = {
            'model_status': {
                'status': 'healthy',
                'uptime': system_metrics['uptime'],
                'version': 'v2.1.3',
                'last_trained': '2 hours ago'
            },
            'performance': model_stats,
            'system': {
                'cpu_usage': system_metrics['cpu_usage'],
                'memory_usage': system_metrics['memory_usage'],
                'network_io': system_metrics['network_io'],
                'requests_per_sec': system_metrics['requests_per_sec'],
                'active_containers': system_metrics['active_containers'],
                'success_rate': system_metrics['success_rate'],
                'avg_latency': system_metrics['avg_latency']
            },
            'predictions_24h': {
                'total': 2847,
                'by_class': {
                    'roses': 435,
                    'tulips': 312,
                    'sunflowers': 198
                }
            }
        }
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/single', methods=['POST'])
def predict_single_image():
    """Predict flower type for a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Process the image
        start_time = datetime.now()
        
        # Read image data
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        prediction_result = predictor.predict(image)
        
        end_time = datetime.now()
        response_time = int((end_time - start_time).total_seconds() * 1000)
        
        result = {
            'predicted_class': prediction_result['class'],
            'confidence': prediction_result['confidence'],
            'probabilities': prediction_result['probabilities'],
            'response_time_ms': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed: {prediction_result['class']} ({prediction_result['confidence']:.1f}%)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/bulk', methods=['POST'])
def predict_bulk_images():
    """Handle bulk image upload and processing"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        flower_class = request.form.get('flower_class')
        
        if not flower_class:
            return jsonify({'error': 'Flower class not specified'}), 400
        
        results = {
            'total': len(files),
            'successful': 0,
            'failed': 0,
            'processing_time': 0,
            'uploaded_files': []
        }
        
        start_time = datetime.now()
        
        for file in files:
            if allowed_file(file.filename):
                try:
                    # Save file
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    
                    # Process for training data
                    preprocessor.add_training_image(file_path, flower_class)
                    
                    results['successful'] += 1
                    results['uploaded_files'].append({
                        'filename': filename,
                        'class': flower_class,
                        'status': 'success'
                    })
                except Exception as e:
                    results['failed'] += 1
                    results['uploaded_files'].append({
                        'filename': file.filename,
                        'class': flower_class,
                        'status': 'failed',
                        'error': str(e)
                    })
            else:
                results['failed'] += 1
                results['uploaded_files'].append({
                    'filename': file.filename,
                    'status': 'failed',
                    'error': 'Invalid file type'
                })
        
        end_time = datetime.now()
        results['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Bulk upload completed: {results['successful']} successful, {results['failed']} failed")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in bulk upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Trigger model retraining"""
    try:
        # Get retraining parameters
        data = request.get_json() or {}
        epochs = data.get('epochs', 50)
        learning_rate = data.get('learning_rate', 0.001)
        
        # Start retraining process
        training_id = model_manager.start_retraining(epochs=epochs, learning_rate=learning_rate)
        
        result = {
            'training_id': training_id,
            'status': 'started',
            'message': 'Model retraining has been initiated',
            'estimated_time': '2-3 hours',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model retraining started with ID: {training_id}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting retraining: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain/status/<training_id>', methods=['GET'])
def get_retraining_status(training_id):
    """Get retraining progress status"""
    try:
        status = model_manager.get_training_status(training_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/versions', methods=['GET'])
def get_model_versions():
    """Get all model versions"""
    try:
        versions = model_manager.get_all_versions()
        return jsonify(versions)
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/deploy/<version>', methods=['POST'])
def deploy_model(version):
    """Deploy a specific model version"""
    try:
        result = model_manager.deploy_version(version)
        
        logger.info(f"Model {version} deployment initiated")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error deploying model {version}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/insights', methods=['GET'])
def get_data_insights():
    """Get data analytics and insights"""
    try:
        insights = {
            'class_distribution': [
                {'name': 'Roses', 'count': 1247, 'percentage': 44.2},
                {'name': 'Tulips', 'count': 956, 'percentage': 33.9},
                {'name': 'Sunflowers', 'count': 644, 'percentage': 22.8}
            ],
            'confidence_distribution': [
                {'range': '90-100%', 'count': 1456, 'percentage': 51.6},
                {'range': '80-90%', 'count': 892, 'percentage': 31.6},
                {'range': '70-80%', 'count': 334, 'percentage': 11.8},
                {'range': '60-70%', 'count': 142, 'percentage': 5.0}
            ],
            'upload_trends': [
                {'date': '2024-01-01', 'roses': 45, 'tulips': 32, 'sunflowers': 28},
                {'date': '2024-01-02', 'roses': 52, 'tulips': 38, 'sunflowers': 31},
                {'date': '2024-01-03', 'roses': 48, 'tulips': 41, 'sunflowers': 25},
                {'date': '2024-01-04', 'roses': 61, 'tulips': 35, 'sunflowers': 33},
                {'date': '2024-01-05', 'roses': 55, 'tulips': 42, 'sunflowers': 29},
                {'date': '2024-01-06', 'roses': 58, 'tulips': 45, 'sunflowers': 35},
                {'date': '2024-01-07', 'roses': 63, 'tulips': 39, 'sunflowers': 31}
            ],
            'performance_metrics': {
                'total_predictions': 2847,
                'avg_confidence': 87.3,
                'active_users': 23,
                'daily_uploads': 156
            }
        }
        
        return jsonify(insights)
    except Exception as e:
        logger.error(f"Error getting data insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/logs', methods=['GET'])
def get_system_logs():
    """Get system logs"""
    try:
        # Read logs from file or database
        logs = [
            {
                'id': '1',
                'timestamp': datetime.now().isoformat(),
                'level': 'success',
                'category': 'api',
                'message': 'Prediction request processed successfully',
                'details': 'Image: rose_001.jpg, Confidence: 98.7%, Response time: 145ms'
            },
            {
                'id': '2',
                'timestamp': datetime.now().isoformat(),
                'level': 'info',
                'category': 'system',
                'message': 'Auto-scaling triggered: Container instance added',
                'details': 'New container: ml-worker-04, CPU threshold exceeded (75%)'
            }
        ]
        
        return jsonify(logs)
    except Exception as e:
        logger.error(f"Error getting system logs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monitoring/metrics', methods=['GET'])
def get_system_monitoring():
    """Get real-time system monitoring data"""
    try:
        metrics = {
            'system': {
                'cpu_usage': system_metrics['cpu_usage'],
                'memory_usage': system_metrics['memory_usage'],
                'network_io': system_metrics['network_io'],
                'requests_per_min': int(system_metrics['requests_per_sec'] * 60)
            },
            'performance_timeline': [
                {'time': '00:00', 'cpu': 45, 'memory': 62, 'network': 12, 'requests': 89},
                {'time': '00:15', 'cpu': 52, 'memory': 58, 'network': 18, 'requests': 124},
                {'time': '00:30', 'cpu': 48, 'memory': 61, 'network': 15, 'requests': 98},
                {'time': '00:45', 'cpu': 67, 'memory': 45, 'network': 23, 'requests': 156}
            ],
            'containers': [
                {
                    'name': 'ml-worker-01',
                    'status': 'healthy',
                    'cpu': 52,
                    'memory': 67,
                    'uptime': '23h 45m',
                    'requests': 1247
                },
                {
                    'name': 'ml-worker-02',
                    'status': 'healthy',
                    'cpu': 48,
                    'memory': 61,
                    'uptime': '18h 12m',
                    'requests': 978
                },
                {
                    'name': 'ml-worker-03',
                    'status': 'warning',
                    'cpu': 89,
                    'memory': 76,
                    'uptime': '2h 34m',
                    'requests': 2156
                }
            ]
        }
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)