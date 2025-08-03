# Flask Backend API for Flower Classification Dashboard
# Connects frontend to your existing flower classification pipeline

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import threading
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

# Import your existing flower classification modules
# Assuming these are available from your notebook/script
try:
    from your_flower_module import (
        Config, 
        preprocessor, 
        model_instance, 
        predictor, 
        retraining_manager,
        train_flower_model
    )
    print("✅ Flower classification modules imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: Could not import flower modules: {e}")
    print("Please ensure your flower classification code is available as a module")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connections

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global training status
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0.0,
    'start_time': None
}

# Statistics tracking
stats = {
    'predictions_made': 0,
    'retrainings_completed': 0,
    'start_time': datetime.now()
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_extension(filename):
    """Get file extension"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

# ================================================================================================
# API Endpoints
# ================================================================================================

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# ================================================================================================
# Model Status and Information
# ================================================================================================

@app.route('/api/status')
def get_status():
    """Get overall system status"""
    try:
        # Check if model is loaded
        model_loaded = predictor.model is not None if 'predictor' in globals() else False
        
        return jsonify({
            'success': True,
            'model_loaded': model_loaded,
            'training_status': training_status,
            'uptime': (datetime.now() - stats['start_time']).total_seconds(),
            'predictions_made': stats['predictions_made'],
            'retrainings_completed': stats['retrainings_completed']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info')
def get_model_info():
    """Get detailed model information"""
    try:
        if 'model_instance' not in globals():
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 404

        info = model_instance.get_model_info()
        
        # Add additional info
        info.update({
            'architecture': 'Enhanced CNN',
            'last_training': 'N/A',  # You can add this from training history
            'classes': Config.CLASS_NAMES if 'Config' in globals() else ['roses', 'tulips', 'sunflowers']
        })

        return jsonify({
            'success': True,
            **info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/dataset-stats')
def get_dataset_stats():
    """Get dataset statistics"""
    try:
        if 'preprocessor' not in globals():
            return jsonify({
                'success': False,
                'error': 'Preprocessor not available'
            }), 404

        # Get training data stats
        train_stats = preprocessor.analyze_dataset(Config.TRAIN_DIR)
        
        stats_data = {
            'roses': train_stats['class_distribution'].get('roses', 0),
            'tulips': train_stats['class_distribution'].get('tulips', 0),
            'sunflowers': train_stats['class_distribution'].get('sunflowers', 0),
            'total': train_stats['total_images']
        }

        return jsonify({
            'success': True,
            **stats_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Image Prediction
# ================================================================================================

@app.route('/api/predict', methods=['POST'])
def predict_image():
    """Predict flower type from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file format'
            }), 400

        # Save file temporarily
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Make prediction
            if 'predictor' not in globals():
                return jsonify({
                    'success': False,
                    'error': 'Predictor not available'
                }), 500

            result = predictor.predict_single_image(filepath)
            
            # Update statistics
            if result.get('success', False):
                stats['predictions_made'] += 1

            return jsonify(result)

        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Bulk Data Upload
# ================================================================================================

@app.route('/api/upload', methods=['POST'])
def upload_bulk_images():
    """Upload multiple images for training"""
    try:
        if 'retraining_manager' not in globals():
            return jsonify({
                'success': False,
                'error': 'Retraining manager not available'
            }), 500

        class_name = request.form.get('class_name')
        if not class_name:
            return jsonify({
                'success': False,
                'error': 'Class name is required'
            }), 400

        if class_name not in Config.CLASS_NAMES:
            return jsonify({
                'success': False,
                'error': f'Invalid class name. Must be one of: {Config.CLASS_NAMES}'
            }), 400

        # Get uploaded files
        files = request.files.getlist('images')
        if not files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400

        # Save files temporarily and collect paths
        temp_files = []
        uploaded_paths = []

        try:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    temp_files.append(filepath)
                    uploaded_paths.append(filepath)

            # Upload to retraining manager
            if uploaded_paths:
                result = retraining_manager.upload_images(uploaded_paths, class_name, source="web")
                
                return jsonify({
                    'success': True,
                    'uploaded_count': result['uploaded_count'],
                    'failed_count': result['failed_count'],
                    'class_name': class_name
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No valid images found'
                }), 400

        finally:
            # Clean up temporary files
            for filepath in temp_files:
                if os.path.exists(filepath):
                    os.remove(filepath)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Model Retraining
# ================================================================================================

@app.route('/api/retraining-status')
def get_retraining_status():
    """Get retraining readiness status"""
    try:
        if 'retraining_manager' not in globals():
            return jsonify({
                'success': False,
                'error': 'Retraining manager not available'
            }), 500

        status = retraining_manager.get_retraining_status()

        return jsonify({
            'success': True,
            **status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/retrain', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    try:
        if 'retraining_manager' not in globals():
            return jsonify({
                'success': False,
                'error': 'Retraining manager not available'
            }), 500

        data = request.get_json() or {}
        epochs = data.get('epochs', 10)

        # Validate epochs
        if not isinstance(epochs, int) or epochs < 5 or epochs > 100:
            return jsonify({
                'success': False,
                'error': 'Epochs must be an integer between 5 and 100'
            }), 400

        # Check if already training
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training already in progress'
            }), 409

        # Start training in background thread
        def training_worker():
            global training_status
            try:
                training_status.update({
                    'is_training': True,
                    'current_epoch': 0,
                    'total_epochs': epochs,
                    'current_loss': 0.0,
                    'start_time': datetime.now()
                })

                # Trigger retraining
                success = retraining_manager.trigger_retraining(epochs=epochs)
                
                if success:
                    stats['retrainings_completed'] += 1

            except Exception as e:
                print(f"Training error: {e}")
            finally:
                training_status['is_training'] = False

        # Start training thread
        training_thread = threading.Thread(target=training_worker)
        training_thread.daemon = True
        training_thread.start()

        return jsonify({
            'success': True,
            'message': f'Retraining started with {epochs} epochs'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/training-status')
def get_training_status():
    """Get current training status"""
    return jsonify({
        'success': True,
        **training_status
    })

# ================================================================================================
# Model Evaluation
# ================================================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate_model():
    """Evaluate model performance"""
    try:
        if 'model_instance' not in globals():
            return jsonify({
                'success': False,
                'error': 'Model not available'
            }), 500

        if 'preprocessor' not in globals():
            return jsonify({
                'success': False,
                'error': 'Preprocessor not available'
            }), 500

        # Check if test data exists
        test_stats = preprocessor.analyze_dataset(Config.TEST_DIR)
        if test_stats['total_images'] < 5:
            return jsonify({
                'success': False,
                'error': 'Insufficient test data for evaluation'
            }), 400

        # Create test generator
        test_gen = preprocessor.create_data_generators(
            Config.TEST_DIR,
            target_size=Config.IMAGE_SIZE,
            batch_size=32,
            augment=False,
            shuffle=False
        )

        # Evaluate model
        evaluation_results = model_instance.evaluate(test_gen)

        return jsonify({
            'success': True,
            **evaluation_results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Utility Endpoints
# ================================================================================================

@app.route('/api/clear-pending', methods=['POST'])
def clear_pending_data():
    """Clear pending upload data"""
    try:
        if 'retraining_manager' not in globals():
            return jsonify({
                'success': False,
                'error': 'Retraining manager not available'
            }), 500

        data = request.get_json() or {}
        class_name = data.get('class_name')

        retraining_manager.clear_pending_data(class_name)

        return jsonify({
            'success': True,
            'message': f'Cleared pending data for {class_name}' if class_name else 'Cleared all pending data'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/download-sample-dataset', methods=['POST'])
def download_sample_dataset():
    """Download sample dataset"""
    try:
        # Import the download function from your existing code
        from your_flower_module import download_sample_data
        
        success = download_sample_data()
        
        return jsonify({
            'success': success,
            'message': 'Sample dataset downloaded successfully!' if success else 'Failed to download dataset'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Health Check and Monitoring
# ================================================================================================

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': (datetime.now() - stats['start_time']).total_seconds()
    })

@app.route('/api/logs')
def get_logs():
    """Get recent application logs"""
    try:
        # Read recent logs (implement based on your logging setup)
        logs = []
        log_file = f"{Config.LOGS_DIR}/pipeline.log" if 'Config' in globals() else "logs/pipeline.log"
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                logs = lines[-100:]  # Last 100 lines

        return jsonify({
            'success': True,
            'logs': logs
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================================================================================
# Error Handlers
# ================================================================================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ================================================================================================
# Enhanced Training Progress Tracking
# ================================================================================================

class TrainingProgressCallback:
    """Custom callback to track training progress"""
    
    def __init__(self):
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        global training_status
        training_status.update({
            'current_epoch': epoch + 1,
            'current_loss': logs.get('loss', 0.0) if logs else 0.0
        })

# ================================================================================================
# WebSocket Support for Real-time Updates (Optional)
# ================================================================================================

try:
    from flask_socketio import SocketIO, emit
    
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        emit('status', {'message': 'Connected to Flower Classification API'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
    
    @socketio.on('request_status')
    def handle_status_request():
        # Send current status to client
        emit('status_update', {
            'training_status': training_status,
            'stats': stats
        })
    
    # Function to broadcast training updates
    def broadcast_training_update():
        if hasattr(app, 'socketio'):
            socketio.emit('training_update', training_status)
    
    print("✅ WebSocket support enabled")
    
except ImportError:
    print("⚠️ WebSocket support not available (flask-socketio not installed)")
    socketio = None

# ================================================================================================
# Main Application Runner
# ================================================================================================

if __name__ == '__main__':
    print("🚀 Starting Flower Classification API Server...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Max file size: {MAX_CONTENT_LENGTH / (1024*1024)}MB")
    
    # Check if required modules are available
    required_modules = ['Config', 'preprocessor', 'model_instance', 'predictor', 'retraining_manager']
    missing_modules = [mod for mod in required_modules if mod not in globals()]
    
    if missing_modules:
        print(f"⚠️ Warning: Missing modules: {missing_modules}")
        print("Please ensure your flower classification pipeline is properly imported")
    else:
        print("✅ All required modules loaded successfully")
    
    # Run the application
    if socketio:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)

# ================================================================================================
# Additional Utility Functions
# ================================================================================================

def initialize_ml_components():
    """Initialize ML components if not already loaded"""
    global Config, preprocessor, model_instance, predictor, retraining_manager
    
    try:
        # Your initialization code here
        # This is where you'd load your trained model and setup components
        print("🔧 Initializing ML components...")
        
        # Example initialization (adjust based on your actual setup)
        # Config = load_config()
        # preprocessor = FlowerDataPreprocessor()
        # model_instance = FlowerClassificationModel()
        # predictor = FlowerPredictor()
        # retraining_manager = FlowerRetrainingManager()
        
        print("✅ ML components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize ML components: {e}")
        return False

def setup_directories():
    """Setup required directories"""
    directories = [
        'static',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"📁 Created directories: {directories}")

# Initialize on import
setup_directories()

# Try to initialize ML components
try:
    initialize_ml_components()
except Exception as e:
    print(f"⚠️ Could not auto-initialize ML components: {e}")
    print("Please call initialize_ml_components() manually after importing your modules")