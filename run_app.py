
# Main Application Runner for Flower Classification Dashboard
# Run this file to start the web interface

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing flower classification modules
# YOU NEED TO MODIFY THESE IMPORTS BASED ON YOUR ACTUAL FILE STRUCTURE
try:
    # Option 1: If your code is in a separate file (e.g., flower_classification.py)
    from flower_classification import (
        Config, 
        preprocessor, 
        model_instance, 
        predictor, 
        retraining_manager,
        train_flower_model
    )
    print("✅ Successfully imported flower classification modules")
    
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("📝 Please ensure your flower classification code is available.")
    print("   You can either:")
    print("   1. Save your notebook code as 'flower_classification.py'")
    print("   2. Modify the imports above to match your file structure")
    print("   3. Copy-paste your classes directly into this file")
    sys.exit(1)

# Import the Flask backend
from flask_backend import app, socketio

if __name__ == '__main__':
    print("🌸 Starting Flower Classification Dashboard...")
    print("📍 Dashboard will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    
    # Start the application
    if socketio:
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
