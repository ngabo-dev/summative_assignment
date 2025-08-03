#!/usr/bin/env python3
"""
Setup script for Flower Classification ML System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'data/train/rose',
        'data/train/tulip', 
        'data/train/sunflower',
        'data/test/rose',
        'data/test/tulip',
        'data/test/sunflower',
        'data/uploads',
        'models',
        'logs',
        'tests',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def setup_environment():
    """Setup environment variables"""
    env_content = """# Flower Classification ML System Environment Variables
SECRET_KEY=your-secret-key-here-change-in-production
FLASK_ENV=development
FLASK_APP=app.py
UPLOAD_FOLDER=data/uploads
MODEL_DIR=models
DATA_DIR=data
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("Created .env file with default configuration")
    else:
        print(".env file already exists")

def download_sample_data():
    """Download sample flower images for testing"""
    print("Sample data download would be implemented here")
    print("For now, please manually add some flower images to:")
    print("  - data/train/rose/")
    print("  - data/train/tulip/") 
    print("  - data/train/sunflower/")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
logs/
data/uploads/*
models/*.tf
models/*.pkl
!models/.gitkeep
!data/uploads/.gitkeep

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
"""
    
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("Created .gitignore file")

def create_placeholder_files():
    """Create placeholder files for git"""
    placeholders = [
        'data/uploads/.gitkeep',
        'models/.gitkeep',
        'logs/.gitkeep'
    ]
    
    for placeholder in placeholders:
        Path(placeholder).touch()

def main():
    parser = argparse.ArgumentParser(description='Setup Flower Classification ML System')
    parser.add_argument('--skip-install', action='store_true', 
                       help='Skip installing requirements')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip sample data download')
    
    args = parser.parse_args()
    
    print("Setting up Flower Classification ML System...")
    print("=" * 50)
    
    # Create directory structure
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    # Create placeholder files
    print("\n2. Creating placeholder files...")
    create_placeholder_files()
    
    # Setup environment
    print("\n3. Setting up environment...")
    setup_environment()
    
    # Create .gitignore
    print("\n4. Creating .gitignore...")
    create_gitignore()
    
    # Install requirements
    if not args.skip_install:
        print("\n5. Installing requirements...")
        if not install_requirements():
            print("Failed to install requirements. Please run 'pip install -r requirements.txt' manually")
    else:
        print("\n5. Skipping requirements installation")
    
    # Download sample data
    if not args.skip_data:
        print("\n6. Setting up sample data...")
        download_sample_data()
    else:
        print("\n6. Skipping sample data setup")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Add flower images to the data/train/ directories")
    print("2. Run the Jupyter notebook: jupyter notebook notebook/flower_prediction.ipynb")
    print("3. Start the Flask server: python app.py")
    print("4. Access the API at: http://localhost:5000")

if __name__ == '__main__':
    main()