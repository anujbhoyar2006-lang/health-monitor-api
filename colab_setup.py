"""
Google Colab Setup Script for Remote Health Monitoring Backend

Copy and run this entire script in a Google Colab cell to set up and run the backend.
"""

# Install required packages
import subprocess
import sys
import os

def install_packages():
    """Install required Python packages"""
    packages = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0", 
        "pydantic==2.5.0",
        "scikit-learn==1.3.2",
        "tensorflow==2.15.0",
        "pandas==2.1.4",
        "numpy==1.24.3",
        "joblib==1.3.2",
        "python-multipart==0.0.6",
        "pyngrok==7.0.0"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("✅ All packages installed successfully!")

def setup_directory_structure():
    """Create the backend directory structure"""
    directories = [
        "/content/backend",
        "/content/backend/ml", 
        "/content/backend/artifacts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directory structure created!")

def create_files():
    """Create all necessary Python files"""
    
    # Create __init__.py files
    init_files = [
        "/content/backend/__init__.py",
        "/content/backend/ml/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
    
    print("✅ Package files created!")

def run_server():
    """Start the FastAPI server with ngrok tunnel"""
    from pyngrok import ngrok
    import uvicorn
    import threading
    import time
    
    # Set up ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f"🌐 Public URL: {public_url}")
    print(f"📋 API Documentation: {public_url}/docs")
    print(f"🔍 Health Check: {public_url}/")
    
    # Start FastAPI server in a separate thread
    def start_server():
        # Add backend to Python path
        sys.path.append('/content')
        
        # Import and run the app
        from backend.main import app
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    print("🚀 Server starting...")
    time.sleep(5)  # Give server time to start
    print("✅ Server is running!")
    
    return public_url

# Main setup function
def main():
    """Run the complete setup"""
    print("🔧 Setting up Remote Health Monitoring Backend...")
    
    # Step 1: Install packages
    print("\n📦 Installing packages...")
    install_packages()
    
    # Step 2: Create directory structure  
    print("\n📁 Creating directories...")
    setup_directory_structure()
    
    # Step 3: Create package files
    print("\n📄 Creating package files...")
    create_files()
    
    print("\n✅ Setup complete!")
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("1. Copy the backend code files to their respective locations")
    print("2. Run: run_server() to start the API server")
    print("="*50)

if __name__ == "__main__":
    main()
