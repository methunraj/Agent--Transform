#!/usr/bin/env python3
"""
Development server startup script for IntelliExtract backend
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found!")
        print("📋 Creating .env from .env.example...")
        
        example_file = Path(".env.example")
        if example_file.exists():
            # Copy example to .env
            with open(example_file, 'r') as src, open(env_file, 'w') as dst:
                content = src.read()
                dst.write(content)
            print("✅ Created .env file from .env.example")
            print("🔧 Please edit .env and add your GOOGLE_API_KEY")
            return False
        else:
            print("❌ .env.example file not found!")
            return False
    
    # Check if GOOGLE_API_KEY is set
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not set in .env file!")
        print("🔧 Please add your Google API key to the .env file")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import agno
        import pandas
        print("✅ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Please install dependencies:")
        print("   pip install -r requirements.txt")
        return False

def start_server(host="127.0.0.1", port=8000, reload=True):
    """Start the FastAPI server"""
    print(f"🚀 Starting IntelliExtract backend server...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main startup function"""
    print("🔧 IntelliExtract Backend - Development Server")
    print("=" * 50)
    
    # Check environment
    if not check_env_file():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Parse command line arguments
    host = "127.0.0.1"
    port = 8000
    reload = True
    
    if len(sys.argv) > 1:
        if "--host" in sys.argv:
            idx = sys.argv.index("--host")
            if idx + 1 < len(sys.argv):
                host = sys.argv[idx + 1]
        
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        
        if "--no-reload" in sys.argv:
            reload = False
    
    # Start the server
    start_server(host, port, reload)

if __name__ == "__main__":
    main()