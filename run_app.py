#!/usr/bin/env python3
"""
Simple launcher script for the Streamlit Agentic Data Analysis App
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import pandas
        import langchain
        import langchain_google_genai
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements using: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("Please create a .env file with your GOOGLE_API_KEY")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'GOOGLE_API_KEY' not in content:
            print("❌ GOOGLE_API_KEY not found in .env file")
            print("Please add GOOGLE_API_KEY=your_api_key_here to your .env file")
            return False
    
    print("✅ Environment file configured")
    return True

def main():
    print("🚀 Starting Agentic Data Analysis App...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Run Streamlit app
    print("🌟 Launching Streamlit app...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()