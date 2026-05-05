#!/bin/bash

# Sentiguard-ML Startup Script
# This script sets up the environment and starts both the FastAPI backend and Streamlit frontend

set -e

echo "🚀 Starting Sentiguard-ML..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Start FastAPI backend in background
echo "⚙️ Starting FastAPI backend on port 8000..."
uvicorn main:app --port 8000 --reload &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Failed to start FastAPI backend"
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend on port 8501..."
streamlit run app.py --server.port=8501

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null || true" EXIT