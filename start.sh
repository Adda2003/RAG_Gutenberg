#!/bin/bash
# Simple startup script

echo "🐰 Starting Alice in Wonderland QA System"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create .env file with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Check if knowledge base exists
if [ ! -d "data/vector_store" ]; then
    echo "📚 Building knowledge base..."
    python main.py --build
fi

# Start the server
echo "🚀 Starting Flask server..."
python run_server.py