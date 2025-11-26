#!/bin/bash

# PoolGuy CV Service - Run Script
# Activates virtual environment and starts Flask server

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start Flask app
echo "Starting PoolGuy CV Service..."
echo "Service will be available at http://localhost:${PORT:-5000}"
echo "Press Ctrl+C to stop"
echo ""

python app.py







