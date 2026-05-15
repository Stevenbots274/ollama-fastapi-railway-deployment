#!/bin/bash
set -e

# Configure Ollama for low memory
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..90}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
done

# Pull default model if not exists in background
DEFAULT_MODEL=${DEFAULT_MODEL:-qwen2.5:0.5b}
echo "Pulling model: $DEFAULT_MODEL in background..."
ollama pull $DEFAULT_MODEL > /dev/null 2>&1 &

# Start FastAPI
echo "Starting FastAPI on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
