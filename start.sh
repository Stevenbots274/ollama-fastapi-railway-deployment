#!/bin/bash
set -e

# Configure Ollama for low memory
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

echo "Starting Ollama server in background..."
ollama serve &

echo "Starting FastAPI on port 8080 immediately..."
# Using python3 -m uvicorn to ensure it's in the path
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 &
FASTAPI_PID=$!

echo "Waiting for Ollama to be ready in background..."
# Try to pull model only if Ollama is ready
for i in $(seq 1 120); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready! Pulling default model in background..."
        DEFAULT_MODEL=${DEFAULT_MODEL:-qwen2.5:0.5b}
        ollama pull $DEFAULT_MODEL > /dev/null 2>&1 &
        break
    fi
    sleep 1
done

# Keep the script running by waiting for the FastAPI process
wait $FASTAPI_PID
