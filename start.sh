#!/bin/bash
set -e

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

echo "Starting Ollama server in background..."
ollama serve &

echo "Starting FastAPI on port 8000 immediately..."
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

echo "Waiting for Ollama to be ready in background..."
for i in $(seq 1 120); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready! Pulling default model in background..."
        DEFAULT_MODEL=${DEFAULT_MODEL:-qwen2.5:0.5b}
        ollama pull $DEFAULT_MODEL > /dev/null 2>&1 &
        break
    fi
    sleep 1
done

wait $FASTAPI_PID
