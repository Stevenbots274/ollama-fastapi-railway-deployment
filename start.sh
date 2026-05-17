#!/bin/bash
set -e

export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=-1

case "$FLY_PROCESS_GROUP" in
    "qwen")
        MODEL_TO_PULL="qwen2.5:0.5b"
        ;;
    "tinyllama")
        MODEL_TO_PULL="tinyllama:latest"
        ;;
    *)
        MODEL_TO_PULL="qwen2.5:0.5b"
        ;;
esac

echo "Starting Ollama server in background (Process Group: ${FLY_PROCESS_GROUP:-default})..."
ollama serve &

echo "Starting FastAPI on port 8080 immediately..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4 &
FASTAPI_PID=$!

echo "Waiting for Ollama to be ready in background..."
for i in $(seq 1 120); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready! Pulling model: $MODEL_TO_PULL..."
        ollama pull "$MODEL_TO_PULL"
        echo "Model pulled. Warming up $MODEL_TO_PULL in memory..."
        ollama run "$MODEL_TO_PULL" "Hello" > /dev/null 2>&1 &
        echo "Warm-up initiated. Model will stay loaded in RAM (KEEP_ALIVE=-1)."
        break
    fi
    sleep 1
done

wait $FASTAPI_PID
