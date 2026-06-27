#!/bin/bash
set -e

export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NOPRUNE=1

case "$FLY_PROCESS_GROUP" in
    "qwen")
        MODEL_TO_PULL="qwen2.5:0.5b"
        ;;
    "tinyllama")
        MODEL_TO_PULL="tinyllama:latest"
        ;;
    "llama3")
        MODEL_TO_PULL="llama3.2:1b"
        ;;
    "phi3")
        MODEL_TO_PULL="phi3:3.8b-mini-4k-instruct-q4_K_M"
        ;;
    *)
        MODEL_TO_PULL="qwen2.5:0.5b"
        ;;
esac

echo "Starting Ollama server in background (Process Group: ${FLY_PROCESS_GROUP:-default})..."
ollama serve &

echo "Starting FastAPI on port 8080 immediately..."
python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 8 &
FASTAPI_PID=$!

echo "Waiting for Ollama to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready! Pulling model: $MODEL_TO_PULL..."
        ollama pull "$MODEL_TO_PULL"

        echo "Aggressive Warm-up: Loading $MODEL_TO_PULL into RAM with multiple pings..."
        for j in {1..3}; do
            curl -s -X POST http://localhost:11434/api/chat -d "{\"model\":\"$MODEL_TO_PULL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" > /dev/null
            echo "Warm-up ping $j complete."
            sleep 1
        done

        echo "Warm-up complete. Model is locked in RAM (KEEP_ALIVE=-1)."
        break
    fi
    sleep 1
done

wait $FASTAPI_PID
