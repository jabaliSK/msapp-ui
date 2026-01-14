#!/bin/bash

# Define the log directory (mounted volume)
LOG_DIR="/var/log/vllm"
mkdir -p "$LOG_DIR"

# Function to start a model and wait for it to be ready
launch_model() {
    local cmd="$1"
    local port="$2"
    local name="$3"
    local log_file="$LOG_DIR/${name}.log"

    echo "---------------------------------------------------"
    echo "🚀 Starting $name on port $port..."
    echo "Logs: $log_file"
    
    # Run the command in the background and redirect both stdout and stderr to the log file
    eval "$cmd" > "$log_file" 2>&1 &
    
    local pid=$!
    
    # Wait logic: Poll the health endpoint until it returns 200
    echo "Waiting for $name to initialize..."
    while true; do
        if ! kill -0 $pid 2>/dev/null; then
            echo "❌ $name (PID $pid) crashed during startup. Check $log_file"
            exit 1
        fi

        # vLLM exposes /health or /v1/models. We check /health.
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
        
        if [ "$HTTP_STATUS" == "200" ]; then
            echo "✅ $name is UP and READY!"
            break
        fi
        sleep 5
    done
}

# --- MODEL 1: Qwen3-8B-FP8 (Port 8000) ---
CMD_1="CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B-FP8 --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.4 --max-model-len 4096 --quantization fp8 --trust-remote-code"
launch_model "$CMD_1" "8000" "qwen-8b"

# --- MODEL 2: Qwen3Guard (Port 8001) ---
CMD_2="CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3Guard-Gen-0.6B --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.1 --max-model-len 23000 --trust-remote-code"
launch_model "$CMD_2" "8001" "qwen-guard"

# --- MODEL 3: Granite Embedding (Port 8006) ---
CMD_3="CUDA_VISIBLE_DEVICES=0 vllm serve ibm-granite/granite-embedding-278m-multilingual --trust-remote-code --port 8006 --host 0.0.0.0 --gpu-memory-utilization 0.1 --max-model-len 512"
launch_model "$CMD_3" "8006" "granite-embedding"

# --- MODEL 4: Qwen3-VL (Port 8003) ---
# Note: Escaped quotes for JSON argument
CMD_4="CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct-FP8 --host 0.0.0.0 --port 8003 --gpu-memory-utilization 0.2 --max-model-len 4096 --limit-mm-per-prompt '{\"image\": 1, \"video\": 0}' --mm-processor-cache-gb 0 --trust-remote-code --tensor-parallel-size 1"
launch_model "$CMD_4" "8003" "qwen-vl"

# --- MODEL 5: Qwen3-Reranker (Port 8007) ---
CMD_5="CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Reranker-0.6B  --host 0.0.0.0 --port 8007 --gpu-memory-utilization 0.1 --max-model-len 4096 --trust-remote-code"
launch_model "$CMD_5" "8007" "qwen-reranker"

echo "---------------------------------------------------"
echo "🎉 All models are running."
echo "Keeping container alive..."

# Keep the script running to prevent the container from exiting
wait
