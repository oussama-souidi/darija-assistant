#!/bin/bash

# Script to run all three servers for the Olive Health Assistant

echo "🫒 Starting Olive Health Assistant Servers..."

# Function to kill all background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $ASR_PID $CNN_PID $RAG_PID
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# 1. Start ASR Server (Port 8001 by default)
echo "🎙️ Starting ASR Server..."
python asr_server.py &
ASR_PID=$!

# 2. Start CNN Server (Port 8000 by default)
echo "📸 Starting CNN Server..."
python cnn_server.py &
CNN_PID=$!

# 3. Start RAG Server (Port 8002 by default)
echo "📚 Starting RAG Server..."
python rag_server.py &
RAG_PID=$!

echo "✅ All servers are running!"
echo "   - ASR: http://localhost:8001"
echo "   - CNN: http://localhost:8000"
echo "   - RAG: http://localhost:8002"
echo ""
echo "Press [Ctrl+C] to stop all servers."

# Keep the script running to maintain the background processes
wait
