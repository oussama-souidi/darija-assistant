@echo off
title Olive Health Assistant - All Servers
echo 🫒 Starting Olive Health Assistant Servers...

echo 🎙️ Starting ASR Server...
start /b python asr_server.py

echo 📸 Starting CNN Server...
start /b python cnn_server.py

echo 📚 Starting RAG Server...
start /b python rag_server.py

echo.
echo ✅ All servers are starting in the background.
echo    - ASR: http://localhost:8001
echo    - CNN: http://localhost:8000
echo    - RAG: http://localhost:8002
echo.
echo NOTE: To stop the servers, you may need to close this terminal or use Task Manager to kill the python processes.
echo.
pause
