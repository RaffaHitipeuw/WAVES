#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "============================================================"
echo "HYDROSIGNAL - FLOOD EARLY WARNING SYSTEM"
echo "============================================================"

if [ -f "data/asset.mp4" ]; then
    mkdir -p frontend/public
    cp data/asset.mp4 frontend/public/
    echo "Copied video to public directory"
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

echo "Starting backend server..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

sleep 2

echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "============================================================"
echo "SYSTEM RUNNING"
echo "============================================================"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait