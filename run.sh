#!/usr/bin/env bash

# Exit immediately if any command fails
set -e

# Start the frontend in the background
cd frontend
npm start &

# Navigate back to the root directory
cd ..

# Start the backend in the background
python -m backend.app &

# Start the worker in the foreground (keeps the script alive)
python -m src.worker
