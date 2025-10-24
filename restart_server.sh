#!/bin/bash

# Kill any existing server on port 8000
echo "Stopping any existing server on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1

# Start the server
echo "Starting SSE server..."
/Users/leiliu/projects/agent_sdk/.venv/bin/python start_sse_server.py

