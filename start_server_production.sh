#!/bin/bash

# Kill any existing server on port 8000
echo "Stopping any existing server on port 7000..."
lsof -ti:7000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1

# Start the server
echo "Starting Streamable HTTP server..."
/home/ec2-user/agent_sdk/.venv/bin/python start_streamable_http_server.py

