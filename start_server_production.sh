#!/bin/bash

# Kill any existing server on port 8000
echo "Stopping any existing server on port 7000..."
lsof -ti:7000 | xargs kill -9 2>/dev/null || true

# Wait a moment
sleep 1


# Start the server with SSL certificates from Let's Encrypt
echo "Starting Streamable HTTP server with SSL (Let's Encrypt)..."
sudo /home/ec2-user/agent_sdk/.venv/bin/python start_streamable_http_server.py --ssl-keyfile /etc/letsencrypt/live/homipilot.com/privkey.pem --ssl-certfile /etc/letsencrypt/live/homipilot.com/fullchain.pem

