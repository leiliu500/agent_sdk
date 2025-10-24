"""
Start MCP server with custom SSE streaming support
"""
import sys
import os
import json
import asyncio
import uuid
from typing import Dict, Any
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import tool functions directly
import sdk_workflow

app = FastAPI(title="Real Estate MCP SSE Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions with tokens
sessions: Dict[str, Dict[str, Any]] = {}
session_tokens: Dict[str, str] = {}  # Map session_id to token

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "run_workflow": sdk_workflow.run_workflow,
    "buyer_intake_step": sdk_workflow.buyer_intake_step,
    "search_and_match_tool": sdk_workflow.search_and_match_tool,
    "tour_plan_tool": sdk_workflow.tour_plan_tool,
    "disclosure_qa_tool": sdk_workflow.disclosure_qa_tool,
    "offer_drafter_tool": sdk_workflow.offer_drafter_tool,
    "negotiation_coach_tool": sdk_workflow.negotiation_coach_tool,
    "health": sdk_workflow.health,
    "reset_session": sdk_workflow.reset_session,
}

@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE endpoint that provides session endpoint URLs"""
    session_id = str(uuid.uuid4())
    token = str(uuid.uuid4())  # Generate a session token
    
    sessions[session_id] = {"created": True, "token": token}
    session_tokens[token] = session_id
    
    # Get the base URL from the request
    base_url = str(request.base_url).rstrip('/')
    
    async def event_generator():
        # Send the endpoint event with full URL and token
        # The MCP Inspector expects the endpoint URL with sessionId parameter
        # Use /messages/ (with trailing slash) for better routing compatibility
        endpoint_url = f"{base_url}/messages/?sessionId={session_id}"
        
        yield f"event: endpoint\n"
        yield f"data: {endpoint_url}\n\n"
        
        # Keep connection alive with periodic heartbeats
        try:
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                yield f": heartbeat\n\n"
        except asyncio.CancelledError:
            # Client disconnected
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )

@app.post("/messages/")
@app.post("/messages")
async def handle_message(request: Request):
    """Handle JSON-RPC messages and stream responses via SSE"""
    # Support both session_id and sessionId parameter names
    session_id = request.query_params.get("sessionId") or request.query_params.get("session_id")
    
    # Log incoming request for debugging
    print(f"\n[DEBUG] Incoming request to /messages")
    print(f"[DEBUG] Session ID from URL: {session_id}")
    print(f"[DEBUG] Query params: {dict(request.query_params)}")
    print(f"[DEBUG] Headers: Authorization={request.headers.get('Authorization')}")
    
    if not session_id or session_id not in sessions:
        print(f"[DEBUG] Invalid session: {session_id}")
        return JSONResponse({"error": "Invalid session", "sessionId": session_id}, status_code=400)
    
    # Optional: Validate session token if provided
    # The MCP Inspector might send a token, but it's not required for direct SSE connections
    session_token = request.headers.get("Authorization") or request.query_params.get("token")
    if session_token:
        token = session_token.replace("Bearer ", "").strip()
        print(f"[DEBUG] Token provided: {token}")
        # Only validate if we have tokens configured (for proxy mode)
        if token and token not in session_tokens:
            print(f"[DEBUG] Token not found in session_tokens")
            # Don't fail - the Inspector might be sending its own token
            # Just log it for debugging
        elif token and session_tokens.get(token) != session_id:
            print(f"[DEBUG] Token doesn't match session")
            # Don't fail - allow the connection to proceed
    
    body = await request.json()
    method = body.get("method")
    msg_id = body.get("id")
    params = body.get("params", {})
    
    # Log the request for debugging
    print(f"\n[DEBUG] Received request: method={method}, id={msg_id}")
    print(f"[DEBUG] Full body: {json.dumps(body, indent=2)}")
    
    # Handle initialize
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "real-estate-mcp",
                    "version": "1.0.0"
                }
            }
        }
        print(f"[DEBUG] Sending initialize response: {json.dumps(response, indent=2)}")
        return JSONResponse(response)
    
    # Handle notifications/initialized (no response needed)
    if method == "notifications/initialized":
        print("[DEBUG] Received initialized notification")
        # Notifications don't require a response, but we need to return 200 OK
        # Return empty response with proper status
        return JSONResponse({}, status_code=200)
    
    # Handle ping
    if method == "ping":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {}
        })
    
    # Handle tools/list
    if method == "tools/list":
        tools_list = []
        for tool_name, tool_func in TOOL_FUNCTIONS.items():
            tool_info = {
                "name": tool_name,
                "description": tool_func.fn.__doc__ or f"Tool: {tool_name}",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
            tools_list.append(tool_info)
        
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": tools_list
            }
        }
        return JSONResponse(response)
    
    # Handle tool calls - return JSON response (not SSE for this endpoint)
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            # Get the tool function
            tool = TOOL_FUNCTIONS.get(tool_name)
            if not tool:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }, status_code=400)
            
            # Call the wrapped function
            result = tool.fn(**arguments)
            
            # Format response as JSON-RPC
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result) if not isinstance(result, str) else result
                        }
                    ]
                }
            }
            
            return JSONResponse(response)
            
        except Exception as e:
            import traceback
            error_response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(e),
                    "data": traceback.format_exc()
                }
            }
            return JSONResponse(error_response, status_code=500)
    
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }, status_code=400)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "sessions": len(sessions)}

if __name__ == "__main__":
    print("Starting Custom SSE MCP server on http://localhost:8000")
    print("Endpoints:")
    print("  GET  /sse        - Get session endpoint")
    print("  POST /messages/  - Send JSON-RPC messages")
    print("  GET  /health     - Health check")
    print("\nPress Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
