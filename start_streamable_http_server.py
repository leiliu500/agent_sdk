"""
Start MCP server with Streamable HTTP transport (SSE fallback support)
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
import real_estate_workflow

app = FastAPI(title="Real Estate MCP Streamable HTTP Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    print(f"\n{'='*70}")
    print(f"[REQUEST] {request.method} {request.url}")
    print(f"[HEADERS] {dict(request.headers)}")
    print(f"{'='*70}")
    response = await call_next(request)
    print(f"[RESPONSE] Status: {response.status_code}")
    return response

# Store active sessions with tokens
sessions: Dict[str, Dict[str, Any]] = {}
session_tokens: Dict[str, str] = {}  # Map session_id to token
client_sessions: Dict[str, str] = {}  # Map client IP to most recent session_id (port may change)
last_session_by_ip: Dict[str, str] = {}  # Map IP address to last used session

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "run_workflow": real_estate_workflow.run_workflow,
    "buyer_intake_step": real_estate_workflow.buyer_intake_step,
    "search_and_match_tool": real_estate_workflow.search_and_match_tool,
    "tour_plan_tool": real_estate_workflow.tour_plan_tool,
    "disclosure_qa_tool": real_estate_workflow.disclosure_qa_tool,
    "offer_drafter_tool": real_estate_workflow.offer_drafter_tool,
    "negotiation_coach_tool": real_estate_workflow.negotiation_coach_tool,
    "health": real_estate_workflow.health,
    "reset_session": real_estate_workflow.reset_session,
}

# Manual schemas for tools (since FastMCP wrapping may not expose schemas properly)
TOOL_SCHEMAS = {
    "run_workflow": {
        "type": "object",
        "properties": {
            "user_message": {
                "type": "string",
                "description": "The user's message to process through the workflow"
            },
            "session_id": {
                "type": "string",
                "description": "Optional session ID to maintain conversation context"
            }
        },
        "required": ["user_message"]
    },
    "buyer_intake_step": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            },
            "user_message": {
                "type": "string",
                "description": "User's response to intake questions"
            }
        }
    },
    "search_and_match_tool": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            }
        }
    },
    "tour_plan_tool": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            },
            "open_houses": {
                "type": "array",
                "description": "Open house options with availability",
                "items": {
                    "type": "object",
                    "properties": {
                        "property_id": {"type": "string"},
                        "address": {"type": "string"},
                        "slots": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "start": {"type": "string"},
                                    "end": {"type": "string"}
                                },
                                "required": ["start", "end"]
                            }
                        }
                    },
                    "required": ["property_id", "address", "slots"]
                }
            },
            "preferred_windows": {
                "type": "array",
                "description": "Preferred time windows to attend tours",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"}
                    },
                    "required": ["start", "end"]
                }
            }
        },
        "required": ["open_houses", "preferred_windows"]
    },
    "disclosure_qa_tool": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            },
            "user_question": {
                "type": "string",
                "description": "Question about property disclosures"
            }
        }
    },
    "offer_drafter_tool": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            }
        }
    },
    "negotiation_coach_tool": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID"
            },
            "user_input": {
                "type": "string",
                "description": "User's negotiation question or scenario"
            }
        }
    },
    "health": {
        "type": "object",
        "properties": {}
    },
    "reset_session": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID to reset"
            }
        }
    }
}

@app.options("/sse")
async def sse_options():
    """Handle OPTIONS preflight for SSE endpoint"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

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
        # Try sending ONLY the path, not the full URL - some SSE clients expect relative paths
        endpoint_path = f"/messages/?sessionId={session_id}"
        
        yield f"event: endpoint\n"
        yield f"data: {endpoint_path}\n\n"
        
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

@app.options("/messages/")
@app.options("/messages")
async def messages_options():
    """Handle OPTIONS preflight for messages endpoint"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/messages/")
@app.post("/messages")
async def handle_message(request: Request):
    """Handle JSON-RPC messages - supports both old SSE and new StreamableHttp"""
    # Check if this is StreamableHttp (has Accept header with application/json)
    accept_header = request.headers.get("Accept", "")
    is_streamable_http = "application/json" in accept_header or "text/event-stream" in accept_header
    
    # For StreamableHttp, session can be in header or query param
    session_id = (
        request.headers.get("Mcp-Session-Id") or 
        request.query_params.get("sessionId") or 
        request.query_params.get("session_id")
    )
    
    # Log incoming request for debugging
    print(f"\n{'='*60}")
    print(f"[DEBUG] Incoming request to /messages")
    print(f"[DEBUG] Session ID from URL: {session_id}")
    print(f"[DEBUG] Query params: {dict(request.query_params)}")
    print(f"[DEBUG] Headers: Authorization={request.headers.get('Authorization')}")
    print(f"[DEBUG] All Headers: {dict(request.headers)}")
    print(f"{'='*60}")
    
    # Read body first to check if this is an initialize request
    body = await request.json()
    method = body.get("method")
    
    # For StreamableHttp, we may not have a session yet (on first initialize)
    if not session_id and is_streamable_http:
        # Create a new session for StreamableHttp
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"created": True, "initialized": False}
        print(f"[DEBUG] Created new StreamableHttp session: {session_id}")
    
    # Don't fail on missing session for initialize - that's expected
    if not session_id and method != "initialize":
        print(f"[DEBUG] No session ID and not initialize request")
        return JSONResponse({"error": "No session ID provided", "method": method}, status_code=400)
    
    if session_id and session_id not in sessions and method != "initialize":
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
    
    # Body already read above
    msg_id = body.get("id")
    params = body.get("params", {})
    
    # Log the request for debugging
    print(f"\n[DEBUG] Received request: method={method}, id={msg_id}")
    print(f"[DEBUG] Full body: {json.dumps(body, indent=2)}")
    
    # Handle initialize
    if method == "initialize":
        # Ensure we're responding to the exact protocol version the client requested
        client_protocol = params.get("protocolVersion", "2024-11-05")
        
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": client_protocol,  # Echo back the client's version
                "capabilities": {
                    "tools": {},  # Empty object indicates tools are supported
                    "logging": {}  # Empty object indicates logging is supported
                },
                "serverInfo": {
                    "name": "real-estate-mcp",
                    "version": "1.0.0"
                }
            }
        }
        print(f"[DEBUG] Sending initialize response: {json.dumps(response, indent=2)}")
        print(f"[DEBUG] Client requested protocol version: {client_protocol}")
        
        # For StreamableHttp, include session ID in response header
        response_headers = {"Content-Type": "application/json"}
        if is_streamable_http:
            response_headers["Mcp-Session-Id"] = session_id
            print(f"[DEBUG] Adding Mcp-Session-Id header: {session_id}")
        
        return JSONResponse(
            response,
            status_code=200,
            headers=response_headers
        )
    
    # Handle notifications/initialized (no response needed)
    if method == "notifications/initialized":
        print("[DEBUG] ✅ Received initialized notification - handshake complete!")
        # Mark session as fully initialized
        if session_id in sessions:
            sessions[session_id]["initialized"] = True
        # Notifications require 202 Accepted for StreamableHttp, or 200 OK for old SSE
        status_code = 202 if is_streamable_http else 200
        print(f"[DEBUG] Returning {status_code} (StreamableHttp={is_streamable_http})")
        return JSONResponse({}, status_code=status_code)
    
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
            # Use manual schema if available, otherwise try to get from tool wrapper
            input_schema = TOOL_SCHEMAS.get(tool_name, {
                "type": "object",
                "properties": {}
            })
            
            # Check if the tool has an input_schema attribute (from FastMCP)
            if hasattr(tool_func, 'input_schema'):
                input_schema = tool_func.input_schema
            
            tool_info = {
                "name": tool_name,
                "description": tool_func.fn.__doc__ or f"Tool: {tool_name}",
                "inputSchema": input_schema
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

@app.post("/")
async def root_endpoint(request: Request):
    """
    Root MCP endpoint for StreamableHttp transport.
    This handles the new transport method where clients POST directly.
    """
    # Get session ID from header, or fallback to client connection tracking
    session_id = request.headers.get("Mcp-Session-Id")
    
    # Track client by IP (port may change between requests due to connection pooling)
    client_ip = request.client.host
    client_key = f"{client_ip}:{request.client.port}"
    
    # Read body to check method
    body = await request.json()
    method = body.get("method")
    msg_id = body.get("id")
    params = body.get("params", {})
    
    print(f"\n[DEBUG] Received request at /: method={method}, id={msg_id}")
    print(f"[DEBUG] Session ID from header: {session_id}")
    print(f"[DEBUG] Client: {client_key}")
    print(f"[DEBUG] Full body: {json.dumps(body, indent=2)}")
    
    # For initialize, create new session if needed
    if method == "initialize":
        if not session_id:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": True, "initialized": False}
            # Track by IP address (not IP:port since port changes)
            last_session_by_ip[client_ip] = session_id
            client_sessions[client_key] = session_id  # Also track specific connection
            print(f"[DEBUG] Created new session for initialize: {session_id}")
            print(f"[DEBUG] Mapped client IP {client_ip} -> session {session_id}")
        
        # Ensure we're responding to the exact protocol version the client requested
        client_protocol = params.get("protocolVersion", "2024-11-05")
        
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": client_protocol,
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
        print(f"[DEBUG] Sending initialize response with session: {session_id}")
        
        return JSONResponse(
            response,
            status_code=200,
            headers={"Mcp-Session-Id": session_id}
        )
    
    # For all other methods, require valid session
    # If no session header, try to find session by client IP (most recent session for this IP)
    if not session_id:
        # Try by IP:port first, then by IP only
        session_id = client_sessions.get(client_key) or last_session_by_ip.get(client_ip)
        print(f"[DEBUG] No session header, looked up by client IP {client_ip}: {session_id}")
    
    if not session_id:
        print(f"[DEBUG] No session ID for method: {method}")
        return JSONResponse(
            {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": "No session ID provided"}},
            status_code=400
        )
    
    if session_id not in sessions:
        print(f"[DEBUG] Invalid session: {session_id}")
        return JSONResponse(
            {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32000, "message": "Invalid session"}},
            status_code=400
        )
    
    # Handle notifications/initialized
    if method == "notifications/initialized":
        print(f"[DEBUG] ✅ Received initialized notification for session: {session_id}")
        sessions[session_id]["initialized"] = True
        return JSONResponse({}, status_code=202)
    
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
            # Use manual schema if available, otherwise try to get from tool wrapper
            input_schema = TOOL_SCHEMAS.get(tool_name, {
                "type": "object",
                "properties": {}
            })
            
            # Check if the tool has an input_schema attribute (from FastMCP)
            if hasattr(tool_func, 'input_schema'):
                input_schema = tool_func.input_schema
            
            tool_info = {
                "name": tool_name,
                "description": tool_func.fn.__doc__ or f"Tool: {tool_name}",
                "inputSchema": input_schema
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
    
    # Handle logging/setLevel
    if method == "logging/setLevel":
        level = params.get("level", "info")
        print(f"[DEBUG] Setting log level to: {level} for session: {session_id}")
        # Store log level in session
        sessions[session_id]["log_level"] = level
        response = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {}
        }
        return JSONResponse(response)
    
    # Handle tool calls
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
    
    # Unknown method
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Unknown method: {method}"
        }
    }, status_code=400)

@app.get("/")
async def root_endpoint_get(request: Request):
    """
    GET on root endpoint - for StreamableHttp listening to server-initiated messages
    """
    # For StreamableHttp, we can optionally provide an SSE stream for server-initiated messages
    session_id = request.headers.get("Mcp-Session-Id")
    
    # If no session header, try to find by client IP
    client_ip = request.client.host
    if not session_id:
        session_id = last_session_by_ip.get(client_ip)
    
    print(f"[DEBUG] GET / request, session_id: {session_id}, client IP: {client_ip}")
    
    if not session_id:
        # No session yet - client should initialize first with POST
        print(f"[DEBUG] No session ID in GET request - client needs to initialize first")
        return JSONResponse(
            {
                "error": "No session ID provided", 
                "message": "Please initialize first by POSTing to / with an 'initialize' request"
            },
            status_code=400
        )
    
    if session_id not in sessions:
        print(f"[DEBUG] Invalid session ID: {session_id}")
        return JSONResponse(
            {"error": "Invalid session ID", "sessionId": session_id},
            status_code=400
        )
    
    # Return SSE stream for server-initiated messages (optional feature)
    print(f"[DEBUG] Starting SSE stream for session: {session_id}")
    async def event_generator():
        try:
            while True:
                await asyncio.sleep(30)
                yield f": heartbeat\n\n"
        except asyncio.CancelledError:
            print(f"[DEBUG] SSE stream closed for session: {session_id}")
            pass
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    print("Starting MCP server with StreamableHttp transport (SSE fallback) on http://localhost:8000")
    print("Endpoints:")
    print("  POST /               - StreamableHttp MCP endpoint (primary)")
    print("  GET  /               - StreamableHttp SSE stream (optional)")
    print("  GET  /sse            - Legacy SSE endpoint (deprecated)")
    print("  POST /messages/      - Legacy message endpoint (deprecated)")
    print("  GET  /health         - Health check")
    print("\nConnect with:")
    print("  npx @modelcontextprotocol/inspector http://127.0.0.1:8000")
    print("\nPress Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
