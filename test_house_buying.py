"""
Test the MCP workflow with a house buying request
Demonstrates SSE streaming responses from the MCP server
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_workflow():
    """Test the run_workflow tool with house buying request"""
    print("=" * 70)
    print("Testing MCP Workflow: House Buying Request")
    print("=" * 70)
    
    # Step 1: Get SSE endpoint (establishes session)
    print("\n[1] Connecting to SSE endpoint...")
    sse_response = requests.get(f"{BASE_URL}/sse", stream=True)
    
    session_id = None
    endpoint_url = None
    
    # Read SSE events
    for line in sse_response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            print(f"    SSE: {line_str}")
            
            if line_str.startswith("data: "):
                endpoint_url = line_str[6:]  # Remove "data: " prefix
                # Extract session ID from URL
                if "sessionId=" in endpoint_url:
                    session_id = endpoint_url.split("sessionId=")[1].split("&")[0]
                    print(f"    ✓ Session established: {session_id}")
                    break
    
    if not session_id:
        print("    ✗ Failed to get session ID")
        return
    
    # Step 2: Initialize the MCP protocol
    print("\n[2] Initializing MCP protocol...")
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    init_response = requests.post(
        f"{BASE_URL}/messages/?sessionId={session_id}",
        json=init_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if init_response.status_code == 200:
        print(f"    ✓ Initialized: {init_response.json()}")
    else:
        print(f"    ✗ Init failed: {init_response.status_code}")
        return
    
    # Step 3: Send initialized notification
    print("\n[3] Sending initialized notification...")
    notif_payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    
    notif_response = requests.post(
        f"{BASE_URL}/messages/?sessionId={session_id}",
        json=notif_payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"    ✓ Notification sent: {notif_response.status_code}")
    
    # Step 4: List available tools
    print("\n[4] Listing available tools...")
    tools_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    tools_response = requests.post(
        f"{BASE_URL}/messages/?sessionId={session_id}",
        json=tools_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if tools_response.status_code == 200:
        tools = tools_response.json().get("result", {}).get("tools", [])
        print(f"    ✓ Available tools ({len(tools)}):")
        for tool in tools:
            print(f"      - {tool['name']}")
    
    # Step 5: Call run_workflow with house buying request
    print("\n[5] Calling run_workflow with house buying request...")
    print("    Request: 'I want to buy a single family house under 2 million'")
    
    workflow_payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "run_workflow",
            "arguments": {
                "user_message": "I want to buy a single family house under 2 million"
            }
        }
    }
    
    workflow_response = requests.post(
        f"{BASE_URL}/messages/?sessionId={session_id}",
        json=workflow_payload,
        headers={"Content-Type": "application/json"}
    )
    
    print("\n" + "=" * 70)
    print("WORKFLOW RESPONSE (Streaming via JSON-RPC over HTTP)")
    print("=" * 70)
    
    if workflow_response.status_code == 200:
        result = workflow_response.json()
        print(json.dumps(result, indent=2))
        
        # Extract and display key information
        if "result" in result and "content" in result["result"]:
            content = result["result"]["content"][0]["text"]
            workflow_data = json.loads(content)
            
            print("\n" + "=" * 70)
            print("WORKFLOW ANALYSIS")
            print("=" * 70)
            print(f"\nSession ID: {workflow_data.get('session_id')}")
            print(f"Status: {workflow_data.get('status')}")
            
            router = workflow_data.get('router', {})
            print(f"\n📍 Intent: {router.get('intent')}")
            print(f"📊 Confidence: {router.get('confidence')}")
            
            slots = workflow_data.get('slots_obj', {})
            print(f"\n🏠 Extracted Information:")
            print(f"   Budget: {slots.get('budget')}")
            print(f"   Property Type: single family house (from request)")
            print(f"   Areas: {slots.get('areas')}")
            
            print(f"\n➡️  Next Step: {workflow_data.get('next_node')}")
            
            return workflow_data.get('session_id'), session_id  # Return both IDs
    else:
        print(f"    ✗ Workflow failed: {workflow_response.status_code}")
        print(f"    Response: {workflow_response.text}")
        return None, None

def test_intake_process(workflow_session_id, mcp_session_id):
    """Continue with buyer intake process"""
    if not workflow_session_id:
        print("\n⚠️  No session ID available, skipping intake test")
        return
    
    print("\n\n" + "=" * 70)
    print("Testing Buyer Intake Process")
    print("=" * 70)
    print(f"Using MCP session: {mcp_session_id}")
    print(f"Using workflow session: {workflow_session_id}")
    
    # Call buyer_intake_step to start the intake questionnaire
    intake_payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "buyer_intake_step",
            "arguments": {
                "session_id": workflow_session_id,
                "user_message": "under 2 million"
            }
        }
    }
    
    intake_response = requests.post(
        f"{BASE_URL}/messages/?sessionId={mcp_session_id}",
        json=intake_payload,
        headers={"Content-Type": "application/json"}
    )
    
    if intake_response.status_code == 200:
        result = intake_response.json()
        content = result["result"]["content"][0]["text"]
        intake_data = json.loads(content)
        
        print("\n📋 Intake Step Response:")
        print(json.dumps(intake_data, indent=2))
        
        result_data = intake_data.get('result', {})
        print(f"\n❓ Next Question: {result_data.get('message')}")
        print(f"📍 Current Field: {result_data.get('field')}")
        print(f"✅ Intake Complete: {intake_data.get('intake_complete')}")
    else:
        print(f"✗ Intake step failed: {intake_response.status_code}")

if __name__ == "__main__":
    workflow_session_id, mcp_session_id = test_workflow()
    test_intake_process(workflow_session_id, mcp_session_id)
    
    print("\n" + "=" * 70)
    print("✅ MCP Workflow Test Complete")
    print("=" * 70)
    print("\nNote: The MCP server uses HTTP with JSON-RPC for tool calls.")
    print("SSE (Server-Sent Events) is used for session establishment.")
    print("For true streaming responses, the tools would need to yield chunks.")
