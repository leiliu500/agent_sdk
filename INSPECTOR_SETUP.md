# MCP Inspector Connection Guide

## ✅ YOUR SERVER IS WORKING!

If you're seeing the initialize response like this:
```json
{
    "jsonrpc": "2.0",
    "id": 0,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {...},
        "serverInfo": {...}
    }
}
```

**Your server is correctly implementing MCP!** The issue is likely the Inspector UI not updating its status.

## Quick Fix Steps

### 1. Try the Inspector in the Browser Console

Open the browser DevTools (F12) on the Inspector page and run:

```javascript
// This forces the Inspector to check the connection status
location.reload()
```

### 2. Check What the Inspector is Doing

In DevTools → Network tab:
1. Filter by "messages"
2. Look for POST requests to `/messages/?sessionId=...`
3. Check if they're returning 200 OK
4. If yes, but UI shows "not connected", it's an Inspector UI bug

### 3. Use the Command Line Inspector (Recommended)

Instead of the web UI, use the official CLI inspector:

```bash
npx @modelcontextprotocol/inspector --sse http://127.0.0.1:8000/sse
```

This opens a more reliable Inspector interface.

## Connection Configuration

### Method 1: CLI Inspector (Most Reliable) ⭐

```bash
npx @modelcontextprotocol/inspector --sse http://127.0.0.1:8000/sse
```

This is the recommended way! It opens a web interface that's more stable than connecting manually.

### Method 2: Web UI Manual Connection

### Step 1: Start the Server
```bash
./restart_server.sh
```

### Step 2: Open MCP Inspector
Open in your browser:
```
http://localhost:6274
```

### Step 3: Configure Connection in Inspector

**IMPORTANT: Fill in these fields EXACTLY:**

1. **Transport Type**: Select `sse` (SSE - Server-Sent Events)
2. **URL**: Enter `http://localhost:8000/sse` (or `http://127.0.0.1:8000/sse`)
3. **Use Proxy**: ❌ **LEAVE UNCHECKED** (This is critical!)

### Step 4: Click "Connect"

After clicking Connect:
- The Inspector will call `/sse` to get a session endpoint
- Then it will call the session endpoint with `initialize` method
- You should see the connection status change to "Connected"
- Tools should appear in the Inspector UI

## Common Issues & Solutions

### Issue: "Connection Error - Did you add the proxy session token"

**Cause**: You checked the "Use Proxy" checkbox
**Solution**: 
1. **Uncheck "Use Proxy"**
2. Make sure URL is exactly: `http://localhost:8000/sse`
3. Transport Type is `sse`
4. Click Connect again

### Issue: Inspector shows "Not Connected" but initialize returns 200 OK

**Cause**: The Inspector UI might be waiting for specific capabilities or events

**Check these**:
1. Open Browser DevTools (F12) → Console tab
2. Look for any JavaScript errors
3. Check Network tab to see if requests are succeeding
4. Verify the initialize response has `protocolVersion: "2024-11-05"`

**Test manually**:
```bash
# Run this test script to verify server is working
./test_mcp_connection.sh
```

### Issue: CORS Errors in Browser Console

**Solution**: The server already has CORS enabled. Try:
- Use `http://127.0.0.1:8000/sse` instead of `http://localhost:8000/sse`
- Or restart your browser

### Issue: Server works with curl but Inspector shows "Not Connected"

This means the server is working correctly, but the Inspector UI might have internal issues.

**Debugging steps**:

1. **Test with our flow tester**: Open `test_inspector_flow.html` in your browser
   - This mimics exactly what the Inspector does
   - If this works, your server is fine

2. **Check Inspector Console**:
   - Open DevTools (F12) in the Inspector page
   - Look for JavaScript errors or failed requests
   - Check if the Inspector is making requests to the correct URL

3. **Verify Inspector Version**:
   - The MCP Inspector might have version-specific issues
   - Try the official MCP Inspector from: https://github.com/modelcontextprotocol/inspector

4. **Alternative: Use the Direct Link**:
   - Open `inspector_connect.html` in browser
   - This provides a direct link to connect

## Testing Files

- `test_mcp_connection.sh` - Command-line test of full MCP flow
- `test_inspector_flow.html` - Browser-based test mimicking Inspector behavior
- `inspector_connect.html` - Quick connect link for Inspector

## Server Endpoints

- **GET /sse** - SSE connection endpoint (returns session endpoint URL)
- **POST /messages/?sessionId={id}** - JSON-RPC message endpoint
- **GET /health** - Health check

## Expected Flow

1. Client connects to `/sse` via EventSource
2. Server sends `event: endpoint` with URL like `http://localhost:8000/messages/?sessionId=xxx`
3. Client sends `initialize` request to that URL
4. Server responds with capabilities
5. Client can now call `tools/list` and `tools/call`

## Debug Mode

The server now logs all requests. Check the terminal where you ran `./restart_server.sh` to see:
- All incoming JSON-RPC requests
- Request method and ID
- Response data

## If All Else Fails

Your server is working correctly (as verified by `test_mcp_connection.sh`). The issue is likely:

1. **Inspector UI bug**: The Inspector might have state management issues
2. **Browser cache**: Try hard refresh (Cmd+Shift+R) or incognito mode
3. **Inspector expectations**: The Inspector might expect different capability format

**Workaround**: Use the command-line MCP client or integrate directly with your application using the working endpoints.

## Server Endpoints

- **GET /sse** - SSE connection endpoint (returns session endpoint URL)
- **POST /messages/?sessionId={id}** - JSON-RPC message endpoint
- **GET /health** - Health check

## Expected Flow

1. Client connects to `/sse`
2. Server sends `event: endpoint` with URL like `/messages/?sessionId=xxx`
3. Client uses that URL to send JSON-RPC requests
4. Server responds with tool results or errors
