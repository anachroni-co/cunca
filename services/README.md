# Capibara Services Module

**Production-ready services for text-to-speech, automation, and model coordination**

## 📋 Table of Contents

1. [Overview](#overview)
2. [Service Categories](#service-categories)
3. [Architecture](#architecture)
4. [TTS Service](#tts-service)
5. [MCP Integration](#mcp-integration)
6. [N8N Automation Service](#n8n-automation-service)
7. [Quick Start](#quick-start)
8. [Configuration](#configuration)
9. [Deployment](#deployment)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## 🎯 Overview

The `capibara/services` module provides production-ready microservices for CapibaraGPT-v2:

### Available Services

| Service | Purpose | Port | Status |
|---------|---------|------|--------|
| **TTS Service** | Text-to-Speech with FastSpeech + HiFi-GAN | 8765 | ✅ Production |
| **MCP Integration** | Model Context Protocol for distributed coordination | N/A | ✅ Production |
| **N8N Automation** | Natural language to workflow automation | 8080 | ✅ Production |
| **E2B Sandbox** | Secure code execution environment | N/A | ✅ Production |

### Key Features

✅ **Async-First**: Built on asyncio for high concurrency
✅ **WebSocket Support**: Real-time streaming for TTS and events
✅ **REST APIs**: FastAPI-based REST endpoints
✅ **Distributed**: MCP protocol for multi-node coordination
✅ **Secure**: E2B sandboxing for untrusted code execution
✅ **Production-Ready**: Logging, monitoring, error handling
✅ **Configurable**: Environment-based configuration

---

## 🏗️ Service Categories

### Service Taxonomy

```
capibara/services/
├── 🎤 TTS (Text-to-Speech)
│   ├── capibara_tts_service.py     # FastSpeech + HiFi-GAN TTS
│   ├── __init__.py                 # TTS module exports
│   └── tts.py                      # TTS interface
│
├── 📡 MCP Integration
│   └── mcp_integration.py          # Model Context Protocol
│
└── 🤖 Automation
    ├── n8n_service.py              # N8N workflow service
    ├── workflow_builder.py         # AI-powered workflow construction
    ├── agent_executor.py           # Agent-based execution
    ├── e2b_manager.py              # E2B sandbox manager
    ├── web_ui.py                   # Web interface
    ├── models.py                   # Data models
    ├── config.py                   # Configuration
    └── main.py                     # Main entry point
```

---

## 🏛️ Architecture

### Overall System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Client Applications                   │
│          (Web UI, CLI, External Services)               │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│   TTS    │  │   MCP    │  │   N8N    │  │   E2B    │
│ Service  │  │Protocol  │  │Automation│  │ Sandbox  │
│ (8765)   │  │  (Async) │  │  (8080)  │  │  (Secure)│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │             │
     ▼             ▼              ▼             ▼
┌─────────────────────────────────────────────────────────┐
│              Core Capibara Model Services               │
│   (Inference, Training, Data Processing, Monitoring)    │
└─────────────────────────────────────────────────────────┘
```

### Service Communication

```
┌──────────────┐      WebSocket       ┌──────────────┐
│  TTS Client  │◄───────────────────►│  TTS Service │
└──────────────┘   Audio Streaming    └──────────────┘

┌──────────────┐      MCP Messages    ┌──────────────┐
│  Model Node  │◄───────────────────►│  MCP Broker  │
└──────────────┘   Context Sync       └──────────────┘

┌──────────────┐      REST API        ┌──────────────┐
│Automation UI │◄───────────────────►│ N8N Service  │
└──────────────┘   Workflow Execute   └──────────────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │ E2B Sandbox  │
                                      │ (Isolated)   │
                                      └──────────────┘
```

---

## 🎤 TTS Service

### Overview

The **TTS (Text-to-Speech) Service** converts text to natural-sounding speech using:
- **FastSpeech**: ONNX-based spectrogram generation
- **HiFi-GAN**: High-fidelity vocoder for audio synthesis
- **pyttsx3**: Fallback TTS engine
- **WebSocket Server**: Real-time audio streaming

### Architecture

```
Text Input
    │
    ▼
┌──────────────────────────────┐
│  CapibaraTextToSpeech        │
│  ┌────────────────────────┐  │
│  │  FastSpeech (ONNX)     │  │ → Mel Spectrogram
│  └────────────┬───────────┘  │
│               ▼              │
│  ┌────────────────────────┐  │
│  │  HiFi-GAN (ONNX)       │  │ → Audio Waveform
│  └────────────┬───────────┘  │
└───────────────┼──────────────┘
                │
                ▼
        WebSocket Server (8765)
                │
                ▼
         Connected Clients
```

### Configuration

**Environment Variables** (`.env`):
```bash
# Model paths
FASTSPEECH_MODEL_PATH=/path/to/fastspeech.onnx
HIFIGAN_MODEL_PATH=/path/to/hifigan.onnx

# Server configuration
CAPIBARA_TTS_HOST=localhost
CAPIBARA_TTS_PORT=8765
CAPIBARA_TTS_SAMPLE_RATE=22050

# SSL/TLS (optional)
CAPIBARA_TTS_CERT_FILE=/path/to/cert.pem
CAPIBARA_TTS_KEY_FILE=/path/to/key.pem
```

### Quick Start

**1. Install Dependencies**
```bash
pip install onnxruntime websockets pyttsx3 numpy
```

**2. Download Models**
```bash
# FastSpeech model
wget https://example.com/fastspeech.onnx -O models/fastspeech.onnx

# HiFi-GAN vocoder
wget https://example.com/hifigan.onnx -O models/hifigan.onnx
```

**3. Start TTS Service**
```python
from capibara.services.tts import CapibaraTextToSpeech

# Initialize TTS
tts = CapibaraTextToSpeech(
    fastspeech_model_path="models/fastspeech.onnx",
    hifigan_model_path="models/hifigan.onnx",
    sample_rate=22050
)

# Generate audio
audio = tts.synthesize("Hello, I am Capibara!")

# Save to file
import soundfile as sf
sf.write("output.wav", audio, tts.sample_rate)
```

**4. Start WebSocket Server**
```python
import asyncio
from capibara.services.tts import start_tts_server

async def main():
    await start_tts_server(
        host="localhost",
        port=8765,
        tts=tts
    )

asyncio.run(main())
```

### WebSocket API

**Client Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log('Connected to TTS service');

    // Send text to synthesize
    ws.send(JSON.stringify({
        text: "Hello, world!",
        streaming: true
    }));
};

ws.onmessage = (event) => {
    // Receive audio chunks
    const audioData = event.data;
    playAudio(audioData);
};
```

**Request Format:**
```json
{
    "text": "Text to synthesize",
    "streaming": true,
    "sample_rate": 22050,
    "format": "wav"
}
```

**Response Format:**
```json
{
    "status": "streaming",
    "chunk_id": 1,
    "total_chunks": 10,
    "audio_data": "base64_encoded_audio"
}
```

### Python Client Example

```python
import asyncio
import websockets
import json
import base64
import numpy as np

async def tts_client(text):
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "text": text,
            "streaming": True
        }))

        # Receive audio chunks
        audio_chunks = []

        async for message in websocket:
            data = json.loads(message)

            if data["status"] == "streaming":
                # Decode audio chunk
                audio_chunk = base64.b64decode(data["audio_data"])
                audio_chunks.append(audio_chunk)

            elif data["status"] == "complete":
                break

        # Combine chunks
        audio = np.concatenate(audio_chunks)
        return audio

# Usage
audio = asyncio.run(tts_client("Hello from Capibara!"))
```

### Fallback TTS

If ONNX models are unavailable, the service automatically falls back to pyttsx3:

```python
from capibara.services.tts import FallbackTTS

fallback_tts = FallbackTTS()
audio = fallback_tts.synthesize("Fallback TTS engine")
```

---

## 📡 MCP Integration

### Overview

**MCP (Model Context Protocol)** enables distributed coordination between multiple Capibara model instances and external services.

### Key Concepts

**MCP Node**: A service or model instance that participates in the MCP network

**MCP Message**: Standardized message format for inter-node communication

**MCP Broker**: Central coordinator for message routing (optional)

### Message Types

| Type | Purpose | Priority |
|------|---------|----------|
| `HANDSHAKE` | Initial node connection | Medium |
| `CONTEXT_SYNC` | Synchronize model context | High |
| `TRAINING_UPDATE` | Share training progress | Medium |
| `MODEL_STATE` | Broadcast model state | Medium |
| `PERFORMANCE_REPORT` | Share metrics | Low |
| `CONTROL_COMMAND` | Remote control | Critical |
| `HEARTBEAT` | Keep-alive ping | Low |
| `ERROR` | Error notification | High |

### Architecture

```
┌─────────────┐       MCP Network        ┌─────────────┐
│  Trainer    │◄────────────────────────►│ Inference   │
│  Node       │   Context Sync           │ Node        │
└──────┬──────┘                          └──────┬──────┘
       │                                        │
       │         ┌──────────────────┐          │
       └────────►│   MCP Broker     │◄─────────┘
                 │  (Optional Hub)  │
                 └────────┬─────────┘
                          │
                 ┌────────┴─────────┐
                 ▼                  ▼
          ┌─────────────┐    ┌─────────────┐
          │  Monitor    │    │ Coordinator │
          │  Node       │    │ Node        │
          └─────────────┘    └─────────────┘
```

### Quick Start

**1. Create MCP Node**
```python
from capibara.services.mcp_integration import (
    MCPNode,
    MCPMessageHandler,
    MCPMessage,
    MCPMessageType
)

# Create node
node = MCPNode(
    node_id="inference_1",
    node_type="inference",
    capabilities=["text_generation", "embeddings"],
    status="online"
)

# Create message handler
handler = MCPMessageHandler(node_id=node.node_id)
```

**2. Register Message Handlers**
```python
async def handle_context_sync(message: MCPMessage):
    """Handle context synchronization messages"""
    context_data = message.payload.get("context")

    # Update local context
    update_model_context(context_data)

    # Send acknowledgment
    return MCPMessage(
        message_type=MCPMessageType.CONTEXT_SYNC,
        sender_id=node.node_id,
        recipient_id=message.sender_id,
        payload={"status": "synced"}
    )

# Register handler
handler.register_handler(
    MCPMessageType.CONTEXT_SYNC,
    handle_context_sync
)
```

**3. Send Messages**
```python
# Create message
msg = MCPMessage(
    message_type=MCPMessageType.TRAINING_UPDATE,
    sender_id=node.node_id,
    recipient_id="coordinator_1",
    payload={
        "epoch": 10,
        "loss": 0.245,
        "accuracy": 0.892
    },
    priority=2
)

# Send message
await send_mcp_message(msg)
```

**4. Broadcast Messages**
```python
# Broadcast to all nodes
broadcast_msg = MCPMessage(
    message_type=MCPMessageType.MODEL_STATE,
    sender_id=node.node_id,
    recipient_id=None,  # None = broadcast
    payload={
        "model_version": "v2.3",
        "checkpoint": "epoch_50.ckpt"
    }
)

await broadcast_mcp_message(broadcast_msg)
```

### Use Cases

**1. Distributed Training Coordination**
```python
# Trainer node sends gradient updates
gradient_update = MCPMessage(
    message_type=MCPMessageType.TRAINING_UPDATE,
    sender_id="trainer_1",
    recipient_id="parameter_server",
    payload={
        "gradients": compressed_gradients,
        "batch_id": 1234
    }
)
```

**2. Model Serving Coordination**
```python
# Load balancer queries node health
heartbeat = MCPMessage(
    message_type=MCPMessageType.HEARTBEAT,
    sender_id="load_balancer",
    recipient_id="inference_1",
    payload={"timestamp": time.time()}
)

# Node responds with status
status_response = MCPMessage(
    message_type=MCPMessageType.PERFORMANCE_REPORT,
    sender_id="inference_1",
    recipient_id="load_balancer",
    payload={
        "cpu_usage": 0.45,
        "memory_usage": 0.62,
        "requests_per_sec": 120,
        "avg_latency_ms": 45.3
    }
)
```

**3. Context Synchronization**
```python
# Sync conversation context across nodes
context_sync = MCPMessage(
    message_type=MCPMessageType.CONTEXT_SYNC,
    sender_id="inference_1",
    recipient_id="inference_2",
    payload={
        "conversation_id": "conv_12345",
        "history": [...],
        "embeddings": [...]
    }
)
```

---

## 🤖 N8N Automation Service

### Overview

The **N8N Automation Service** provides intelligent workflow automation using natural language:
- Convert natural language to N8N workflows
- Agent-based intelligent execution
- Secure code execution in E2B sandboxes
- REST API for integration
- Web UI for management

### Architecture

```
Natural Language Input
        │
        ▼
┌────────────────────────┐
│  WorkflowBuilder       │ ← AI-Powered Analysis
│  - Parse intent        │
│  - Generate nodes      │
│  - Create connections  │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  N8N Workflow (JSON)   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  AgentExecutor         │
│  - Intelligent routing │
│  - Parameter inference │
│  - Error handling      │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│  E2BSandboxManager     │ ← Secure Execution
│  - Isolate code exec   │
│  - Resource limits     │
│  - Security policies   │
└────────────────────────┘
```

### Components

#### 1. WorkflowBuilder

**Purpose**: Convert natural language to N8N workflows

**Example:**
```python
from capibara.services.automation import WorkflowBuilder

builder = WorkflowBuilder(config={
    "llm_model": "capibara-v2",
    "max_nodes": 50
})

# Natural language input
description = """
Send a daily email at 9 AM with:
1. Weather forecast from OpenWeather API
2. Top 3 Hacker News stories
3. My calendar events for today from Google Calendar
"""

# Generate workflow
workflow = builder.build_workflow(description)

print(workflow.to_json())
```

**Generated Workflow:**
```json
{
    "nodes": [
        {
            "name": "Schedule Trigger",
            "type": "n8n-nodes-base.cron",
            "parameters": {
                "cronExpression": "0 9 * * *"
            }
        },
        {
            "name": "Get Weather",
            "type": "n8n-nodes-base.httpRequest",
            "parameters": {
                "url": "https://api.openweathermap.org/..."
            }
        },
        {
            "name": "Get HN Stories",
            "type": "n8n-nodes-base.httpRequest",
            "parameters": {
                "url": "https://hacker-news.firebaseio.com/..."
            }
        },
        {
            "name": "Get Calendar",
            "type": "n8n-nodes-base.googleCalendar",
            "parameters": {...}
        },
        {
            "name": "Format Email",
            "type": "n8n-nodes-base.function",
            "parameters": {
                "functionCode": "..."
            }
        },
        {
            "name": "Send Email",
            "type": "n8n-nodes-base.emailSend",
            "parameters": {...}
        }
    ],
    "connections": {...}
}
```

#### 2. AgentExecutor

**Purpose**: Intelligent workflow execution with agents

**Features:**
- Dynamic parameter inference
- Error recovery
- Adaptive routing
- Performance optimization

**Example:**
```python
from capibara.services.automation import AgentExecutor

executor = AgentExecutor(config={
    "max_retries": 3,
    "timeout": 300,
    "enable_learning": True
})

# Execute workflow
result = await executor.execute_workflow(
    workflow=workflow,
    inputs={"user_id": "12345"},
    mode="intelligent"  # vs "standard"
)

print(f"Status: {result.status}")
print(f"Output: {result.output}")
print(f"Execution time: {result.execution_time_ms}ms")
```

**Intelligent Features:**
```python
# Agent automatically infers missing parameters
workflow_with_missing_params = {
    "nodes": [
        {
            "name": "Send Email",
            "type": "emailSend",
            "parameters": {
                "subject": "Hello",
                # Missing: recipient, body
            }
        }
    ]
}

# Agent infers from context
result = await executor.execute_workflow(
    workflow=workflow_with_missing_params,
    context={
        "user": {"email": "user@example.com"},
        "previous_conversation": "...send report to john..."
    }
)

# Agent inferred:
# - recipient: john@company.com (from conversation)
# - body: "Report attached" (from context)
```

#### 3. E2BSandboxManager

**Purpose**: Secure code execution in isolated sandboxes

**Features:**
- Process isolation
- Resource limits (CPU, memory, network)
- Timeout enforcement
- Security policies

**Example:**
```python
from capibara.services.automation import E2BSandboxManager

sandbox = E2BSandboxManager(config={
    "timeout": 30,
    "memory_limit_mb": 512,
    "allow_network": False,
    "allowed_imports": ["pandas", "numpy", "requests"]
})

# Execute code safely
code = """
import pandas as pd
import requests

# Fetch data
response = requests.get('https://api.example.com/data')
data = response.json()

# Process with pandas
df = pd.DataFrame(data)
result = df.groupby('category').sum()

print(result)
"""

result = await sandbox.execute_code(
    code=code,
    language="python",
    timeout=30
)

print(f"Output: {result.stdout}")
print(f"Errors: {result.stderr}")
print(f"Exit code: {result.exit_code}")
```

**Security Features:**
```python
# Dangerous code is blocked
dangerous_code = """
import os
os.system('rm -rf /')  # Blocked!
"""

result = await sandbox.execute_code(dangerous_code)
# Result: SecurityError - Forbidden operation detected
```

### REST API

**Start API Server:**
```python
from capibara.services.automation import create_automation_service
import uvicorn

# Create service
service = create_automation_service(config={
    "host": "0.0.0.0",
    "port": 8080,
    "workers": 4
})

# Run server
uvicorn.run(service.app, host="0.0.0.0", port=8080)
```

**API Endpoints:**

**1. Create Workflow**
```bash
POST /api/v1/workflows/create

{
    "description": "Send daily summary email at 9 AM",
    "user_id": "user_123"
}

Response:
{
    "workflow_id": "wf_abc123",
    "workflow": {...},
    "status": "created"
}
```

**2. Execute Workflow**
```bash
POST /api/v1/workflows/{workflow_id}/execute

{
    "inputs": {"user_id": "123"},
    "mode": "intelligent"
}

Response:
{
    "execution_id": "exec_xyz789",
    "status": "running",
    "progress": 0.5
}
```

**3. Get Execution Status**
```bash
GET /api/v1/executions/{execution_id}

Response:
{
    "execution_id": "exec_xyz789",
    "status": "completed",
    "output": {...},
    "execution_time_ms": 1234,
    "steps_completed": 6,
    "steps_total": 6
}
```

**4. List Workflows**
```bash
GET /api/v1/workflows?user_id=user_123

Response:
{
    "workflows": [
        {
            "workflow_id": "wf_abc123",
            "name": "Daily Email Summary",
            "created_at": "2025-11-16T10:00:00Z",
            "status": "active"
        }
    ],
    "total": 1
}
```

### Web UI

**Start Web Interface:**
```bash
python -m capibara.services.automation.web_ui --port 3000
```

**Features:**
- Visual workflow builder
- Execution monitoring
- Performance analytics
- User management
- Workflow templates

**Access**: `http://localhost:3000`

---

## 🚀 Quick Start

### Installation

```bash
# Install service dependencies
pip install -r requirements.txt

# Install service-specific dependencies
pip install -r requirements-services.txt
```

### Running All Services

```bash
# 1. Start TTS Service
python -m capibara.services.tts --port 8765

# 2. Start MCP Broker
python -m capibara.services.mcp_broker --port 5000

# 3. Start N8N Automation
python -m capibara.services.automation --port 8080
```

### Docker Deployment

```bash
# Build services
docker-compose -f docker-compose.services.yml build

# Start all services
docker-compose -f docker-compose.services.yml up -d

# Check status
docker-compose -f docker-compose.services.yml ps
```

---

## ⚙️ Configuration

### Service Configuration File

**`config/services.toml`:**
```toml
[tts]
enabled = true
host = "0.0.0.0"
port = 8765
sample_rate = 22050
fastspeech_model = "models/fastspeech.onnx"
hifigan_model = "models/hifigan.onnx"
enable_ssl = false

[mcp]
enabled = true
node_id = "capibara_main"
node_type = "coordinator"
broker_host = "localhost"
broker_port = 5000

[automation]
enabled = true
host = "0.0.0.0"
port = 8080
n8n_url = "http://localhost:5678"
e2b_api_key = "${E2B_API_KEY}"
max_concurrent_executions = 10
workflow_timeout = 300

[automation.security]
enable_sandbox = true
sandbox_timeout = 30
sandbox_memory_limit = 512
allow_network = true
allowed_imports = ["pandas", "numpy", "requests", "json"]
```

---

## 📊 Monitoring & Metrics

### Prometheus Metrics

```python
from prometheus_client import start_http_server, Counter, Histogram

# TTS metrics
tts_requests = Counter('tts_requests_total', 'Total TTS requests')
tts_latency = Histogram('tts_latency_seconds', 'TTS generation latency')

# MCP metrics
mcp_messages = Counter('mcp_messages_total', 'Total MCP messages', ['type'])
mcp_errors = Counter('mcp_errors_total', 'MCP errors', ['type'])

# Automation metrics
workflow_executions = Counter('workflow_executions_total', 'Workflow executions')
workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution time')

# Start metrics server
start_http_server(9090)
```

**Grafana Dashboard**: Import `dashboards/services.json`

---

## 🔧 Troubleshooting

### TTS Service Issues

**Problem**: WebSocket connection fails
```bash
# Check if port is available
lsof -i :8765

# Check SSL configuration
openssl s_client -connect localhost:8765
```

**Problem**: Poor audio quality
```python
# Increase sample rate
tts = CapibaraTextToSpeech(sample_rate=44100)  # vs 22050

# Use better models
# Download high-quality FastSpeech/HiFi-GAN models
```

### MCP Integration Issues

**Problem**: Nodes not discovering each other
```python
# Enable debug logging
import logging
logging.getLogger('capibara.services.mcp').setLevel(logging.DEBUG)

# Check node registration
print(mcp_broker.list_nodes())
```

### Automation Service Issues

**Problem**: Workflow execution timeout
```python
# Increase timeout
executor = AgentExecutor(config={"timeout": 600})  # 10 minutes

# Enable async execution
result = await executor.execute_workflow(workflow, async_mode=True)
```

**Problem**: Sandbox security errors
```python
# Check allowed imports
sandbox.config.allowed_imports.append('new_module')

# Disable network if not needed
sandbox.config.allow_network = False
```

---

## 📚 References

### Documentation

- [FastSpeech Paper](https://arxiv.org/abs/1905.09263)
- [HiFi-GAN Paper](https://arxiv.org/abs/2010.05646)
- [N8N Documentation](https://docs.n8n.io/)
- [E2B Sandboxes](https://e2b.dev/docs)
- [Model Context Protocol](https://modelcontextprotocol.io/)

### Related Modules

- [Core Module](../core/README.md) - Core model architecture
- [Inference Module](../inference/README.md) - Inference optimization
- [Pipeline Module](../pipeline/README.md) - Data pipelines

---

## 🤝 Contributing

Contributions welcome! Priority areas:

1. **New Services**: Add new microservices (ASR, video processing, etc.)
2. **Performance**: Optimize TTS latency and throughput
3. **MCP Features**: Add new message types and coordination patterns
4. **Automation**: More N8N integrations and workflow templates
5. **Testing**: Comprehensive unit and integration tests

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

Part of the capibaraGPT-v2 project. See [LICENSE](../../LICENSE) for details.

---

**Maintained by**: Capibara ML Team
**Last Updated**: 2025-11-16

## Ejemplo rápido

Ejemplo (pseudo-comando) para levantar un servicio:

```bash
python services/api_server.py --config config/configs_toml/service.toml
```
