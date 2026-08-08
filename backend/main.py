import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import MonitoringNode, RiskLevel
from simulator import AsyncWaterLevelSimulator, SimulatorConfig
from engine import FloodEngine

app = FastAPI(
    title="WAVES",
    description="real-time flood early warning system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

node = MonitoringNode(
    id="NODE-001",
    name="Jakarta Basin A",
    latitude=-6.2,
    longitude=106.8,
    thresholds={
        "watch": 30,
        "warning": 50,
        "critical": 70
    }
)

engine = FloodEngine(node=node)

simulator = AsyncWaterLevelSimulator(
    SimulatorConfig(
        node_id=node.id,
        interval_ms=1000,
        start_level=20,
        rise_rate=0.5,
        noise=0.3,
        mode="rising"
    )
)

class ConnectionManager:
    """manages WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WebSocket] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Send message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                print(f"[WebSocket] Error sending: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

async def simulator_loop():
    """Background loop that runs the simulator and broadcasts results"""
    print("[App] Starting simulator loop...")

    def on_reading(reading):
        result = engine.process(reading)
        asyncio.create_task(manager.broadcast(result))

    simulator.on_data(on_reading)
    simulator._running = True
    simulator.current_level = simulator.start_level

    reading_count = 0
    while simulator._running:
        await asyncio.sleep(simulator.interval_ms / 1000)
        if simulator._running:
            reading = simulator._generate_reading()
            if reading:
                reading_count += 1

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "HydroSignal",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "simulator": simulator.get_stats(),
        "engine": engine.get_stats(),
        "node": engine.node.to_dict(),
        "state": engine.get_state(),
        "websocket_clients": len(manager.active_connections)
    }


@app.get("/api/node")
async def get_node():
    """Get node configuration"""
    return engine.node.to_dict()


@app.post("/api/simulator/mode")
async def set_simulator_mode(mode: str):
    """Change simulator mode"""
    valid_modes = ["rising", "falling", "stable", "fluctuating", "rapid", "emergency"]
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    simulator.set_mode(mode)
    return {"status": "ok", "mode": mode}


@app.post("/api/simulator/preset")
async def set_simulator_preset(preset: str):
    """Apply a simulator preset"""
    try:
        simulator.use_preset(preset)
        return {"status": "ok", "preset": preset, "config": simulator.get_stats()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulator/inject")
async def inject_level(level: float):
    """Inject a specific water level"""
    if level < 0:
        raise HTTPException(status_code=400, detail="Level cannot be negative")

    reading = simulator.inject_level(level)
    result = engine.process(reading)
    await manager.broadcast(result)

    return {"status": "ok", "reading": reading.to_dict()}


@app.post("/api/simulator/start")
async def start_simulator():
    """Start the simulator"""
    if simulator._running:
        return {"status": "already_running"}

    # Start background task
    asyncio.create_task(simulator_loop())
    return {"status": "started"}


@app.post("/api/simulator/stop")
async def stop_simulator():
    """Stop the simulator"""
    simulator.stop()
    return {"status": "stopped"}


@app.post("/api/simulator/reset")
async def reset_simulator():
    """Reset the simulator"""
    simulator.reset()
    return {"status": "reset"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming.

    Clients connect here to receive:
    - Water level readings
    - Risk state changes
    - Alert events

    Message format:
    {
        "type": "reading" | "alert" | "status",
        "data": { ... }
    }
    """
    await manager.connect(websocket)

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "data": {
            "message": "Connected to HydroSignal",
            "node": engine.node.to_dict(),
            "state": engine.get_state()
        }
    })

    try:
        while True:
            data = await websocket.receive_text()
            try:
                command = json.loads(data)
                await handle_websocket_command(websocket, command)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON"}
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
        manager.disconnect(websocket)


async def handle_websocket_command(websocket: WebSocket, command: Dict):
    """Handle commands received via WebSocket"""
    action = command.get("action")

    if action == "ping":
        await websocket.send_json({"type": "pong", "data": {}})

    elif action == "get_status":
        await websocket.send_json({
            "type": "status",
            "data": engine.get_state()
        })

    elif action == "set_mode":
        mode = command.get("mode")
        if mode:
            simulator.set_mode(mode)
            await websocket.send_json({
                "type": "mode_changed",
                "data": {"mode": mode}
            })

    elif action == "inject_level":
        level = command.get("level")
        if level is not None:
            reading = simulator.inject_level(level)
            result = engine.process(reading)
            await websocket.send_json({
                "type": "reading",
                "data": result
            })

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 50)
    print("WAVES - FLOOD EARLY WARNING SYSTEM")
    print("=" * 50)
    print(f"Node: {node.id} ({node.name})")
    print(f"Thresholds: {node.thresholds}")
    print(f"WebSocket: ws://localhost:8000/ws")
    print(f"REST API: http://localhost:8000/api")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """run on application shutdown"""
    print("[App] Shutting down...")
    simulator.stop()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
