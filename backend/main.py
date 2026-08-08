import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import MonitoringNode
from simulator import AsyncWaterLevelSimulator, SimulatorConfig
from engine import FloodEngine
from video_processor import VideoProcessor, ProcessorConfig


app = FastAPI(
    title="HydroSignal",
    description="Real-time Flood Early Warning System",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Node configuration
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

# Core engine for risk calculation
engine = FloodEngine(node=node)

# Video path - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_PATH = PROJECT_ROOT / "data" / "assets.mp4"

# Processor configuration
video_config = ProcessorConfig(
    video_path=str(VIDEO_PATH),
    process_every_n_frames=30,  # Process ~1 frame per second
    roi_config={
        'x_min': 200,
        'x_max': 1720,
        'y_min': 400,
        'y_max': 1080
    }
)

# Create video processor
video_processor: Optional[VideoProcessor] = None

# Simulator (for development/testing)
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

# Current mode: 'video' or 'simulator'
current_mode = 'video'


class ConnectionManager:

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        if not self.active_connections:
            return

        message_json = json.dumps(
            message,
            default=str
        )

        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            self.disconnect(connection)


manager = ConnectionManager()


async def video_loop():
    """Async loop for video processing."""
    global video_processor

    if video_processor is None:
        video_processor = VideoProcessor(video_config, engine)
        video_processor.initialize()

    video_processor._running = True

    print(f"[Video Loop] Started (video mode)")

    while video_processor._running:
        # Process next frame
        output = video_processor.process_frame()

        if output is None:
            # Video ended
            print("[Video Loop] Video ended")
            break

        # Get measurement
        measurement = video_processor.get_current_measurement()

        if measurement:
            # Broadcast to WebSocket clients
            await manager.broadcast({
                'type': 'reading',
                'data': {
                    'timestamp': measurement['timestamp'],
                    'frameIndex': measurement['frameIndex'],
                    'state': engine.get_state(),
                    'node': engine.node.to_dict(),
                    'measurement': measurement
                }
            })

        # Wait for next frame interval
        await asyncio.sleep(1.0 / 30 * video_config.process_every_n_frames)


async def simulator_loop():
    """Async loop for simulator (development mode)."""
    def on_reading(reading):
        result = engine.process(reading)
        asyncio.create_task(
            manager.broadcast({
                'type': 'reading',
                'data': result
            })
        )

    simulator.on_data(on_reading)

    simulator._running = True
    simulator.current_level = simulator.start_level

    print(f"[Simulator Loop] Started (simulator mode)")

    while simulator._running:
        await asyncio.sleep(
            simulator.interval_ms / 1000
        )

        if simulator._running:
            simulator._generate_reading()


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "HydroSignal",
        "version": "2.0.0",
        "mode": current_mode,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/status")
async def get_status():
    """Get system status."""
    response = {
        "mode": current_mode,
        "simulator": simulator.get_stats(),
        "engine": engine.get_stats(),
        "node": engine.node.to_dict(),
        "websocket_clients": len(manager.active_connections)
    }

    # Add video processor info if in video mode
    if current_mode == 'video' and video_processor:
        response["video"] = {
            "path": VIDEO_PATH,
            "progress": video_processor.progress,
            "frameIndex": video_processor.frame_index,
            "isInitialized": video_processor.pipeline is not None
        }

    return response


@app.get("/api/node")
async def get_node():
    return engine.node.to_dict()


@app.post("/api/mode")
async def set_mode(mode: str):
    """Set processing mode: 'video' or 'simulator'."""
    global current_mode, video_processor

    valid_modes = ['video', 'simulator']

    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    # Stop current processing
    if current_mode == 'video' and video_processor:
        video_processor._running = False
        video_processor.release()

    if current_mode == 'simulator':
        simulator.stop()

    # Set new mode
    current_mode = mode

    return {
        "status": "ok",
        "mode": mode
    }


@app.post("/api/start")
async def start_processing():
    """Start processing (video or simulator based on mode)."""
    global video_processor

    if current_mode == 'video':
        if video_processor is None:
            video_processor = VideoProcessor(video_config, engine)
            video_processor.initialize()

        if video_processor._running:
            return {"status": "already_running"}

        asyncio.create_task(video_loop())
        return {"status": "started", "mode": "video"}

    else:
        if simulator._running:
            return {"status": "already_running"}

        asyncio.create_task(simulator_loop())
        return {"status": "started", "mode": "simulator"}


@app.post("/api/stop")
async def stop_processing():
    """Stop processing."""
    global video_processor

    if current_mode == 'video' and video_processor:
        video_processor._running = False
        return {"status": "stopped", "mode": "video"}

    simulator.stop()
    return {"status": "stopped", "mode": "simulator"}


@app.post("/api/reset")
async def reset_processing():
    """Reset processing."""
    global video_processor

    if current_mode == 'video' and video_processor:
        video_processor.reset()
        return {"status": "reset", "mode": "video"}

    simulator.reset()
    return {"status": "reset", "mode": "simulator"}


@app.post("/api/seek")
async def seek_to_frame(frame: int):
    """Seek video to specific frame (video mode only)."""
    global video_processor

    if current_mode != 'video':
        raise HTTPException(
            status_code=400,
            detail="Seek only available in video mode"
        )

    if video_processor is None:
        raise HTTPException(
            status_code=400,
            detail="Video processor not initialized"
        )

    video_processor.seek_to(frame)

    return {
        "status": "ok",
        "frameIndex": frame
    }


@app.post("/api/simulator/mode")
async def set_simulator_mode(mode: str):
    """Set simulator mode (only in simulator mode)."""
    if current_mode != 'simulator':
        raise HTTPException(
            status_code=400,
            detail="Simulator mode not active"
        )

    valid_modes = [
        "rising", "falling", "stable", "fluctuating", "rapid", "emergency"
    ]

    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Must be one of: {valid_modes}"
        )

    simulator.set_mode(mode)

    return {
        "status": "ok",
        "mode": mode
    }


@app.post("/api/simulator/preset")
async def set_simulator_preset(preset: str):
    """Set simulator preset (only in simulator mode)."""
    if current_mode != 'simulator':
        raise HTTPException(
            status_code=400,
            detail="Simulator mode not active"
        )

    try:
        simulator.use_preset(preset)
        return {
            "status": "ok",
            "preset": preset,
            "config": simulator.get_stats()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for realtime data."""
    await manager.connect(websocket)

    # Send initial connection message
    initial_state = {
        "type": "connected",
        "data": {
            "message": "Connected to HydroSignal",
            "mode": current_mode,
            "node": engine.node.to_dict(),
            "state": engine.get_state()
        }
    }

    if current_mode == 'video' and video_processor:
        initial_state["data"]["videoInfo"] = {
            "path": str(VIDEO_PATH),
            "progress": video_processor.progress
        }

    await websocket.send_json(initial_state)

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
    """Handle WebSocket commands."""
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
        if mode in ['video', 'simulator']:
            # Would require stopping current processing
            await websocket.send_json({
                "type": "mode_changed",
                "data": {"mode": mode}
            })

    elif action == "get_info":
        info = {
            "mode": current_mode,
            "engine": engine.get_stats()
        }
        if current_mode == 'video' and video_processor:
            info["video"] = video_processor.get_info()
        await websocket.send_json({"type": "info", "data": info})


@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("HYDROSIGNAL - FLOOD EARLY WARNING SYSTEM")
    print("=" * 50)
    print(f"Node: {node.id} ({node.name})")
    print(f"Thresholds: {node.thresholds}")
    print(f"Mode: {current_mode}")
    print(f"Video path: {VIDEO_PATH}")
    print(f"WebSocket: ws://localhost:8000/ws")
    print(f"REST API: http://localhost:8000/api")
    print("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    print("[App] Shutting down...")
    if video_processor:
        video_processor._running = False
        video_processor.release()
    simulator.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
