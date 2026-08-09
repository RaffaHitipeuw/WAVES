from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.routing import WebSocketRoute
from typing import Optional
import asyncio
import uvicorn
import cv2
import numpy as np
from datetime import datetime

from .models import WaterLevelReading, MonitoringNode, RiskLevel, DataSource
from .simulator import WaterLevelSimulator
from .engine import FloodEngine
from vision import CVPipeline


app = FastAPI(title="HydroSignal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModeRequest(BaseModel):
    mode: str


class SimulatorPresetRequest(BaseModel):
    preset: str


simulator = WaterLevelSimulator()
engine = FloodEngine()
pipeline = CVPipeline(frame_width=1920, frame_height=1080)
processing = False
mode = "simulator"
video_cap = None


@app.get("/")
async def root():
    return {"status": "running", "service": "HydroSignal API"}


@app.get("/api/status")
async def get_status():
    return {
        "processing": processing,
        "mode": mode,
        "simulator_stats": simulator.get_stats(),
        "engine_stats": engine.get_stats(),
        "pipeline_stats": pipeline.get_stats() if hasattr(pipeline, 'get_stats') else {}
    }


@app.post("/api/start")
async def start():
    global processing, video_cap
    if processing:
        return {"status": "already_running"}
    processing = True
    if mode == "video":
        video_cap = cv2.VideoCapture("asset.mp4")
    elif mode == "simulator":
        simulator.start()
    asyncio.create_task(process_loop())
    return {"status": "started"}


@app.post("/api/stop")
async def stop():
    global processing, video_cap
    processing = False
    simulator.stop()
    if video_cap:
        video_cap.release()
        video_cap = None
    return {"status": "stopped"}


@app.post("/api/reset")
async def reset():
    global processing, video_cap
    processing = False
    if video_cap:
        video_cap.release()
        video_cap = None
    simulator.reset()
    pipeline.reset()
    engine.state.water_level = 0.0
    engine.state.smoothed_level = 0.0
    engine.state.rate_of_change = 0.0
    engine.state.risk = RiskLevel.SAFE
    return {"status": "reset"}


@app.post("/api/mode")
async def set_mode(request: ModeRequest):
    global mode
    if request.mode not in ["simulator", "video"]:
        raise HTTPException(status_code=400, detail="Mode must be 'simulator' or 'video'")
    mode = request.mode
    return {"status": "ok", "mode": mode}


@app.post("/api/simulator/preset")
async def set_simulator_preset(request: SimulatorPresetRequest):
    simulator.use_preset(request.preset)
    return {"status": "ok", "preset": request.preset}


@app.post("/api/simulator/mode")
async def set_simulator_mode(request: ModeRequest):
    simulator.set_mode(request.mode)
    return {"status": "ok", "mode": request.mode}


@app.post("/api/simulator/inject")
async def inject_level(level: float):
    reading = simulator.inject_level(level)
    result = engine.process(reading)
    return result


async def process_loop():
    global processing, video_cap
    while processing:
        if mode == "simulator":
            reading = simulator._generate_reading()
            if reading:
                result = engine.process(reading)
                result["diagnostics"] = {
                    "state": "SIMULATOR",
                    "reasons": [],
                    "permitted_inferences": ["simulated_water_level"],
                    "blocked_inferences": []
                }
                result["evidence"] = {}
                result["signals"] = {}
                result["candidates"] = []
                await broadcast(result)
        elif mode == "video":
            if video_cap is None or not video_cap.isOpened():
                break
            ret, frame = video_cap.read()
            if not ret:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            result = pipeline.process_frame(frame, frame_index=int(video_cap.get(cv2.CAP_PROP_POS_FRAMES)))
            cv_measurement = result.get("measurement", {})
            cv_water_level = cv_measurement.get("waterLevel")
            cv_confidence = cv_measurement.get("confidence", 0.0)
            diagnostics = result.get("diagnostics", {})
            blocked = set(diagnostics.get("blocked_inferences", []))

            # Gate risk on blocked_inferences: do not escalate if risk_level is blocked
            if 'risk_level' in blocked or 'water_level' in blocked:
                effective_risk = 'SAFE'
                effective_risk_confidence = 0.0
            else:
                effective_risk = result.get("risk", "SAFE")
                effective_risk_confidence = result.get("risk_confidence", 0.0)

            # Only feed to engine if detection succeeded; skip on no-detection to avoid fabricating 0
            if cv_water_level is not None:
                water_reading = WaterLevelReading(
                    node_id="NODE-001",
                    water_level=cv_water_level,
                    source=DataSource.SENSOR
                )
                engine_result = engine.process(water_reading, cv_confidence=cv_confidence)
            else:
                # No detection this frame — engine holds last known state
                engine_result = engine.process(
                    WaterLevelReading(node_id="NODE-001", water_level=engine.state.water_level, source=DataSource.SENSOR),
                    cv_confidence=0.0
                )
            signals = result.get("signals", {})
            for sig_key in ["edge", "color", "texture"]:
                sig = signals.get(sig_key)
                if sig and sig.get("data"):
                    sig_data = sig["data"]
                    step = max(1, len(sig_data) // 40)
                    sig["data"] = sig_data[::step]
                    sig["downsampled"] = True
                    sig["original_length"] = len(sig_data)
            full_result = {
                **engine_result,
                "video": {
                    "frameIndex": result.get("frame_index", 0),
                    "measurement": cv_measurement,
                    "progress": video_cap.get(cv2.CAP_PROP_POS_FRAMES) / video_cap.get(cv2.CAP_PROP_FRAME_COUNT) if video_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
                },
                "detection": result.get("detection", {}),
                "temporal": result.get("temporal", {}),
                "diagnostics": result.get("diagnostics", {}),
                "evidence": result.get("evidence", {}),
                "signals": signals,
                "candidates": result.get("detection", {}).get("candidates", []),
                "risk": effective_risk,
                "risk_confidence": effective_risk_confidence,
                "risk_blocked": 'risk_level' in blocked,
                # Top-level measurement for frontend compatibility
                "measurement": cv_measurement,
            }
            await broadcast(full_result)
        await asyncio.sleep(simulator.interval_ms / 1000)


connected_websockets = set()


async def ws_handler(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        connected_websockets.discard(websocket)


app.router.routes.append(WebSocketRoute("/ws", ws_handler))


async def broadcast(data):
    payload = {"type": "reading", "data": data}
    for ws in connected_websockets.copy():
        try:
            await ws.send_json(payload)
        except Exception:
            connected_websockets.discard(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
