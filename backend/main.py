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


@app.post("/api/calibration/baseline")
async def set_calibration_baseline(request: dict):
    """
    Manual baseline calibration endpoint.
    Sets the calibration baseline to a specific pixel Y coordinate.
    This overrides the automatic baseline that may be locked on wrong features.

    Use this when the scene is DRY (no flooding) and you want to set
    the baseline reference point manually.

    Body: {"pixel_y": 800}  # dry reference Y from ROI
    """
    pixel_y = request.get("pixel_y")
    if pixel_y is None:
        raise HTTPException(status_code=400, detail="pixel_y required")

    try:
        pixel_y = int(pixel_y)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="pixel_y must be an integer")

    # Set baseline in both pipeline calibration AND detector
    if mode == "video" and pipeline is not None:
        pipeline.calibration.set_baseline(float(pixel_y))
        # Also reset calibration state so it uses the manual baseline
        pipeline.calibration._baseline_established = True
        pipeline.calibration._calibration_samples.clear()

        # Also update the detector's baseline expectation
        # Force re-calibration by resetting the calibration quality
        return {
            "status": "ok",
            "baseline_y": pixel_y,
            "calibration_method": "manual",
            "message": f"Baseline set to pixel Y={pixel_y}. Water level will be delta from this reference."
        }
    elif mode == "simulator":
        return {
            "status": "ok",
            "baseline_y": pixel_y,
            "mode": "simulator",
            "message": "Simulator mode — baseline not applicable"
        }
    else:
        raise HTTPException(status_code=400, detail="Pipeline not initialized")


async def process_loop():
    global processing, video_cap
    while processing:
        if mode == "simulator":
            reading = simulator._generate_reading()
            if reading:
                result = engine.process(reading)
                processed = result["processed"]
                rate = processed.get("rateOfChange", 0.0) or 0.0
                # Derive trend from rate (engine doesn't compute this)
                abs_rate = abs(rate)
                if abs_rate < 0.5:
                    trend = "STABLE"
                elif abs_rate < 3.0:
                    trend = "RISING" if rate > 0 else "FALLING"
                elif rate > 0:
                    trend = "RISING_FAST"
                else:
                    trend = "FALLING_FAST"
                simulator_diagnostics = {
                    "state": "SIMULATOR",
                    "reasons": [],
                    "permitted_inferences": ["simulated_water_level"],
                    "blocked_inferences": []
                }
                # Build complete broadcast matching frontend expectations
                full_result = {
                    **result,
                    # Frontend uses data.measurement — provide it from engine result
                    "measurement": {
                        "waterLevel": processed.get("rawWaterLevel"),
                        "smoothedLevel": processed.get("smoothedWaterLevel"),
                        "confidence": processed.get("confidence", 0.0),
                        "isValid": True,
                        "measurementStatus": "SIMULATOR",
                        "measurementValidity": "VALID",
                        "trend": trend,
                    },
                    "temporal": {
                        "trend": trend,
                        "rate_px_per_sec": None,
                        "rate_cm_per_min": rate,  # engine rate is already cm/min
                        "waterline_y": None,
                        "raw_waterline_y": None,
                        "valid_detections": 0,
                        "invalid_detections": 0,
                        "confidence": processed.get("confidence", 0.0),
                    },
                    "detection": {},  # van-mode guard: data.detection exists but empty
                    "evidence": {
                        "detection": processed.get("confidence", 0.0),
                        "temporal": processed.get("confidence", 0.0),
                        "stability": 1.0,
                        "calibration": 1.0,
                        "lighting": 1.0,
                        "plausibility": 1.0,
                    },
                    "signals": {},
                    "candidates": [],
                    "risk_confidence": processed.get("confidence", 0.0),
                    "diagnostics": simulator_diagnostics,
                    "rateCmPerMin": rate,
                    "absoluteDepthStatus": "SIMULATOR",
                    "video": None,
                }
                await broadcast(full_result)
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

            # Base risk from pipeline — gated by measurement confidence.
            # This is what the pipeline decided given current level + rate + evidence quality.
            effective_risk = result.get("risk", "SAFE")
            effective_risk_confidence = result.get("risk_confidence", 0.0)

            # P1 FIX: Prediction-based risk override.
            # Even if current measurement confidence is low, if the 5-minute forecast
            # crosses a threshold, escalate risk. This is the actual "early warning" logic.
            # The pipeline computes ETA; we use it here to override risk level.
            prediction = result.get("prediction")
            if prediction and prediction.get("predictedLevel5min") is not None:
                pred = prediction["predictedLevel5min"]
                # Override risk if forecast crosses a higher severity threshold.
                # rate can't always catch sudden rises, but forecast can.
                if pred >= 70.0:
                    effective_risk = "CRITICAL"
                    effective_risk_confidence = max(effective_risk_confidence, 0.3)
                elif pred >= 50.0 and effective_risk in ["SAFE", "WATCH"]:
                    effective_risk = "WARNING"
                    effective_risk_confidence = max(effective_risk_confidence, 0.2)
                elif pred >= 30.0 and effective_risk == "SAFE":
                    effective_risk = "WATCH"
                    effective_risk_confidence = max(effective_risk_confidence, 0.15)

            # Blocked inference: can't trust the level — demote risk regardless.
            if 'risk_level' in blocked or 'water_level' in blocked:
                effective_risk = 'SAFE'
                effective_risk_confidence = 0.0

            # Only feed to engine if detection succeeded.
            # P1 FIX: When CV fails, pass confidence=0 to reset engine confidence.
            # Previously used `continue` which kept stale confidence in engine state.
            if cv_water_level is not None:
                water_reading = WaterLevelReading(
                    node_id="NODE-001",
                    water_level=cv_water_level,
                    source=DataSource.SENSOR
                )
                engine_result = engine.process(water_reading, cv_confidence=cv_confidence)
            else:
                # No detection — update engine with zero confidence so stale value doesn't persist.
                # This is not a "new reading" — it's a gap acknowledgment.
                engine_result = engine.process(
                    WaterLevelReading(node_id="NODE-001", water_level=engine.state.water_level, source=DataSource.SENSOR),
                    cv_confidence=0.0
                )
                # Sleep at video rate for no-detection frames too.
                fps = video_cap.get(cv2.CAP_PROP_FPS)
                if fps and fps > 0:
                    await asyncio.sleep(1.0 / fps)
                else:
                    await asyncio.sleep(0)
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
                # Use pipeline's physical rate (cm/min) — not engine's px/s
                # Pipeline converts px/s to cm/min using calibration pixels_per_cm
                "rateCmPerMin": result.get("temporal", {}).get("rate_cm_per_min"),
                # Absolute depth trust status
                "absoluteDepthStatus": result.get("diagnostics", {}).get("absolute_depth_status", "UNAVAILABLE"),
                # Actual prediction: ETA to thresholds + projected level
                "prediction": result.get("prediction"),
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
