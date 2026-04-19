"""
app/server.py
-------------
FastAPI backend:
  - Serves live dashboard at http://localhost:8080/
  - WebSocket at ws://localhost:8080/ws  (panic scores, sensor data, ML scores)
  - REST API for status, episodes, override, baseline, contacts
"""

import asyncio
import time
import json
import os
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from loguru import logger

from core.event_bus import EventBus, Topics
from core.state_machine import PanicState


app = FastAPI(title="PanicGuard Companion API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bus: EventBus | None = None
_episode_log: list[dict] = []
_live_data: dict = {}
_ws_clients: set[WebSocket] = set()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def init_server(bus: EventBus):
    global _bus
    _bus = bus
    bus.subscribe(Topics.PANIC_SCORE,      _on_score)
    bus.subscribe(Topics.PANIC_STATE,      _on_state)
    bus.subscribe(Topics.PREDICTION_SCORE, _on_prediction_score)
    bus.subscribe(Topics.ANOMALY_SCORE,    _on_anomaly_score)
    bus.subscribe(Topics.FUSION_RISK,      _on_fusion_risk)
    bus.subscribe("sensor.*",              _on_sensor)


# ── Event handlers ────────────────────────────────────────────────────────────

async def _on_score(topic: str, score: float):
    _live_data["panic_score"] = score
    _live_data["score_ts"] = time.time()
    await _broadcast({"type": "panic_score", "score": score})


async def _on_state(topic: str, state: PanicState):
    _live_data["state"] = state.value
    if state in (PanicState.EPISODE, PanicState.SOS):
        entry = {"ts": time.time(), "state": state.value,
                 "score": _live_data.get("panic_score", 0)}
        _episode_log.append(entry)
        _episode_log[:] = _episode_log[-500:]
        await _broadcast({"type": "episode", **entry})
    await _broadcast({"type": "state_change", "state": state.value})


async def _on_prediction_score(topic: str, score: float):
    _live_data["lstm_score"] = score
    await _broadcast({"type": "ml_scores", "lstm": score,
                      "anomaly": _live_data.get("anomaly_score", 0)})


async def _on_anomaly_score(topic: str, score: float):
    _live_data["anomaly_score"] = score
    await _broadcast({"type": "ml_scores",
                      "lstm": _live_data.get("lstm_score", 0), "anomaly": score})


async def _on_fusion_risk(topic: str, payload: dict):
    _live_data["fusion"] = payload
    await _broadcast({"type": "fusion_risk", **payload})


async def _on_sensor(topic: str, sample: Any):
    key = topic.replace("sensor.", "")
    d = sample.__dict__ if hasattr(sample, "__dict__") else {"value": sample}
    _live_data[key] = d
    await _broadcast({"type": "sensor", "sensor": key, "data": d})


async def _broadcast(msg: dict):
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(msg, default=str))
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard():
    path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(path):
        return FileResponse(path, media_type="text/html")
    return {"message": "Dashboard not found. Run from project root."}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_feed(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    logger.info(f"[API] WebSocket connected ({len(_ws_clients)} clients)")
    try:
        await ws.send_text(json.dumps({"type": "init", "data": _live_data}, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        logger.info(f"[API] WebSocket disconnected ({len(_ws_clients)} clients)")


# ── REST API ──────────────────────────────────────────────────────────────────

@app.get("/status")
async def get_status():
    return {
        "state":             _live_data.get("state", "idle"),
        "panic_score":       _live_data.get("panic_score", 0.0),
        "lstm_score":        _live_data.get("lstm_score", 0.0),
        "anomaly_score":     _live_data.get("anomaly_score", 0.0),
        "fusion":            _live_data.get("fusion", {}),
        "connected_sensors": [k for k in _live_data if k in ("ppg", "gsr", "respiration")],
        "uptime":            time.time(),
    }


@app.get("/episodes")
async def get_episodes(limit: int = 50):
    return {"episodes": _episode_log[-limit:], "total": len(_episode_log)}


class OverrideRequest(BaseModel):
    action: str

@app.post("/override")
async def override(req: OverrideRequest):
    if _bus is None:
        raise HTTPException(503, "Event bus not initialised")
    if req.action == "cancel":
        await _bus.publish(Topics.USER_CANCEL)
    elif req.action == "trigger_sos":
        await _bus.publish(Topics.SOS_CMD, {"action": "escalate"})
    else:
        raise HTTPException(400, f"Unknown action: {req.action}")
    return {"ok": True}


class BaselineUpdate(BaseModel):
    hr_resting:    float | None = None
    rmssd_resting: float | None = None
    scl_resting:   float | None = None
    temp_baseline: float | None = None

@app.post("/baseline")
async def update_baseline(update: BaselineUpdate):
    if _bus is None:
        raise HTTPException(503, "Event bus not initialised")
    changes = {k: v for k, v in update.model_dump().items() if v is not None}
    await _bus.publish("system.baseline_update", changes)
    return {"ok": True, "updated": changes}


class ContactConfig(BaseModel):
    name: str
    phone: str | None = None
    ntfy_topic: str | None = None

@app.post("/contacts")
async def add_contact(contact: ContactConfig):
    if _bus is None:
        raise HTTPException(503, "Event bus not initialised")
    await _bus.publish("system.add_contact", contact.model_dump())
    return {"ok": True}


@app.post("/voice-recording")
async def upload_voice_recording(name: str = "calm_voice", file: UploadFile = File(...)):
    import aiofiles
    os.makedirs("assets/audio", exist_ok=True)
    path = f"assets/audio/{name}.wav"
    async with aiofiles.open(path, "wb") as f:
        await f.write(await file.read())
    if _bus:
        await _bus.publish("system.voice_recording", {"name": name, "path": path})
    return {"ok": True, "path": path}


@app.get("/features/latest")
async def get_latest_features():
    return _live_data.get("features", {"message": "No features computed yet"})


@app.delete("/episodes")
async def clear_episodes():
    _episode_log.clear()
    return {"ok": True}
