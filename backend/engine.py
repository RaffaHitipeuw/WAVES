from typing import Callable, List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque

from .models import (
    WaterLevelReading,
    MonitoringNode,
    ProcessedReading,
    Alert,
    RiskLevel
)


@dataclass
class EngineState:
    node_id: str
    water_level: float = 0.0
    smoothed_level: float = 0.0
    rate_of_change: float = 0.0
    confidence: float = 0.0
    risk: RiskLevel = RiskLevel.SAFE
    last_reading_at: Optional[datetime] = None
    readings_processed: int = 0


class CoreEngine:
    def __init__(
        self,
        node: Optional[MonitoringNode] = None
    ):
        self.node = node or MonitoringNode(
            id="NODE-001",
            name="Primary Sensor",
            thresholds={
                "watch": 30,
                "warning": 50,
                "critical": 70
            }
        )
        self.state = EngineState(
            node_id=self.node.id
        )
        self.buffer_size = 10
        self.reading_buffer: deque = deque(
            maxlen=self.buffer_size
        )
        self._processing_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []
        self._previous_level: Optional[float] = None
        self._previous_timestamp: Optional[datetime] = None

    def on_processed(
        self,
        callback: Callable
    ) -> None:
        self._processing_callbacks.append(callback)

    def on_alert(
        self,
        callback: Callable
    ) -> None:
        self._alert_callbacks.append(callback)

    def _emit_processed(
        self,
        result: Dict
    ) -> None:
        for callback in self._processing_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"[Engine] Callback error: {e}")

    def _emit_alert(
        self,
        alert: Alert
    ) -> None:
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"[Engine] Alert callback error: {e}")

    def process(
        self,
        reading: WaterLevelReading,
        cv_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        self.state.readings_processed += 1
        self.state.last_reading_at = datetime.now()
        self.state.water_level = reading.water_level
        self.reading_buffer.append(reading)
        smoothed = self._calculate_smoothed_level()
        rate = self._calculate_rate_of_change(reading)
        self.state.smoothed_level = smoothed
        self.state.rate_of_change = rate
        risk = self.determine_risk(reading.water_level, rate)
        self.state.risk = risk

        # Confidence: use CV pipeline value if available (video mode),
        # otherwise derive from buffer fill (simulator mode).
        # Simulator cap at 0.8 — buffer fill is internal consistency,
        # not evidential reliability about external truth.
        if cv_confidence is not None:
            self.state.confidence = round(cv_confidence, 3)
        else:
            buf_size = len(self.reading_buffer)
            self.state.confidence = round(min(buf_size / 10.0, 0.8), 3)

        result = {
            "reading": reading.to_dict(),
            "processed": {
                "rawWaterLevel": reading.water_level,
                "smoothedWaterLevel": smoothed,
                "rateOfChange": rate,
                "confidence": self.state.confidence,
                "risk": self.state.risk.value,
                "processedAt": datetime.now().isoformat()
            },
            "node": self.node.to_dict(),
            "state": self.get_state()
        }
        print(
            f"[Flood Engine] Reading #{self.state.readings_processed}: "
            f"{reading.water_level} cm | Risk: {risk.value} | Conf: {self.state.confidence:.2f}"
        )
        self._emit_processed(result)
        return result

    def _calculate_smoothed_level(self) -> float:
        if not self.reading_buffer:
            return self.state.water_level
        levels = [reading.water_level for reading in self.reading_buffer]
        return round(sum(levels) / len(levels), 2)

    def _calculate_rate_of_change(
        self,
        reading: WaterLevelReading
    ) -> float:
        if self._previous_level is None:
            self._previous_level = reading.water_level
            self._previous_timestamp = reading.timestamp
            return 0.0
        current_level = reading.water_level
        current_time = reading.timestamp
        time_diff = (current_time - self._previous_timestamp).total_seconds()
        # Guard against back-to-back calls: require at least 0.1s between readings
        # to compute a meaningful rate. Fast successive calls (same tick) produce
        # spurious large rates.
        if time_diff < 0.1:
            self._previous_level = current_level
            self._previous_timestamp = current_time
            return 0.0
        level_change = current_level - self._previous_level
        rate = (level_change / time_diff) * 60
        self._previous_level = current_level
        self._previous_timestamp = current_time
        return round(rate, 2)

    def get_state(self) -> Dict[str, Any]:
        return {
            "nodeId": self.state.node_id,
            "waterLevel": self.state.water_level,
            "smoothedLevel": self.state.smoothed_level,
            "rateOfChange": self.state.rate_of_change,
            "confidence": self.state.confidence,
            "risk": self.state.risk.value,
            "lastReadingAt": (
                self.state.last_reading_at.isoformat()
                if self.state.last_reading_at else None
            ),
            "readingsProcessed": self.state.readings_processed,
            "bufferSize": len(self.reading_buffer)
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "readings_processed": self.state.readings_processed,
            "last_processed_at": (
                self.state.last_reading_at.isoformat()
                if self.state.last_reading_at else None
            ),
            "buffer_size": len(self.reading_buffer)
        }


class FloodEngine(CoreEngine):
    def __init__(
        self,
        node: Optional[MonitoringNode] = None
    ):
        super().__init__(node)

    def determine_risk(
        self,
        water_level: float,
        rate: float
    ) -> RiskLevel:
        """
        Rate-unit: cm/min (from _calculate_rate_of_change).
        If rate is None or 0, fall back to level-only determination.
        """
        thresholds = self.node.thresholds

        # Rate thresholds in cm/min
        HIGH_RATE = 5.0   # cm/min — rapid rise warrants escalation
        MOD_RATE = 2.0    # cm/min — moderate rise

        # Level-only escalation
        if water_level >= thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif water_level >= thresholds["warning"]:
            # Rate can escalate or de-escalate WARNING
            if rate is not None and abs(rate) >= HIGH_RATE:
                return RiskLevel.CRITICAL  # rapid rise
            return RiskLevel.WARNING
        elif water_level >= thresholds["watch"]:
            if rate is not None and rate >= HIGH_RATE:
                return RiskLevel.WARNING   # early escalation for rapid rise
            return RiskLevel.WATCH

        # Below watch threshold
        if rate is not None and rate >= HIGH_RATE * 1.5:
            return RiskLevel.WATCH  # watch for potential escalation

        return RiskLevel.SAFE

    def create_alert(
        self,
        risk: RiskLevel,
        confidence: float
    ) -> Alert:
        messages = {
            RiskLevel.SAFE: "Water level is within normal range.",
            RiskLevel.WATCH: "Water level is increasing. Monitoring intensified.",
            RiskLevel.WARNING: "Rapid water-level increase detected. Prepare for potential flooding.",
            RiskLevel.CRITICAL: "Critical water level detected. Immediate local warning triggered."
        }
        titles = {
            RiskLevel.SAFE: "Normal",
            RiskLevel.WATCH: "Watch",
            RiskLevel.WARNING: "Warning",
            RiskLevel.CRITICAL: "Critical"
        }
        return Alert(
            severity=risk,
            title=titles[risk],
            message=messages[risk],
            node_id=self.node.id,
            confidence=confidence,
            water_level=self.state.water_level
        )
