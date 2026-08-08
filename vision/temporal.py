import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque


@dataclass
class TemporalState:
    waterline_y: Optional[float]
    raw_waterline_y: Optional[float]
    smoothed_waterline_y: Optional[float]
    rate_of_change: Optional[float]
    trend: str
    confidence: float
    valid_detections: int
    invalid_detections: int
    detection_history: List[bool]


class TemporalBuffer:
    def __init__(
        self,
        max_history: int = 30,
        smoothing_window: int = 5,
        rate_window: int = 10,
        invalid_threshold: int = 5,
        rate_confidence_threshold: int = 5
    ):
        self.max_history = max_history
        self.smoothing_window = smoothing_window
        self.rate_window = rate_window
        self.invalid_threshold = invalid_threshold
        self.rate_confidence_threshold = rate_confidence_threshold
        self._timestamps: deque = deque(maxlen=max_history)
        self._waterline_y: deque = deque(maxlen=max_history)
        self._detections: deque = deque(maxlen=max_history)
        self._confidences: deque = deque(maxlen=max_history)

    def add(
        self,
        timestamp: float,
        waterline_y: Optional[float],
        confidence: float = 1.0
    ) -> TemporalState:
        self._timestamps.append(timestamp)
        self._waterline_y.append(waterline_y)
        self._detections.append(waterline_y is not None)
        self._confidences.append(confidence)
        return self.get_state()

    def get_state(self) -> TemporalState:
        if len(self._waterline_y) == 0:
            return TemporalState(
                waterline_y=None,
                raw_waterline_y=None,
                smoothed_waterline_y=None,
                rate_of_change=None,
                trend='NO_DETECTION',
                confidence=0.0,
                valid_detections=0,
                invalid_detections=0,
                detection_history=list(self._detections)
            )
        raw_waterline_y = self._waterline_y[-1]
        valid_detections = 0
        invalid_detections = 0
        for detection in reversed(self._detections):
            if detection:
                valid_detections += 1
            else:
                break
        for i, detection in enumerate(self._detections):
            if not detection:
                invalid_detections = len(self._detections) - i
                break
        smoothed_waterline_y = self._calculate_smoothed()
        rate_of_change = self._calculate_rate()
        trend = self._determine_trend(rate_of_change, valid_detections)
        confidence = self._calculate_confidence(valid_detections, invalid_detections)
        return TemporalState(
            waterline_y=smoothed_waterline_y,
            raw_waterline_y=raw_waterline_y,
            smoothed_waterline_y=smoothed_waterline_y,
            rate_of_change=rate_of_change,
            trend=trend,
            confidence=confidence,
            valid_detections=valid_detections,
            invalid_detections=invalid_detections,
            detection_history=list(self._detections)
        )

    def _calculate_smoothed(self) -> Optional[float]:
        valid_readings = [
            (y, c)
            for y, c in zip(self._waterline_y, self._confidences)
            if y is not None
        ]
        if not valid_readings:
            return None
        recent = valid_readings[-self.smoothing_window:]
        if not recent:
            return None
        total_weight = sum(c for _, c in recent)
        if total_weight <= 0:
            return recent[-1][0] if recent else None
        weighted_sum = sum(y * c for y, c in recent)
        return weighted_sum / total_weight

    def _calculate_rate(self) -> Optional[float]:
        valid_readings = [
            (t, y)
            for t, y in zip(self._timestamps, self._waterline_y)
            if y is not None
        ]
        if len(valid_readings) < self.rate_confidence_threshold:
            return None
        recent = valid_readings[-self.rate_window:]
        if len(recent) < 2:
            return None
        times = np.array([t for t, _ in recent])
        values = np.array([y for _, y in recent])
        if times[-1] - times[0] <= 0:
            return None
        try:
            slope, _ = np.polyfit(times, values, 1)
            return float(slope)
        except:
            return None

    def _determine_trend(
        self,
        rate: Optional[float],
        valid_detections: int
    ) -> str:
        if valid_detections < 3:
            if valid_detections == 0:
                return 'NO_DETECTION'
            return 'UNCERTAIN'
        if rate is None:
            return 'UNCERTAIN'
        abs_rate = abs(rate)
        if abs_rate < 1.0:
            return 'STABLE'
        elif rate > 0:
            if abs_rate > 10:
                return 'RISING_FAST'
            return 'RISING'
        else:
            if abs_rate > 10:
                return 'FALLING_FAST'
            return 'FALLING'

    def _calculate_confidence(
        self,
        valid_detections: int,
        invalid_detections: int
    ) -> float:
        if valid_detections >= self.rate_confidence_threshold:
            base_confidence = 0.8
        elif valid_detections >= 2:
            base_confidence = 0.5
        elif valid_detections == 1:
            base_confidence = 0.3
        else:
            base_confidence = 0.0
        if invalid_detections > 0:
            reduction = min(0.3, invalid_detections * 0.1)
            base_confidence = max(0.0, base_confidence - reduction)
        return base_confidence

    def reset(self):
        self._timestamps.clear()
        self._waterline_y.clear()
        self._detections.clear()
        self._confidences.clear()

    @property
    def has_data(self) -> bool:
        return len(self._waterline_y) > 0

    @property
    def detection_rate(self) -> float:
        if not self._detections:
            return 0.0
        return sum(1 for d in self._detections if d) / len(self._detections)
