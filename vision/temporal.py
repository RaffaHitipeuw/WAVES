"""
Temporal Buffer Module

Maintains a history of waterline detections and provides temporal smoothing.

Features:
- Bounded history (configurable max length)
- Weighted averaging (recent readings more important)
- Detection state tracking
- Trend analysis (rising, falling, stable, uncertain)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from collections import deque


@dataclass
class TemporalState:
    """Current state of temporal analysis."""
    waterline_y: Optional[float]  # Smoothed waterline position
    raw_waterline_y: Optional[float]  # Latest raw detection
    smoothed_waterline_y: Optional[float]  # Smoothed value
    rate_of_change: Optional[float]  # Pixels per second
    trend: str  # 'RISING', 'FALLING', 'STABLE', 'UNCERTAIN', 'NO_DETECTION'
    confidence: float  # 0-1 based on detection quality
    valid_detections: int  # Number of consecutive valid detections
    invalid_detections: int  # Number of consecutive invalid detections
    detection_history: List[bool]  # Recent detection status


class TemporalBuffer:
    """
    Maintains temporal history of waterline detections.

    Provides:
    - Smoothing of noisy detections
    - Rate of rise/fall calculation
    - Trend detection
    - Confidence based on detection consistency
    """

    def __init__(
        self,
        max_history: int = 30,
        smoothing_window: int = 5,
        rate_window: int = 10,
        invalid_threshold: int = 5,
        rate_confidence_threshold: int = 5
    ):
        """
        Initialize temporal buffer.

        Args:
            max_history: Maximum number of readings to keep
            smoothing_window: Window for moving average
            rate_window: Window for rate calculation
            invalid_threshold: Consecutive invalid detections before marking uncertain
            rate_confidence_threshold: Minimum valid detections for rate calculation
        """
        self.max_history = max_history
        self.smoothing_window = smoothing_window
        self.rate_window = rate_window
        self.invalid_threshold = invalid_threshold
        self.rate_confidence_threshold = rate_confidence_threshold

        # History storage
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
        """
        Add a new detection to the buffer.

        Args:
            timestamp: Time of detection (seconds)
            waterline_y: Y coordinate of detected waterline (None if no detection)
            confidence: Detection confidence (0-1)

        Returns:
            Current temporal state
        """
        # Store in history
        self._timestamps.append(timestamp)
        self._waterline_y.append(waterline_y)
        self._detections.append(waterline_y is not None)
        self._confidences.append(confidence)

        return self.get_state()

    def get_state(self) -> TemporalState:
        """
        Get current temporal state.

        Returns:
            TemporalState with smoothed values and trend
        """
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

        # Raw value (most recent)
        raw_waterline_y = self._waterline_y[-1]

        # Count consecutive valid/invalid detections
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

        # Calculate smoothed value
        smoothed_waterline_y = self._calculate_smoothed()

        # Calculate rate of change
        rate_of_change = self._calculate_rate()

        # Determine trend
        trend = self._determine_trend(rate_of_change, valid_detections)

        # Calculate overall confidence
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
        """Calculate smoothed waterline using weighted moving average."""
        # Get recent valid detections
        valid_readings = [
            (y, c)
            for y, c in zip(self._waterline_y, self._confidences)
            if y is not None
        ]

        if not valid_readings:
            return None

        # Limit to smoothing window
        recent = valid_readings[-self.smoothing_window:]

        if not recent:
            return None

        # Weighted average (higher confidence = higher weight)
        total_weight = sum(c for _, c in recent)
        if total_weight <= 0:
            return recent[-1][0] if recent else None

        weighted_sum = sum(y * c for y, c in recent)
        return weighted_sum / total_weight

    def _calculate_rate(self) -> Optional[float]:
        """
        Calculate rate of change in pixels per second.

        Uses linear regression over recent valid detections.
        """
        # Get recent valid detections with timestamps
        valid_readings = [
            (t, y)
            for t, y in zip(self._timestamps, self._waterline_y)
            if y is not None
        ]

        # Need minimum detections for rate calculation
        if len(valid_readings) < self.rate_confidence_threshold:
            return None

        # Use limited window
        recent = valid_readings[-self.rate_window:]

        if len(recent) < 2:
            return None

        times = np.array([t for t, _ in recent])
        values = np.array([y for _, y in recent])

        # Simple linear regression
        # rate = dy/dt
        if times[-1] - times[0] <= 0:
            return None

        # Calculate slope using numpy polyfit
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
        """
        Determine trend from rate of change.

        Args:
            rate: Rate of change in pixels per second
            valid_detections: Number of consecutive valid detections

        Returns:
            Trend string
        """
        # Not enough valid detections
        if valid_detections < 3:
            if valid_detections == 0:
                return 'NO_DETECTION'
            return 'UNCERTAIN'

        # No rate calculated
        if rate is None:
            return 'UNCERTAIN'

        # Classify based on rate (pixels per second)
        # Positive rate = water rising (waterline moving down in image)
        # Negative rate = water falling (waterline moving up in image)
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
        """
        Calculate overall confidence.

        Based on:
        - Consecutive valid detections
        - Recent invalid detections
        """
        # High confidence if many recent valid detections
        if valid_detections >= self.rate_confidence_threshold:
            base_confidence = 0.8
        elif valid_detections >= 2:
            base_confidence = 0.5
        elif valid_detections == 1:
            base_confidence = 0.3
        else:
            base_confidence = 0.0

        # Reduce confidence if recent invalid detections
        if invalid_detections > 0:
            reduction = min(0.3, invalid_detections * 0.1)
            base_confidence = max(0.0, base_confidence - reduction)

        return base_confidence

    def reset(self):
        """Clear all history."""
        self._timestamps.clear()
        self._waterline_y.clear()
        self._detections.clear()
        self._confidences.clear()

    @property
    def has_data(self) -> bool:
        """Check if buffer has any data."""
        return len(self._waterline_y) > 0

    @property
    def detection_rate(self) -> float:
        """Detection success rate over history."""
        if not self._detections:
            return 0.0
        return sum(1 for d in self._detections if d) / len(self._detections)
