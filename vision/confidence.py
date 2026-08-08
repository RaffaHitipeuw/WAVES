"""
Confidence Calculator Module

Calculates confidence scores based on multiple quality signals.

Confidence is NOT arbitrary. It reflects:
1. Detection quality (from waterline detector)
2. Temporal consistency (from temporal buffer)
3. Detection stability (consecutive valid/invalid)
4. Physical plausibility (no impossible jumps)
"""

import numpy as np
from typing import Dict, Any, Optional


class ConfidenceCalculator:
    """
    Calculates confidence scores from multiple signals.

    The confidence score (0-1) represents how trustworthy
    the current measurement is.
    """

    def __init__(
        self,
        detection_weight: float = 0.35,
        temporal_weight: float = 0.35,
        stability_weight: float = 0.20,
        plausibility_weight: float = 0.10
    ):
        """
        Initialize confidence calculator.

        Args:
            detection_weight: Weight for detection quality
            temporal_weight: Weight for temporal consistency
            stability_weight: Weight for detection stability
            plausibility_weight: Weight for physical plausibility
        """
        self.detection_weight = detection_weight
        self.temporal_weight = temporal_weight
        self.stability_weight = stability_weight
        self.plausibility_weight = plausibility_weight

        # Track for plausibility check
        self._previous_level: Optional[float] = None
        self._max_plausible_rate = 20.0  # cm per second

    def calculate(
        self,
        detection_result: Optional[Dict],
        temporal_state: Dict,
        measurement_result: Dict,
        previous_confidence: Optional[float] = None
    ) -> float:
        """
        Calculate overall confidence.

        Args:
            detection_result: Result from waterline detector
            temporal_state: Current temporal state
            measurement_result: Current measurement result
            previous_confidence: Previous confidence (for temporal smoothing)

        Returns:
            Confidence score 0-1
        """
        # Detection quality
        detection_conf = self._detection_confidence(detection_result)

        # Temporal consistency
        temporal_conf = self._temporal_confidence(temporal_state)

        # Stability (consecutive valid detections)
        stability_conf = self._stability_confidence(temporal_state)

        # Physical plausibility
        plausibility_conf = self._plausibility_confidence(
            measurement_result,
            previous_confidence
        )

        # Weighted combination
        confidence = (
            detection_conf * self.detection_weight +
            temporal_conf * self.temporal_weight +
            stability_conf * self.stability_weight +
            plausibility_conf * self.plausibility_weight
        )

        # Smooth over time (prevents jarring changes)
        if previous_confidence is not None:
            confidence = 0.7 * confidence + 0.3 * previous_confidence

        # Clamp to 0-1
        return max(0.0, min(1.0, confidence))

    def _detection_confidence(self, result: Optional[Dict]) -> float:
        """Confidence from detection quality."""
        if result is None or not result.get('detected'):
            return 0.0

        # Use detector's confidence
        base_conf = result.get('confidence', 0)

        # Quality score affects confidence
        quality = result.get('quality_score', 0.5)

        # Combine
        return base_conf * 0.7 + quality * 0.3

    def _temporal_confidence(self, state: Dict) -> float:
        """Confidence from temporal analysis."""
        if not state:
            return 0.0

        # Trend quality affects confidence
        trend = state.get('trend', 'NO_DETECTION')

        trend_confidence = {
            'STABLE': 0.9,
            'RISING': 0.8,
            'FALLING': 0.8,
            'RISING_FAST': 0.6,
            'FALLING_FAST': 0.6,
            'UNCERTAIN': 0.3,
            'NO_DETECTION': 0.0
        }.get(trend, 0.0)

        # Rate plausibility affects confidence
        rate = state.get('rate_of_change')
        if rate is not None:
            # Normalize rate to 0-1 (0-20 cm/s is normal)
            rate_normalized = 1.0 - min(1.0, abs(rate) / 30.0)
            return (trend_confidence * 0.7 + rate_normalized * 0.3)

        return trend_confidence * 0.8  # No rate, reduce confidence

    def _stability_confidence(self, state: Dict) -> float:
        """Confidence from detection stability."""
        valid = state.get('valid_detections', 0)
        invalid = state.get('invalid_detections', 0)

        if valid == 0 and invalid > 0:
            return 0.0

        if invalid > 0:
            # Recently invalid, low confidence
            return max(0.1, 1.0 - invalid * 0.15)

        # More consecutive valid = higher confidence
        # Cap at 10 valid detections
        stability = min(1.0, valid / 10.0)

        return stability

    def _plausibility_confidence(
        self,
        measurement: Dict,
        previous: Optional[float]
    ) -> float:
        """Confidence from physical plausibility."""
        if previous is None:
            return 0.5  # No previous, medium confidence

        level = measurement.get('waterLevel')
        if level is None:
            return 0.3  # No level, low confidence

        if self._previous_level is None:
            self._previous_level = level
            return 0.7

        # Check for impossible changes
        delta = abs(level - self._previous_level)

        # Update tracking
        self._previous_level = level

        # Rate check
        # This is already handled in measurement, so give benefit of doubt
        if delta > 50:
            return 0.2  # Very suspicious
        elif delta > 20:
            return 0.5  # Unusual but possible
        else:
            return 0.9  # Normal

    def reset(self):
        """Reset internal state."""
        self._previous_level = None

    def get_info(self) -> Dict[str, Any]:
        """Get configuration info."""
        return {
            'weights': {
                'detection': self.detection_weight,
                'temporal': self.temporal_weight,
                'stability': self.stability_weight,
                'plausibility': self.plausibility_weight
            },
            'maxPlausibleRate': self._max_plausible_rate
        }
