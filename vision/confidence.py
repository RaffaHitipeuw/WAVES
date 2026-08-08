import numpy as np
from typing import Dict, Any, Optional


class ConfidenceCalculator:
    def __init__(
        self,
        detection_weight: float = 0.35,
        temporal_weight: float = 0.35,
        stability_weight: float = 0.20,
        plausibility_weight: float = 0.10
    ):
        self.detection_weight = detection_weight
        self.temporal_weight = temporal_weight
        self.stability_weight = stability_weight
        self.plausibility_weight = plausibility_weight
        self._previous_level: Optional[float] = None
        self._max_plausible_rate = 20.0

    def calculate(
        self,
        detection_result: Optional[Dict],
        temporal_state: Dict,
        measurement_result: Dict,
        previous_confidence: Optional[float] = None
    ) -> float:
        detection_conf = self._detection_confidence(detection_result)
        temporal_conf = self._temporal_confidence(temporal_state)
        stability_conf = self._stability_confidence(temporal_state)
        plausibility_conf = self._plausibility_confidence(
            measurement_result,
            previous_confidence
        )
        confidence = (
            detection_conf * self.detection_weight +
            temporal_conf * self.temporal_weight +
            stability_conf * self.stability_weight +
            plausibility_conf * self.plausibility_weight
        )
        if previous_confidence is not None:
            confidence = 0.7 * confidence + 0.3 * previous_confidence
        return max(0.0, min(1.0, confidence))

    def _detection_confidence(self, result: Optional[Dict]) -> float:
        if result is None or not result.get('detected'):
            return 0.0
        base_conf = result.get('confidence', 0)
        quality = result.get('quality_score', 0.5)
        return base_conf * 0.7 + quality * 0.3

    def _temporal_confidence(self, state: Dict) -> float:
        if not state:
            return 0.0
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
        rate = state.get('rate_of_change')
        if rate is not None:
            rate_normalized = 1.0 - min(1.0, abs(rate) / 30.0)
            return (trend_confidence * 0.7 + rate_normalized * 0.3)
        return trend_confidence * 0.8

    def _stability_confidence(self, state: Dict) -> float:
        valid = state.get('valid_detections', 0)
        invalid = state.get('invalid_detections', 0)
        if valid == 0 and invalid > 0:
            return 0.0
        if invalid > 0:
            return max(0.1, 1.0 - invalid * 0.15)
        stability = min(1.0, valid / 10.0)
        return stability

    def _plausibility_confidence(
        self,
        measurement: Dict,
        previous: Optional[float]
    ) -> float:
        if previous is None:
            return 0.5
        level = measurement.get('waterLevel')
        if level is None:
            return 0.3
        if self._previous_level is None:
            self._previous_level = level
            return 0.7
        delta = abs(level - self._previous_level)
        self._previous_level = level
        if delta > 50:
            return 0.2
        elif delta > 20:
            return 0.5
        else:
            return 0.9

    def reset(self):
        self._previous_level = None

    def get_info(self) -> Dict[str, Any]:
        return {
            'weights': {
                'detection': self.detection_weight,
                'temporal': self.temporal_weight,
                'stability': self.stability_weight,
                'plausibility': self.plausibility_weight
            },
            'maxPlausibleRate': self._max_plausible_rate
        }
