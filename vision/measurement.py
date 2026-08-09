from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class MeasurementResult:
    water_level: Optional[float]
    pixel_waterline: Optional[float]
    confidence: float
    measurement_status: str
    is_valid: bool
    details: Dict[str, Any]


class MeasurementProcessor:
    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        max_pixel_rate: float = 50.0
    ):
        self.min_confidence = min_confidence_threshold
        self.max_pixel_rate = max_pixel_rate

    def process(
        self,
        raw_detection: Optional[Dict],
        temporal_state: Dict,
        calibration_result: Dict,
        frame_index: int,
        timestamp: float
    ) -> MeasurementResult:
        status, is_valid = self._determine_status(
            raw_detection,
            temporal_state,
            calibration_result
        )
        pixel_waterline = temporal_state.get('raw_waterline_y')
        smoothed_waterline = temporal_state.get('waterline_y')
        if smoothed_waterline is not None:
            measurement_pixel = smoothed_waterline
        elif pixel_waterline is not None:
            measurement_pixel = pixel_waterline
        else:
            measurement_pixel = None
        confidence = self._calculate_confidence(
            raw_detection,
            temporal_state,
            is_valid
        )
        water_level = calibration_result.get('waterLevel')
        details = {
            'frameIndex': frame_index,
            'timestamp': timestamp,
            'pixelWaterline': pixel_waterline,
            'smoothedWaterline': smoothed_waterline,
            'temporalTrend': temporal_state.get('trend', 'UNKNOWN'),
            'rateOfChange': temporal_state.get('rate_of_change'),
            'validDetections': temporal_state.get('valid_detections', 0),
            'invalidDetections': temporal_state.get('invalid_detections', 0),
            'calibrationMethod': calibration_result.get('calibrationMethod', 'unknown'),
            'detectionConfidence': raw_detection.get('confidence', 0) if raw_detection else 0,
            'detectionMethod': raw_detection.get('method', 'none') if raw_detection else 'none',
            'calibrationStatus': calibration_result.get('status', 'unknown'),
            **calibration_result
        }
        return MeasurementResult(
            water_level=water_level,
            pixel_waterline=measurement_pixel,
            confidence=confidence,
            measurement_status=status,
            is_valid=is_valid,
            details=details
        )

    def _determine_status(
        self,
        raw_detection: Optional[Dict],
        temporal_state: Dict,
        calibration_result: Dict
    ) -> tuple:
        if not raw_detection or not raw_detection.get('detected'):
            return 'NO_VALID_WATERLINE', False
        detection_confidence = raw_detection.get('confidence', 0)
        if detection_confidence < self.min_confidence:
            return 'LOW_CONFIDENCE', False
        temporal_confidence = temporal_state.get('confidence', 0)
        if temporal_confidence < self.min_confidence:
            return 'UNCERTAIN', False
        rate = temporal_state.get('rate_of_change')
        if rate is not None and abs(rate) > self.max_pixel_rate:
            return 'UNCERTAIN', False
        calibration_status = calibration_result.get('status', '')
        if 'CALIBRATING' in calibration_status or 'SAMPLING' in calibration_status:
            return 'CALIBRATING', False
        if not calibration_result.get('calibrated', False):
            return 'UNCERTAIN', False
        return 'VALID', True

    def _calculate_confidence(
        self,
        raw_detection: Optional[Dict],
        temporal_state: Dict,
        is_valid: bool
    ) -> float:
        if not is_valid:
            return 0.0
        detection_conf = raw_detection.get('confidence', 0) if raw_detection else 0
        temporal_conf = temporal_state.get('confidence', 0)
        # Detection confidence is already counted in temporal_conf via the buffer.
        # Weight it at 0.35 here to avoid double-counting signal strength.
        confidence = detection_conf * 0.35 + temporal_conf * 0.35
        return min(0.85, max(0.0, confidence))
