"""
Measurement Processing Module

Converts detection results into meaningful measurements.

Output:
{
    "waterLevel": float | null,       # Estimated water level (cm or relative)
    "pixelWaterline": float | null,     # Raw pixel position
    "confidence": float,               # 0-1 confidence
    "measurementStatus": str,          # Status code
    "details": {...}                   # Additional details
}

Measurement Status Codes:
- NO_VALID_WATERLINE: No waterline detected
- LOW_CONFIDENCE: Detection exists but confidence low
- CALIBRATING: Establishing baseline
- VALID: Valid measurement
- UNCERTAIN: Measurement may be unreliable
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class MeasurementResult:
    """Result of a measurement."""
    water_level: Optional[float]
    pixel_waterline: Optional[float]
    confidence: float
    measurement_status: str
    is_valid: bool
    details: Dict[str, Any]


class MeasurementProcessor:
    """
    Processes raw detections into measurements.

    Combines:
    - Waterline detection results
    - Temporal smoothing
    - Calibration
    - Confidence assessment
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.3,
        max_pixel_rate: float = 50.0  # Max plausible pixel movement per second
    ):
        """
        Initialize measurement processor.

        Args:
            min_confidence_threshold: Minimum confidence for valid measurement
            max_pixel_rate: Maximum plausible pixel rate (for outlier rejection)
        """
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
        """
        Process raw detection into measurement.

        Args:
            raw_detection: Result from waterline detector
            temporal_state: Current state from temporal buffer
            calibration_result: Result from calibration
            frame_index: Current frame index
            timestamp: Current timestamp

        Returns:
            MeasurementResult with processed measurement
        """
        # Determine measurement status
        status, is_valid = self._determine_status(
            raw_detection,
            temporal_state,
            calibration_result
        )

        # Extract values
        pixel_waterline = temporal_state.get('raw_waterline_y')
        smoothed_waterline = temporal_state.get('waterline_y')

        # Use smoothed value for measurement
        if smoothed_waterline is not None:
            measurement_pixel = smoothed_waterline
        elif pixel_waterline is not None:
            measurement_pixel = pixel_waterline
        else:
            measurement_pixel = None

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            raw_detection,
            temporal_state,
            is_valid
        )

        # Get water level from calibration
        water_level = calibration_result.get('waterLevel')

        # Build details
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
        """
        Determine measurement status.

        Returns:
            Tuple of (status_code, is_valid)
        """
        # No detection
        if not raw_detection or not raw_detection.get('detected'):
            return 'NO_VALID_WATERLINE', False

        # Check detection confidence
        detection_confidence = raw_detection.get('confidence', 0)
        if detection_confidence < self.min_confidence:
            return 'LOW_CONFIDENCE', False

        # Check temporal confidence
        temporal_confidence = temporal_state.get('confidence', 0)
        if temporal_confidence < self.min_confidence:
            return 'UNCERTAIN', False

        # Check for rapid changes (potential errors)
        rate = temporal_state.get('rate_of_change')
        if rate is not None and abs(rate) > self.max_pixel_rate:
            return 'UNCERTAIN', False

        # Check calibration status
        calibration_status = calibration_result.get('status', '')

        if 'CALIBRATING' in calibration_status or 'SAMPLING' in calibration_status:
            return 'CALIBRATING', False

        if not calibration_result.get('calibrated', False):
            return 'UNCERTAIN', False

        # All checks passed
        return 'VALID', True

    def _calculate_confidence(
        self,
        raw_detection: Optional[Dict],
        temporal_state: Dict,
        is_valid: bool
    ) -> float:
        """
        Calculate overall measurement confidence.

        Combines:
        - Detection confidence
        - Temporal confidence
        - Validity
        """
        if not is_valid:
            return 0.0

        # Detection confidence
        detection_conf = raw_detection.get('confidence', 0) if raw_detection else 0

        # Temporal confidence
        temporal_conf = temporal_state.get('confidence', 0)

        # Combine with weights
        # Detection: 40%, Temporal: 40%, Base: 20%
        confidence = (
            detection_conf * 0.4 +
            temporal_conf * 0.4 +
            0.2
        )

        return min(1.0, max(0.0, confidence))

    def validate_measurement(
        self,
        measurement: MeasurementResult,
        previous_measurement: Optional[MeasurementResult]
    ) -> MeasurementResult:
        """
        Validate measurement against previous measurement.

        Detects sudden jumps or impossible changes.

        Args:
            measurement: Current measurement
            previous_measurement: Previous measurement

        Returns:
            Validated measurement (may modify status if invalid)
        """
        if previous_measurement is None:
            return measurement

        if not measurement.is_valid:
            return measurement

        if measurement.water_level is None or previous_measurement.water_level is None:
            return measurement

        # Check for sudden jump
        delta = abs(measurement.water_level - previous_measurement.water_level)

        # Sanity check: water level shouldn't change by more than 20cm in 1 second
        # (unless there's an obvious calibration issue)
        if delta > 20:
            measurement.measurement_status = 'SUDDEN_CHANGE'
            measurement.is_valid = False
            measurement.details['warning'] = f'Sudden change: {delta:.1f}cm'

        return measurement
