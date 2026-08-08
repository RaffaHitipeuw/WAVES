import cv2
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .waterline import WaterlineDetector, create_default_detector
from .temporal import TemporalBuffer
from .calibration import CalibrationModel, estimate_calibration_from_scene
from .measurement import MeasurementProcessor, MeasurementResult
from .confidence import ConfidenceCalculator


@dataclass
class PipelineResult:
    frame_index: int
    timestamp: float
    detection: Dict[str, Any]
    temporal: Dict[str, Any]
    measurement: Dict[str, Any]
    risk: str
    risk_confidence: float


class CVPipeline:
    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        roi_config: dict = None,
        calibration_config: dict = None
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.detector = create_default_detector(frame_width, frame_height, roi_config)

        calibration_cfg = calibration_config or {}
        self.calibration = CalibrationModel(
            config=estimate_calibration_from_scene(frame_height, calibration_cfg)
        )

        self.temporal = TemporalBuffer(
            max_history=30,
            smoothing_window=5,
            rate_window=10,
            invalid_threshold=5,
            rate_confidence_threshold=5
        )

        self.measurement = MeasurementProcessor(
            min_confidence_threshold=0.3,
            max_pixel_rate=50.0
        )

        self.confidence_calc = ConfidenceCalculator(
            detection_weight=0.35,
            temporal_weight=0.35,
            stability_weight=0.20,
            plausibility_weight=0.10
        )

        self._frame_count = 0
        self._start_time = None
        self._last_confidence = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None
    ) -> Dict[str, Any]:
        self._frame_count += 1
        if self._start_time is None:
            self._start_time = self._frame_count

        current_frame = frame_index or self._frame_count
        timestamp = current_frame / 30.0

        detection_result = self.detector.detect(frame)

        temporal_state = self.temporal.get_state()
        if detection_result.detected:
            temporal_state = self.temporal.add(
                timestamp=timestamp,
                waterline_y=detection_result.waterline_y,
                confidence=detection_result.confidence
            )
        else:
            self.temporal.add(
                timestamp=timestamp,
                waterline_y=None,
                confidence=0.0
            )
            temporal_state = self.temporal.get_state()

        waterline_y = temporal_state.get('waterline_y') or temporal_state.get('raw_waterline_y')
        if waterline_y is not None:
            calibration_result = self.calibration.calibrate(waterline_y, timestamp)
        else:
            calibration_result = {
                'waterLevel': None,
                'calibrationMethod': 'unknown',
                'confidence': 0.0,
                'calibrated': False
            }

        measurement_result = self.measurement.process(
            raw_detection=detection_result.raw_signal if detection_result.detected else None,
            temporal_state=temporal_state,
            calibration_result=calibration_result,
            frame_index=current_frame,
            timestamp=timestamp
        )

        measurement_result.is_valid = detection_result.detected
        measurement_result.measurement_status = detection_result.method.upper() if detection_result.detected else 'NO_DETECTION'
        if detection_result.detected:
            measurement_result.water_level = calibration_result.get('waterLevel')
            measurement_result.pixel_waterline = detection_result.waterline_y
            measurement_result.confidence = detection_result.confidence * 0.6 + temporal_state.get('confidence', 0) * 0.4

        risk, risk_confidence = self._determine_risk(measurement_result, temporal_state)

        detection_dict = {
            'detected': detection_result.detected,
            'waterline_y': detection_result.waterline_y,
            'confidence': detection_result.confidence,
            'method': detection_result.method,
            'quality_score': detection_result.quality_score
        }

        temporal_dict = {
            'waterline_y': temporal_state.waterline_y,
            'raw_waterline_y': temporal_state.raw_waterline_y,
            'trend': temporal_state.trend,
            'rate_of_change': temporal_state.rate_of_change,
            'confidence': temporal_state.confidence,
            'valid_detections': temporal_state.valid_detections
        }

        measurement_dict = {
            'waterLevel': measurement_result.water_level,
            'pixelWaterline': measurement_result.pixel_waterline,
            'confidence': measurement_result.confidence,
            'measurementStatus': measurement_result.measurement_status,
            'isValid': measurement_result.is_valid,
            'smoothedLevel': temporal_state.waterline_y
        }

        return {
            'frame_index': current_frame,
            'timestamp': timestamp,
            'detection': detection_dict,
            'temporal': temporal_dict,
            'measurement': measurement_dict,
            'risk': risk,
            'risk_confidence': risk_confidence
        }

    def _determine_risk(
        self,
        measurement: MeasurementResult,
        temporal_state
    ) -> tuple:
        if not measurement.is_valid or measurement.water_level is None:
            return 'SAFE', 0.0

        level = measurement.water_level

        if level >= 70:
            return 'CRITICAL', 0.9
        elif level >= 50:
            return 'WARNING', 0.85
        elif level >= 30:
            return 'WATCH', 0.8
        else:
            return 'SAFE', 0.95

    def reset(self):
        self._frame_count = 0
        self._start_time = None
        self._last_confidence = None
        self.temporal.reset()
        self.calibration.reset()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'frames_processed': self._frame_count,
            'temporal_has_data': self.temporal.has_data,
            'calibration_is_calibrated': self.calibration.is_calibrated,
            'calibration_info': self.calibration.get_info()
        }
