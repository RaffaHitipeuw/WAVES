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
    diagnostics: Dict[str, Any]
    candidates: list
    signals: Dict[str, Any]
    evidence: Dict[str, float]


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
        self._last_raw_detection: Optional[Dict] = None
        self._temporal_history: list = []

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

        self._temporal_history.append({
            'frame': current_frame,
            'timestamp': timestamp,
            'detected': detection_result.detected,
            'waterline_y': detection_result.waterline_y,
            'smoothed': temporal_state.waterline_y,
            'confidence': detection_result.confidence,
            'trend': temporal_state.trend,
            'is_valid': detection_result.detected
        })
        if len(self._temporal_history) > 60:
            self._temporal_history = self._temporal_history[-60:]

        waterline_y = temporal_state.waterline_y or temporal_state.raw_waterline_y
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

        self._last_raw_detection = {
            'detected': detection_result.detected,
            'waterline_y': detection_result.waterline_y,
            'confidence': detection_result.confidence,
            'method': detection_result.method,
            'quality_score': detection_result.quality_score
        }

        risk, risk_confidence = self._determine_risk(measurement_result, temporal_state)

        evidence = self._calculate_evidence(
            detection_result, temporal_state, calibration_result
        )

        detection_dict = {
            'detected': detection_result.detected,
            'waterline_y': detection_result.waterline_y,
            'confidence': detection_result.confidence,
            'method': detection_result.method,
            'quality_score': detection_result.quality_score,
            'candidates': detection_result.candidates,
            'stability': self.detector.get_diagnostics().get('detection_stability', 'unknown')
        }

        temporal_dict = {
            'waterline_y': temporal_state.waterline_y,
            'raw_waterline_y': temporal_state.raw_waterline_y,
            'trend': temporal_state.trend,
            'rate_of_change': temporal_state.rate_of_change,
            'confidence': temporal_state.confidence,
            'valid_detections': temporal_state.valid_detections,
            'invalid_detections': temporal_state.invalid_detections,
            'detection_history': temporal_state.detection_history[-20:],
            'buffer_size': len(temporal_state.detection_history),
            'detection_rate': round(self.temporal.detection_rate_value, 3)
        }

        measurement_dict = {
            'waterLevel': measurement_result.water_level,
            'pixelWaterline': measurement_result.pixel_waterline,
            'confidence': measurement_result.confidence,
            'measurementStatus': measurement_result.measurement_status,
            'isValid': measurement_result.is_valid,
            'smoothedLevel': temporal_state.waterline_y
        }

        diagnostics = self._build_diagnostics(
            detection_result, temporal_state, calibration_result, measurement_result
        )

        signals = self._extract_signals()

        return {
            'frame_index': current_frame,
            'timestamp': round(timestamp, 3),
            'detection': detection_dict,
            'temporal': temporal_dict,
            'measurement': measurement_dict,
            'risk': risk,
            'risk_confidence': risk_confidence,
            'diagnostics': diagnostics,
            'evidence': evidence,
            'signals': signals
        }

    def _calculate_evidence(
        self,
        detection_result,
        temporal_state,
        calibration_result
    ) -> Dict[str, float]:
        detection_conf = (
            detection_result.confidence * 0.7 + detection_result.quality_score * 0.3
            if detection_result.detected else 0.0
        )
        trend = temporal_state.trend
        trend_conf_map = {
            'STABLE': 0.9, 'RISING': 0.8, 'FALLING': 0.8,
            'RISING_FAST': 0.6, 'FALLING_FAST': 0.6,
            'UNCERTAIN': 0.3, 'NO_DETECTION': 0.0
        }
        temporal_conf = trend_conf_map.get(trend, 0.0)
        rate = temporal_state.rate_of_change
        if rate is not None:
            rate_norm = max(0.0, 1.0 - abs(rate) / 30.0)
            temporal_conf = temporal_conf * 0.7 + rate_norm * 0.3
        valid = temporal_state.valid_detections
        invalid = temporal_state.invalid_detections
        if valid == 0 and invalid > 0:
            stability_conf = 0.0
        elif invalid > 0:
            stability_conf = max(0.1, 1.0 - invalid * 0.15)
        else:
            stability_conf = min(1.0, valid / 10.0)
        calibration_conf = 1.0 if calibration_result.get('calibrated') else 0.0
        brightness = self.detector.get_diagnostics().get('avg_brightness', 128.0)
        lighting_conf = 1.0 if 30 <= brightness <= 220 else max(0.0, min(1.0, brightness / 128.0))
        plausibility_conf = 0.5
        if self._last_confidence is not None and detection_result.detected:
            level = calibration_result.get('waterLevel')
            if level is not None and self._last_confidence.get('level') is not None:
                delta = abs(level - self._last_confidence['level'])
                plausibility_conf = 0.9 if delta < 5 else (0.5 if delta < 20 else 0.2)
        self._last_confidence = {
            'detection': detection_conf,
            'temporal': temporal_conf,
            'stability': stability_conf,
            'calibration': calibration_conf,
            'lighting': lighting_conf,
            'plausibility': plausibility_conf,
            'level': calibration_result.get('waterLevel')
        }
        return {
            'detection': round(detection_conf, 3),
            'temporal': round(temporal_conf, 3),
            'stability': round(stability_conf, 3),
            'calibration': round(calibration_conf, 3),
            'lighting': round(lighting_conf, 3),
            'plausibility': round(plausibility_conf, 3)
        }

    def _build_diagnostics(
        self,
        detection_result,
        temporal_state,
        calibration_result,
        measurement_result
    ) -> Dict[str, Any]:
        state = self._classify_state(detection_result, temporal_state, calibration_result)
        reasons = []
        permitted = []
        blocked = []
        if not detection_result.detected:
            reasons.append('no valid waterline candidate detected')
            blocked.extend(['water_level', 'risk_level', 'absolute_depth'])
            permitted.extend(['temporal_accumulation', 'relative_trend'])
        elif detection_result.confidence < 0.3:
            reasons.append(f'detection confidence too low ({detection_result.confidence:.2f})')
            blocked.append('water_level')
        if temporal_state.trend == 'NO_DETECTION':
            reasons.append('no detection history')
            permitted.append('buffer_accumulation')
            blocked.extend(['temporal_trend', 'rate_calculation'])
        elif temporal_state.trend == 'UNCERTAIN':
            reasons.append('insufficient detection history for stable trend')
            permitted.append('temporal_accumulation')
            blocked.append('rate_calculation')
        if temporal_state.invalid_detections > 0:
            reasons.append(f'{temporal_state.invalid_detections} recent invalid detections in buffer')
            permitted.append('temporal_accumulation')
        if not calibration_result.get('calibrated'):
            calib_status = calibration_result.get('status', 'unknown')
            reasons.append(f'calibration not established: {calib_status}')
            blocked.extend(['absolute_depth'])
            permitted.append('relative_level')
        if not self.detector.get_diagnostics().get('detection_stability') == 'stable':
            permitted.append('trend_direction')
        return {
            'state': state,
            'reasons': reasons,
            'permitted_inferences': list(set(permitted)),
            'blocked_inferences': list(set(blocked)),
            'calibration_status': calibration_result.get('status', 'unknown'),
            'calibration_valid': calibration_result.get('calibrated', False),
            'calibration_method': calibration_result.get('calibrationMethod', 'unknown'),
            'detector_info': self.detector.get_diagnostics(),
            'buffer_full': len(temporal_state.detection_history) >= 30
        }

    def _classify_state(
        self,
        detection_result,
        temporal_state,
        calibration_result
    ) -> str:
        if not detection_result.detected:
            if not calibration_result.get('calibrated'):
                return 'CALIBRATION_INVALID'
            return 'NO_DETECTION'
        if detection_result.confidence < 0.3:
            return 'CANDIDATE'
        stability = self.detector.get_diagnostics().get('detection_stability', 'unknown')
        if stability == 'jittering':
            return 'UNSTABLE'
        if stability == 'unstable':
            return 'UNSTABLE'
        if temporal_state.trend == 'NO_DETECTION':
            return 'CANDIDATE'
        if temporal_state.trend == 'UNCERTAIN':
            return 'UNCERTAIN'
        if not calibration_result.get('calibrated'):
            return 'CANDIDATE'
        if temporal_state.valid_detections >= 10 and stability == 'stable':
            return 'STABLE'
        if temporal_state.valid_detections >= 3:
            return 'DETECTED'
        return 'CANDIDATE'

    def _extract_signals(self) -> Dict[str, Any]:
        diag = self.detector.get_diagnostics()
        edge_signal = None
        color_signal = None
        texture_signal = None
        if diag.get('has_edge_signal') and self.detector._last_edge_signal is not None:
            signal = self.detector._last_edge_signal
            roi_h = diag['roi']['height']
            normalized = signal.tolist() if hasattr(signal, 'tolist') else list(signal)
            edge_signal = {
                'data': normalized,
                'length': len(normalized),
                'peak_idx': int(np.argmax(signal)) if len(signal) > 0 else 0,
                'peak_value': float(np.max(signal)) if len(signal) > 0 else 0.0,
                'mean': float(np.mean(signal)) if len(signal) > 0 else 0.0
            }
        if diag.get('has_color_signal') and self.detector._last_color_signal is not None:
            signal = self.detector._last_color_signal
            normalized = signal.tolist() if hasattr(signal, 'tolist') else list(signal)
            color_signal = {
                'data': normalized,
                'length': len(normalized),
                'peak_idx': int(np.argmax(signal)) if len(signal) > 0 else 0,
                'peak_value': float(np.max(signal)) if len(signal) > 0 else 0.0,
                'mean': float(np.mean(signal)) if len(signal) > 0 else 0.0
            }
        if diag.get('has_texture_signal') and self.detector._last_texture_signal is not None:
            signal = self.detector._last_texture_signal
            normalized = signal.tolist() if hasattr(signal, 'tolist') else list(signal)
            texture_signal = {
                'data': normalized,
                'length': len(normalized),
                'peak_idx': int(np.argmin(signal)) if len(signal) > 0 else 0,
                'peak_value': float(np.min(signal)) if len(signal) > 0 else 0.0,
                'mean': float(np.mean(signal)) if len(signal) > 0 else 0.0
            }
        return {
            'edge': edge_signal,
            'color': color_signal,
            'texture': texture_signal,
            'roi': diag['roi'],
            'brightness': diag['avg_brightness']
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
        self._last_raw_detection = None
        self._temporal_history.clear()
        self.temporal.reset()
        self.calibration.reset()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'frames_processed': self._frame_count,
            'temporal_has_data': self.temporal.has_data,
            'calibration_is_calibrated': self.calibration.is_calibrated,
            'calibration_info': self.calibration.get_info(),
            'buffer_history_size': len(self._temporal_history),
            'last_detection_state': self._classify_state(
                type('D', (), {'detected': self._last_raw_detection is not None,
                               'confidence': self._last_raw_detection.get('confidence', 0) if self._last_raw_detection else 0,
                               'quality_score': self._last_raw_detection.get('quality_score', 0) if self._last_raw_detection else 0})(),
                self.temporal.get_state(),
                self.calibration.get_info()
            )
        }
