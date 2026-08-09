import cv2
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .waterline import WaterlineDetector, create_default_detector
from .temporal import TemporalBuffer
from .calibration import CalibrationModel, estimate_calibration_from_scene
from .measurement import MeasurementProcessor, MeasurementResult


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

        # Compute calibration quality early so it can be used in measurement confidence
        calibration_conf = self._calibration_quality(calibration_result, temporal_state)

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

            det_signal = min(0.8, detection_result.confidence)
            calib_quality = calibration_conf
            evidence_penalty = (
                1.0 if temporal_state.invalid_detections == 0
                else max(0.2, 1.0 - temporal_state.invalid_detections * 0.1)
            )

            # Plausibility: penalise disagreement between detection and quality metrics.
            # If quality_score is much lower than detection confidence, the signal
            # may be strong but positionally unreliable.
            quality_ratio = detection_result.quality_score / max(det_signal, 0.05)
            quality_agreement = min(1.0, quality_ratio)
            plausibility_penalty = max(0.3, quality_agreement)

            measurement_result.confidence = round(
                det_signal * calib_quality * evidence_penalty * plausibility_penalty, 3
            )

        self._last_raw_detection = {
            'detected': detection_result.detected,
            'waterline_y': detection_result.waterline_y,
            'confidence': detection_result.confidence,
            'method': detection_result.method,
            'quality_score': detection_result.quality_score
        }

        # Convert pixel-rate to physical-rate for risk determination.
        # temporal_state.rate_of_change is px/s.
        # Calibration result gives pixels_per_cm.
        px_per_cm = calibration_result.get('pixelsPerCm')
        if temporal_state.rate_of_change is not None and px_per_cm and px_per_cm > 0:
            rate_cm_per_min = round(temporal_state.rate_of_change / px_per_cm * 60, 2)
        else:
            rate_cm_per_min = None

        risk, risk_confidence = self._determine_risk(
            measurement_result, temporal_state, calibration_result, rate_cm_per_min
        )

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
            # Note: rate_of_change is in px/s (not physical units)
            'rate_px_per_sec': temporal_state.rate_of_change,
            'rate_cm_per_min': rate_cm_per_min,
            'confidence': temporal_state.confidence,
            'valid_detections': temporal_state.valid_detections,
            'invalid_detections': temporal_state.invalid_detections,
            'detection_history': temporal_state.detection_history[-20:],
            'buffer_size': len(temporal_state.detection_history),
            'detection_rate': round(self.temporal.detection_rate_value, 3)
        }

        # Categorical measurement validity — can this observation become a trusted physical measurement?
        # Separate from measurement_confidence (internal pipeline quality).
        if not calibration_result.get('calibrated'):
            meas_validity = 'UNCALIBRATED'
        elif temporal_state.invalid_detections >= 3:
            meas_validity = 'UNSTABLE'
        elif calibration_conf < 0.3:
            meas_validity = 'LOW_QUALITY'
        elif not measurement_result.is_valid:
            meas_validity = 'NO_DETECTION'
        else:
            meas_validity = 'VALID'

        measurement_dict = {
            'waterLevel': measurement_result.water_level,
            'pixelWaterline': measurement_result.pixel_waterline,
            'confidence': measurement_result.confidence,
            'calibrationConfidence': calibration_conf,
            'measurementStatus': measurement_result.measurement_status,
            'measurementValidity': meas_validity,
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

    def _calibration_quality(
        self,
        calibration_result: Dict,
        temporal_state
    ) -> float:
        """Compute continuous calibration quality score.
        Returns 0.0–1.0 based on calibration method, sample quality,
        establishment age, and scene consistency.
        """
        if not calibration_result.get('calibrated'):
            status = calibration_result.get('status', '')
            if 'SAMPLING' in status:
                # Count samples from status string e.g. "SAMPLING_BASELINE (2/5)"
                try:
                    parts = status.split('(')[1].split('/')
                    current = int(parts[0])
                    total = int(parts[1].rstrip(')'))
                    # Partial calibration: linear ramp from 0 to 0.3
                    return round((current / total) * 0.3, 3)
                except (IndexError, ValueError):
                    return 0.0
            return 0.0

        method = calibration_result.get('calibrationMethod', '')
        base_conf = {
            'relative': 0.5,
            'relative_baseline': 0.6,
            'absolute': 0.8
        }.get(method, 0.4)

        # Penalise for recent invalid detections (scene may have changed)
        invalid = temporal_state.invalid_detections if hasattr(temporal_state, 'invalid_detections') else 0
        invalid_penalty = min(0.2, invalid * 0.05)

        # Penalise if baseline is stale (too many frames since establishment)
        # relative calibration is sensitive to camera/scene changes
        if method.startswith('relative'):
            status = calibration_result.get('status', '')
            if status == 'BASELINE_ESTABLISHED':
                # Newly established — full confidence
                return round(base_conf - invalid_penalty, 3)
            elif status == 'CALIBRATED':
                # Older calibration — reduce slightly
                return round((base_conf - 0.1) - invalid_penalty, 3)

        return round(max(0.0, base_conf - invalid_penalty), 3)

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

        # Penalise correlated noise: if all recent detections have near-identical
        # waterline_y values, the detector may be tracking a static wrong feature.
        # Use std of recent detection_history (True=valid) waterline_y positions.
        recent_y_values = []
        for i, detected in enumerate(reversed(temporal_state.detection_history)):
            if detected and self.temporal._waterline_y:
                y_val = self.temporal._waterline_y[-(i + 1)]
                if y_val is not None:
                    recent_y_values.append(y_val)
        plausibility_conf = 0.5
        if len(recent_y_values) >= 5:
            y_std = float(np.std(recent_y_values))
            # Very low std means all detections are at nearly the same pixel position.
            # This is suspicious — penalise it. Allow ~3px std as minimum for "real" water.
            if y_std < 2.0:
                stability_conf = min(stability_conf, 0.2)
            elif y_std < 5.0:
                stability_conf = min(stability_conf, 0.5)
            # Detector-lock: unique positions barely change — detector stuck on static feature.
            # This is a different signal from std-based correlated-noise penalty.
            # The std penalty catches near-identical values; this catches TOTAL stagnation.
            unique_y = len(set(round(y, 1) for y in recent_y_values))
            if unique_y <= 2:
                plausibility_conf = min(plausibility_conf, 0.2)

        calibration_conf = self._calibration_quality(
            calibration_result, temporal_state
        )
        brightness = self.detector.get_diagnostics().get('avg_brightness', 128.0)
        lighting_conf = 1.0 if 30 <= brightness <= 220 else max(0.0, min(1.0, brightness / 128.0))
        # Level-delta plausibility: if the detector IS moving (delta >= 5), we can be more confident
        # that it's not locked. But ONLY raise plausibility — never lower it below detector-lock floor.
        if self._last_confidence is not None and detection_result.detected:
            level = calibration_result.get('waterLevel')
            if level is not None and self._last_confidence.get('level') is not None:
                delta = abs(level - self._last_confidence['level'])
                if delta >= 5:
                    plausibility_conf = max(plausibility_conf, 0.5)
                if delta >= 20:
                    plausibility_conf = max(plausibility_conf, 0.9)
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

        # Detector lock detection: raw positions are identical for many consecutive frames.
        # This is distinct from plausibility (which checks delta between frames).
        # Plausibility: "did the level jump suddenly?" -> catches discontinuities
        # Detector lock: "is the detector returning the same value repeatedly?" -> catches stagnation
        # Both can indicate a broken observation channel.
        recent_y = list(self.temporal._waterline_y)[-20:] if hasattr(self.temporal, '_waterline_y') else []
        non_none_y = [y for y in recent_y if y is not None]
        if len(non_none_y) >= 10:
            unique_y = len(set(round(y, 1) for y in non_none_y))
            if unique_y <= 2:
                reasons.append('detector may be locked: only %d unique position(s) in last %d frames' % (unique_y, len(non_none_y)))
                permitted.append('temporal_accumulation')
                blocked.append('water_level')
        if not calibration_result.get('calibrated'):
            calib_status = calibration_result.get('status', 'unknown')
            reasons.append(f'calibration not established: {calib_status}')
            blocked.extend(['absolute_depth'])
            permitted.append('relative_level')
        elif calibration_result.get('calibrationMethod', '').startswith('relative'):
            # Even with relative calibration, cm is approximate
            reasons.append('calibration is relative-only: cm values are linear approximation')
            permitted.append('relative_level')
            # Don't block absolute_depth, but mark it as approximate in the diagnostics
        if not self.detector.get_diagnostics().get('detection_stability') == 'stable':
            permitted.append('trend_direction')
        # Determine absolute_depth trust level
        # Current calibration is a LINEAR MODEL (pixels_per_cm ratio).
        # This does NOT account for perspective distortion, lens distortion,
        # or camera angle. The cm value is therefore APPROXIMATE.
        if not calibration_result.get('calibrated'):
            abs_depth_status = 'UNAVAILABLE'
        elif calibration_result.get('calibrationMethod') == 'absolute':
            abs_depth_status = 'TRUSTED'  # Uses physical reference — higher confidence
        elif calibration_result.get('calibrationMethod', '').startswith('relative'):
            abs_depth_status = 'APPROXIMATE'  # Linear model only — perspective unmodeled
        else:
            abs_depth_status = 'APPROXIMATE'

        return {
            'state': state,
            'reasons': reasons,
            'permitted_inferences': list(set(permitted)),
            'blocked_inferences': list(set(blocked)),
            'calibration_status': calibration_result.get('status', 'unknown'),
            'calibration_valid': calibration_result.get('calibrated', False),
            'calibration_method': calibration_result.get('calibrationMethod', 'unknown'),
            'absolute_depth_status': abs_depth_status,
            'absolute_depth_note': (
                'Pixel-to-cm conversion uses linear model only. '
                'Perspective distortion, lens distortion, and camera angle '
                'are NOT compensated. cm values are APPROXIMATE.' if abs_depth_status == 'APPROXIMATE'
                else 'Absolute calibration with physical reference.' if abs_depth_status == 'TRUSTED'
                else 'Calibration not established — cm values unavailable.'
            ),
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
        temporal_state,
        calibration_result: Optional[Dict] = None,
        rate_cm_per_min: Optional[float] = None
    ) -> tuple:
        """
        Determines risk level based on physical reading, evidence quality, and rate of change.

        rate_cm_per_min: physical rate of change in cm/min (converted from px/s by pipeline).
                         None if calibration not established (cannot convert px to cm).
        """
        if not measurement.is_valid or measurement.water_level is None:
            return 'SAFE', 0.0
        level = measurement.water_level
        meas_conf = measurement.confidence

        # Evidential gate: do not escalate to WARNING/CRITICAL unless
        # measurement confidence is above minimum threshold.
        MIN_EVIDENCE_FOR_WARNING = 0.15
        if meas_conf < MIN_EVIDENCE_FOR_WARNING:
            # Level is elevated but evidence is insufficient — downgrade severity
            if level >= 70:
                return 'WATCH', round(meas_conf * 0.5, 3)
            elif level >= 50:
                return 'WATCH', round(meas_conf * 0.4, 3)
            elif level >= 30:
                return 'WATCH', round(meas_conf * 0.3, 3)
            else:
                return 'SAFE', round(meas_conf * 0.2, 3)

        # Rate-aware escalation (rate_cm_per_min is in cm/min)
        # HIGH_RATE: rapid rise warrants escalation
        HIGH_RATE = 5.0   # cm/min
        MOD_RATE = 2.0    # cm/min

        # Level + rate determination
        if level >= 70:
            # CRITICAL threshold — rate doesn't de-escalate from CRITICAL
            return 'CRITICAL', round(meas_conf, 3)
        elif level >= 50:
            if rate_cm_per_min is not None and abs(rate_cm_per_min) >= HIGH_RATE:
                return 'CRITICAL', round(meas_conf, 3)
            return 'WARNING', round(meas_conf * 0.85, 3)
        elif level >= 30:
            if rate_cm_per_min is not None and rate_cm_per_min >= HIGH_RATE:
                return 'WARNING', round(meas_conf * 0.8, 3)
            return 'WATCH', round(meas_conf * 0.8, 3)
        else:
            # Below WATCH threshold — only escalate if rate is very high
            if rate_cm_per_min is not None and rate_cm_per_min >= HIGH_RATE * 1.5:
                return 'WATCH', round(meas_conf * 0.7, 3)
            return 'SAFE', round(meas_conf * 0.9, 3)

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
