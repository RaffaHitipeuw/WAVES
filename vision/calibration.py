import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CalibrationConfig:
    reference_y: Optional[float] = None
    reference_level: Optional[float] = None
    pixels_per_cm: Optional[float] = None
    roi_y_min: float = 540
    roi_y_max: float = 1080
    method: str = 'relative'


class CalibrationModel:
    def __init__(
        self,
        config: Optional[CalibrationConfig] = None
    ):
        self.config = config or CalibrationConfig()
        self._baseline_y: Optional[float] = None
        self._baseline_established = False
        self._calibration_samples = []

    def calibrate(
        self,
        waterline_y: float,
        timestamp: float
    ) -> Dict[str, Any]:
        result = {
            'waterLevel': None,
            'relativeLevel': None,
            'calibrationMethod': 'unknown',
            'confidence': 0.0,
            'calibrated': False
        }
        if self.config.method == 'absolute':
            return self._calibrate_absolute(waterline_y, result)
        elif self.config.method == 'relative':
            return self._calibrate_relative(waterline_y, timestamp, result)
        else:
            return result

    def _calibrate_absolute(
        self,
        waterline_y: float,
        result: Dict
    ) -> Dict:
        if self.config.pixels_per_cm is None:
            if self.config.reference_y and self.config.reference_level:
                ref_distance_pixels = self.config.roi_y_max - self.config.reference_y
                self.config.pixels_per_cm = ref_distance_pixels / self.config.reference_level
        if self.config.pixels_per_cm and self.config.pixels_per_cm > 0:
            distance_from_bottom = self.config.roi_y_max - waterline_y
            water_level = distance_from_bottom / self.config.pixels_per_cm
            result.update({
                'waterLevel': round(water_level, 2),
                'relativeLevel': None,
                'calibrationMethod': 'absolute',
                'confidence': 0.8,
                'calibrated': True
            })
        return result

    def _calibrate_relative(
        self,
        waterline_y: float,
        timestamp: float,
        result: Dict
    ) -> Dict:
        if not self._baseline_established:
            self._calibration_samples.append((waterline_y, timestamp))
            if len(self._calibration_samples) >= 5:
                y_values = [y for y, _ in self._calibration_samples]
                self._baseline_y = np.median(y_values)
                self._baseline_established = True
                self._calibration_samples.clear()
                result.update({
                    'waterLevel': 0.0,
                    'relativeLevel': 0.0,
                    'calibrationMethod': 'relative_baseline',
                    'baselineY': self._baseline_y,
                    'confidence': 0.7,
                    'calibrated': True,
                    'status': 'BASELINE_ESTABLISHED'
                })
                return result
            else:
                result.update({
                    'calibrationMethod': 'relative_establishing',
                    'confidence': 0.2,
                    'calibrated': False,
                    'status': f'SAMPLING_BASELINE ({len(self._calibration_samples)}/5)'
                })
                return result
        delta_y = self._baseline_y - waterline_y
        relative_level = delta_y
        estimated_cm = relative_level / 10.0
        result.update({
            'waterLevel': round(estimated_cm, 2),
            'relativeLevel': round(delta_y, 1),
            'calibrationMethod': 'relative',
            'baselineY': self._baseline_y,
            'currentY': waterline_y,
            'deltaFromBaseline': round(delta_y, 1),
            'confidence': 0.6,
            'calibrated': True,
            'status': 'CALIBRATED'
        })
        return result

    def set_baseline(self, waterline_y: float):
        self._baseline_y = waterline_y
        self._baseline_established = True
        self._calibration_samples.clear()

    def reset(self):
        self._baseline_y = None
        self._baseline_established = False
        self._calibration_samples.clear()

    @property
    def is_calibrated(self) -> bool:
        if self.config.method == 'absolute':
            return self.config.pixels_per_cm is not None
        elif self.config.method == 'relative':
            return self._baseline_established
        return False

    def get_info(self) -> Dict[str, Any]:
        return {
            'method': self.config.method,
            'isCalibrated': self.is_calibrated,
            'baselineEstablished': self._baseline_established,
            'baselineY': self._baseline_y,
            'pixelsPerCm': self.config.pixels_per_cm,
            'referenceY': self.config.reference_y,
            'referenceLevel': self.config.reference_level
        }


def estimate_calibration_from_scene(
    frame_height: int,
    roi_config: Dict
) -> CalibrationConfig:
    return CalibrationConfig(
        method='relative',
        roi_y_min=roi_config.get('y_min', int(frame_height * 0.4)),
        roi_y_max=roi_config.get('y_max', frame_height)
    )
