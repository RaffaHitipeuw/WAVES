"""
Calibration Module

Converts pixel coordinates to physical measurements.

IMPORTANT: Pixel position is NOT automatically centimeters.
This module provides calibration between image-space and physical-space.

Calibration Approaches:
1. Known reference in frame (preferred)
2. Relative measurement (baseline tracking)
3. Camera geometry estimation (if calibratable)

The system will use the strongest defensible approach based on available evidence.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""
    # Reference points (if known)
    reference_y: Optional[float] = None  # Y position of known reference
    reference_level: Optional[float] = None  # Physical level at reference

    # Scale parameters
    pixels_per_cm: Optional[float] = None  # Manual calibration

    # ROI parameters
    roi_y_min: float = 540  # Top of detection region
    roi_y_max: float = 1080  # Bottom of detection region

    # Calibration method
    method: str = 'relative'  # 'absolute', 'relative', 'unknown'


class CalibrationModel:
    """
    Calibration model for pixel-to-physical conversion.

    Supports multiple calibration approaches:
    1. ABSOLUTE: Known reference in frame with physical measurements
    2. RELATIVE: Baseline tracking without absolute physical units
    3. UNKNOWN: Explicitly no calibration
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize calibration model.

        Args:
            config: Calibration configuration. If None, uses default relative calibration.
        """
        self.config = config or CalibrationConfig()

        # Calibration state
        self._baseline_y: Optional[float] = None
        self._baseline_established = False
        self._calibration_samples = []

    def calibrate(
        self,
        waterline_y: float,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Perform calibration step.

        Args:
            waterline_y: Detected waterline Y position
            timestamp: Detection timestamp

        Returns:
            Calibration result with physical level
        """
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
        """Absolute calibration using known reference."""
        if self.config.pixels_per_cm is None:
            # Try to establish from reference
            if self.config.reference_y and self.config.reference_level:
                # Calculate scale from reference
                ref_distance_pixels = self.config.roi_y_max - self.config.reference_y
                # Assume reference is at water level 0
                self.config.pixels_per_cm = ref_distance_pixels / self.config.reference_level

        if self.config.pixels_per_cm and self.config.pixels_per_cm > 0:
            # Convert to physical level
            # Distance from bottom reference point
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
        """
        Relative calibration using baseline tracking.

        Establishes a baseline from first valid detection,
        then reports changes relative to that baseline.
        """
        # Establish baseline from first stable detection
        if not self._baseline_established:
            # Wait for a few samples to establish baseline
            self._calibration_samples.append((waterline_y, timestamp))

            if len(self._calibration_samples) >= 5:
                # Use median of first samples as baseline
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

        # Calculate relative level
        # Positive = above baseline (rising in frame = falling water)
        # Negative = below baseline (falling in frame = rising water)
        delta_y = self._baseline_y - waterline_y  # Inverted: rising water = positive

        # Convert to relative units (1 unit = 1 pixel change)
        relative_level = delta_y

        # Estimate physical level assuming some scale
        # Without calibration, we report relative units
        # A reasonable assumption: ~10 pixels ≈ 1 cm (rough estimate)
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
        """
        Manually set the baseline.

        Useful when starting from a known state.
        """
        self._baseline_y = waterline_y
        self._baseline_established = True
        self._calibration_samples.clear()

    def reset(self):
        """Reset calibration state."""
        self._baseline_y = None
        self._baseline_established = False
        self._calibration_samples.clear()

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration is established."""
        if self.config.method == 'absolute':
            return self.config.pixels_per_cm is not None
        elif self.config.method == 'relative':
            return self._baseline_established
        return False

    def get_info(self) -> Dict[str, Any]:
        """Get calibration information."""
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
    """
    Estimate calibration parameters from scene configuration.

    This is a placeholder for more sophisticated calibration.
    Returns a relative calibration by default.

    Args:
        frame_height: Video frame height
        roi_config: ROI configuration

    Returns:
        Estimated calibration configuration
    """
    return CalibrationConfig(
        method='relative',
        roi_y_min=roi_config.get('y_min', int(frame_height * 0.4)),
        roi_y_max=roi_config.get('y_max', frame_height)
    )
