"""
CV Pipeline Module

Orchestrates the complete computer vision pipeline:

VIDEO FRAME
    ↓
ROI EXTRACTION
    ↓
WATERLINE DETECTION
    ↓
TEMPORAL VALIDATION
    ↓
CALIBRATION
    ↓
MEASUREMENT
    ↓
CONFIDENCE
    ↓
OUTPUT

The pipeline outputs a single coherent measurement object.
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .waterline import WaterlineDetector, ROI, WaterlineResult
from .temporal import TemporalBuffer, TemporalState
from .calibration import CalibrationModel, CalibrationConfig, estimate_calibration_from_scene
from .measurement import MeasurementProcessor, MeasurementResult
from .confidence import ConfidenceCalculator


@dataclass
class PipelineOutput:
    """Output from the CV pipeline."""
    timestamp: float
    frame_index: int
    fps: float

    # Measurement
    water_level: Optional[float]
    pixel_waterline: Optional[float]
    smoothed_waterline: Optional[float]
    rate_of_change: Optional[float]

    # Status
    measurement_status: str
    trend: str
    confidence: float
    is_valid: bool

    # Detection info
    detection_method: str
    detection_confidence: float

    # Debug
    raw_detection: Optional[Dict] = field(default_factory=dict)
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket transmission."""
        return {
            'timestamp': self.timestamp,
            'frameIndex': self.frame_index,
            'fps': self.fps,

            'waterLevel': self.water_level,
            'pixelWaterline': self.pixel_waterline,
            'smoothedWaterline': self.smoothed_waterline,
            'rateOfChange': self.rate_of_change,

            'measurementStatus': self.measurement_status,
            'trend': self.trend,
            'confidence': round(self.confidence, 3),
            'isValid': self.is_valid,

            'detectionMethod': self.detection_method,
            'detectionConfidence': round(self.detection_confidence, 3),

            'details': self.details
        }


class CVPipeline:
    """
    Complete computer vision pipeline for water level estimation.

    Processes video frames and produces measurements with:
    - Waterline detection
    - Temporal smoothing
    - Calibration
    - Confidence assessment
    """

    def __init__(
        self,
        video_path: str,
        roi: Optional[ROI] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        process_every_n_frames: int = 1
    ):
        """
        Initialize CV pipeline.

        Args:
            video_path: Path to video file
            roi: Region of Interest for detection
            calibration_config: Calibration configuration
            process_every_n_frames: Process every N frames (for performance)
        """
        self.video_path = video_path
        self.process_every_n_frames = process_every_n_frames

        # Open video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # Components
        self.detector = WaterlineDetector(
            roi=roi or ROI(
                x_min=int(self.frame_width * 0.1),
                x_max=int(self.frame_width * 0.9),
                y_min=int(self.frame_height * 0.4),
                y_max=self.frame_height
            )
        )

        self.temporal = TemporalBuffer(
            max_history=60,
            smoothing_window=5,
            rate_window=15,
            invalid_threshold=5
        )

        roi_config = {
            'y_min': self.detector.roi.y_min,
            'y_max': self.detector.roi.y_max
        }
        calib_config = calibration_config or estimate_calibration_from_scene(
            self.frame_height,
            roi_config
        )
        self.calibration = CalibrationModel(calib_config)

        self.measurement = MeasurementProcessor(
            min_confidence_threshold=0.25
        )

        self.confidence_calc = ConfidenceCalculator()

        # State
        self.frame_index = 0
        self.start_time = time.time()
        self.processed_count = 0
        self._previous_confidence = None
        self._previous_measurement = None

    def process_frame(self, frame: np.ndarray) -> Tuple[WaterlineResult, float]:
        """Process a single frame."""
        timestamp = time.time() - self.start_time

        # Detect waterline
        detection_result = self.detector.detect(frame)

        return detection_result, timestamp

    def process_next(self) -> Optional[PipelineOutput]:
        """
        Process next frame and return measurement.

        Returns:
            PipelineOutput if frame processed, None if video ended
        """
        self.frame_index += 1

        # Skip frames if configured
        if self.process_every_n_frames > 1:
            skip = self.process_every_n_frames - 1
            for _ in range(skip):
                ret = self.cap.grab()
                if not ret:
                    return None
                self.frame_index += 1

        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            return None

        timestamp = time.time() - self.start_time
        self.processed_count += 1

        # Step 1: Detect waterline
        detection_result = self.detector.detect(frame)

        # Step 2: Add to temporal buffer
        self.temporal.add(
            timestamp=timestamp,
            waterline_y=detection_result.waterline_y,
            confidence=detection_result.confidence
        )

        temporal_state = self.temporal.get_state()

        # Step 3: Calibrate
        if detection_result.detected and detection_result.waterline_y is not None:
            calibration_result = self.calibration.calibrate(
                detection_result.waterline_y,
                timestamp
            )
        else:
            calibration_result = {
                'waterLevel': None,
                'calibrated': False,
                'calibrationMethod': 'none'
            }

        # Build raw detection dict
        raw_detection_dict = None
        if detection_result:
            raw_detection_dict = {
                'detected': detection_result.detected,
                'waterline_y': detection_result.waterline_y,
                'confidence': detection_result.confidence,
                'method': detection_result.method,
                'quality_score': detection_result.quality_score
            }

        # Step 4: Process measurement
        measurement_result = self.measurement.process(
            raw_detection=raw_detection_dict,
            temporal_state={
                'waterline_y': temporal_state.waterline_y,
                'raw_waterline_y': temporal_state.raw_waterline_y,
                'smoothed_waterline_y': temporal_state.smoothed_waterline_y,
                'rate_of_change': temporal_state.rate_of_change,
                'trend': temporal_state.trend,
                'confidence': temporal_state.confidence,
                'valid_detections': temporal_state.valid_detections,
                'invalid_detections': temporal_state.invalid_detections
            },
            calibration_result=calibration_result,
            frame_index=self.frame_index,
            timestamp=timestamp
        )

        # Step 5: Calculate confidence
        confidence = self.confidence_calc.calculate(
            detection_result=raw_detection_dict,
            temporal_state={
                'trend': temporal_state.trend,
                'rate_of_change': temporal_state.rate_of_change,
                'confidence': temporal_state.confidence,
                'valid_detections': temporal_state.valid_detections,
                'invalid_detections': temporal_state.invalid_detections
            },
            measurement_result={
                'waterLevel': measurement_result.water_level,
                'measurementStatus': measurement_result.measurement_status
            },
            previous_confidence=self._previous_confidence
        )

        # Step 6: Validate measurement
        if self._previous_measurement:
            measurement_result = self.measurement.validate_measurement(
                measurement_result,
                self._previous_measurement
            )

        # Update tracking
        self._previous_confidence = confidence
        self._previous_measurement = measurement_result

        # Build output
        return PipelineOutput(
            timestamp=timestamp,
            frame_index=self.frame_index,
            fps=self.fps,

            water_level=measurement_result.water_level,
            pixel_waterline=measurement_result.pixel_waterline,
            smoothed_waterline=temporal_state.smoothed_waterline_y,
            rate_of_change=temporal_state.rate_of_change,

            measurement_status=measurement_result.measurement_status,
            trend=temporal_state.trend,
            confidence=confidence,
            is_valid=measurement_result.is_valid,

            detection_method=detection_result.method if detection_result else 'none',
            detection_confidence=detection_result.confidence if detection_result else 0.0,

            raw_detection=raw_detection_dict or {},
            details=measurement_result.details
        )

    def reset(self):
        """Reset pipeline state."""
        self.frame_index = 0
        self.start_time = time.time()
        self.processed_count = 0
        self.temporal.reset()
        self.calibration.reset()
        self.confidence_calc.reset()
        self._previous_confidence = None
        self._previous_measurement = None

        # Reset video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek_to(self, frame_index: int):
        """Seek to specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.frame_index = frame_index

    def release(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def is_open(self) -> bool:
        """Check if video is still open."""
        return self.cap is not None and self.cap.isOpened()

    @property
    def progress(self) -> float:
        """Video progress (0-1)."""
        if self.frame_count <= 0:
            return 0.0
        return self.frame_index / self.frame_count

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            'video': {
                'path': self.video_path,
                'fps': self.fps,
                'frameCount': self.frame_count,
                'width': self.frame_width,
                'height': self.frame_height,
                'duration': self.duration
            },
            'roi': {
                'x_min': self.detector.roi.x_min,
                'x_max': self.detector.roi.x_max,
                'y_min': self.detector.roi.y_min,
                'y_max': self.detector.roi.y_max
            },
            'calibration': self.calibration.get_info(),
            'state': {
                'frameIndex': self.frame_index,
                'processedCount': self.processed_count,
                'progress': self.progress,
                'isOpen': self.is_open
            }
        }
