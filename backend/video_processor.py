"""
Video Processor Module

Processes video frames through the CV pipeline and produces measurements.

This replaces the simulator for the real demo path.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.pipeline import CVPipeline, PipelineOutput
from vision.waterline import ROI
from engine import FloodEngine


@dataclass
class ProcessorConfig:
    """Configuration for video processor."""
    video_path: str
    process_every_n_frames: int = 30  # ~1 second at 30fps
    roi_config: Optional[Dict] = None
    use_simulator_fallback: bool = True  # Fall back to simulator if CV fails


class VideoProcessor:
    """
    Video processor using CV pipeline.

    Produces measurements from actual video processing.
    """

    def __init__(
        self,
        config: ProcessorConfig,
        engine: FloodEngine
    ):
        """
        Initialize video processor.

        Args:
            config: Processor configuration
            engine: FloodEngine for risk calculation
        """
        self.config = config
        self.engine = engine
        self.pipeline: Optional[CVPipeline] = None
        self._running = False
        self._last_output: Optional[PipelineOutput] = None

    def initialize(self):
        """Initialize the CV pipeline."""
        # Set up ROI
        if self.config.roi_config:
            roi = ROI(**self.config.roi_config)
        else:
            # Default ROI for 1920x1080 video
            roi = ROI(
                x_min=200,
                x_max=1720,
                y_min=400,
                y_max=1080
            )

        # Create pipeline
        self.pipeline = CVPipeline(
            video_path=self.config.video_path,
            roi=roi,
            process_every_n_frames=self.config.process_every_n_frames
        )

        print(f"[VideoProcessor] Initialized with video: {self.config.video_path}")
        print(f"[VideoProcessor] ROI: {roi.x_min}-{roi.x_max}, {roi.y_min}-{roi.y_max}")
        print(f"[VideoProcessor] Processing every {self.config.process_every_n_frames} frames")

    def process_frame(self) -> Optional[PipelineOutput]:
        """
        Process next frame.

        Returns:
            PipelineOutput or None if video ended
        """
        if not self.pipeline:
            self.initialize()

        output = self.pipeline.process_next()

        if output:
            self._last_output = output

            # Process through FloodEngine for risk calculation
            # Convert to WaterLevelReading-like format
            water_level = output.water_level if output.is_valid else 0

            # Create a fake reading for the engine
            from models import WaterLevelReading, DataSource

            reading = WaterLevelReading(
                node_id=self.engine.node.id,
                water_level=water_level,
                source=DataSource.FILE
            )

            # Process through engine
            result = self.engine.process(reading)

            # Enrich with CV data
            result['cv'] = output.to_dict()
            result['measurementStatus'] = output.measurement_status
            result['isValid'] = output.is_valid
            result['frameIndex'] = output.frame_index
            result['fps'] = output.fps

        return output

    def get_current_measurement(self) -> Optional[Dict[str, Any]]:
        """Get current measurement from last processed frame."""
        if not self._last_output:
            return None

        output = self._last_output

        # Build measurement dict
        measurement = {
            'timestamp': output.timestamp,
            'frameIndex': output.frame_index,
            'fps': output.fps,

            'waterLevel': output.water_level,
            'pixelWaterline': output.pixel_waterline,
            'smoothedLevel': output.smoothed_waterline,
            'rateOfChange': output.rate_of_change,

            'measurementStatus': output.measurement_status,
            'trend': output.trend,
            'confidence': output.confidence,
            'isValid': output.is_valid,

            'detectionMethod': output.detection_method,
            'detectionConfidence': output.detection_confidence,

            'videoProgress': self.pipeline.progress if self.pipeline else 0
        }

        return measurement

    def get_engine_state(self) -> Dict[str, Any]:
        """Get current engine state with CV enrichment."""
        state = self.engine.get_state()

        if self._last_output:
            state['cv'] = self._last_output.to_dict()
            state['measurementStatus'] = self._last_output.measurement_status
            state['isValid'] = self._last_output.is_valid

        return state

    def reset(self):
        """Reset processor state."""
        if self.pipeline:
            self.pipeline.reset()
        self._last_output = None

    def seek_to(self, frame_index: int):
        """Seek to specific frame."""
        if self.pipeline:
            self.pipeline.seek_to(frame_index)

    def release(self):
        """Release resources."""
        if self.pipeline:
            self.pipeline.release()

    @property
    def progress(self) -> float:
        """Get video progress (0-1)."""
        if self.pipeline:
            return self.pipeline.progress
        return 0.0

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        if self.pipeline:
            return self.pipeline.frame_index
        return 0

    def get_info(self) -> Dict[str, Any]:
        """Get processor info."""
        info = {
            'type': 'video',
            'video_path': self.config.video_path,
            'is_initialized': self.pipeline is not None
        }

        if self.pipeline:
            info.update(self.pipeline.get_info())

        return info
