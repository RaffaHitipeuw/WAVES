"""
Vision module for HydroSignal.

This module handles computer vision processing for flood water level estimation.

Architecture:
    VIDEO FRAME
        ↓
    ROI EXTRACTION
        ↓
    WATERLINE DETECTION
        ↓
    CALIBRATION
        ↓
    MEASUREMENT
        ↓
    TEMPORAL SMOOTHING
        ↓
    RISK DETERMINATION
"""

from .waterline import WaterlineDetector
from .temporal import TemporalBuffer
from .calibration import CalibrationModel
from .measurement import MeasurementProcessor
from .confidence import ConfidenceCalculator
from .pipeline import CVPipeline

__all__ = [
    'WaterlineDetector',
    'TemporalBuffer',
    'CalibrationModel',
    'MeasurementProcessor',
    'ConfidenceCalculator',
    'CVPipeline'
]
