"""
Waterline Detection Module

Detects the horizontal boundary between water and other surfaces in video frames.

The detector uses multiple methods:
1. Edge-based detection - finds horizontal edges where brightness changes significantly
2. Color-based detection - identifies water-like colors
3. Motion-based detection (if applicable) - detects moving water surface

Output:
    {
        "detected": bool,
        "waterline_y": float | None,  # Y coordinate from top (pixels)
        "confidence": float,  # 0-1 confidence score
        "method": str,  # Which method was used
        "raw_signal": dict  # Raw detection signals
    }
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class WaterlineResult:
    """Result of waterline detection on a single frame."""
    detected: bool
    waterline_y: Optional[float]  # Y coordinate from top
    confidence: float  # 0-1
    method: str
    quality_score: float  # Internal quality metric
    raw_signal: Dict[str, Any]


@dataclass
class ROI:
    """Region of Interest for detection."""
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    def contains(self, x: int, y: int) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


class WaterlineDetector:
    """
    Detects waterline boundary in video frames.

    Uses a combination of:
    - Horizontal edge detection (Sobel)
    - Color analysis (blue channel excess)
    - Texture analysis (water is typically smooth)
    """

    def __init__(
        self,
        roi: Optional[ROI] = None,
        edge_threshold: float = 30.0,
        blue_excess_threshold: float = 5.0,
        min_confidence: float = 0.3
    ):
        """
        Initialize waterline detector.

        Args:
            roi: Region of Interest. If None, uses default (bottom 60% of frame).
            edge_threshold: Minimum edge strength to consider (0-255)
            blue_excess_threshold: Minimum blue excess to consider water-like
            min_confidence: Minimum confidence to report detection
        """
        # Default ROI: focus on bottom 60% where water would appear
        self.roi = roi or ROI(x_min=0, x_max=1920, y_min=540, y_max=1080)

        self.edge_threshold = edge_threshold
        self.blue_excess_threshold = blue_excess_threshold
        self.min_confidence = min_confidence

        # Statistics for adaptive thresholds
        self._frame_count = 0
        self._avg_brightness = 128.0

    def detect(self, frame: np.ndarray) -> WaterlineResult:
        """
        Detect waterline in a frame.

        Args:
            frame: BGR image (HxWx3)

        Returns:
            WaterlineResult with detection information
        """
        self._frame_count += 1

        # Extract ROI
        roi_frame = self._extract_roi(frame)
        if roi_frame is None or roi_frame.size == 0:
            return WaterlineResult(
                detected=False,
                waterline_y=None,
                confidence=0.0,
                method="none",
                quality_score=0.0,
                raw_signal={}
            )

        # Update brightness statistics
        self._update_statistics(roi_frame)

        # Try multiple detection methods
        results = []

        # Method 1: Edge-based detection
        edge_result = self._detect_by_edges(roi_frame)
        if edge_result:
            results.append(edge_result)

        # Method 2: Color-based detection
        color_result = self._detect_by_color(roi_frame)
        if color_result:
            results.append(color_result)

        # Method 3: Texture-based detection
        texture_result = self._detect_by_texture(roi_frame)
        if texture_result:
            results.append(texture_result)

        # Combine results
        return self._combine_results(results, frame.shape[0])

    def _extract_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract Region of Interest from frame."""
        h, w = frame.shape[:2]

        # Ensure ROI is within frame bounds
        x_min = max(0, min(self.roi.x_min, w - 1))
        x_max = max(0, min(self.roi.x_max, w))
        y_min = max(0, min(self.roi.y_min, h - 1))
        y_max = max(0, min(self.roi.y_max, h))

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame[y_min:y_max, x_min:x_max]

    def _update_statistics(self, roi_frame: np.ndarray):
        """Update running statistics for adaptive thresholds."""
        brightness = np.mean(roi_frame)
        # Smooth update
        self._avg_brightness = 0.9 * self._avg_brightness + 0.1 * brightness

    def _detect_by_edges(self, roi_frame: np.ndarray) -> Optional[Dict]:
        """
        Detect waterline using horizontal edge detection.

        Water surfaces create a horizontal edge where they meet other surfaces
        or where depth changes.
        """
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Sobel edge detection for horizontal edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Horizontal edge magnitude
        horizontal_edges = np.abs(sobely)

        # Average edge strength per row
        row_edge_strength = np.mean(horizontal_edges, axis=1)

        # Find strongest horizontal edge (significant brightness change)
        # Look for downward transitions (bright -> dark)
        max_edge_idx = np.argmax(row_edge_strength)
        max_edge_strength = row_edge_strength[max_edge_idx]

        # Calculate quality based on edge strength vs noise
        noise_estimate = np.std(row_edge_strength)
        if noise_estimate > 0:
            quality = min(1.0, max_edge_strength / (noise_estimate * 3))
        else:
            quality = 0.0

        # Confidence based on edge strength relative to threshold
        confidence = min(1.0, max_edge_strength / 100.0)

        if max_edge_strength > self.edge_threshold and confidence > self.min_confidence:
            # Convert ROI-relative Y to frame-relative Y
            waterline_y = self.roi.y_min + max_edge_idx

            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': quality,
                'method': 'edge',
                'edge_strength': max_edge_strength,
                'edge_idx_roi': max_edge_idx
            }

        return None

    def _detect_by_color(self, roi_frame: np.ndarray) -> Optional[Dict]:
        """
        Detect waterline using color analysis.

        Water typically has:
        - Blue excess compared to land
        - Lower saturation in reflective areas
        - Distinct blue-green tint
        """
        h, w = roi_frame.shape[:2]

        b, g, r = cv2.split(roi_frame.astype(np.float32))

        # Blue excess (B - R)
        blue_excess = b - r

        # Look for regions with significant blue excess
        row_blue_excess = np.mean(blue_excess, axis=1)

        # Find row with maximum blue excess
        max_blue_idx = np.argmax(row_blue_excess)
        max_blue_excess = row_blue_excess[max_blue_idx]

        # Quality based on how much blue excess there is
        quality = min(1.0, max_blue_excess / 20.0) if max_blue_excess > 0 else 0.0

        # Confidence
        confidence = min(1.0, max_blue_excess / self.blue_excess_threshold)

        if max_blue_excess > self.blue_excess_threshold and confidence > self.min_confidence:
            waterline_y = self.roi.y_min + max_blue_idx

            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': quality,
                'method': 'color',
                'blue_excess': max_blue_excess
            }

        return None

    def _detect_by_texture(self, roi_frame: np.ndarray) -> Optional[Dict]:
        """
        Detect waterline using texture analysis.

        Water surfaces are typically smooth (low texture) compared to
        roads, vegetation, or debris.
        """
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Laplacian gives texture measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_strength = np.abs(laplacian)

        # Average texture per row
        row_texture = np.mean(texture_strength, axis=1)

        # Find region with lowest texture (smooth = water)
        min_texture_idx = np.argmin(row_texture)
        min_texture = row_texture[min_texture_idx]

        # Mean texture for comparison
        mean_texture = np.mean(row_texture)

        # Quality: how much smoother than average
        if mean_texture > 0:
            smoothness = 1.0 - (min_texture / mean_texture)
        else:
            smoothness = 0.0

        # Confidence based on smoothness
        confidence = smoothness * 0.7  # Slightly lower weight for texture

        if smoothness > 0.3 and confidence > self.min_confidence:
            waterline_y = self.roi.y_min + min_texture_idx

            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': smoothness,
                'method': 'texture',
                'smoothness': smoothness
            }

        return None

    def _combine_results(
        self,
        results: list,
        frame_height: int
    ) -> WaterlineResult:
        """
        Combine multiple detection results.

        Uses weighted voting based on confidence.
        """
        if not results:
            return WaterlineResult(
                detected=False,
                waterline_y=None,
                confidence=0.0,
                method="none",
                quality_score=0.0,
                raw_signal={}
            )

        # Weighted average of waterline positions
        total_weight = sum(r['confidence'] for r in results)

        if total_weight > 0:
            weighted_y = sum(
                r['waterline_y'] * r['confidence']
                for r in results
            ) / total_weight

            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            avg_quality = sum(r['quality'] for r in results) / len(results)

            # Use method of highest confidence result
            best_method = max(results, key=lambda r: r['confidence'])['method']

            return WaterlineResult(
                detected=True,
                waterline_y=weighted_y,
                confidence=avg_confidence,
                method=best_method,
                quality_score=avg_quality,
                raw_signal={'detections': results}
            )

        return WaterlineResult(
            detected=False,
            waterline_y=None,
            confidence=0.0,
            method="none",
            quality_score=0.0,
            raw_signal={}
        )


def create_default_detector(
    frame_width: int = 1920,
    frame_height: int = 1080,
    roi_config: dict = None
) -> WaterlineDetector:
    """
    Create a detector with sensible defaults.

    Args:
        frame_width: Video frame width
        frame_height: Video frame height
        roi_config: Optional ROI configuration

    Returns:
        Configured WaterlineDetector
    """
    if roi_config is None:
        # Default: bottom 60% where water would appear
        roi = ROI(
            x_min=int(frame_width * 0.1),
            x_max=int(frame_width * 0.9),
            y_min=int(frame_height * 0.4),
            y_max=frame_height
        )
    else:
        roi = ROI(**roi_config)

    return WaterlineDetector(
        roi=roi,
        edge_threshold=30.0,
        blue_excess_threshold=5.0,
        min_confidence=0.3
    )
