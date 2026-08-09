import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class WaterlineResult:
    detected: bool
    waterline_y: Optional[float]
    confidence: float
    method: str
    quality_score: float
    raw_signal: Dict[str, Any]
    candidates: List[Dict] = field(default_factory=list)


@dataclass
class ROI:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
            "usable_pixels": self.width * self.height if self.width > 0 and self.height > 0 else 0
        }


class WaterlineDetector:
    def __init__(
        self,
        roi: Optional[ROI] = None,
        edge_threshold: float = 30.0,
        blue_excess_threshold: float = 5.0,
        min_confidence: float = 0.3
    ):
        self.roi = roi or ROI(x_min=0, x_max=1920, y_min=540, y_max=1080)
        self.edge_threshold = edge_threshold
        self.blue_excess_threshold = blue_excess_threshold
        self.min_confidence = min_confidence
        self._frame_count = 0
        self._avg_brightness = 128.0
        self._last_candidates: List[Dict] = []
        self._last_edge_signal: Optional[np.ndarray] = None
        self._last_color_signal: Optional[np.ndarray] = None
        self._last_texture_signal: Optional[np.ndarray] = None
        self._previous_waterline: Optional[float] = None

    def detect(self, frame: np.ndarray) -> WaterlineResult:
        self._frame_count += 1
        roi_frame = self._extract_roi(frame)
        if roi_frame is None or roi_frame.size == 0:
            result = WaterlineResult(
                detected=False,
                waterline_y=None,
                confidence=0.0,
                method="none",
                quality_score=0.0,
                raw_signal={},
                candidates=[]
            )
            self._last_candidates = []
            return result
        self._update_statistics(roi_frame)
        results: List[Dict] = []
        edge_result = self._detect_by_edges(roi_frame)
        if edge_result:
            results.append(edge_result)
        color_result = self._detect_by_color(roi_frame)
        if color_result:
            results.append(color_result)
        texture_result = self._detect_by_texture(roi_frame)
        if texture_result:
            results.append(texture_result)
        self._last_candidates = results
        combined = self._combine_results(results, frame.shape[0])
        self._previous_waterline = combined.waterline_y
        return combined

    def _extract_roi(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x_min = max(0, min(self.roi.x_min, w - 1))
        x_max = max(0, min(self.roi.x_max, w))
        y_min = max(0, min(self.roi.y_min, h - 1))
        y_max = max(0, min(self.roi.y_max, h))
        if x_max <= x_min or y_max <= y_min:
            return None
        return frame[y_min:y_max, x_min:x_max]

    def _update_statistics(self, roi_frame: np.ndarray):
        brightness = np.mean(roi_frame)
        self._avg_brightness = 0.9 * self._avg_brightness + 0.1 * brightness

    def _detect_by_edges(self, roi_frame: np.ndarray) -> Optional[Dict]:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_edges = np.abs(sobely)
        row_edge_strength = np.mean(horizontal_edges, axis=1)
        self._last_edge_signal = row_edge_strength.copy()
        max_edge_idx = np.argmax(row_edge_strength)
        max_edge_strength = row_edge_strength[max_edge_idx]
        noise_estimate = np.std(row_edge_strength)
        quality = min(1.0, max_edge_strength / (noise_estimate * 3)) if noise_estimate > 0 else 0.0
        confidence = min(1.0, max_edge_strength / 100.0)
        if max_edge_strength > self.edge_threshold and confidence > self.min_confidence:
            waterline_y = self.roi.y_min + max_edge_idx
            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': quality,
                'method': 'edge',
                'edge_strength': float(max_edge_strength),
                'noise_estimate': float(noise_estimate),
                'edge_idx_roi': int(max_edge_idx),
                'signal': row_edge_strength.tolist(),
                'signal_peak': float(max_edge_strength),
                'selected': False
            }
        return None

    def _detect_by_color(self, roi_frame: np.ndarray) -> Optional[Dict]:
        h, w = roi_frame.shape[:2]
        b, g, r = cv2.split(roi_frame.astype(np.float32))
        blue_excess = b - r
        row_blue_excess = np.mean(blue_excess, axis=1)
        self._last_color_signal = row_blue_excess.copy()
        max_blue_idx = np.argmax(row_blue_excess)
        max_blue_excess = row_blue_excess[max_blue_idx]
        quality = min(1.0, max_blue_excess / 20.0) if max_blue_excess > 0 else 0.0
        confidence = min(1.0, max_blue_excess / self.blue_excess_threshold)
        if max_blue_excess > self.blue_excess_threshold and confidence > self.min_confidence:
            waterline_y = self.roi.y_min + max_blue_idx
            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': quality,
                'method': 'color',
                'blue_excess': float(max_blue_excess),
                'color_idx_roi': int(max_blue_idx),
                'signal': row_blue_excess.tolist(),
                'signal_peak': float(max_blue_excess),
                'selected': False
            }
        return None

    def _detect_by_texture(self, roi_frame: np.ndarray) -> Optional[Dict]:
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_strength = np.abs(laplacian)
        row_texture = np.mean(texture_strength, axis=1)
        self._last_texture_signal = row_texture.copy()
        min_texture_idx = np.argmin(row_texture)
        min_texture = row_texture[min_texture_idx]
        mean_texture = np.mean(row_texture)
        smoothness = 1.0 - (min_texture / mean_texture) if mean_texture > 0 else 0.0
        confidence = smoothness * 0.7
        if smoothness > 0.3 and confidence > self.min_confidence:
            waterline_y = self.roi.y_min + min_texture_idx
            return {
                'waterline_y': waterline_y,
                'confidence': confidence,
                'quality': smoothness,
                'method': 'texture',
                'smoothness': float(smoothness),
                'texture_idx_roi': int(min_texture_idx),
                'signal': row_texture.tolist(),
                'signal_peak': float(min_texture),
                'selected': False
            }
        return None

    def _combine_results(self, results: List[Dict], frame_height: int) -> WaterlineResult:
        if not results:
            self._last_candidates = []
            return WaterlineResult(
                detected=False,
                waterline_y=None,
                confidence=0.0,
                method="none",
                quality_score=0.0,
                raw_signal={},
                candidates=[]
            )
        total_weight = sum(r['confidence'] for r in results)
        if total_weight > 0:
            weighted_y = sum(r['waterline_y'] * r['confidence'] for r in results) / total_weight
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            avg_quality = sum(r['quality'] for r in results) / len(results)
            best_method = max(results, key=lambda r: r['confidence'])['method']
            best_idx = max(range(len(results)), key=lambda i: results[i]['confidence'])
            for i, r in enumerate(results):
                r['selected'] = (i == best_idx)
            return WaterlineResult(
                detected=True,
                waterline_y=weighted_y,
                confidence=avg_confidence,
                method=best_method,
                quality_score=avg_quality,
                raw_signal={'detections': results},
                candidates=results
            )
        return WaterlineResult(
            detected=False,
            waterline_y=None,
            confidence=0.0,
            method="none",
            quality_score=0.0,
            raw_signal={},
            candidates=[]
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        stability = 'unknown'
        if self._previous_waterline is not None and self._last_candidates:
            latest = self._last_candidates[0]['waterline_y'] if self._last_candidates else None
            if latest is not None:
                delta = abs(latest - self._previous_waterline)
                if delta < 5:
                    stability = 'stable'
                elif delta < 20:
                    stability = 'unstable'
                else:
                    stability = 'jittering'
        return {
            'roi': self.roi.to_dict(),
            'frame_count': self._frame_count,
            'avg_brightness': round(self._avg_brightness, 1),
            'candidate_count': len(self._last_candidates),
            'previous_waterline': self._previous_waterline,
            'detection_stability': stability,
            'edge_threshold': self.edge_threshold,
            'blue_excess_threshold': self.blue_excess_threshold,
            'min_confidence': self.min_confidence,
            'has_edge_signal': self._last_edge_signal is not None,
            'has_color_signal': self._last_color_signal is not None,
            'has_texture_signal': self._last_texture_signal is not None,
        }


def create_default_detector(
    frame_width: int = 1920,
    frame_height: int = 1080,
    roi_config: dict = None
) -> WaterlineDetector:
    if roi_config is None:
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
