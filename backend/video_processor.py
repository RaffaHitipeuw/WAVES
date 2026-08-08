import cv2
import numpy as np
from typing import Optional, Tuple


class VideoFrameProcessor:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.total_frames = 0

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def reset(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_count = 0

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_info(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "progress": self.frame_count / self.total_frames if self.total_frames > 0 else 0
        }
