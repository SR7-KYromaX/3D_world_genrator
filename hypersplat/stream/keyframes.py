"""Keyframe gate: keep only frames that are sharp and show something new.

This is where the stream's footprint is decided: a 30 fps camera produces ~1800 frames a
minute, nearly all redundant. A frame is accepted when

    novelty    |gray difference| (0-255) of 96 px thumbnails vs the last keyframe
               >= min_motion (ACE-Zero's own motion measure, video.avg_motion, uses the
               same scale), and
    sharpness  variance of the Laplacian >= blur_ratio x the running median of recent
               frames (a blurry frame is skipped; the next sharp one with enough novelty
               is taken instead).

max_interval forces a keyframe after that many seconds without one, so a slowly drifting
view still gets refreshed.

Novelty should measure camera motion, not video noise. A phone's auto exposure brightens or
darkens the whole frame, and a person walking through changes part of it; with a plain mean
difference both read as "new", and ACE then fails to register the redundant keyframes (a
handheld room scan accepted 381 of 1640 frames and registered 112). So:

    exposure_invariant  the frame's thumbnail is gain/offset-matched to the last keyframe's
                        (same mean and std) before differencing
    cells               the difference is averaged per cell of a rows x cols grid and the
                        median cell is the novelty: camera motion changes most cells, a
                        moving person or a flickering light only a few

exposure_invariant=False, cells=None is the original global mean difference.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

THUMB_WIDTH = 96
SHARPNESS_WIDTH = 320


@dataclass
class GateResult:
    accept: bool
    reason: str
    novelty: float
    sharpness: float


def _gray(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(image, SHARPNESS_WIDTH), cv2.CV_32F).var())


class KeyframeGate:
    def __init__(self, min_motion: float = 8.0, blur_ratio: float = 0.6,
                 max_interval: Optional[float] = 10.0, history: int = 30,
                 exposure_invariant: bool = True, cells: Optional[Tuple[int, int]] = (4, 6)):
        self.min_motion = min_motion
        self.blur_ratio = blur_ratio
        self.max_interval = max_interval
        self.exposure_invariant = exposure_invariant
        self.cells = cells
        self._sharp = deque(maxlen=history)
        self._last_thumb: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self._motion_sum = 0.0
        self.offered = 0
        self.accepted = 0

    @property
    def avg_motion(self) -> Optional[float]:
        """Mean novelty between consecutive keyframes (ACE schedule input)."""
        return self._motion_sum / (self.accepted - 1) if self.accepted > 1 else None

    def offer(self, image: np.ndarray, t: float) -> GateResult:
        self.offered += 1
        thumb = _gray(image, THUMB_WIDTH).astype(np.float32)
        sharp = sharpness(image)
        self._sharp.append(sharp)

        if self._last_thumb is None:
            return self._accept(thumb, t, 0.0, sharp, "first")
        novelty = self._novelty(thumb)
        median = float(np.median(self._sharp))
        if len(self._sharp) >= 5 and sharp < self.blur_ratio * median:
            return GateResult(False, "blurry", novelty, sharp)
        if novelty >= self.min_motion:
            return self._accept(thumb, t, novelty, sharp, "novel")
        if self.max_interval is not None and t - self._last_t >= self.max_interval:
            return self._accept(thumb, t, novelty, sharp, "interval")
        return GateResult(False, "redundant", novelty, sharp)

    def _novelty(self, thumb: np.ndarray) -> float:
        last = self._last_thumb
        if self.exposure_invariant:
            std = float(thumb.std())
            if std > 1e-3:
                thumb = (thumb - thumb.mean()) * (float(last.std()) / std) + float(last.mean())
        diff = np.abs(thumb - last)
        if not self.cells:
            return float(diff.mean())
        rows, cols = self.cells
        means = [c.mean() for band in np.array_split(diff, rows, axis=0)
                 for c in np.array_split(band, cols, axis=1) if c.size]
        return float(np.median(means))

    def _accept(self, thumb, t, novelty, sharp, reason) -> GateResult:
        if self._last_thumb is not None:
            self._motion_sum += novelty
        self._last_thumb, self._last_t = thumb, t
        self.accepted += 1
        return GateResult(True, reason, novelty, sharp)
