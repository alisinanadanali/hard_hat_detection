from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .detector_tracked import TrackedDetection


@dataclass
class TrackState:
    track_id: int
    hits: int = 0
    missed: int = 0
    last_seen_sample_idx: int = -1

    score_baretli: float = 0.0
    score_baretsiz: float = 0.0

    # baretsiz maksimum güven anı
    best_baretsiz_conf: float = 0.0
    best_baretsiz_time_sec: Optional[float] = None
    best_baretsiz_frame: Optional[np.ndarray] = None
    best_baretsiz_dets: Optional[List[TrackedDetection]] = None

    def vote(self, det: TrackedDetection) -> None:
        if det.label == "baretli":
            self.score_baretli += det.conf ** 2
        elif det.label == "baretsiz":
            self.score_baretsiz += det.conf ** 2

    def final_label(self) -> str:
        return "baretsiz" if self.score_baretsiz > self.score_baretli else "baretli"
