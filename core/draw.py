from __future__ import annotations
from typing import List
import cv2
import numpy as np
from .detector_tracked import TrackedDetection


def expand_bbox(xyxy: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(float)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = max(0, int(round(cx - bw / 2.0)))
    ny1 = max(0, int(round(cy - bh / 2.0)))
    nx2 = min(w - 1, int(round(cx + bw / 2.0)))
    ny2 = min(h - 1, int(round(cy + bh / 2.0)))
    return np.array([nx1, ny1, nx2, ny2], dtype=np.int32)


def format_time(sec: float) -> str:
    mm = int(sec // 60)
    ss = sec - mm * 60
    return f"{mm:02d}:{ss:06.3f}"  # 00:12.340 gibi


def annotate_frame(
    frame_bgr: np.ndarray,
    dets: List[TrackedDetection],
    bbox_scale: float = 1.2,
    focus_track_id: int | None = None,
    show_others: bool = False,
    **kwargs,  # <-- önemli: beklenmeyen keyword'leri yutar
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for d in dets:
        if focus_track_id is not None and d.track_id != focus_track_id and not show_others:
            continue

        bb = expand_bbox(d.xyxy, bbox_scale, w, h)
        x1, y1, x2, y2 = map(int, bb.tolist())

        is_alarm = bool(getattr(d, "is_alarm", False))

        # focus dışındakiler soluk olsun istersen
        if focus_track_id is not None and d.track_id != focus_track_id and show_others:
            color = (180, 180, 180)
            thickness = 1
            tag = ""
        else:
            if is_alarm:
                color = (0, 0, 255)       # kırmızı
                thickness = 4
                tag = "ALARM"
            elif d.label == "baretsiz":
                color = (0, 165, 255)     # turuncu
                thickness = 3
                tag = "ADAY"
            else:
                color = (0, 255, 0)       # yeşil
                thickness = 2
                tag = ""

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        txt = f"{tag} ID:{d.track_id} {d.label} {d.conf:.2f}".strip()
        cv2.putText(out, txt, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    return out
