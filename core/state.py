from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector_tracked import TrackedDetection


@dataclass
class AlarmEvent:
    """
    Bir kişi/track için TEK kez üretilen alarm çıktısı.
    Alarm kare olarak: o track'te alarm koşulu oluşan frameler içinde
    en yüksek conf'a sahip olan frame seçilir.
    """
    track_id: int
    conf: float
    time_sec: Optional[float]
    sample_idx: int
    xyxy: np.ndarray
    label: str  # "baretsiz" (alarm üretildiği için pratikte baretsiz olur)
    frame_bgr: Optional[np.ndarray] = None         # saklamak istersen (opsiyonel)
    crop_bgr: Optional[np.ndarray] = None          # daha hafif (önerilir)
    det_snapshot: Optional[TrackedDetection] = None
    frame_dets_snapshot: Optional[List[TrackedDetection]] = None  # o frame’deki tüm dets (opsiyonel)


@dataclass
class TrackState:
    track_id: int

    hits: int = 0
    missed: int = 0
    last_seen_sample_idx: int = -1
    last_seen_time_sec: Optional[float] = None

    last_label: Optional[str] = None
    last_conf: float = 0.0

    # alarm gözlemi oldu mu?
    alarm_armed: bool = False

    # bu track için alarm üretildi mi? (tek sefer)
    alarm_emitted: bool = False

    # alarm için en iyi kare (max conf)
    best_alarm_conf: float = 0.0
    best_alarm_time_sec: Optional[float] = None
    best_alarm_sample_idx: int = -1
    best_alarm_xyxy: Optional[np.ndarray] = None
    best_alarm_label: Optional[str] = None

    best_alarm_frame_bgr: Optional[np.ndarray] = None
    best_alarm_crop_bgr: Optional[np.ndarray] = None
    best_alarm_det: Optional[TrackedDetection] = None
    best_alarm_frame_dets: Optional[List[TrackedDetection]] = None

    def update_seen(
        self,
        det: TrackedDetection,
        sample_idx: int,
        time_sec: Optional[float],
    ) -> None:
        self.hits += 1
        self.missed = 0
        self.last_seen_sample_idx = sample_idx
        self.last_seen_time_sec = time_sec
        self.last_label = det.label
        self.last_conf = det.conf

    def update_missed(self) -> None:
        self.missed += 1

    def _make_crop(self, frame_bgr: np.ndarray, xyxy: np.ndarray, pad: int = 6) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
        return frame_bgr[y1:y2, x1:x2].copy()

    def consider_alarm_candidate(
        self,
        det: TrackedDetection,
        sample_idx: int,
        time_sec: Optional[float],
        frame_bgr: Optional[np.ndarray],
        frame_dets: Optional[List[TrackedDetection]],
        store_full_frame: bool,
        store_crop: bool,
        crop_pad: int,
    ) -> None:
        """
        Alarm adayı sadece det.is_alarm True ise işlenir.
        Bu track içinde alarm-adayı frameler arasından en yüksek conf seçilir.
        """
        if not det.is_alarm:
            return

        self.alarm_armed = True

        # en iyi alarm frame'i: conf maksimize
        if det.conf <= self.best_alarm_conf:
            return

        self.best_alarm_conf = det.conf
        self.best_alarm_time_sec = time_sec
        self.best_alarm_sample_idx = sample_idx
        self.best_alarm_xyxy = det.xyxy.copy()
        self.best_alarm_label = det.label
        self.best_alarm_det = det

        if frame_dets is not None:
            # hafif kopya (TrackedDetection listesi küçük olur genelde)
            self.best_alarm_frame_dets = list(frame_dets)

        if frame_bgr is not None:
            if store_crop:
                self.best_alarm_crop_bgr = self._make_crop(frame_bgr, det.xyxy, pad=crop_pad)
            if store_full_frame:
                self.best_alarm_frame_bgr = frame_bgr.copy()


class TrackStateManager:
    """
    Detector çıktılarından (TrackedDetection) track bazlı state yönetimi.

    Amaç:
    - bir track için TEK alarm üretmek
    - alarm kare olarak: det.is_alarm True olan frameler içinde max conf'u seçmek
    - alarmı track sona erince (missed >= end_after_missed) emit etmek
    """

    def __init__(
        self,
        end_after_missed: int = 15,
        store_full_frame: bool = False,
        store_crop: bool = True,
        crop_pad: int = 8,
    ):
        self.end_after_missed = int(end_after_missed)
        self.store_full_frame = bool(store_full_frame)
        self.store_crop = bool(store_crop)
        self.crop_pad = int(crop_pad)

        self.tracks: Dict[int, TrackState] = {}

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        dets: List[TrackedDetection],
        sample_idx: int,
        time_sec: Optional[float] = None,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[AlarmEvent]:
        """
        Her frame/sample çağrılır.

        Girdi:
        - dets: detector_tracked.detect() çıktısı (logical track_id + stabilize label + is_alarm)
        - sample_idx: işlenen frame index'i (senin pipeline'daki sample index)
        - time_sec: opsiyonel timestamp
        - frame_bgr: opsiyonel frame (alarm görseli saklanacaksa gerekli)

        Çıktı:
        - Bu update çağrısında kapanan track'lerden üretilen alarm event listesi
        """
        alarms: List[AlarmEvent] = []

        seen_ids = set()

        # 1) görülen track'leri güncelle
        for det in dets:
            tid = int(det.track_id)
            seen_ids.add(tid)

            st = self.tracks.get(tid)
            if st is None:
                st = TrackState(track_id=tid)
                self.tracks[tid] = st

            st.update_seen(det, sample_idx=sample_idx, time_sec=time_sec)
            st.consider_alarm_candidate(
                det=det,
                sample_idx=sample_idx,
                time_sec=time_sec,
                frame_bgr=frame_bgr,
                frame_dets=dets,
                store_full_frame=self.store_full_frame,
                store_crop=self.store_crop,
                crop_pad=self.crop_pad,
            )

        # 2) görülmeyen track'ler missed++
        for tid, st in list(self.tracks.items()):
            if tid not in seen_ids:
                st.update_missed()

        # 3) sona eren track'leri kapat + alarm emit
        for tid, st in list(self.tracks.items()):
            if st.missed < self.end_after_missed:
                continue

            # track kapanıyor: eğer alarm_armed ise ve daha önce emit edilmediyse tek alarm üret
            if st.alarm_armed and (not st.alarm_emitted) and st.best_alarm_xyxy is not None:
                st.alarm_emitted = True

                alarms.append(
                    AlarmEvent(
                        track_id=st.track_id,
                        conf=st.best_alarm_conf,
                        time_sec=st.best_alarm_time_sec,
                        sample_idx=st.best_alarm_sample_idx,
                        xyxy=st.best_alarm_xyxy.copy(),
                        label=st.best_alarm_label or "baretsiz",
                        frame_bgr=st.best_alarm_frame_bgr,
                        crop_bgr=st.best_alarm_crop_bgr,
                        det_snapshot=st.best_alarm_det,
                        frame_dets_snapshot=st.best_alarm_frame_dets,
                    )
                )

            # track state'i temizle (memory şişmesin)
            self.tracks.pop(tid, None)

        return alarms

    def get_active_tracks(self) -> List[TrackState]:
        return list(self.tracks.values())
