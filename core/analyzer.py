from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional
import os
import json
import csv
import cv2

from .detector_tracked import YoloTrackedHelmetDetector, TrackedDetection
from .state import TrackState
from .draw import annotate_frame, format_time


@dataclass
class PersonResult:
    track_id: int
    final_label: str
    hits: int
    score_baretli: float
    score_baretsiz: float

    # baretsiz ise:
    best_baretsiz_conf: Optional[float] = None
    best_baretsiz_time_sec: Optional[float] = None
    best_baretsiz_time_str: Optional[str] = None
    best_frame_path: Optional[str] = None


@dataclass
class AnalysisResult:
    video_path: str
    fps: float
    sample_every_sec: float
    total_people: int
    baretli_count: int
    baretsiz_count: int
    people: List[PersonResult]


class VideoAnalyzer:
    def __init__(
        self,
        detector: YoloTrackedHelmetDetector,
        sample_every_sec: float = 0.1,
        bbox_scale: float = 1.2,
        max_missed_samples: int = 15,  # 0.1s örnekleme -> 1.5s yoksa finalize
        min_hits: int = 3,
    ):
        self.detector = detector
        self.sample_every_sec = sample_every_sec
        self.bbox_scale = bbox_scale
        self.max_missed_samples = max_missed_samples
        self.min_hits = min_hits

        self._states: Dict[int, TrackState] = {}
        self._finalized: List[TrackState] = []

    def _finalize_state(self, st: TrackState, frames_dir: str) -> Optional[PersonResult]:
        if st.hits < self.min_hits:
            return None

        label = st.final_label()
        pr = PersonResult(
            track_id=st.track_id,
            final_label=label,
            hits=st.hits,
            score_baretli=st.score_baretli,
            score_baretsiz=st.score_baretsiz,
        )

        if label == "baretsiz" and st.best_baretsiz_frame is not None and st.best_baretsiz_dets is not None:
            annotated = annotate_frame(st.best_baretsiz_frame, st.best_baretsiz_dets, bbox_scale=self.bbox_scale)
            tsec = float(st.best_baretsiz_time_sec or 0.0)
            tstr = format_time(tsec)
            fname = f"id_{st.track_id:05d}_t_{tsec:.2f}s_conf_{st.best_baretsiz_conf:.2f}.jpg"
            fpath = os.path.join(frames_dir, fname)
            cv2.imwrite(fpath, annotated)

            pr.best_baretsiz_conf = float(st.best_baretsiz_conf)
            pr.best_baretsiz_time_sec = tsec
            pr.best_baretsiz_time_str = tstr
            pr.best_frame_path = fpath

        return pr

    def analyze(
        self,
        video_path: str,
        out_dir: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> AnalysisResult:
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Yeni video için state reset
        self._states.clear()
        self._finalized.clear()
        self.detector.reset_tracker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps * self.sample_every_sec)))

        frame_idx = 0
        sample_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                time_sec = frame_idx / fps
                dets = self.detector.detect(frame)

                # Bu örnek frame’de görünen ID seti
                seen_ids = set()

                # Update seen states
                for d in dets:
                    seen_ids.add(d.track_id)
                    st = self._states.get(d.track_id)
                    if st is None:
                        st = TrackState(track_id=d.track_id)
                        self._states[d.track_id] = st

                    st.hits += 1
                    st.missed = 0
                    st.last_seen_sample_idx = sample_idx

                    st.vote(d)

                    # baretsiz maksimum anı sakla (bu frame’in tüm dets’i ile)
                    if d.label == "baretsiz" and d.conf > st.best_baretsiz_conf:
                        st.best_baretsiz_conf = d.conf
                        st.best_baretsiz_time_sec = time_sec
                        st.best_baretsiz_frame = frame.copy()
                        st.best_baretsiz_dets = [
                            TrackedDetection(
                                track_id=dd.track_id,
                                xyxy=dd.xyxy.copy(),
                                conf=dd.conf,
                                label=dd.label,
                                cls_id=dd.cls_id,
                            )
                            for dd in dets
                        ]

                # Miss update (görünmeyenler)
                to_finalize = []
                for tid, st in self._states.items():
                    if tid not in seen_ids:
                        st.missed += 1
                        if st.missed > self.max_missed_samples:
                            to_finalize.append(tid)

                # Finalize
                for tid in to_finalize:
                    st = self._states.pop(tid, None)
                    if st is not None:
                        self._finalized.append(st)

                sample_idx += 1

            frame_idx += 1
            if progress_cb and total_frames > 0:
                progress_cb(min(1.0, frame_idx / total_frames))

        cap.release()

        # Kalanları finalize et
        for st in list(self._states.values()):
            self._finalized.append(st)
        self._states.clear()

        # Person results
        people: List[PersonResult] = []
        for st in self._finalized:
            pr = self._finalize_state(st, frames_dir)
            if pr is not None:
                people.append(pr)

        total_people = len(people)
        baretli_count = sum(1 for p in people if p.final_label == "baretli")
        baretsiz_count = sum(1 for p in people if p.final_label == "baretsiz")

        result = AnalysisResult(
            video_path=video_path,
            fps=fps,
            sample_every_sec=self.sample_every_sec,
            total_people=total_people,
            baretli_count=baretli_count,
            baretsiz_count=baretsiz_count,
            people=people,
        )

        # write reports
        self._write_json(result, os.path.join(out_dir, "report.json"))
        self._write_csv(result, os.path.join(out_dir, "report.csv"))

        return result

    def _write_json(self, result: AnalysisResult, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    def _write_csv(self, result: AnalysisResult, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "track_id", "final_label", "hits",
                "score_baretli", "score_baretsiz",
                "best_baretsiz_conf", "best_baretsiz_time_sec", "best_baretsiz_time_str",
                "best_frame_path"
            ])
            for p in result.people:
                w.writerow([
                    p.track_id, p.final_label, p.hits,
                    f"{p.score_baretli:.6f}", f"{p.score_baretsiz:.6f}",
                    "" if p.best_baretsiz_conf is None else f"{p.best_baretsiz_conf:.3f}",
                    "" if p.best_baretsiz_time_sec is None else f"{p.best_baretsiz_time_sec:.3f}",
                    p.best_baretsiz_time_str or "",
                    p.best_frame_path or "",
                ])
