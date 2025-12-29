from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional
import os
import json
import csv

import cv2

from .detector_tracked import YoloTrackedHelmetDetector, TrackedDetection
from .draw import annotate_frame, format_time


@dataclass
class PersonResult:
    track_id: int
    final_label: str  # "baretli" | "baretsiz"

    # Sadece rapor eşiği (>= report_min_conf) içindeki sayımlar:
    hits_report: int

    # UI için "final label" best'i (>= report_min_conf):
    best_conf: float
    best_time_sec: float
    best_time_str: str
    best_frame_path: str

    # Detay (rapor eşiği içinde label bazlı):
    best_baretli_conf: Optional[float] = None
    best_baretli_time_sec: Optional[float] = None
    best_baretli_frame_path: Optional[str] = None

    best_baretsiz_conf: Optional[float] = None
    best_baretsiz_time_sec: Optional[float] = None
    best_baretsiz_frame_path: Optional[str] = None


@dataclass
class AnalysisResult:
    video_path: str
    out_dir: str
    total_people: int
    baretli_count: int
    baretsiz_count: int
    people: List[PersonResult]


@dataclass
class _Snap:
    conf: float = -1.0
    time_sec: float = 0.0
    path: str = ""


class VideoAnalyzer:
    """
    Kurallar (Edge-case KALDIRILDI):
      - report_min_conf (default 0.79) altı rapora/UI'a/KAYIT karelerine girmez.
      - 0.79 altındaki tespitler tamamen yok sayılır (ne rapora, ne frames'e).
      - Final label kararı SADECE report oyları üzerinden verilir.
      - Rapora ancak en az 1 adet report det'i olan track girer.

    Not:
      - Detector tarafı tracking için daha düşük conf ile çalışabilir (örn 0.15),
        ama analyzer ürün çıktısı için 0.79 filtresini “kesin” uygular.
    """

    def __init__(
        self,
        detector: YoloTrackedHelmetDetector,
        sample_every_sec: float = 0.1,
        bbox_scale: float = 1.2,
        report_min_conf: float = 0.79,
        store_baretli_frames: bool = False,
        **kwargs,  # geri uyum
    ):
        self.detector = detector
        self.sample_every_sec = float(sample_every_sec)
        self.bbox_scale = float(bbox_scale)

        self.report_min_conf = float(report_min_conf)
        self.store_baretli_frames = bool(store_baretli_frames)

        # track istatistikleri (sadece report bandı)
        self._hits_report: Dict[int, int] = {}
        self._votes_report: Dict[int, Dict[str, int]] = {}

        # report best (label bazlı)
        self._best_report: Dict[int, Dict[str, _Snap]] = {}

    def _ensure(self, tid: int):
        if tid not in self._hits_report:
            self._hits_report[tid] = 0
            self._votes_report[tid] = {"baretli": 0, "baretsiz": 0}
            self._best_report[tid] = {"baretli": _Snap(), "baretsiz": _Snap()}

    def _write_best(
        self,
        out_path: str,
        tid: int,
        label: str,
        frame_bgr,
        dets: List[TrackedDetection],
        tag: str,
    ) -> str:
        annotated = annotate_frame(
            frame_bgr,
            dets,
            bbox_scale=self.bbox_scale,
            focus_track_id=tid,
            show_others=False,
        )
        fname = f"id_{tid:05d}_{label}_{tag}_best.jpg"
        fpath = os.path.join(out_path, fname)
        cv2.imwrite(fpath, annotated)
        return fpath

    def analyze(
        self,
        video_path: str,
        out_dir: str,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> AnalysisResult:
        os.makedirs(out_dir, exist_ok=True)
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # reset
        self.detector.reset_tracker()
        self._hits_report.clear()
        self._votes_report.clear()
        self._best_report.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(float(fps) * self.sample_every_sec)))

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % step == 0:
                time_sec = frame_idx / float(fps)
                dets = self.detector.detect(frame)

                for d in dets:
                    # ÜRÜN KURALI: 0.79 altı tamamen yok say
                    if float(d.conf) < self.report_min_conf:
                        continue

                    tid = int(d.track_id)
                    self._ensure(tid)

                    self._hits_report[tid] += 1

                    if d.label in self._votes_report[tid]:
                        self._votes_report[tid][d.label] += 1

                    # best snapshot güncelle (label bazlı)
                    if d.label == "baretli" and not self.store_baretli_frames:
                        # conf/time sakla ama dosya yazma
                        snap_r = self._best_report[tid]["baretli"]
                        if d.conf > snap_r.conf:
                            snap_r.conf = float(d.conf)
                            snap_r.time_sec = float(time_sec)
                            snap_r.path = ""
                    else:
                        snap_r = self._best_report[tid][d.label]
                        if d.conf > snap_r.conf:
                            snap_r.conf = float(d.conf)
                            snap_r.time_sec = float(time_sec)
                            snap_r.path = self._write_best(
                                frames_dir, tid, d.label, frame, dets, tag="report"
                            )

            frame_idx += 1
            if progress_cb and total_frames > 0:
                progress_cb(min(1.0, frame_idx / total_frames))

        cap.release()

        # --- rapor üret (sadece report hit'i olanlar) ---
        people: List[PersonResult] = []

        for tid in sorted(self._hits_report.keys()):
            if self._hits_report.get(tid, 0) <= 0:
                continue

            votes = self._votes_report[tid]
            final_label = "baretsiz" if votes["baretsiz"] > votes["baretli"] else "baretli"

            snap_final = self._best_report[tid][final_label]
            best_conf = float(snap_final.conf if snap_final.conf >= 0 else 0.0)
            best_time_sec = float(snap_final.time_sec)
            best_path = snap_final.path or ""

            pr = PersonResult(
                track_id=tid,
                final_label=final_label,
                hits_report=int(self._hits_report.get(tid, 0)),
                best_conf=best_conf,
                best_time_sec=best_time_sec,
                best_time_str=format_time(best_time_sec),
                best_frame_path=best_path,
            )

            # label bazlı report best (dosya yazılmamış olabilir)
            br = self._best_report[tid]["baretli"]
            if br.conf >= 0:
                pr.best_baretli_conf = float(br.conf)
                pr.best_baretli_time_sec = float(br.time_sec)
                pr.best_baretli_frame_path = br.path or ""

            bzr = self._best_report[tid]["baretsiz"]
            if bzr.conf >= 0:
                pr.best_baretsiz_conf = float(bzr.conf)
                pr.best_baretsiz_time_sec = float(bzr.time_sec)
                pr.best_baretsiz_frame_path = bzr.path or ""

            people.append(pr)

        total_people = len(people)
        baretsiz_count = sum(1 for p in people if p.final_label == "baretsiz")
        baretli_count = total_people - baretsiz_count

        result = AnalysisResult(
            video_path=video_path,
            out_dir=out_dir,
            total_people=total_people,
            baretli_count=baretli_count,
            baretsiz_count=baretsiz_count,
            people=people,
        )

        self._write_json(result, os.path.join(out_dir, "report.json"))
        self._write_csv(result, os.path.join(out_dir, "report.csv"))

        return result

    def _write_json(self, result: AnalysisResult, path: str) -> None:
        data = asdict(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _write_csv(self, result: AnalysisResult, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "track_id", "final_label",
                "hits_report",
                "best_conf", "best_time_sec", "best_time_str", "best_frame_path",
                "best_baretli_conf", "best_baretli_time_sec", "best_baretli_frame_path",
                "best_baretsiz_conf", "best_baretsiz_time_sec", "best_baretsiz_frame_path",
            ])
            for p in result.people:
                w.writerow([
                    p.track_id, p.final_label,
                    p.hits_report,
                    f"{p.best_conf:.3f}",
                    f"{p.best_time_sec:.3f}",
                    p.best_time_str,
                    p.best_frame_path,
                    "" if p.best_baretli_conf is None else f"{p.best_baretli_conf:.3f}",
                    "" if p.best_baretli_time_sec is None else f"{p.best_baretli_time_sec:.3f}",
                    p.best_baretli_frame_path or "",
                    "" if p.best_baretsiz_conf is None else f"{p.best_baretsiz_conf:.3f}",
                    "" if p.best_baretsiz_time_sec is None else f"{p.best_baretsiz_time_sec:.3f}",
                    p.best_baretsiz_frame_path or "",
                ])


