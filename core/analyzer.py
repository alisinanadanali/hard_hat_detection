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
    final_label: str          # "baretli" | "baretsiz"
    hits: int

    # Her kişi için: final_label'e ait maksimum conf kare
    best_conf: float
    best_time_sec: float
    best_time_str: str
    best_frame_path: str

    # Baretsiz alarm varsa (opsiyonel)
    has_alarm: bool = False
    best_alarm_conf: Optional[float] = None
    best_alarm_time_sec: Optional[float] = None
    best_alarm_time_str: Optional[str] = None
    best_alarm_frame_path: Optional[str] = None


@dataclass
class AnalysisResult:
    video_path: str
    fps: float
    sample_every_sec: float
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
    ALL-TRACKS rapor:
      - Tüm track_id'ler rapora girer (hits/conf eşiği yok)
      - Baretli için de max-conf kare kaydedilir
      - Baretsiz alarm (is_alarm=True) için ayrıca alarm_best kaydedilir

    Not: init imzası geriye uyumlu tutuldu (app/run_cli kırılmasın).
    """

    def __init__(
        self,
        detector: YoloTrackedHelmetDetector,
        sample_every_sec: float = 0.1,
        bbox_scale: float = 1.2,
        max_missed_samples: int = 15,          # geriye uyum için var (kullanılmıyor)
        min_hits: int = 3,                      # geriye uyum için var (kullanılmıyor)
        store_full_frame_for_alarm: bool = True,  # geriye uyum için var (kullanılmıyor)
        include_all_tracks: bool = True,        # NEW: varsayılan True
        search_every_sec: float = 0.4,            # SEARCH modu örnekleme
        empty_time_to_search_sec: float = 1.0,    # TRACK modunda bu kadar süre boşsa SEARCH'e geç
    ):
        self.detector = detector
        self.sample_every_sec = sample_every_sec
        self.bbox_scale = bbox_scale
        self.include_all_tracks = include_all_tracks

        self._hits: Dict[int, int] = {}
        self._votes: Dict[int, Dict[str, int]] = {}
        self._best_by_label: Dict[int, Dict[str, _Snap]] = {}
        self._has_alarm: Dict[int, bool] = {}
        self._best_alarm: Dict[int, _Snap] = {}
        self.search_every_sec = float(search_every_sec)
        self.empty_time_to_search_sec = float(empty_time_to_search_sec)

    def _ensure(self, tid: int):
        if tid not in self._hits:
            self._hits[tid] = 0
            self._votes[tid] = {"baretli": 0, "baretsiz": 0}
            self._best_by_label[tid] = {"baretli": _Snap(), "baretsiz": _Snap()}
            self._has_alarm[tid] = False
            self._best_alarm[tid] = _Snap()

    def _write_best(self, frames_dir: str, tid: int, tag: str,
                    frame_bgr, dets: List[TrackedDetection]) -> str:
        # Aynı isim: üstüne yazar, klasör şişmez
        annotated = annotate_frame(
            frame_bgr,
            dets,
            bbox_scale=self.bbox_scale,
            focus_track_id=tid,      # <-- sadece bu ID
            show_others=False
        )

        fpath = os.path.join(frames_dir, f"id_{tid:05d}_{tag}_best.jpg")
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

        self.detector.reset_tracker()
        self._hits.clear()
        self._votes.clear()
        self._best_by_label.clear()
        self._has_alarm.clear()
        self._best_alarm.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        track_step = max(1, int(round(fps * self.sample_every_sec)))       # örn 0.1s
        search_step = max(1, int(round(fps * self.search_every_sec)))      # örn 0.4s

        mode = "track"  # "track" | "search"
        empty_track_streak = 0
        search_retry_left = 0          # kaç kez hızlı tekrar denenecek
        SEARCH_RETRY_N = 2             # 2 iyi başlangıç
        jitter_flip = False            # search step'i 1 frame dither etmek için
        warmup_left = 0
        WARMUP_N = int(round(1.2 / self.sample_every_sec))  # ~1.2 saniye


        frame_idx = 0                 # stream'de bulunduğumuz index (0..)
        next_process_idx = 0          # bir sonraki "decode+YOLO" yapılacak frame index

        while True:
            # 1) İşlemeyeceğimiz frame'leri decode etmeden geç
            while frame_idx < next_process_idx:
                if not cap.grab():    # decode yok (genelde çok hızlı)
                    # döngü bitir
                    frame_idx = None
                    break
                frame_idx += 1

            if frame_idx is None:
                break

            # 2) Hedef frame'i decode et + işle
            ok, frame = cap.read()
            if not ok:
                break

            cur_idx = frame_idx       # bu okunan frame'in index'i
            frame_idx += 1
            time_sec = cur_idx / fps

            if mode == "search":
                dets = self.detector.detect_only(frame)
            else:
                dets = self.detector.detect(frame)

            
            for d in dets:
                if d.track_id < 0:
                    continue
                tid = int(d.track_id)
                self._ensure(tid)

                self._hits[tid] += 1
                if d.label in self._votes[tid]:
                    self._votes[tid][d.label] += 1

                # label bazlı best (baretli + baretsiz)
                snap = self._best_by_label[tid][d.label]
                if d.conf > snap.conf:
                    snap.conf = float(d.conf)
                    snap.time_sec = float(time_sec)
                    snap.path = self._write_best(frames_dir, tid, d.label, frame, dets)

                # alarm bazlı best
                if bool(getattr(d, "is_alarm", False)):
                    self._has_alarm[tid] = True
                    asnap = self._best_alarm[tid]
                    if d.conf > asnap.conf:
                        asnap.conf = float(d.conf)
                        asnap.time_sec = float(time_sec)
                        asnap.path = self._write_best(frames_dir, tid, "alarm", frame, dets)


            # 3) Mod değişimi (SEARCH/TRACK)
            if mode == "search":
                if len(dets) > 0:
                    mode = "track"
                    warmup_left = WARMUP_N
                    empty_track_streak = 0
                    self.detector.reset_tracker()
            else:
                # mode == "track"
                if len(dets) == 0:
                    empty_track_streak += 1
                    if empty_track_streak * self.sample_every_sec >= self.empty_time_to_search_sec:
                        mode = "search"
                        warmup_left = 0
                else:
                    empty_track_streak = 0


            # 4) Bir sonraki işlenecek frame index'i (TRACK warmup + SEARCH burst/jitter)

            if mode == "track":
            # Track modunda her zaman sık örnekle
                search_retry_left = 0
                next_process_idx = cur_idx + track_step

                # warmup sayacı (mode geçişinde set ediliyor)
                if warmup_left > 0:
                    warmup_left -= 1

            else:
                # SEARCH modu (detect_only çalışıyor)
                if len(dets) == 0 and search_retry_left < SEARCH_RETRY_N:
                # boş geldiyse kısa aralıkla tekrar dene
                    search_retry_left += 1
                    next_process_idx = cur_idx + track_step
                else:
                    # normal search adımı + jitter
                    search_retry_left = 0
                    jitter = 1 if jitter_flip else 0
                    jitter_flip = not jitter_flip
                    next_process_idx = cur_idx + search_step + jitter


            # progress
            if progress_cb and total_frames > 0:
                progress_cb(min(1.0, frame_idx / total_frames))

        cap.release()

        # Tüm track'ler rapora
        people: List[PersonResult] = []
        for tid in sorted(self._hits.keys()):
            hits = self._hits[tid]
            votes = self._votes[tid]

            has_alarm = bool(self._has_alarm.get(tid, False))
            final_label = "baretsiz" if votes["baretsiz"] > votes["baretli"] else "baretli"
            if has_alarm:
                final_label = "baretsiz"

            snap_final = self._best_by_label[tid][final_label]
            if snap_final.conf < 0 or not snap_final.path:
                # edge-case fallback
                other = "baretli" if final_label == "baretsiz" else "baretsiz"
                snap_final = self._best_by_label[tid][other]
            if snap_final.conf < 0 or not snap_final.path:
                continue  # çok nadir

            pr = PersonResult(
                track_id=tid,
                final_label=final_label,
                hits=hits,
                best_conf=float(snap_final.conf),
                best_time_sec=float(snap_final.time_sec),
                best_time_str=format_time(float(snap_final.time_sec)),
                best_frame_path=snap_final.path,
                has_alarm=has_alarm,
            )

            if has_alarm:
                asnap = self._best_alarm[tid]
                if asnap.conf >= 0 and asnap.path:
                    pr.best_alarm_conf = float(asnap.conf)
                    pr.best_alarm_time_sec = float(asnap.time_sec)
                    pr.best_alarm_time_str = format_time(float(asnap.time_sec))
                    pr.best_alarm_frame_path = asnap.path

            people.append(pr)

        total_people = len(people)
        baretsiz_count = sum(1 for p in people if p.final_label == "baretsiz")
        baretli_count = total_people - baretsiz_count

        result = AnalysisResult(
            video_path=video_path,
            fps=fps,
            sample_every_sec=self.sample_every_sec,
            total_people=total_people,
            baretli_count=baretli_count,
            baretsiz_count=baretsiz_count,
            people=people,
        )

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
                "best_conf", "best_time_sec", "best_time_str", "best_frame_path",
                "has_alarm", "best_alarm_conf", "best_alarm_time_sec", "best_alarm_time_str", "best_alarm_frame_path"
            ])
            for p in result.people:
                w.writerow([
                    p.track_id, p.final_label, p.hits,
                    f"{p.best_conf:.3f}",
                    f"{p.best_time_sec:.3f}",
                    p.best_time_str,
                    p.best_frame_path,
                    "1" if p.has_alarm else "0",
                    "" if p.best_alarm_conf is None else f"{p.best_alarm_conf:.3f}",
                    "" if p.best_alarm_time_sec is None else f"{p.best_alarm_time_sec:.3f}",
                    p.best_alarm_time_str or "",
                    p.best_alarm_frame_path or "",
                ])

