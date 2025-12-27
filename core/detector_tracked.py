from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedDetection:
    track_id: int
    xyxy: np.ndarray   # float32 [x1,y1,x2,y2]
    conf: float
    label: str         # "baretli" | "baretsiz"
    cls_id: int
    is_alarm: bool = False  # NEW: alarm/ihlal tetik flag'i

@dataclass
class TrackState:
    hits: int = 0
    last_seen: int = 0
    vote_baretli: int = 0
    vote_baretsiz: int = 0
    baretsiz_streak: int = 0
    last_xyxy: Optional[np.ndarray] = None  # NEW: ID merge için son bbox

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)

class YoloTrackedHelmetDetector:
    """
    Ultralytics YOLOv8 + tracker wrapper.
    Classes expected: "baretli" and "baretsiz"
    """
    def __init__(
        self,
        model_path: str,
        imgsz: int = 1024,
        conf_thres: float = 0.15,
        iou_thres: float = 0.45,
        device: Optional[str] = None,
        tracker_cfg: str = "botsort.yaml",  # moving camera için daha sağlam
    ):
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.tracker_cfg = tracker_cfg
        self.names = self.model.names

        # ---- hızlı saha ayarları ----
        self.class_conf = {"baretli": 0.55, "baretsiz": 0.70}  # FP kırıcı
        self.min_w = 12
        self.min_h = 12
        self.min_area = 12 * 12
        self.border_margin = 8
        self.ar_min, self.ar_max = 0.6, 1.8

        # (2) Alarm mantığı (NEW)
        # -------------------------
        self.IMMEDIATE_BARETSIZ_CONF = 0.80  # NEW: "çok eminsen" tek frame alarm
        self.BARETSIZ_STREAK_N = 2           # NEW: değilse 3 ardışık baretsiz alarm
        # Not: Overlay için minimum hit beklemiyoruz (anlık görünsün diye)

        # Track stabilizasyon
        self.frame_idx = 0
        # NEW: raw track_id -> logical_id map + logical state
        self.raw_to_logical: Dict[int, int] = {}
        self.logical_states: Dict[int, TrackState] = {}
        self.logical_next_id = 1
        self.stale_after = 30  # 30 frame görünmeyen state'i sil

        # (4) ID parçalanmasına hızlı pansuman: ID merge (NEW)
        # -------------------------
        self.MERGE_MAX_GAP = 15     # NEW: son 15 frame içinde kaybolduysa birleştir
        self.MERGE_IOU = 0.75       # NEW: bbox IoU bu değerden yüksekse aynı kişi say
        # Not: Bu "hızlı pansuman". tracker param tuning ile birlikte daha da iyileşir.

        # in-frame dedup
        self.dedup_iou = 0.7

    def _canonical_label(self, raw_name: str) -> Optional[str]:
        n = raw_name.strip().lower()
        if n == "baretli":
            return "baretli"
        if n == "baretsiz":
            return "baretsiz"
        return None

    def reset_tracker(self) -> None:
        """
        Aynı process içinde arka arkaya farklı video analiz edeceksen iyi olur.
        En garantisi yeni detector instance oluşturmaktır.
        Bu method best-effort bir reset denemesidir.
        """
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers = None
        except Exception:
            pass
        self.frame_idx = 0
        self.raw_to_logical.clear()
        self.logical_states.clear()
        self.logical_next_id = 1

    def _valid_det(self, xyxy: np.ndarray, label: str, conf: float, W: int, H: int) -> bool:
        if conf < self.class_conf.get(label, 0.6):
            return False

        x1, y1, x2, y2 = map(float, xyxy)
        w, h = x2 - x1, y2 - y1
        if w < self.min_w or h < self.min_h or (w * h) < self.min_area:
            return False

        m = self.border_margin
        if x1 < m or y1 < m or x2 > (W - m) or y2 > (H - m):
            return False

        ar = w / (h + 1e-6)
        if ar < self.ar_min or ar > self.ar_max:
            return False

        return True

    def _dedup(self, dets: List[TrackedDetection]) -> List[TrackedDetection]:
        # conf'a göre azalan sırala, IoU yüksek olanları ele
        dets = sorted(dets, key=lambda d: d.conf, reverse=True)
        kept: List[TrackedDetection] = []
        for d in dets:
            ok = True
            for k in kept:
                if iou_xyxy(d.xyxy, k.xyxy) >= self.dedup_iou:
                    ok = False
                    break
            if ok:
                kept.append(d)
        return kept
    
    # NEW: raw track_id parçalanınca logical_id ile birleştirme
    def _assign_logical_id(self, raw_id: int, xyxy: np.ndarray, active_logicals: set[int]) -> int:
        # raw_id daha önce maplenmişse direkt kullan
        if raw_id in self.raw_to_logical:
            return self.raw_to_logical[raw_id]

        # Aksi halde "yakın zamanda kaybolan" logical'larla IoU eşleştir
        best_lid = None
        best_iou = 0.0
        for lid, st in self.logical_states.items():
            if lid in active_logicals:
                continue  # aynı frame'de zaten aktif olanı kullanma (yanlış merge riskini azaltır)
            if st.last_xyxy is None:
                continue
            if (self.frame_idx - st.last_seen) > self.MERGE_MAX_GAP:
                continue

            iou = iou_xyxy(xyxy, st.last_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_lid = lid

        if best_lid is not None and best_iou >= self.MERGE_IOU:
            self.raw_to_logical[raw_id] = best_lid
            return best_lid

        # Yeni logical id oluştur
        lid = self.logical_next_id
        self.logical_next_id += 1
        self.raw_to_logical[raw_id] = lid
        return lid

    def detect(self, frame_bgr: np.ndarray) -> List[TrackedDetection]:
        self.frame_idx += 1
        H, W = frame_bgr.shape[:2]
        # persist=True => ID sürekliliği
        res = self.model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.tracker_cfg,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            agnostic_nms=True,   # <-- kritik
            max_det=300,
        )[0]


        if res.boxes is None or len(res.boxes) == 0:
            self._cleanup_states()
            return []

        if getattr(res.boxes, "id", None) is None:
            # Tracker ID üretemedi ama detection var: varlık sinyali dönelim.
            out: List[TrackedDetection] = []
            for b in res.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                raw_name = self.names.get(cls_id, str(cls_id))
                label = self._canonical_label(raw_name)
                if label is None:
                    continue

                xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float32)
                out.append(
                    TrackedDetection(
                        track_id=-1,  # geçici idsiz
                        xyxy=xyxy,
                        conf=conf,
                        label=label,
                        cls_id=cls_id,
                    )
                )
            return out


        ids = res.boxes.id.detach().cpu().numpy().astype(int)

        # 1) ham det -> canonical label + filtre
        raw: List[TrackedDetection] = []
        for i, b in enumerate(res.boxes):
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            raw_name = self.names.get(cls_id, str(cls_id))
            label = self._canonical_label(raw_name)
            if label is None:
                continue

            xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float32)
            
            if not self._valid_det(xyxy, label, conf, W, H):
                continue

            raw.append(
                TrackedDetection(
                    track_id=int(ids[i]),
                    xyxy=xyxy,
                    conf=conf,
                    label=label,
                    cls_id=cls_id,
                    is_alarm=False,
                )
            )

        if not raw:
            # stale cleanup
            self._cleanup_states()
            return []
        
        # 2) in-frame dedup (ek güvenlik)
        raw = self._dedup(raw)

        active_logicals: set[int] = set()

        # 3) Track-level stabilizasyon (min_hits + majority vote)
        out: List[TrackedDetection] = []
        for d in raw:
            raw_id = d.track_id
            logical_id = self._assign_logical_id(raw_id, d.xyxy, active_logicals)
            active_logicals.add(logical_id)

            st = self.logical_states.get(logical_id, TrackState())
            st.hits += 1
            st.last_seen = self.frame_idx
            st.last_xyxy = d.xyxy.copy()
            if d.label == "baretsiz":
                st.vote_baretsiz += 1
                st.baretsiz_streak += 1
            else:
                st.vote_baretli += 1
                st.baretsiz_streak = 0
            
            self.logical_states[logical_id] = st

            total_votes = st.vote_baretli + st.vote_baretsiz
            if total_votes < 3:
                stable_label = d.label
            else:
                stable_label = "baretsiz" if st.vote_baretsiz > st.vote_baretli else "baretli"
            # majority
            
            # (B) Alarm (ihlal) mantığı (NEW)
            # -----------------------------
            is_alarm = False
            if stable_label == "baretsiz":
                # 1) Çok yüksek güven: anında alarm
                if d.conf >= self.IMMEDIATE_BARETSIZ_CONF:
                    is_alarm = True
                # 2) Değilse: kısa streak ile alarm
                elif st.baretsiz_streak >= self.BARETSIZ_STREAK_N:
                    is_alarm = True

            
            # çıktı objesini yaz
            d.track_id = logical_id          # NEW: dışarıya logical_id veriyoruz
            d.label = stable_label
            d.is_alarm = is_alarm
            out.append(d)

        self._cleanup_states()
        return out
    
    def detect_only(self, frame_bgr: np.ndarray) -> List[TrackedDetection]:
        """
        Tracker olmadan sadece YOLO detect.
        Bazı ultralytics sürümlerinde predict() list/generator/tek Result dönebilir.
        Bu yüzden sonuç okuması robust yapıldı.
        track_id = -1 döner.
        """
        pred = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            stream=False,   # <-- özellikle ekledim
        )

        # pred None olabilir -> boş dön
        if pred is None:
            return []

        # pred list / generator / tek Result olabilir
        res = None
        if isinstance(pred, list):
            if len(pred) == 0:
                return []
            res = pred[0]
        else:
            # iterator/generator ise ilk sonucu çek
            try:
                res = next(iter(pred))
            except Exception:
                # tek Result olabilir
                res = pred

        if res is None or res.boxes is None or len(res.boxes) == 0:
            return []

        # names dict veya list olabilir
        names = getattr(self.model, "names", None) or self.names
        out: List[TrackedDetection] = []

        for b in res.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())

            if isinstance(names, dict):
                raw_name = names.get(cls_id, str(cls_id))
            elif isinstance(names, (list, tuple)):
                raw_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
            else:
                raw_name = str(cls_id)

            label = self._canonical_label(raw_name)
            if label is None:
                continue

            xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float32)
            out.append(
                TrackedDetection(
                    track_id=-1,  # SEARCH modunda ID yok
                    xyxy=xyxy,
                    conf=conf,
                    label=label,
                    cls_id=cls_id,
                )
            )

        return out



    def _cleanup_states(self):
        # stale logical states temizle
        dead_logicals = []
        for lid, st in self.logical_states.items():
            if (self.frame_idx - st.last_seen) > self.stale_after:
                dead_logicals.append(lid)
        for lid in dead_logicals:
            self.logical_states.pop(lid, None)

        # raw_to_logical map'ten artık olmayan logical'lara gidenleri temizle
        alive = set(self.logical_states.keys())
        dead_raw = [rid for rid, lid in self.raw_to_logical.items() if lid not in alive]
        for rid in dead_raw:
            self.raw_to_logical.pop(rid, None)
