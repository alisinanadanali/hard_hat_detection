from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackedDetection:
    track_id: int
    xyxy: np.ndarray   # float32 [x1,y1,x2,y2]
    conf: float
    label: str         # "baretli" | "baretsiz"
    cls_id: int


class YoloTrackedHelmetDetector:
    """
    Ultralytics YOLOv8 + tracker wrapper.
    Classes expected: "baretli" and "baretsiz"
    """
    def __init__(
        self,
        model_path: str,
        imgsz: int = 1024,
        conf_thres: float = 0.25,
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

    def detect(self, frame_bgr: np.ndarray) -> List[TrackedDetection]:
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
        )[0]

        out: List[TrackedDetection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out

        # tracker id yoksa (bazı edge-case) boş dön
        if getattr(res.boxes, "id", None) is None:
            return out

        ids = res.boxes.id.detach().cpu().numpy().astype(int)

        for i, b in enumerate(res.boxes):
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            raw_name = self.names.get(cls_id, str(cls_id))
            label = self._canonical_label(raw_name)
            if label is None:
                continue

            xyxy = b.xyxy[0].detach().cpu().numpy().astype(np.float32)
            out.append(
                TrackedDetection(
                    track_id=int(ids[i]),
                    xyxy=xyxy,
                    conf=conf,
                    label=label,
                    cls_id=cls_id,
                )
            )
        return out
