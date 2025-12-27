import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

from core.detector_tracked import YoloTrackedHelmetDetector
from core.analyzer import VideoAnalyzer


def resource_path(relative: str) -> Path:
    """
    Geliştirmede: proje klasörü
    PyInstaller gibi paketlemede: sys._MEIPASS içi
    """
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent / relative


def default_model_path() -> Path:
    return resource_path("assets/models/best.pt")


def default_output_dir_for_video(video_path: Path) -> Path:
    """
    Videonun bulunduğu klasöre: hardhat_reports/<videoStem_timestamp>/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = video_path.parent / "hardhat_reports" / f"{video_path.stem}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _set_if_has(obj, name: str, value):
    """Detector patch'lerinde attribute'lar varsa set et, yoksa sessiz geç."""
    if hasattr(obj, name):
        setattr(obj, name, value)


def main():
    parser = argparse.ArgumentParser()

    # Zorunlu / temel
    parser.add_argument("--video", type=str, required=True, help="Analiz edilecek video dosyası")
    parser.add_argument("--model", type=str, default=str(default_model_path()), help="best.pt yolu")
    parser.add_argument("--out", type=str, default="", help="Çıktı klasörü (boşsa videonun yanına)")
    parser.add_argument("--device", type=str, default="", help="cuda:0 gibi, boşsa auto")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="botsort.yaml veya bytetrack.yaml")

    # YOLO inference
    parser.add_argument("--imgsz", type=int, default=1024, help="YOLO imgsz (örn: 1024/1280)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO track conf (tracker aday eşiği)")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU")

    # Analyzer sampling / finalize
    parser.add_argument("--sample-every-sec", type=float, default=0.1, help="Örnekleme aralığı (sn)")
    parser.add_argument("--max-missed-samples", type=int, default=15, help="Track kapanma eşiği (sample sayısı)")
    parser.add_argument("--min-hits", type=int, default=3, help="Rapora girmek için min görüm sayısı")
    parser.add_argument("--bbox-scale", type=float, default=1.2, help="Kaydedilen karelerde bbox büyütme")
    parser.add_argument("--store-full-frame", action="store_true", help="Alarm karesinde full frame sakla (annotate için önerilir)")

    # NEW: Detector tuning (patch’li detector_tracked için)
    parser.add_argument("--baretli-conf", type=float, default=0.55, help="Baretli class conf filtresi")
    parser.add_argument("--baretsiz-conf", type=float, default=0.75, help="Baretsiz class conf filtresi")
    parser.add_argument("--immediate-baretsiz-conf", type=float, default=0.88, help="Tek frame alarm için baretsiz conf")
    parser.add_argument("--baretsiz-streak-n", type=int, default=3, help="Alarm için ardışık baretsiz sayısı")
    parser.add_argument("--border-margin", type=int, default=8, help="Kenar FP kırmak için margin(px)")
    parser.add_argument("--min-w", type=int, default=12, help="Min bbox width(px)")
    parser.add_argument("--min-h", type=int, default=12, help="Min bbox height(px)")
    parser.add_argument("--min-area", type=int, default=144, help="Min bbox area(px^2)")
    parser.add_argument("--ar-min", type=float, default=0.6, help="Min aspect ratio (w/h)")
    parser.add_argument("--ar-max", type=float, default=1.8, help="Max aspect ratio (w/h)")

    # ID merge (opsiyonel)
    parser.add_argument("--merge-iou", type=float, default=0.75, help="ID merge IoU eşiği")
    parser.add_argument("--merge-max-gap", type=int, default=15, help="ID merge max gap (frame)")

    args = parser.parse_args()

    video_path = Path(args.video).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model bulunamadı: {model_path}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else default_output_dir_for_video(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YoloTrackedHelmetDetector(
        model_path=str(model_path),
        imgsz=int(args.imgsz),
        conf_thres=float(args.conf),
        iou_thres=float(args.iou),
        device=(args.device if args.device else None),
        tracker_cfg=args.tracker,
    )

    # ---- NEW: Patch’li detector ayarlarını enjekte et ----
    # class_conf
    if hasattr(detector, "class_conf") and isinstance(detector.class_conf, dict):
        detector.class_conf["baretli"] = float(args.baretli_conf)
        detector.class_conf["baretsiz"] = float(args.baretsiz_conf)

    _set_if_has(detector, "IMMEDIATE_BARETSIZ_CONF", float(args.immediate_baretsiz_conf))
    _set_if_has(detector, "BARETSIZ_STREAK_N", int(args.baretsiz_streak_n))
    _set_if_has(detector, "border_margin", int(args.border_margin))
    _set_if_has(detector, "min_w", int(args.min_w))
    _set_if_has(detector, "min_h", int(args.min_h))
    _set_if_has(detector, "min_area", int(args.min_area))
    _set_if_has(detector, "ar_min", float(args.ar_min))
    _set_if_has(detector, "ar_max", float(args.ar_max))
    _set_if_has(detector, "MERGE_IOU", float(args.merge_iou))
    _set_if_has(detector, "MERGE_MAX_GAP", int(args.merge_max_gap))

    analyzer = VideoAnalyzer(
        detector=detector,
        sample_every_sec=float(args.sample_every_sec),
        bbox_scale=float(args.bbox_scale),
        max_missed_samples=int(args.max_missed_samples),
        min_hits=int(args.min_hits),
        store_full_frame_for_alarm=bool(args.store_full_frame),
    )

    def progress(p):
        print(f"\rProgress: {p*100:.1f}%", end="")

    result = analyzer.analyze(str(video_path), str(out_dir), progress_cb=progress)

    print("\n--- DONE ---")
    print(f"Toplam kişi: {result.total_people} | Baretli: {result.baretli_count} | Baretsiz: {result.baretsiz_count}")
    print(f"Rapor klasörü: {out_dir}")


if __name__ == "__main__":
    main()

