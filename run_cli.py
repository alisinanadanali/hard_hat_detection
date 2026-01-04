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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Analiz edilecek video dosyası")
    parser.add_argument("--model", type=str, default=str(default_model_path()), help="best.pt yolu")
    parser.add_argument("--out", type=str, default="", help="Çıktı klasörü (boşsa videonun yanına)")
    parser.add_argument("--device", type=str, default="", help="cuda:0 gibi, boşsa auto")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="botsort.yaml veya bytetrack.yaml")
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
        imgsz=960,
        conf_thres=0.25,
        iou_thres=0.45,
        device=(args.device if args.device else None),
        tracker_cfg=args.tracker,
    )

    analyzer = VideoAnalyzer(
        detector=detector,
        sample_every_sec=0.1,
        bbox_scale=1.2,
        max_missed_samples=15,
        min_hits=3,
    )

    def progress(p):
        print(f"\rProgress: {p*100:.1f}%", end="")

    result = analyzer.analyze(str(video_path), str(out_dir), progress_cb=progress)
    print("\n--- DONE ---")
    print(f"Toplam kişi: {result.total_people} | Baretli: {result.baretli_count} | Baretsiz: {result.baretsiz_count}")
    print(f"Rapor klasörü: {out_dir}")


if __name__ == "__main__":
    main()
