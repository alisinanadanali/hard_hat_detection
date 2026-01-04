import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QComboBox, QCheckBox,
)

from core.detector_tracked import YoloTrackedHelmetDetector
from core.analyzer import VideoAnalyzer


# Ürün kuralı: UI/Rapor/Kare tarafında bu eşik altı ASLA görünmesin
REPORT_MIN_CONF = 0.70

# Tracker bozulmasın / ID kopmasın diye: state güncellemek için daha düşük “iç eşik”
# (UI/Rapor/Kare ile ilgisi yok)
TRACK_KEEP_MIN_CONF = 0.15


def resource_path(relative: str) -> Path:
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / relative
    return Path(__file__).resolve().parent / relative


def default_model_path() -> Path:
    return resource_path("assets/models/best.pt")


def default_output_dir_for_video(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = video_path.parent / "hardhat_reports" / f"{video_path.stem}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _set_if_has(obj, name: str, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def _getattr(obj, name: str, default=None):
    return getattr(obj, name, default)


def make_proxy_video_10fps(
    src_video: str,
    cache_root: Path,
    target_fps: int = 10,
    max_width: int | None = 1280,
    crf: int = 23,
    preset: str = "veryfast",
) -> str | None:
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if not ffmpeg:
        return None

    src = Path(src_video)
    if not src.exists():
        return None

    cache_root.mkdir(parents=True, exist_ok=True)

    st = src.stat()
    key = f"{st.st_size}_{st.st_mtime_ns}_{target_fps}_{max_width}_{crf}_{preset}"
    proxy_path = cache_root / f"{src.stem}_proxy_{key}.mp4"

    if proxy_path.exists():
        return str(proxy_path)

    vf = f"fps={int(target_fps)}"
    if max_width is not None:
        vf += f",scale={int(max_width)}:-2"

    cmd = [
        ffmpeg, "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(proxy_path),
    ]
    subprocess.run(cmd, check=True)
    return str(proxy_path)


class AnalyzeWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(object, str)
    error = Signal(str)

    def __init__(
        self,
        video_path: str,
        model_path: str,
        out_dir: str,
        use_proxy_10fps: bool,
        sample_every_sec: float,
        store_baretli_frames: bool,
        parent=None
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.out_dir = out_dir
        self.use_proxy_10fps = bool(use_proxy_10fps)
        self.sample_every_sec = float(sample_every_sec)
        self.store_baretli_frames = bool(store_baretli_frames)

    @Slot()
    def run(self):
        try:
            import core.detector_tracked as det_mod
            import core.analyzer as analyzer_mod

            self.log.emit(f"[DEBUG] detector_tracked.py path = {det_mod.__file__}")
            self.log.emit(f"[DEBUG] analyzer.py path = {analyzer_mod.__file__}")
            self.log.emit(f"[DEBUG] VideoAnalyzer init args = {analyzer_mod.VideoAnalyzer.__init__.__code__.co_varnames}")
            self.log.emit(f"[DEBUG] Detector init args = {det_mod.YoloTrackedHelmetDetector.__init__.__code__.co_varnames}")

            self.log.emit(f"Video: {self.video_path}")
            self.log.emit(f"Model: {self.model_path}")
            self.log.emit(f"Çıktı: {self.out_dir}")
            self.log.emit(f"Ayarlar: tracker=botsort | imgsz=1024 | sample={self.sample_every_sec:.2f}s")
            self.log.emit(f"Kural: UI/Rapor/Kare >= {REPORT_MIN_CONF:.2f} (altı tamamen yok sayılır)")

            video_to_analyze = self.video_path

            if self.use_proxy_10fps:
                try:
                    out_dir_p = Path(self.out_dir)
                    reports_root = out_dir_p.parent
                    proxy_cache = reports_root / "_proxy_cache"

                    self.log.emit("Hızlandırma: 10 FPS proxy hazırlanıyor (FFmpeg)...")
                    proxy = make_proxy_video_10fps(
                        src_video=self.video_path,
                        cache_root=proxy_cache,
                        target_fps=10,
                        max_width=1280,
                        crf=23,
                        preset="veryfast",
                    )
                    if proxy:
                        video_to_analyze = proxy
                        self.log.emit(f"Proxy hazır: {proxy}")
                    else:
                        self.log.emit("FFmpeg bulunamadı -> proxy atlandı, normal video ile devam.")
                except Exception as e:
                    self.log.emit(f"Proxy üretilemedi -> normal video ile devam: {e}")

            detector = YoloTrackedHelmetDetector(
                model_path=self.model_path,
                imgsz=960,
                conf_thres=0.15,   # tracker adayı (düşük kalabilir)
                iou_thres=0.45,
                device=None,
                tracker_cfg="botsort.yaml",
            )

            # 0.79 altı ürün çıktısı: görünmesin
            _set_if_has(detector, "report_min_conf", REPORT_MIN_CONF)
            _set_if_has(detector, "min_output_conf", REPORT_MIN_CONF)  # bazı sürümlerde bu isim kullanılıyor olabilir

            # Track kopmasın diye (UI/rapor değil): düşük iç eşik
            _set_if_has(detector, "edge_min_conf", TRACK_KEEP_MIN_CONF)         # bazı sürümlerde “edge_min_conf” vardı
            _set_if_has(detector, "track_keep_min_conf", TRACK_KEEP_MIN_CONF)   # bazı sürümlerde bu isim tercih edilir

            # class_conf varsa: tracking tarafını öldürmesin diye düşük tut
            if hasattr(detector, "class_conf") and isinstance(detector.class_conf, dict):
                detector.class_conf["baretli"] = TRACK_KEEP_MIN_CONF
                detector.class_conf["baretsiz"] = TRACK_KEEP_MIN_CONF

            # geometri/kalite filtreleri
            _set_if_has(detector, "border_margin", 8)
            _set_if_has(detector, "min_w", 12)
            _set_if_has(detector, "min_h", 12)
            _set_if_has(detector, "min_area", 144)
            _set_if_has(detector, "ar_min", 0.6)
            _set_if_has(detector, "ar_max", 1.8)

            _set_if_has(detector, "MERGE_IOU", 0.75)
            _set_if_has(detector, "MERGE_MAX_GAP", 15)

            analyzer = VideoAnalyzer(
                detector=detector,
                sample_every_sec=self.sample_every_sec,
                bbox_scale=1.2,
            )

            # Analyzer seviyesinde de rapor eşiğini set et
            _set_if_has(analyzer, "report_min_conf", REPORT_MIN_CONF)

            # “edge-case” kaldırıldığı için: varsa edge_min_conf’u report ile eşitle (range = boş)
            _set_if_has(analyzer, "edge_min_conf", REPORT_MIN_CONF)

            # baretli kare yazma opsiyonu (varsa)
            _set_if_has(analyzer, "store_baretli_frames", self.store_baretli_frames)

            def cb(p):
                self.progress.emit(int(p * 100))

            result = analyzer.analyze(video_to_analyze, self.out_dir, progress_cb=cb)
            self.finished.emit(result, self.out_dir)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baret Tespit Analizi - Prototip")

        self.video_path: str = ""
        self.model_path: str = str(default_model_path())
        self.out_dir: str = ""
        self._last_result = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        row1 = QHBoxLayout()
        self.lbl_model = QLabel(f"Model: {self.model_path}")
        self.btn_video = QPushButton("Video Seç")
        self.btn_start = QPushButton("Analizi Başlat")
        self.btn_start.setEnabled(False)
        row1.addWidget(self.lbl_model, 1)
        row1.addWidget(self.btn_video)
        row1.addWidget(self.btn_start)
        layout.addLayout(row1)

        self.lbl_video = QLabel("Video: (seçilmedi)")
        layout.addWidget(self.lbl_video)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.lbl_summary = QLabel("Sonuç: -")
        self.lbl_summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.lbl_summary)

        opt = QHBoxLayout()
        opt.addWidget(QLabel("Sıralama:"))
        self.cmb_sort = QComboBox()
        self.cmb_sort.addItem("Güvene göre (yüksek → düşük)", "conf")
        self.cmb_sort.addItem("Zamana göre (erken → geç)", "time")
        opt.addWidget(self.cmb_sort)

        self.chk_show_baretli = QCheckBox("Baretli kayıtları da göster")
        self.chk_show_baretli.setChecked(True)
        opt.addWidget(self.chk_show_baretli)

        opt.addWidget(QLabel("Inference:"))
        self.cmb_sample = QComboBox()
        self.cmb_sample.addItem("0.10 s (10 analiz/sn)", 0.10)
        self.cmb_sample.addItem("0.20 s (5 analiz/sn)", 0.20)
        opt.addWidget(self.cmb_sample)

        self.chk_proxy = QCheckBox("Hızlandır: Proxy video (10 FPS)")
        self.chk_proxy.setChecked(True)
        opt.addWidget(self.chk_proxy)

        opt.addStretch(1)
        layout.addLayout(opt)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Track ID", "Sonuç", "Conf", "Zaman (s)", "Kare Yolu"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 2)

        row2 = QHBoxLayout()
        self.preview = QLabel("Kare önizleme")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setStyleSheet("border: 1px solid #999;")
        row2.addWidget(self.preview, 2)

        right = QVBoxLayout()
        self.btn_open_file = QPushButton("Seçili Kareyi Aç")
        self.btn_open_folder = QPushButton("Çıktı Klasörünü Aç")
        self.btn_open_file.setEnabled(False)
        self.btn_open_folder.setEnabled(False)
        right.addWidget(self.btn_open_file)
        right.addWidget(self.btn_open_folder)
        right.addStretch(1)
        row2.addLayout(right, 1)
        layout.addLayout(row2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self.btn_video.clicked.connect(self.choose_video)
        self.btn_start.clicked.connect(self.start_analysis)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.btn_open_file.clicked.connect(self.open_selected_file)
        self.btn_open_folder.clicked.connect(self.open_output_folder)

        self.cmb_sort.currentIndexChanged.connect(self.refresh_table_from_last)
        self.chk_show_baretli.stateChanged.connect(self.refresh_table_from_last)

        self.thread: QThread | None = None
        self.worker: AnalyzeWorker | None = None

        if not Path(self.model_path).exists():
            QMessageBox.warning(
                self,
                "Model bulunamadı",
                f"Model dosyası bulunamadı:\n{self.model_path}\n\n"
                "Lütfen assets/models/best.pt yoluna koy."
            )

    @Slot()
    def choose_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Video Seç",
            str(Path.home()),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)"
        )
        if not file_path:
            return

        self.video_path = file_path
        self.lbl_video.setText(f"Video: {self.video_path}")

        self.out_dir = str(default_output_dir_for_video(Path(self.video_path)))
        self.log_append(f"Çıktı klasörü hazırlandı: {self.out_dir}")

        self.btn_start.setEnabled(Path(self.model_path).exists())

    @Slot()
    def start_analysis(self):
        if not self.video_path:
            QMessageBox.information(self, "Uyarı", "Önce bir video seç.")
            return
        if not Path(self.model_path).exists():
            QMessageBox.warning(self, "Uyarı", f"Model bulunamadı:\n{self.model_path}")
            return

        sample_every_sec = float(self.cmb_sample.currentData() or 0.10)

        self.btn_start.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_open_file.setEnabled(False)
        self.btn_open_folder.setEnabled(False)

        self.table.setRowCount(0)
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Analiz çalışıyor...")
        self.progress.setValue(0)
        self.lbl_summary.setText("Sonuç: (analiz ediliyor)")
        self._last_result = None

        self.thread = QThread()
        self.worker = AnalyzeWorker(
            video_path=self.video_path,
            model_path=self.model_path,
            out_dir=self.out_dir,
            use_proxy_10fps=self.chk_proxy.isChecked(),
            sample_every_sec=sample_every_sec,
            store_baretli_frames=self.chk_show_baretli.isChecked(),
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_append)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _person_conf_time_path(self, p):
        label = _getattr(p, "final_label", "")

        if label == "baretli":
            conf = _getattr(p, "best_baretli_conf", None)
            tsec = _getattr(p, "best_baretli_time_sec", None)
            path = _getattr(p, "best_baretli_frame_path", None)
        else:
            conf = _getattr(p, "best_baretsiz_conf", None)
            tsec = _getattr(p, "best_baretsiz_time_sec", None)
            path = _getattr(p, "best_baretsiz_frame_path", None)

        if conf is None:
            conf = _getattr(p, "best_conf", None)
        if tsec is None:
            tsec = _getattr(p, "best_time_sec", None)
        if path is None:
            path = _getattr(p, "best_frame_path", "")

        conf_f = float(conf) if conf is not None else 0.0
        tsec_f = float(tsec) if tsec is not None else float("inf")
        path_s = str(path or "")

        return conf_f, tsec_f, path_s

    def populate_table(self, result):
        sort_mode = self.cmb_sort.currentData() or "conf"
        show_baretli = self.chk_show_baretli.isChecked()

        people = list(result.people)

        # Güvenlik: Analyzer eski sürüm olsa bile UI’da 0.79 altı ASLA görünmesin
        people = [p for p in people if self._person_conf_time_path(p)[0] >= REPORT_MIN_CONF]

        if not show_baretli:
            people = [p for p in people if _getattr(p, "final_label", "") == "baretsiz"]

        if sort_mode == "time":
            people.sort(key=lambda p: (_getattr(p, "final_label", "") != "baretsiz", self._person_conf_time_path(p)[1]))
        else:
            people.sort(key=lambda p: (_getattr(p, "final_label", "") != "baretsiz", -self._person_conf_time_path(p)[0]))

        self.table.setRowCount(len(people))

        for r, p in enumerate(people):
            self.table.setItem(r, 0, QTableWidgetItem(str(_getattr(p, "track_id", ""))))
            self.table.setItem(r, 1, QTableWidgetItem(str(_getattr(p, "final_label", ""))))

            conf, tsec, path = self._person_conf_time_path(p)

            self.table.setItem(r, 2, QTableWidgetItem(f"{conf:.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem("" if tsec == float("inf") else f"{tsec:.2f}"))
            self.table.setItem(r, 4, QTableWidgetItem(path))

        selected = False
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 1)
            if it and it.text() == "baretsiz":
                self.table.selectRow(r)
                selected = True
                break
        if not selected and self.table.rowCount() > 0:
            self.table.selectRow(0)

        if self.table.rowCount() == 0:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Kayıt yok (0.79 eşiği nedeniyle boş olabilir).")
            self.btn_open_file.setEnabled(False)
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Kayıt seç: tablodan bir satır seç.")
            self.btn_open_file.setEnabled(True)

    @Slot(object, str)
    def on_finished(self, result, out_dir: str):
        self.log_append("Analiz tamamlandı.")
        self.btn_video.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        self._last_result = result

        self.lbl_summary.setText(
            f"Sonuç: Toplam={result.total_people} | Baretli={result.baretli_count} | Baretsiz={result.baretsiz_count}"
        )

        self.populate_table(result)

        if getattr(result, "baretsiz_count", 0) == 0:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Baretsiz kayıt bulunamadı (0.79 eşiği nedeniyle olabilir).")

    @Slot()
    def refresh_table_from_last(self):
        if self._last_result is None:
            return
        self.populate_table(self._last_result)

    @Slot(str)
    def on_error(self, msg: str):
        self.btn_video.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Hata oluştu.")
        QMessageBox.critical(self, "Hata", msg)
        self.log_append(f"[HATA] {msg}")

    def log_append(self, text: str):
        self.log.append(text)

    @Slot()
    def on_table_select(self):
        items = self.table.selectedItems()
        if not items:
            self.btn_open_file.setEnabled(False)
            return

        row = self.table.currentRow()
        path_item = self.table.item(row, 4)
        frame_path = path_item.text().strip() if path_item else ""

        if frame_path and Path(frame_path).exists():
            pix = QPixmap(frame_path)
            if not pix.isNull():
                scaled = pix.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview.setPixmap(scaled)
                self.preview.setText("")
            else:
                self.preview.setPixmap(QPixmap())
                self.preview.setText("Önizleme yüklenemedi.")
            self.btn_open_file.setEnabled(True)
        else:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Bu kayıt için kare yok (baretli kayıtlar dosya yazmadan raporlanıyor olabilir).")
            self.btn_open_file.setEnabled(False)

    @Slot()
    def open_selected_file(self):
        row = self.table.currentRow()
        if row < 0:
            return
        frame_path = (self.table.item(row, 4).text() or "").strip()
        if not frame_path or not Path(frame_path).exists():
            return
        QDesktopServices.openUrl(Path(frame_path).as_uri())

    @Slot()
    def open_output_folder(self):
        if not self.out_dir:
            return
        p = Path(self.out_dir)
        if p.exists():
            QDesktopServices.openUrl(p.as_uri())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 750)
    w.show()
    sys.exit(app.exec())

