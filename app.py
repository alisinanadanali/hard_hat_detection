import os
import sys
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QTableWidget, QTableWidgetItem, QMessageBox, QCheckBox, QComboBox
)

from core.detector_tracked import YoloTrackedHelmetDetector
from core.analyzer import VideoAnalyzer


def resource_path(relative: str) -> Path:
    """
    Geliştirmede: proje klasörü
    PyInstaller paketlemede: sys._MEIPASS içi
    """
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
    """Detector patch'lerinde attribute'lar varsa set et, yoksa sessiz geç."""
    if hasattr(obj, name):
        setattr(obj, name, value)


class AnalyzeWorker(QObject):
    progress = Signal(int)        # 0..100
    log = Signal(str)
    finished = Signal(object, str)  # (result, out_dir)
    error = Signal(str)

    def __init__(self, video_path: str, model_path: str, out_dir: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.out_dir = out_dir

    @Slot()
    def run(self):
        try:
            import core.analyzer as analyzer_mod
            self.log.emit(f"[DEBUG] analyzer.py path = {analyzer_mod.__file__}")
            self.log.emit(f"[DEBUG] VideoAnalyzer init args = {analyzer_mod.VideoAnalyzer.__init__.__code__.co_varnames}")

            self.log.emit(f"Video: {self.video_path}")
            self.log.emit(f"Model: {self.model_path}")
            self.log.emit(f"Çıktı: {self.out_dir}")

            # Yeni pipeline notu (net ve kısa)
            self.log.emit("Ayarlar: tracker=botsort | imgsz=1024 | sample=0.1s")
            self.log.emit("Mantık: 1 kişi = 1 alarm | alarm karesi = max conf (alarm frameleri içinde)")

            # Tracker aday eşiği düşük; FP'yi biz filtreliyoruz
            detector = YoloTrackedHelmetDetector(
                model_path=self.model_path,
                imgsz=1024,
                conf_thres=0.15,   # <-- önemli (önceden 0.25)
                iou_thres=0.45,
                device=None,                 # istersen "cuda:0"
                tracker_cfg="botsort.yaml",
            )

            # ---- Detector tuning (GUI için sabit, run_cli ile aynı yaklaşım) ----
            if hasattr(detector, "class_conf") and isinstance(detector.class_conf, dict):
                detector.class_conf["baretli"] = 0.55
                detector.class_conf["baretsiz"] = 0.75

            _set_if_has(detector, "IMMEDIATE_BARETSIZ_CONF", 0.88)
            _set_if_has(detector, "BARETSIZ_STREAK_N", 3)

            _set_if_has(detector, "border_margin", 8)
            _set_if_has(detector, "min_w", 12)
            _set_if_has(detector, "min_h", 12)
            _set_if_has(detector, "min_area", 144)
            _set_if_has(detector, "ar_min", 0.6)
            _set_if_has(detector, "ar_max", 1.8)

            _set_if_has(detector, "MERGE_IOU", 0.75)
            _set_if_has(detector, "MERGE_MAX_GAP", 15)

            # Analyzer: alarm karesi için full frame saklansın (annotate yazdırmak için)
            analyzer = VideoAnalyzer(
                detector=detector,
                sample_every_sec=0.1,
                bbox_scale=1.2,
                max_missed_samples=15,
                min_hits=3,
                store_full_frame_for_alarm=True,  # <-- yeni
            )

            def cb(p):
                self.progress.emit(int(p * 100))

            result = analyzer.analyze(self.video_path, self.out_dir, progress_cb=cb)
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

        # UI
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # Top row: model + video select
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

        # Summary
        self.lbl_summary = QLabel("Sonuç: -")
        self.lbl_summary.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.lbl_summary)

        # Filters / Sorting
        row_sort = QHBoxLayout()

        self.chk_show_baretli = QCheckBox("Baretli kayıtları göster")
        self.chk_show_baretli.setChecked(True)

        self.cmb_sort = QComboBox()
        self.cmb_sort.addItems([
            "Baretsiz üstte, zaman (grup içi)",
            "Zamana göre (tümü)",
            "Baretsiz üstte, conf",
        ])

        row_sort.addWidget(self.chk_show_baretli)
        row_sort.addWidget(QLabel("Sıralama:"))
        row_sort.addWidget(self.cmb_sort, 1)
        layout.addLayout(row_sort)

        self.chk_show_baretli.stateChanged.connect(self.refresh_table)
        self.cmb_sort.currentIndexChanged.connect(self.refresh_table)

        # Table
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Track ID", "Sonuç", "Conf", "Zaman (s)", "Kare Yolu"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 2)

        # Preview + open buttons
        row2 = QHBoxLayout()
        self.preview = QLabel("Kare önizleme (baretsiz alarm için seçilen max-conf kare)")
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

        # Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        # Signals
        self.btn_video.clicked.connect(self.choose_video)
        self.btn_start.clicked.connect(self.start_analysis)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.btn_open_file.clicked.connect(self.open_selected_file)
        self.btn_open_folder.clicked.connect(self.open_output_folder)

        # Thread placeholders
        self.thread: QThread | None = None
        self.worker: AnalyzeWorker | None = None
        self._last_result = None

        # Validate model exists
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

        # Default out dir next to video
        self.out_dir = str(default_output_dir_for_video(Path(self.video_path)))
        self.log_append(f"Çıktı klasörü hazırlandı: {self.out_dir}")

        # Enable start if model exists
        self.btn_start.setEnabled(Path(self.model_path).exists())

    @Slot()
    def start_analysis(self):
        if not self.video_path:
            QMessageBox.information(self, "Uyarı", "Önce bir video seç.")
            return
        if not Path(self.model_path).exists():
            QMessageBox.warning(self, "Uyarı", f"Model bulunamadı:\n{self.model_path}")
            return

        # UI lock
        self.btn_start.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_open_file.setEnabled(False)
        self.btn_open_folder.setEnabled(False)
        self.table.setRowCount(0)
        self.preview.setText("Analiz çalışıyor...")
        self.progress.setValue(0)
        self.lbl_summary.setText("Sonuç: (analiz ediliyor)")

        # Thread setup
        self.thread = QThread()
        self.worker = AnalyzeWorker(self.video_path, self.model_path, self.out_dir)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log_append)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        # cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(object, str)
    def on_finished(self, result, out_dir: str):
        self.log_append("Analiz tamamlandı.")
        self.btn_video.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        self.lbl_summary.setText(
            f"Sonuç: Toplam={result.total_people} | Baretli={result.baretli_count} | Baretsiz={result.baretsiz_count}"
        )

        self._last_result = result
        self.refresh_table()

       # Fill table
        people = list(result.people)
        people.sort(key=lambda p: (p.final_label != "baretsiz", -float(getattr(p, "best_conf", getattr(p, "best_baretsiz_conf", 0.0)) or 0.0)))

        def _get(obj, name, default=None):
            return getattr(obj, name, default)

        self.table.setRowCount(len(people))
        for r, p in enumerate(people):
            self.table.setItem(r, 0, QTableWidgetItem(str(p.track_id)))
            self.table.setItem(r, 1, QTableWidgetItem(p.final_label))

            conf = _get(p, "best_conf", _get(p, "best_baretsiz_conf", None))
            tsec = _get(p, "best_time_sec", _get(p, "best_baretsiz_time_sec", None))
            path = _get(p, "best_frame_path", "")

            self.table.setItem(r, 2, QTableWidgetItem("" if conf is None else f"{float(conf):.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem("" if tsec is None else f"{float(tsec):.2f}"))
            self.table.setItem(r, 4, QTableWidgetItem(path or ""))

    @Slot(str)
    def on_error(self, msg: str):
        self.btn_video.setEnabled(True)
        self.btn_start.setEnabled(True)
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
            else:
                self.preview.setText("Önizleme yüklenemedi.")
            self.btn_open_file.setEnabled(True)
        else:
            self.preview.setText("Bu kayıt için kare yok (baretli olabilir).")
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

    def refresh_table(self):
        if self._last_result is None:
            return

        result = self._last_result
        people = list(result.people)

        # 1) filtre: baretsizler kesin kalsın, baretli checkbox'a bağlı
        show_baretli = self.chk_show_baretli.isChecked()
        if not show_baretli:
            people = [p for p in people if p.final_label == "baretsiz"]

        # 2) sıralama
        mode = self.cmb_sort.currentText()

        # güvenli alan okuma (eski/yeni uyum)
        def _conf(p):
            return float(getattr(p, "best_conf", getattr(p, "best_baretsiz_conf", 0.0)) or 0.0)

        def _t(p):
            return float(getattr(p, "best_time_sec", getattr(p, "best_baretsiz_time_sec", 1e18)) or 1e18)

        if mode == "Zamana göre (tümü)":
            # global zaman
            people.sort(key=lambda p: _t(p))
        elif mode == "Baretsiz üstte, conf":
            # baretsiz grup üstte, her grubun içinde conf desc
            people.sort(key=lambda p: (p.final_label != "baretsiz", -_conf(p)))
        else:
            # "Baretsiz üstte, zaman (grup içi)"
            # baretsiz üstte; her grubun içinde zaman asc
            people.sort(key=lambda p: (p.final_label != "baretsiz", _t(p)))

        # 3) tabloya yaz
        self.table.setRowCount(len(people))
        for r, p in enumerate(people):
            self.table.setItem(r, 0, QTableWidgetItem(str(p.track_id)))
            self.table.setItem(r, 1, QTableWidgetItem(p.final_label))

            conf = getattr(p, "best_conf", getattr(p, "best_baretsiz_conf", None))
            tsec = getattr(p, "best_time_sec", getattr(p, "best_baretsiz_time_sec", None))
            path = getattr(p, "best_frame_path", "")

            self.table.setItem(r, 2, QTableWidgetItem("" if conf is None else f"{float(conf):.2f}"))
            self.table.setItem(r, 3, QTableWidgetItem("" if tsec is None else f"{float(tsec):.2f}"))
            self.table.setItem(r, 4, QTableWidgetItem(path or ""))

    # otomatik odak: ilk baretsiz
        for r in range(self.table.rowCount()):
            if self.table.item(r, 1).text() == "baretsiz":
                self.table.selectRow(r)
            break



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1100, 750)
    w.show()
    sys.exit(app.exec())

