"""
Hypertensive Retinopathy Detection — Desktop Application
=========================================================
Modern dark-theme PyQt5 UI with vessel segmentation display.
"""

import sys
import os

# ── IMPORTANT: Import torch BEFORE PyQt5 to avoid DLL conflicts on Windows ──
import torch  # noqa: F401 — must be first
import cv2
import numpy as np
from inference import EnsemblePredictor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar,
    QMessageBox, QFrame, QGraphicsDropShadowEffect, QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QLinearGradient, QPalette, QFontDatabase, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize


# ═══════════════════ Stylesheet ═══════════════════

DARK_STYLE = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #0f0c29, stop:0.5 #302b63, stop:1 #24243e);
}
QLabel {
    color: #e0e0e0;
}
QLabel#header {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    padding: 10px;
}
QLabel#subtitle {
    color: #a0a0c0;
    font-size: 13px;
    padding-bottom: 6px;
}
QLabel#imageTitle {
    color: #c0c0e0;
    font-size: 13px;
    font-weight: 600;
    padding: 4px 0;
}

/* Image panels */
QLabel#imagePanel {
    background: rgba(255, 255, 255, 0.04);
    border: 2px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
}

/* Card frame */
QFrame#card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px;
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7b93ff, stop:1 #8b5fbf);
}
QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5567cc, stop:1 #633d8a);
}
QPushButton:disabled {
    background: rgba(100, 100, 140, 0.3);
    color: rgba(255, 255, 255, 0.3);
}
QPushButton#analyzeBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #11998e, stop:1 #38ef7d);
    font-size: 15px;
}
QPushButton#analyzeBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #15b3a6, stop:1 #4dff91);
}
QPushButton#analyzeBtn:disabled {
    background: rgba(100, 100, 140, 0.3);
    color: rgba(255, 255, 255, 0.3);
}

/* ComboBox */
QComboBox {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 10px 16px;
    font-size: 13px;
    min-width: 200px;
}
QComboBox:hover {
    border: 1px solid rgba(255, 255, 255, 0.3);
}
QComboBox::drop-down {
    border: none;
    padding-right: 12px;
}
QComboBox QAbstractItemView {
    background: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid rgba(255,255,255,0.15);
    selection-background-color: #667eea;
}

/* Progress bar */
QProgressBar {
    background: rgba(255, 255, 255, 0.06);
    border: none;
    border-radius: 8px;
    height: 6px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #667eea, stop:1 #38ef7d);
    border-radius: 8px;
}

/* Result labels */
QLabel#resultHealthy {
    color: #38ef7d;
    font-size: 20px;
    font-weight: 700;
    padding: 12px;
}
QLabel#resultDisease {
    color: #ff6b6b;
    font-size: 20px;
    font-weight: 700;
    padding: 12px;
}
QLabel#resultPending {
    color: #a0a0c0;
    font-size: 16px;
    font-weight: 500;
    padding: 12px;
}
QLabel#statusLabel {
    color: #8888aa;
    font-size: 12px;
}
"""


# ═══════════════════ Worker Threads ═══════════════════

class ModelLoadThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, efn_dir, dense_dir, unetpp_dir):
        super().__init__()
        self.efn_dir = efn_dir
        self.dense_dir = dense_dir
        self.unetpp_dir = unetpp_dir

    def run(self):
        try:
            self.progress.emit("UNet++ segmentasyon modelleri yükleniyor...")
            predictor = EnsemblePredictor(
                self.efn_dir, self.dense_dir, self.unetpp_dir, device="cpu"
            )
            self.finished.emit(predictor)
        except Exception as e:
            self.error.emit(str(e))


class PredictionThread(QThread):
    finished = pyqtSignal(float, np.ndarray, np.ndarray, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, predictor, image_path, model_choice):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.model_choice = model_choice

    def run(self):
        try:
            prob, orig_rgb, vessel_vis, vessel_soft = self.predictor.predict(
                self.image_path, self.model_choice
            )
            self.finished.emit(prob, orig_rgb, vessel_vis, vessel_soft)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════ Main Window ═══════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HR Detection — Hipertansif Retinopati Tespit Sistemi")
        self.resize(1100, 750)
        self.setMinimumSize(900, 600)

        self.predictor = None
        self.current_image_path = None

        self._build_ui()
        self._load_models()

    # ── UI Construction ──

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(12)

        # Header
        header = QLabel("🔬  Hipertansif Retinopati Tespit Sistemi")
        header.setObjectName("header")
        header.setAlignment(Qt.AlignCenter)
        root.addWidget(header)

        subtitle = QLabel("Fundus görüntüsü yükleyin → Damar segmentasyonu → AI ile hastalık tespiti")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(subtitle)

        # ── Image panels ──
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(16)

        self.img_original = self._make_image_panel("Orijinal Fundus")
        self.img_vessel = self._make_image_panel("Damar Segmentasyonu")
        self.img_overlay = self._make_image_panel("Damar Overlay")

        panels_layout.addLayout(self.img_original["layout"])
        panels_layout.addLayout(self.img_vessel["layout"])
        panels_layout.addLayout(self.img_overlay["layout"])
        root.addLayout(panels_layout, stretch=1)

        # ── Controls card ──
        controls_card = QFrame()
        controls_card.setObjectName("card")
        controls_inner = QHBoxLayout(controls_card)
        controls_inner.setSpacing(12)

        self.btn_load = QPushButton("📂  Görüntü Yükle")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.btn_load.setMinimumHeight(46)
        self.btn_load.clicked.connect(self._on_load_image)

        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "🤖  Ensemble (Önerilen)",
            "⚡  EfficientNet-B0",
            "🔬  DenseNet-121",
        ])
        self.combo_model.setMinimumHeight(46)

        self.btn_analyze = QPushButton("▶  Analiz Et")
        self.btn_analyze.setObjectName("analyzeBtn")
        self.btn_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_analyze.setMinimumHeight(46)
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_analyze.setEnabled(False)

        controls_inner.addWidget(self.btn_load, stretch=1)
        controls_inner.addWidget(self.combo_model, stretch=1)
        controls_inner.addWidget(self.btn_analyze, stretch=1)
        root.addWidget(controls_card)

        # ── Result area ──
        result_card = QFrame()
        result_card.setObjectName("card")
        result_inner = QVBoxLayout(result_card)

        self.lbl_result = QLabel("Sonuç bekleniyor...")
        self.lbl_result.setObjectName("resultPending")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        result_inner.addWidget(self.lbl_result)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        result_inner.addWidget(self.progress_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("statusLabel")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        result_inner.addWidget(self.lbl_status)

        root.addWidget(result_card)

    def _make_image_panel(self, title_text):
        layout = QVBoxLayout()
        layout.setSpacing(4)

        title = QLabel(title_text)
        title.setObjectName("imageTitle")
        title.setAlignment(Qt.AlignCenter)

        panel = QLabel()
        panel.setObjectName("imagePanel")
        panel.setAlignment(Qt.AlignCenter)
        panel.setMinimumSize(280, 280)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        panel.setScaledContents(False)

        # Subtle drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 100))
        panel.setGraphicsEffect(shadow)

        layout.addWidget(title)
        layout.addWidget(panel, stretch=1)
        return {"layout": layout, "label": panel, "title": title}

    # ── Model Loading ──

    def _load_models(self):
        self.lbl_result.setText("⏳  AI modelleri yükleniyor... Lütfen bekleyin.")
        self.lbl_result.setObjectName("resultPending")
        self.lbl_result.setStyleSheet("")  # force recalc
        self.progress_bar.show()
        self.btn_load.setEnabled(False)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        efn_dir = os.path.join(base_dir, "efficientnet_b0")
        dense_dir = os.path.join(base_dir, "densenet121")
        unetpp_dir = os.path.join(base_dir, "unetpp")

        # Check paths
        missing = []
        for p, n in [(efn_dir, "efficientnet_b0"), (dense_dir, "densenet121"), (unetpp_dir, "unetpp")]:
            if not os.path.isdir(p):
                missing.append(f"{n}: {p}")
        if missing:
            QMessageBox.critical(
                self, "Hata",
                "Model klasörleri bulunamadı:\n" + "\n".join(missing),
            )
            self.progress_bar.hide()
            return

        self.lbl_status.setText(f"Yükleniyor: {efn_dir}")
        self._load_thread = ModelLoadThread(efn_dir, dense_dir, unetpp_dir)
        self._load_thread.finished.connect(self._on_models_loaded)
        self._load_thread.error.connect(self._on_load_error)
        self._load_thread.progress.connect(lambda msg: self.lbl_status.setText(msg))
        self._load_thread.start()

    def _on_models_loaded(self, predictor):
        self.predictor = predictor
        self.progress_bar.hide()
        self.lbl_result.setText("✅  Hazır — Bir fundus görüntüsü yükleyin.")
        self.lbl_result.setObjectName("resultPending")
        self.lbl_result.setStyleSheet("")
        self.lbl_status.setText("15 model başarıyla yüklendi (5 UNet++ + 5 EfficientNet + 5 DenseNet)")
        self.btn_load.setEnabled(True)

    def _on_load_error(self, err):
        self.progress_bar.hide()
        QMessageBox.critical(self, "Model Yükleme Hatası", f"Modeller yüklenemedi:\n{err}")
        self.lbl_result.setText("❌  Model yükleme başarısız.")
        self.lbl_status.setText(err[:120])

    # ── Image Loading ──

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Fundus Görüntüsü Seç", "",
            "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Tüm Dosyalar (*)",
        )
        if not path:
            return
        self.current_image_path = path

        pixmap = QPixmap(path)
        self._fit_pixmap(self.img_original["label"], pixmap)

        # Clear previous results
        self.img_vessel["label"].clear()
        self.img_overlay["label"].clear()
        self.lbl_result.setText("✅  Görüntü yüklendi — Analiz Et butonuna basın.")
        self.lbl_result.setObjectName("resultPending")
        self.lbl_result.setStyleSheet("")
        self.lbl_status.setText(os.path.basename(path))
        self.btn_analyze.setEnabled(True)

    # ── Analysis ──

    def _on_analyze(self):
        if not self.current_image_path or not self.predictor:
            return

        choice_text = self.combo_model.currentText()
        if "Ensemble" in choice_text:
            model_choice = "Ensemble"
        elif "EfficientNet" in choice_text:
            model_choice = "EfficientNet"
        else:
            model_choice = "DenseNet"

        self.btn_analyze.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.lbl_result.setText("🔄  Analiz ediliyor... Damar segmentasyonu & sınıflandırma")
        self.lbl_result.setObjectName("resultPending")
        self.lbl_result.setStyleSheet("")
        self.lbl_status.setText(f"Model: {model_choice}")

        self._pred_thread = PredictionThread(
            self.predictor, self.current_image_path, model_choice
        )
        self._pred_thread.finished.connect(self._on_prediction_done)
        self._pred_thread.error.connect(self._on_prediction_error)
        self._pred_thread.start()

    def _on_prediction_done(self, prob, orig_rgb, vessel_vis, vessel_soft):
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)

        # Display vessel segmentation
        self._display_numpy(vessel_vis, self.img_vessel["label"])

        # Vessel soft map as grayscale heatmap
        vessel_color = cv2.applyColorMap(
            (vessel_soft * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        vessel_color = cv2.cvtColor(vessel_color, cv2.COLOR_BGR2RGB)
        # Blend overlay on original
        alpha = 0.4
        overlay = (orig_rgb.astype(np.float32) * (1 - alpha)
                   + vessel_color.astype(np.float32) * alpha).astype(np.uint8)
        self._display_numpy(overlay, self.img_overlay["label"])

        # Result
        is_disease = prob >= 0.5
        confidence = prob * 100 if is_disease else (1 - prob) * 100

        if is_disease:
            self.lbl_result.setText(
                f"⚠️  HİPERTANSİF RETİNOPATİ TESPİT EDİLDİ  —  Güven: %{confidence:.1f}"
            )
            self.lbl_result.setObjectName("resultDisease")
        else:
            self.lbl_result.setText(
                f"✅  SAĞLIKLI  —  Güven: %{confidence:.1f}"
            )
            self.lbl_result.setObjectName("resultHealthy")
        self.lbl_result.setStyleSheet("")  # force style refresh
        self.lbl_status.setText(f"Olasılık: {prob:.4f}")

    def _on_prediction_error(self, err):
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)
        QMessageBox.critical(self, "Analiz Hatası", f"Analiz sırasında hata oluştu:\n{err}")
        self.lbl_result.setText("❌  Analiz başarısız.")
        self.lbl_result.setObjectName("resultDisease")
        self.lbl_result.setStyleSheet("")

    # ── Helpers ──

    @staticmethod
    def _fit_pixmap(label, pixmap):
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    @staticmethod
    def _display_numpy(rgb_array, label):
        rgb_array = np.ascontiguousarray(rgb_array)
        h, w, ch = rgb_array.shape
        q_img = QImage(rgb_array.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)


# ═══════════════════ Entry Point ═══════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
