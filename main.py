"""
DIP Studio — A Professional Digital Image Processing Editor
Based on Gonzalez & Woods: "Digital Image Processing"
Requires: PySide6, OpenCV, NumPy, matplotlib
Install: pip install PySide6 opencv-python numpy matplotlib
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QDockWidget,
    QTabWidget, QGroupBox, QGridLayout, QLineEdit, QComboBox,
    QSplitter, QScrollArea, QTextEdit, QFrame, QSpinBox,
    QDoubleSpinBox, QCheckBox, QMenuBar, QMenu, QStatusBar,
    QToolBar, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import (
    QImage, QPixmap, QFont, QColor, QPalette, QIcon,
    QAction, QPainter, QBrush, QPen
)

try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─────────────────────────── STYLESHEET ────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 13px;
}
QMenuBar {
    background-color: #141414;
    color: #c8c8c8;
    border-bottom: 1px solid #2a2a2a;
    padding: 2px 4px;
}
QMenuBar::item:selected { background-color: #2d2d2d; border-radius: 4px; }
QMenu {
    background-color: #1c1c1c;
    color: #d8d8d8;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item { padding: 6px 24px 6px 12px; border-radius: 4px; }
QMenu::item:selected { background-color: #3a86ff22; color: #fff; }
QDockWidget {
    background-color: #141414;
    color: #c0c0c0;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}
QDockWidget::title {
    background-color: #1a1a1a;
    padding: 6px 10px;
    border-bottom: 1px solid #2a2a2a;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #aaa;
    font-size: 11px;
    text-transform: uppercase;
}
QGroupBox {
    background-color: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    font-size: 11px;
    color: #888;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    top: -6px;
}
QPushButton {
    background-color: #222;
    color: #d0d0d0;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2e2e2e;
    border-color: #3a86ff;
    color: #fff;
}
QPushButton:pressed { background-color: #3a86ff33; }
QPushButton#accent {
    background-color: #3a86ff;
    color: white;
    border: none;
    font-weight: 600;
}
QPushButton#accent:hover { background-color: #5599ff; }
QPushButton#danger {
    background-color: #cc3333;
    color: white;
    border: none;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #2a2a2a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3a86ff;
    width: 14px;
    height: 14px;
    border-radius: 7px;
    margin: -5px 0;
}
QSlider::sub-page:horizontal { background: #3a86ff; border-radius: 2px; }
QTabWidget::pane {
    border: 1px solid #252525;
    background-color: #111;
    border-radius: 0 0 8px 8px;
}
QTabBar::tab {
    background-color: #1a1a1a;
    color: #888;
    padding: 8px 16px;
    border: 1px solid #252525;
    border-bottom: none;
    font-size: 12px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #111;
    color: #3a86ff;
    border-bottom: 2px solid #3a86ff;
}
QTabBar::tab:hover { color: #ccc; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1c1c1c;
    color: #d0d0d0;
    border: 1px solid #303030;
    border-radius: 5px;
    padding: 5px 8px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3a86ff;
}
QComboBox::drop-down { border: none; }
QScrollBar:vertical {
    background: #111;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #333;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #3a86ff; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollArea { border: none; background: transparent; }
QLabel#header {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}
QLabel#sub {
    font-size: 11px;
    color: #555;
    letter-spacing: 0.5px;
}
QTextEdit {
    background-color: #111;
    color: #aaa;
    border: 1px solid #222;
    border-radius: 6px;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
}
QStatusBar {
    background-color: #0a0a0a;
    color: #555;
    border-top: 1px solid #1e1e1e;
    padding: 2px 8px;
    font-size: 11px;
}
QToolBar {
    background-color: #141414;
    border-bottom: 1px solid #222;
    spacing: 4px;
    padding: 4px;
}
QSplitter::handle { background: #222; width: 1px; height: 1px; }
"""

# ─────────────────────────── IMAGE PROCESSING ENGINE ───────────────────────

class ImageProcessor:
    """Core DIP operations mapped to Gonzalez & Woods chapters."""

    @staticmethod
    def apply_gamma(image: np.ndarray, c: float, gamma: float) -> np.ndarray:
        """Ch. 3.2 — Power-Law (Gamma) Transform: s = c * r^γ"""
        normalized = image.astype(np.float64) / 255.0
        transformed = c * np.power(np.clip(normalized, 0, 1), gamma)
        return np.clip(transformed * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """Ch. 3.2 — Log Transform: s = c * log(1 + r)"""
        normalized = image.astype(np.float64) / 255.0
        transformed = c * np.log1p(normalized)
        transformed = transformed / transformed.max()
        return (transformed * 255).astype(np.uint8)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """Ch. 3.3.1 — Histogram Equalization"""
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(image)

    @staticmethod
    def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Ch. 3.4 — Spatial Filtering with custom kernel"""
        return cv2.filter2D(image, -1, kernel.astype(np.float32))

    @staticmethod
    def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """Ch. 3.5 — Median Filter (Order-Statistic)"""
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def laplacian_sharpen(image: np.ndarray) -> np.ndarray:
        """Ch. 3.6.1 — Laplacian Sharpening"""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def compute_fft(image: np.ndarray):
        """Ch. 4.3 — 2D FFT Magnitude Spectrum"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        return fshift, magnitude

    @staticmethod
    def ideal_lowpass_filter(image: np.ndarray, cutoff: float) -> np.ndarray:
        """Ch. 4.8.2 — Ideal Lowpass Filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        mask = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                if np.sqrt((r - crow) ** 2 + (c - ccol) ** 2) <= cutoff:
                    mask[r, c] = 1
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = np.clip(img_back / img_back.max() * 255, 0, 255).astype(np.uint8)
        if len(image.shape) == 3:
            img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        return img_back

    @staticmethod
    def butterworth_lowpass(image: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
        """Ch. 4.8.3 — Butterworth Lowpass Filter"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows) - crow
        v = np.arange(cols) - ccol
        V, U = np.meshgrid(v, u)
        D = np.sqrt(U ** 2 + V ** 2)
        H = 1 / (1 + (D / (cutoff + 1e-10)) ** (2 * order))
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        filtered = fshift * H
        img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
        img_back = np.clip(img_back / img_back.max() * 255, 0, 255).astype(np.uint8)
        if len(image.shape) == 3:
            img_back = cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        return img_back

    @staticmethod
    def morphology(image: np.ndarray, operation: str, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        """Ch. 9.2-9.3 — Morphological Operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        ops = {
            "erosion": cv2.MORPH_ERODE,
            "dilation": cv2.MORPH_DILATE,
            "opening": cv2.MORPH_OPEN,
            "closing": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
            "blackhat": cv2.MORPH_BLACKHAT,
        }
        return cv2.morphologyEx(image, ops.get(operation, cv2.MORPH_ERODE), kernel, iterations=iterations)

    @staticmethod
    def edge_detection(image: np.ndarray, method: str) -> np.ndarray:
        """Ch. 10.2 — Edge Detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if method == "sobel":
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            result = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
        elif method == "roberts":
            kx = np.array([[1, 0], [0, -1]], dtype=np.float32)
            ky = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            rx = cv2.filter2D(gray.astype(np.float32), -1, kx)
            ry = cv2.filter2D(gray.astype(np.float32), -1, ky)
            result = np.clip(np.sqrt(rx**2 + ry**2), 0, 255).astype(np.uint8)
        elif method == "canny":
            result = cv2.Canny(gray, 50, 150)
        elif method == "prewitt":
            kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            px = cv2.filter2D(gray.astype(np.float32), -1, kx)
            py = cv2.filter2D(gray.astype(np.float32), -1, ky)
            result = np.clip(np.sqrt(px**2 + py**2), 0, 255).astype(np.uint8)
        else:
            result = gray
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def otsu_threshold(image: np.ndarray):
        """Ch. 10.3.3 — Otsu's Thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return result, thresh_val


# ─────────────────────────── IMAGE CANVAS ──────────────────────────────────

class ImageCanvas(QLabel):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.title = title
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("""
            QLabel {
                background-color: #0a0a0a;
                border: 1px solid #1e1e1e;
                border-radius: 6px;
                color: #333;
            }
        """)
        self.setText(f"[ {title} ]\nNo image loaded")

    def set_image(self, img: np.ndarray):
        if img is None:
            return
        h, w = img.shape[:2]
        if len(img.shape) == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        else:
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_scaled()

    def _update_scaled(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)

    def resizeEvent(self, e):
        self._update_scaled()
        super().resizeEvent(e)


# ─────────────────────────── HISTOGRAM WIDGET ──────────────────────────────

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._data = None
        self.setStyleSheet("background: #0d0d0d; border-radius: 6px;")

    def update_histogram(self, image: np.ndarray):
        self._data = image
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d0d0d"))
        if self._data is None:
            painter.setPen(QColor("#333"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load an image to see histogram")
            return

        w, h = self.width(), self.height()
        pad = 10
        draw_w, draw_h = w - 2 * pad, h - 2 * pad

        if len(self._data.shape) == 3:
            channels = [(0, "#4466ff"), (1, "#44cc66"), (2, "#ff4444")]
            for ch, color in channels:
                hist = cv2.calcHist([self._data], [ch], None, [256], [0, 256]).flatten()
                hist_norm = hist / (hist.max() + 1e-6)
                pts = []
                for i, v in enumerate(hist_norm):
                    x = pad + int(i / 255 * draw_w)
                    y = pad + draw_h - int(v * draw_h)
                    pts.append((x, y))
                pen = QPen(QColor(color))
                pen.setWidth(1)
                painter.setPen(pen)
                for i in range(len(pts) - 1):
                    painter.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
        else:
            hist = cv2.calcHist([self._data], [0], None, [256], [0, 256]).flatten()
            hist_norm = hist / (hist.max() + 1e-6)
            pen = QPen(QColor("#aaa"))
            pen.setWidth(1)
            painter.setPen(pen)
            for i in range(255):
                x1 = pad + int(i / 255 * draw_w)
                x2 = pad + int((i+1) / 255 * draw_w)
                y1 = pad + draw_h - int(hist_norm[i] * draw_h)
                y2 = pad + draw_h - int(hist_norm[i+1] * draw_h)
                painter.drawLine(x1, y1, x2, y2)
        painter.end()


# ─────────────────────────── KERNEL EDITOR ─────────────────────────────────

class KernelEditor(QWidget):
    def __init__(self, size=3, parent=None):
        super().__init__(parent)
        self.size = size
        self.cells = []
        layout = QGridLayout(self)
        layout.setSpacing(3)
        for r in range(size):
            row = []
            for c in range(size):
                cell = QLineEdit("0")
                cell.setFixedSize(42, 36)
                cell.setAlignment(Qt.AlignCenter)
                cell.setStyleSheet("""
                    QLineEdit {
                        background: #1a1a1a; color: #ddd;
                        border: 1px solid #303030; border-radius: 4px;
                        font-family: 'Consolas', monospace; font-size: 13px;
                    }
                    QLineEdit:focus { border-color: #3a86ff; }
                """)
                layout.addWidget(cell, r, c)
                row.append(cell)
            self.cells.append(row)

    def get_kernel(self) -> np.ndarray:
        kernel = np.zeros((self.size, self.size))
        for r in range(self.size):
            for c in range(self.size):
                try:
                    kernel[r, c] = float(self.cells[r][c].text())
                except ValueError:
                    kernel[r, c] = 0
        return kernel

    def set_preset(self, name: str):
        presets = {
            "identity":   [[0,0,0],[0,1,0],[0,0,0]],
            "box_blur":   [[1,1,1],[1,1,1],[1,1,1]],
            "sharpen":    [[0,-1,0],[-1,5,-1],[0,-1,0]],
            "laplacian":  [[0,1,0],[1,-4,1],[0,1,0]],
            "emboss":     [[-2,-1,0],[-1,1,1],[0,1,2]],
            "edge_h":     [[-1,-2,-1],[0,0,0],[1,2,1]],
        }
        if name in presets:
            vals = presets[name]
            for r in range(self.size):
                for c in range(self.size):
                    self.cells[r][c].setText(str(vals[r][c]))


# ─────────────────────────── CHAPTER NAVIGATOR ─────────────────────────────

CHAPTER_INFO = {
    "gamma":     ("Ch. 3.2", "Power-Law (Gamma) Transform", "Applies s = c·r^γ. Useful for correcting display gamma and enhancing dark (γ<1) or bright (γ>1) regions. Mimics the nonlinear response of the human visual system."),
    "log":       ("Ch. 3.2", "Log Transform", "s = c·log(1+r). Compresses dynamic range. Ideal for displaying FFT spectra where high-frequency magnitudes would otherwise dominate."),
    "histeq":    ("Ch. 3.3.1", "Histogram Equalization", "Redistributes intensity values so the output histogram is approximately uniform. Maximizes image contrast by flattening the CDF."),
    "custom_kernel": ("Ch. 3.4.1", "Spatial Convolution", "Each output pixel is the weighted sum of its neighborhood defined by the kernel mask. Convolution is the fundamental operation behind all linear spatial filters."),
    "median":    ("Ch. 3.5", "Median Filter (Order-Statistic)", "Replaces each pixel with the median of its neighborhood. Excellent for removing salt-and-pepper noise while preserving edges — unlike mean filters."),
    "laplacian_s": ("Ch. 3.6.1", "Laplacian Sharpening", "Second derivative operator: ∇²f = ∂²f/∂x² + ∂²f/∂y². Adding the Laplacian to the original image sharpens edges."),
    "fft":       ("Ch. 4.3", "2D Fourier Transform", "F(u,v) = ΣΣ f(x,y)·e^{-j2π(ux/M+vy/N)}. Decomposes the image into frequency components. Low frequencies correspond to slow variations; high frequencies to edges and noise."),
    "ilpf":      ("Ch. 4.8.2", "Ideal Lowpass Filter", "Passes all frequencies within radius D₀ of the origin and attenuates all others. Causes ringing (Gibbs phenomenon) in spatial domain due to sharp frequency cutoff."),
    "blpf":      ("Ch. 4.8.3", "Butterworth Lowpass Filter", "H(u,v) = 1/[1+(D/D₀)^{2n}]. Smooth transition between passband and stopband. Order n controls rolloff steepness. No sharp cutoff → minimal ringing."),
    "erosion":   ("Ch. 9.2", "Erosion", "A⊖B = {z | Bz ⊆ A}. Shrinks foreground objects. Removes small protrusions and disconnects thin bridges. Size of structuring element controls aggressiveness."),
    "dilation":  ("Ch. 9.2", "Dilation", "A⊕B = {z | B̂z∩A≠∅}. Expands foreground objects. Fills small holes and connects nearby regions."),
    "opening":   ("Ch. 9.3", "Morphological Opening", "A∘B = (A⊖B)⊕B. Erosion followed by dilation. Removes small objects/noise while preserving shape of larger structures."),
    "closing":   ("Ch. 9.3", "Morphological Closing", "A•B = (A⊕B)⊖B. Dilation followed by erosion. Closes small holes and gaps within foreground regions."),
    "sobel":     ("Ch. 10.2", "Sobel Edge Detector", "Computes gradient magnitude using 3×3 masks that approximate ∂f/∂x and ∂f/∂y simultaneously. Weighted to reduce noise sensitivity."),
    "roberts":   ("Ch. 10.2", "Roberts Cross Operator", "Uses 2×2 diagonal difference operators. Simplest gradient approximation, computationally efficient but highly sensitive to noise."),
    "canny":     ("Ch. 10.2", "Canny Edge Detector", "Multi-stage: Gaussian smoothing → gradient computation → non-maximum suppression → hysteresis thresholding. Produces thin, well-localized edges with strong noise rejection."),
    "otsu":      ("Ch. 10.3.3", "Otsu's Method", "Automatically selects optimal global threshold by maximizing between-class variance σ²_B(T). Assumes bimodal histogram. Threshold is the T* that maximizes [(μ₁-μ₂)² · ω₁ω₂]."),
}

class ChapterNavigator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        ref_label = QLabel("REFERENCE")
        ref_label.setStyleSheet("color: #3a86ff; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(ref_label)

        self.chapter_lbl = QLabel("—")
        self.chapter_lbl.setStyleSheet("color: #3a86ff; font-size: 13px; font-weight: 600;")
        layout.addWidget(self.chapter_lbl)

        self.title_lbl = QLabel("Apply a filter to see theory")
        self.title_lbl.setStyleSheet("color: #ddd; font-size: 14px; font-weight: 700; margin-top: 2px;")
        self.title_lbl.setWordWrap(True)
        layout.addWidget(self.title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #222;")
        layout.addWidget(sep)

        theory_lbl = QLabel("THEORY")
        theory_lbl.setStyleSheet("color: #555; font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(theory_lbl)

        self.theory_txt = QTextEdit()
        self.theory_txt.setReadOnly(True)
        self.theory_txt.setMinimumHeight(100)
        self.theory_txt.setStyleSheet("""
            QTextEdit {
                background: #111; color: #bbb; border: 1px solid #1e1e1e;
                border-radius: 6px; font-size: 12px; padding: 8px;
            }
        """)
        layout.addWidget(self.theory_txt)
        layout.addStretch()

    def update_info(self, key: str):
        if key in CHAPTER_INFO:
            ch, title, theory = CHAPTER_INFO[key]
            self.chapter_lbl.setText(f"Gonzalez & Woods — {ch}")
            self.title_lbl.setText(title)
            self.theory_txt.setPlainText(theory)


# ─────────────────────────── PANELS ────────────────────────────────────────

def make_slider(min_v, max_v, default, label_text, parent=None):
    """Helper: labeled slider row."""
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    lbl = QLabel(label_text)
    lbl.setFixedWidth(90)
    lbl.setStyleSheet("color: #888; font-size: 11px;")
    slider = QSlider(Qt.Horizontal)
    slider.setRange(min_v, max_v)
    slider.setValue(default)
    val_lbl = QLabel(str(default))
    val_lbl.setFixedWidth(36)
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    val_lbl.setStyleSheet("color: #3a86ff; font-weight: 600; font-size: 11px;")
    slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    layout.addWidget(lbl)
    layout.addWidget(slider)
    layout.addWidget(val_lbl)
    return row, slider


class ExposurePanel(QWidget):
    operation_applied = Signal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # Gamma
        g_box = QGroupBox("Power-Law Transform  |  s = c · r^γ")
        g_layout = QVBoxLayout(g_box)
        self.gamma_row, self.gamma_slider = make_slider(1, 300, 100, "Gamma (×0.01)")
        self.c_row, self.c_slider = make_slider(1, 300, 100, "c (×0.01)")
        g_layout.addWidget(self.gamma_row)
        g_layout.addWidget(self.c_row)
        btn_gamma = QPushButton("Apply Gamma")
        btn_gamma.clicked.connect(self._apply_gamma)
        g_layout.addWidget(btn_gamma)
        layout.addWidget(g_box)

        # Log
        l_box = QGroupBox("Log Transform  |  s = c · log(1 + r)")
        l_layout = QVBoxLayout(l_box)
        self.log_c_row, self.log_c_slider = make_slider(1, 200, 100, "c (×0.01)")
        l_layout.addWidget(self.log_c_row)
        btn_log = QPushButton("Apply Log Transform")
        btn_log.clicked.connect(self._apply_log)
        l_layout.addWidget(btn_log)
        layout.addWidget(l_box)

        # Histogram Equalization
        h_box = QGroupBox("Histogram Equalization")
        h_layout = QVBoxLayout(h_box)
        btn_heq = QPushButton("Equalize Histogram")
        btn_heq.setObjectName("accent")
        btn_heq.clicked.connect(self._apply_heq)
        h_layout.addWidget(btn_heq)
        layout.addWidget(h_box)

        layout.addStretch()
        self.image = None

    def set_image(self, img):
        self.image = img

    def _apply_gamma(self):
        if self.image is None: return
        gamma = self.gamma_slider.value() / 100.0
        c = self.c_slider.value() / 100.0
        result = ImageProcessor.apply_gamma(self.image, c, gamma)
        self.operation_applied.emit("gamma", result)

    def _apply_log(self):
        if self.image is None: return
        c = self.log_c_slider.value() / 100.0
        result = ImageProcessor.log_transform(self.image, c)
        self.operation_applied.emit("log", result)

    def _apply_heq(self):
        if self.image is None: return
        result = ImageProcessor.histogram_equalization(self.image)
        self.operation_applied.emit("histeq", result)


class SpatialPanel(QWidget):
    operation_applied = Signal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # Kernel editor
        k_box = QGroupBox("Custom Convolution Kernel  (Ch. 3.4)")
        k_layout = QVBoxLayout(k_box)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["identity", "box_blur", "sharpen", "laplacian", "emboss", "edge_h"])
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.preset_combo)
        k_layout.addLayout(preset_row)
        self.kernel_editor = KernelEditor(3)
        self.kernel_editor.set_preset("identity")
        k_layout.addWidget(self.kernel_editor)
        norm_row = QHBoxLayout()
        self.normalize_chk = QCheckBox("Normalize kernel")
        self.normalize_chk.setChecked(True)
        norm_row.addWidget(self.normalize_chk)
        k_layout.addLayout(norm_row)
        btn_kernel = QPushButton("Apply Kernel")
        btn_kernel.clicked.connect(self._apply_kernel)
        k_layout.addWidget(btn_kernel)
        layout.addWidget(k_box)

        # Quick filters
        q_box = QGroupBox("Quick Filters")
        q_layout = QVBoxLayout(q_box)
        median_row, self.median_slider = make_slider(1, 9, 3, "Ksize (odd)")
        q_layout.addWidget(median_row)
        btn_median = QPushButton("Apply Median Filter  (Ch. 3.5)")
        btn_median.clicked.connect(self._apply_median)
        q_layout.addWidget(btn_median)
        btn_lap = QPushButton("Laplacian Sharpening  (Ch. 3.6.1)")
        btn_lap.clicked.connect(self._apply_laplacian)
        q_layout.addWidget(btn_lap)
        layout.addWidget(q_box)
        layout.addStretch()
        self.image = None

    def set_image(self, img):
        self.image = img

    def _load_preset(self, name):
        self.kernel_editor.set_preset(name)

    def _apply_kernel(self):
        if self.image is None: return
        k = self.kernel_editor.get_kernel()
        if self.normalize_chk.isChecked() and k.sum() != 0:
            k = k / k.sum()
        result = ImageProcessor.apply_kernel(self.image, k)
        self.operation_applied.emit("custom_kernel", result)

    def _apply_median(self):
        if self.image is None: return
        ksize = self.median_slider.value()
        result = ImageProcessor.median_filter(self.image, ksize)
        self.operation_applied.emit("median", result)

    def _apply_laplacian(self):
        if self.image is None: return
        result = ImageProcessor.laplacian_sharpen(self.image)
        self.operation_applied.emit("laplacian_s", result)


class FrequencyPanel(QWidget):
    operation_applied = Signal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        fft_box = QGroupBox("FFT Magnitude Spectrum  (Ch. 4.3)")
        fft_layout = QVBoxLayout(fft_box)
        btn_fft = QPushButton("Compute & Display FFT")
        btn_fft.clicked.connect(self._show_fft)
        fft_layout.addWidget(btn_fft)
        self.fft_canvas_lbl = QLabel("FFT will appear here")
        self.fft_canvas_lbl.setAlignment(Qt.AlignCenter)
        self.fft_canvas_lbl.setFixedHeight(140)
        self.fft_canvas_lbl.setStyleSheet("background:#0a0a0a; border:1px solid #222; border-radius:4px; color:#333;")
        fft_layout.addWidget(self.fft_canvas_lbl)
        layout.addWidget(fft_box)

        lpf_box = QGroupBox("Frequency Domain Filtering")
        lpf_layout = QVBoxLayout(lpf_box)
        cutoff_row, self.cutoff_slider = make_slider(5, 200, 60, "Cutoff radius")
        order_row, self.order_slider = make_slider(1, 10, 2, "BW order")
        lpf_layout.addWidget(cutoff_row)
        lpf_layout.addWidget(order_row)
        btn_ilpf = QPushButton("Ideal Lowpass Filter  (Ch. 4.8.2)")
        btn_ilpf.clicked.connect(self._apply_ilpf)
        btn_blpf = QPushButton("Butterworth Lowpass  (Ch. 4.8.3)")
        btn_blpf.clicked.connect(self._apply_blpf)
        btn_ihpf = QPushButton("Ideal Highpass Filter")
        btn_ihpf.clicked.connect(self._apply_ihpf)
        lpf_layout.addWidget(btn_ilpf)
        lpf_layout.addWidget(btn_blpf)
        lpf_layout.addWidget(btn_ihpf)
        layout.addWidget(lpf_box)
        layout.addStretch()
        self.image = None

    def set_image(self, img):
        self.image = img

    def _show_fft(self):
        if self.image is None: return
        _, mag = ImageProcessor.compute_fft(self.image)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        h, w = mag_norm.shape
        qimg = QImage(mag_norm.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(self.fft_canvas_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.fft_canvas_lbl.setPixmap(pix)
        self.fft_canvas_lbl.setText("")

    def _apply_ilpf(self):
        if self.image is None: return
        result = ImageProcessor.ideal_lowpass_filter(self.image, self.cutoff_slider.value())
        self.operation_applied.emit("ilpf", result)

    def _apply_blpf(self):
        if self.image is None: return
        result = ImageProcessor.butterworth_lowpass(self.image, self.cutoff_slider.value(), self.order_slider.value())
        self.operation_applied.emit("blpf", result)

    def _apply_ihpf(self):
        if self.image is None: return
        # Ideal highpass = original - lowpass
        lp = ImageProcessor.ideal_lowpass_filter(self.image, self.cutoff_slider.value())
        result = cv2.subtract(self.image, lp)
        self.operation_applied.emit("ilpf", result)


class MorphologyPanel(QWidget):
    operation_applied = Signal(str, np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        morph_box = QGroupBox("Morphological Operations  (Ch. 9)")
        m_layout = QVBoxLayout(morph_box)
        ks_row, self.ks_slider = make_slider(1, 15, 3, "Kernel size")
        it_row, self.iter_slider = make_slider(1, 10, 1, "Iterations")
        m_layout.addWidget(ks_row)
        m_layout.addWidget(it_row)

        ops = [
            ("Erosion  A⊖B", "erosion"),
            ("Dilation  A⊕B", "dilation"),
            ("Opening  A∘B", "opening"),
            ("Closing  A•B", "closing"),
            ("Gradient", "gradient"),
            ("Top Hat", "tophat"),
            ("Black Hat", "blackhat"),
        ]
        for label, op in ops:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, o=op: self._apply(o))
            m_layout.addWidget(btn)

        layout.addWidget(morph_box)
        layout.addStretch()
        self.image = None

    def set_image(self, img):
        self.image = img

    def _apply(self, op):
        if self.image is None: return
        result = ImageProcessor.morphology(self.image, op, self.ks_slider.value(), self.iter_slider.value())
        self.operation_applied.emit(op, result)


class SegmentationPanel(QWidget):
    operation_applied = Signal(str, np.ndarray)
    otsu_threshold_found = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        edge_box = QGroupBox("Edge Detection Suite  (Ch. 10.2)")
        e_layout = QVBoxLayout(edge_box)
        for label, method in [("Sobel", "sobel"), ("Roberts Cross", "roberts"), ("Prewitt", "prewitt"), ("Canny (Multi-stage)", "canny")]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, m=method: self._apply_edge(m))
            e_layout.addWidget(btn)
        layout.addWidget(edge_box)

        thresh_box = QGroupBox("Thresholding  (Ch. 10.3)")
        t_layout = QVBoxLayout(thresh_box)
        btn_otsu = QPushButton("Otsu's Method  (Auto-threshold)")
        btn_otsu.setObjectName("accent")
        btn_otsu.clicked.connect(self._apply_otsu)
        t_layout.addWidget(btn_otsu)
        self.otsu_lbl = QLabel("Optimal threshold: —")
        self.otsu_lbl.setStyleSheet("color: #3a86ff; font-size: 12px; font-weight: 600; padding: 4px;")
        t_layout.addWidget(self.otsu_lbl)
        layout.addWidget(thresh_box)
        layout.addStretch()
        self.image = None

    def set_image(self, img):
        self.image = img

    def _apply_edge(self, method):
        if self.image is None: return
        result = ImageProcessor.edge_detection(self.image, method)
        self.operation_applied.emit(method, result)

    def _apply_otsu(self):
        if self.image is None: return
        result, thresh_val = ImageProcessor.otsu_threshold(self.image)
        self.otsu_lbl.setText(f"Optimal threshold T* = {thresh_val:.1f}")
        self.operation_applied.emit("otsu", result)


# ─────────────────────────── MAIN WINDOW ───────────────────────────────────

class DIPStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.history = []

        self.setWindowTitle("DIP Studio — Gonzalez & Woods Edition")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_docks()
        self._build_statusbar()

        self.setStyleSheet(DARK_STYLE)

    # ── Menu ──
    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("File")
        self._add_action(file_m, "Open Image   (Ch. 2.4)", self.open_image, "Ctrl+O")
        self._add_action(file_m, "Save Processed Image", self.save_image, "Ctrl+S")
        file_m.addSeparator()
        self._add_action(file_m, "Inspect DPI / Metadata  (Ch. 2.4.3)", self.inspect_dpi)
        file_m.addSeparator()
        self._add_action(file_m, "Exit", self.close)

        edit_m = mb.addMenu("Edit")
        self._add_action(edit_m, "Undo", self.undo, "Ctrl+Z")
        self._add_action(edit_m, "Reset to Original", self.reset_image, "Ctrl+R")
        self._add_action(edit_m, "Convert to Grayscale", self.to_gray)
        self._add_action(edit_m, "Swap Original ↔ Processed", self.swap_images)

        view_m = mb.addMenu("View")
        self._add_action(view_m, "Toggle Histogram Panel", self.toggle_histogram)
        self._add_action(view_m, "Toggle Chapter Navigator", self.toggle_navigator)

        help_m = mb.addMenu("Help")
        self._add_action(help_m, "About DIP Studio", self.show_about)

    def _add_action(self, menu, text, slot, shortcut=None):
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut:
            act.setShortcut(shortcut)
        menu.addAction(act)
        return act

    # ── Toolbar ──
    def _build_toolbar(self):
        tb = QToolBar("Quick Actions")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)
        for text, slot in [
            ("📂  Open", self.open_image),
            ("💾  Save", self.save_image),
            ("↩  Undo", self.undo),
            ("🔄  Reset", self.reset_image),
            ("⬛  Grayscale", self.to_gray),
        ]:
            btn = QPushButton(text)
            btn.setStyleSheet("QPushButton { background:transparent; border:none; padding:4px 10px; color:#aaa; font-size:12px; } QPushButton:hover { color:#fff; }")
            btn.clicked.connect(slot)
            tb.addWidget(btn)
        tb.addSeparator()
        self.info_label = QLabel("  No image loaded")
        self.info_label.setStyleSheet("color: #555; font-size: 11px;")
        tb.addWidget(self.info_label)

    # ── Central widget: dual canvas + histogram ──
    def _build_central(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(8)

        # Header
        header_row = QHBoxLayout()
        title_lbl = QLabel("DIP STUDIO")
        title_lbl.setObjectName("header")
        sub_lbl = QLabel("Gonzalez & Woods — Digital Image Processing")
        sub_lbl.setObjectName("sub")
        sub_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_row.addWidget(title_lbl)
        header_row.addStretch()
        header_row.addWidget(sub_lbl)
        main_layout.addLayout(header_row)

        # Canvases
        canvas_splitter = QSplitter(Qt.Horizontal)
        canvas_splitter.setHandleWidth(2)

        left_wrap = QWidget()
        lw_layout = QVBoxLayout(left_wrap)
        lw_layout.setContentsMargins(0, 0, 0, 0)
        orig_lbl = QLabel("ORIGINAL")
        orig_lbl.setStyleSheet("color:#444; font-size:10px; font-weight:700; letter-spacing:1.5px; padding-left:4px;")
        lw_layout.addWidget(orig_lbl)
        self.original_canvas = ImageCanvas("Original Image")
        lw_layout.addWidget(self.original_canvas)

        right_wrap = QWidget()
        rw_layout = QVBoxLayout(right_wrap)
        rw_layout.setContentsMargins(0, 0, 0, 0)
        proc_lbl = QLabel("PROCESSED")
        proc_lbl.setStyleSheet("color:#3a86ff55; font-size:10px; font-weight:700; letter-spacing:1.5px; padding-left:4px;")
        rw_layout.addWidget(proc_lbl)
        self.processed_canvas = ImageCanvas("Processed Image")
        rw_layout.addWidget(self.processed_canvas)

        canvas_splitter.addWidget(left_wrap)
        canvas_splitter.addWidget(right_wrap)
        canvas_splitter.setSizes([600, 600])
        main_layout.addWidget(canvas_splitter, stretch=3)

        # Histogram
        hist_frame = QGroupBox("Histogram  (RGB)")
        hist_layout = QHBoxLayout(hist_frame)
        hist_layout.setContentsMargins(6, 6, 6, 6)
        self.hist_orig = HistogramWidget()
        self.hist_proc = HistogramWidget()
        hist_layout.addWidget(self.hist_orig)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #222;")
        hist_layout.addWidget(sep)
        hist_layout.addWidget(self.hist_proc)
        main_layout.addWidget(hist_frame, stretch=1)

        self.setCentralWidget(central)

    # ── Dock widgets ──
    def _build_docks(self):
        # Left dock: tools
        tools_dock = QDockWidget("Operations", self)
        tools_dock.setObjectName("tools_dock")
        tools_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        tools_container = QWidget()
        tools_layout = QVBoxLayout(tools_container)
        tools_layout.setContentsMargins(4, 4, 4, 4)
        tools_layout.setSpacing(0)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)

        self.exposure_panel = ExposurePanel()
        self.spatial_panel = SpatialPanel()
        self.freq_panel = FrequencyPanel()
        self.morph_panel = MorphologyPanel()
        self.seg_panel = SegmentationPanel()

        tabs.addTab(self._scrolled(self.exposure_panel), "Exposure")
        tabs.addTab(self._scrolled(self.spatial_panel), "Spatial")
        tabs.addTab(self._scrolled(self.freq_panel), "Frequency")
        tabs.addTab(self._scrolled(self.morph_panel), "Morphology")
        tabs.addTab(self._scrolled(self.seg_panel), "Segmentation")

        tools_layout.addWidget(tabs)
        tools_dock.setWidget(tools_container)
        self.addDockWidget(Qt.LeftDockWidgetArea, tools_dock)

        # Right dock: Chapter Navigator
        nav_dock = QDockWidget("Chapter Navigator", self)
        nav_dock.setObjectName("nav_dock")
        nav_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.chapter_nav = ChapterNavigator()
        nav_dock.setWidget(self.chapter_nav)
        self.addDockWidget(Qt.RightDockWidgetArea, nav_dock)
        self.nav_dock = nav_dock

        # Connect signals
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.operation_applied.connect(self._on_operation)

        self.tools_dock = tools_dock

    def _scrolled(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scroll

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — Open an image to begin")

    # ── Image operations ──
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)")
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Error", "Could not read image.")
            return
        self.original_image = img
        self.processed_image = img.copy()
        self.history.clear()
        self._update_display()
        h, w = img.shape[:2]
        self.info_label.setText(f"  {Path(path).name}  •  {w}×{h} px")
        self.status.showMessage(f"Opened: {path}  |  {w}×{h} px  |  Channels: {img.shape[2] if len(img.shape)==3 else 1}")
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.set_image(img)

    def save_image(self):
        if self.processed_image is None:
            QMessageBox.information(self, "Nothing to save", "No processed image available.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "processed.png", "Images (*.png *.jpg *.bmp *.tiff)")
        if path:
            cv2.imwrite(path, self.processed_image)
            self.status.showMessage(f"Saved: {path}")

    def undo(self):
        if self.history:
            self.processed_image = self.history.pop()
            self._update_processed_display()
            self.status.showMessage("Undo applied")
        else:
            self.status.showMessage("Nothing to undo")

    def reset_image(self):
        if self.original_image is None: return
        self.history.append(self.processed_image.copy())
        self.processed_image = self.original_image.copy()
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.set_image(self.original_image)
        self._update_processed_display()
        self.status.showMessage("Reset to original")

    def to_gray(self):
        if self.original_image is None: return
        self.history.append(self.processed_image.copy() if self.processed_image is not None else self.original_image.copy())
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.original_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.processed_image = self.original_image.copy()
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.set_image(self.original_image)
        self._update_display()
        self.status.showMessage("Converted to grayscale")

    def swap_images(self):
        if self.processed_image is None: return
        self.original_image, self.processed_image = self.processed_image, self.original_image
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.set_image(self.original_image)
        self._update_display()

    def inspect_dpi(self):
        if self.original_image is None:
            QMessageBox.information(self, "No Image", "Load an image first.")
            return
        h, w = self.original_image.shape[:2]
        ch = self.original_image.shape[2] if len(self.original_image.shape) == 3 else 1
        bits = 8 * ch
        msg = (
            f"<b>Ch. 2.4.3 — Image Sensing & Acquisition</b><br><br>"
            f"Resolution: <b>{w} × {h} px</b><br>"
            f"Channels: <b>{ch}</b><br>"
            f"Bit depth: <b>{bits}-bit</b><br>"
            f"Total pixels: <b>{w*h:,}</b><br>"
            f"Memory (uncompressed): <b>{w*h*ch/1024/1024:.2f} MB</b><br><br>"
            f"<i>DPI information is embedded in the file metadata (EXIF).<br>"
            f"Standard print DPI: 72 (web), 150 (draft), 300 (print quality).</i>"
        )
        QMessageBox.information(self, "Image Inspector", msg)

    def toggle_histogram(self):
        pass  # Histogram is always visible in central widget

    def toggle_navigator(self):
        self.nav_dock.setVisible(not self.nav_dock.isVisible())

    def show_about(self):
        QMessageBox.about(self, "About DIP Studio",
            "<b>DIP Studio</b> v1.0<br><br>"
            "A professional Digital Image Processing editor<br>"
            "based on Gonzalez & Woods:<br>"
            "<i>Digital Image Processing, 4th Edition</i><br><br>"
            "Built with Python · PySide6 · OpenCV · NumPy"
        )

    # ── Signal handler ──
    def _on_operation(self, op_key: str, result: np.ndarray):
        if self.processed_image is not None:
            self.history.append(self.processed_image.copy())
        self.processed_image = result
        for panel in [self.exposure_panel, self.spatial_panel, self.freq_panel, self.morph_panel, self.seg_panel]:
            panel.set_image(result)
        self._update_processed_display()
        self.chapter_nav.update_info(op_key)
        self.status.showMessage(f"Applied: {CHAPTER_INFO.get(op_key, (op_key,))[0] if op_key in CHAPTER_INFO else op_key}")

    # ── Display helpers ──
    def _update_display(self):
        if self.original_image is not None:
            self.original_canvas.set_image(self.original_image)
            self.hist_orig.update_histogram(self.original_image)
        self._update_processed_display()

    def _update_processed_display(self):
        if self.processed_image is not None:
            self.processed_canvas.set_image(self.processed_image)
            self.hist_proc.update_histogram(self.processed_image)


# ─────────────────────────── ENTRY POINT ───────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DIP Studio")
    app.setOrganizationName("Gonzalez & Woods")

    # High DPI
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    win = DIPStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
