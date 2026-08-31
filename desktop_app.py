import os
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor

from faceNet import (
    device, mtcnn, resnet, load_known_embeddings, recognize_face_embedding,
)
from attendance import check_in, check_out, get_today_records


class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    face_recognized = pyqtSignal(str, float)
    face_unknown = pyqtSignal()
    no_face = pyqtSignal()
    status_update = pyqtSignal(str)
    attendance_updated = pyqtSignal()

    def __init__(self, known_embeddings, camera_index=0):
        super().__init__()
        self.known_embeddings = known_embeddings
        self.camera_index = camera_index
        self.running = False
        self.last_attendance_day = {}
        self.last_seen = {}

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        self.status_update.emit("Loading camera...")
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status_update.emit("Error: Cannot open camera")
            return

        frame_count = 0
        detect_every_n = 3
        self.status_update.emit("Camera active - detecting faces...")
        auto_checkout_after = 15.0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_update.emit("Error: Failed to grab frame")
                break

            frame_count += 1
            if frame_count % detect_every_n == 0:
                frame_small = cv2.resize(frame, (320, 240))
                frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)

                face = mtcnn(img_pil)
                if face is not None:
                    with torch.no_grad():
                        embedding = resnet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()

                    name, sim = recognize_face_embedding(embedding, self.known_embeddings)
                    if name and sim > 0.7:
                        self.face_recognized.emit(name, sim)
                        today = time.strftime("%Y-%m-%d")
                        if self.last_attendance_day.get(name) != today:
                            # Auto-checkout anyone previously seen/in who is not the current person
                            now_ts = time.time()
                            for seen_name, last_ts in list(self.last_seen.items()):
                                if seen_name != name and now_ts - last_ts > 3:
                                    if check_out(seen_name):
                                        self.attendance_updated.emit()
                            check_in(name)
                            self.last_attendance_day[name] = today
                            self.attendance_updated.emit()
                        self.last_seen[name] = time.time()
                    else:
                        self.face_unknown.emit()
                else:
                    self.no_face.emit()

                # Auto-checkout people who have been absent for the threshold
                now_ts = time.time()
                for seen_name, last_ts in list(self.last_seen.items()):
                    if now_ts - last_ts > auto_checkout_after:
                        if check_out(seen_name):
                            self.attendance_updated.emit()
                        del self.last_seen[seen_name]

            self.frame_ready.emit(frame)
            self.msleep(10)

        cap.release()
        self.status_update.emit("Camera stopped")


class FaceWatcherThread(QThread):
    face_present = pyqtSignal(bool)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self._stop_event = None

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False
        self.wait()

    def run(self):
        while self.running:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_small = cv2.resize(frame, (320, 240))
                    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(frame_rgb)
                    face = mtcnn(img_pil)
                    self.face_present.emit(face is not None)
            cap.release()
            if self.running:
                self.msleep(1500)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition App")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QSplitter { background-color: #1e1e2e; }
            QFrame { background-color: #2a2a3e; border-radius: 8px; }
            QLabel { color: #cdd6f4; }
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #74c7ec; }
            QPushButton:disabled { background-color: #585b70; color: #a6adc8; }
            QPushButton#stopBtn {
                background-color: #f38ba8;
            }
            QPushButton#stopBtn:hover {
                background-color: #eba0ac;
            }
            QPushButton#openBtn {
                background-color: #a6e3a1;
                color: #1e1e2e;
            }
            QPushButton#openBtn:hover {
                background-color: #94e2d5;
            }
            QTableWidget {
                background-color: #313244;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                gridline-color: #45475a;
                font-size: 12px;
            }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                padding: 8px;
                font-weight: bold;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QHBoxLayout()
        title = QLabel("Face Recognition")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa; background: transparent;")
        header.addWidget(title)
        header.addStretch()

        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedWidth(120)
        self.start_btn.clicked.connect(self.toggle_camera)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFixedWidth(120)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.toggle_camera)

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet("color: #a6adc8; background: transparent; font-size: 13px;")
        header.addWidget(cam_label)

        self.camera_combo = QComboBox()
        self.camera_combo.setFixedWidth(200)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #45475a;
                color: #cdd6f4;
                border: none;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #cdd6f4;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #45475a;
                color: #cdd6f4;
                selection-background-color: #585b70;
                border: none;
                padding: 4px;
            }
        """)
        self.scan_cameras()
        header.addWidget(self.camera_combo)

        header.addStretch()
        header.addWidget(self.start_btn)
        header.addWidget(self.stop_btn)
        layout.addLayout(header)

        top_section = QHBoxLayout()
        top_section.setSpacing(10)

        cam_frame = QFrame()
        cam_frame.setMinimumHeight(400)
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(4, 4, 4, 4)
        self.camera_label = QLabel("Waiting for camera...")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            background-color: #11111b;
            border-radius: 12px;
            color: #6c7086;
            font-size: 14px;
        """)
        cam_layout.addWidget(self.camera_label)
        top_section.addWidget(cam_frame, stretch=3)

        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(12, 12, 12, 12)
        status_layout.setSpacing(8)

        status_title = QLabel("Status")
        status_title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        status_title.setStyleSheet("color: #89b4fa; background: transparent;")
        status_layout.addWidget(status_title)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #a6adc8;
            font-size: 12px;
        """)
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)

        self.last_person_label = QLabel("None")
        self.last_person_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #cdd6f4;
            font-size: 13px;
            font-weight: bold;
        """)
        self.last_person_label.setWordWrap(True)
        status_layout.addWidget(self.last_person_label)

        self.sim_label = QLabel("")
        self.sim_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #a6e3a1;
            font-size: 12px;
        """)
        status_layout.addWidget(self.sim_label)

        status_layout.addStretch()

        self.time_out_btn = QPushButton("Time Out Current")
        self.time_out_btn.setObjectName("stopBtn")
        self.time_out_btn.clicked.connect(self.time_out_current)
        status_layout.addWidget(self.time_out_btn)

        self.open_csv_btn = QPushButton("Open Attendance CSV")
        self.open_csv_btn.setObjectName("openBtn")
        self.open_csv_btn.clicked.connect(self.open_csv)
        status_layout.addWidget(self.open_csv_btn)

        top_section.addWidget(status_frame, stretch=1)
        layout.addLayout(top_section)

        bottom_label = QLabel("Today's Attendance")
        bottom_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        bottom_label.setStyleSheet("color: #89b4fa; background: transparent; margin-top: 4px;")
        layout.addWidget(bottom_label)

        self.log_table = QTableWidget()
        self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(["Name", "Time-In", "Time-Out", "Status", "Total Hours"])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setStyleSheet("""
            QTableWidget { alternate-background-color: #363749; }
        """)
        layout.addWidget(self.log_table, stretch=1)

        self.camera_thread = None
        self.face_watcher = None
        self.is_running = False
        self.auto_manage = True
        self.current_person = None
        self.last_face_time = None
        self.no_face_timeout = 5.0  # seconds

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_attendance)
        self.refresh_timer.start(2000)

        self.face_check_timer = QTimer()
        self.face_check_timer.timeout.connect(self.check_no_face)
        self.face_check_timer.start(500)

        self.refresh_attendance()

    def get_camera_names(self):
        try:
            from pygrabber.dshow_graph import FilterGraph
            devices = FilterGraph().get_input_devices()
            return devices
        except Exception:
            return []

    def scan_cameras(self):
        self.camera_combo.clear()
        device_names = self.get_camera_names()
        found = 0
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    if found < len(device_names):
                        label = device_names[found]
                    else:
                        label = f"Camera {i}"
                    self.camera_combo.addItem(label, i)
                    found += 1
        if found == 0:
            self.camera_combo.addItem("No cameras found", -1)

    def toggle_camera(self):
        if self.is_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index < 0:
            self.status_label.setText("Error: No camera available")
            return

        if self.face_watcher and self.face_watcher.isRunning():
            self.face_watcher.stop()
            self.face_watcher = None

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.is_running = True
        self.last_face_time = None
        self.status_label.setText("Loading embeddings...")

        QApplication.processEvents()
        known_embeddings = load_known_embeddings("people/")
        self.camera_thread = CameraThread(known_embeddings, camera_index)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.face_recognized.connect(self.on_recognized)
        self.camera_thread.face_unknown.connect(self.on_unknown)
        self.camera_thread.no_face.connect(self.on_no_face)
        self.camera_thread.status_update.connect(self.on_status)
        self.camera_thread.attendance_updated.connect(self.on_attendance_updated)
        self.camera_thread.start()

    def start_watcher(self):
        if self.face_watcher and self.face_watcher.isRunning():
            return
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index < 0:
            return
        self.status_label.setText("Camera off - waiting for a face to appear...")
        self.show_camera_off_message()
        self.face_watcher = FaceWatcherThread(camera_index)
        self.face_watcher.face_present.connect(self.on_watcher_face)
        self.face_watcher.start()

    def on_watcher_face(self, present):
        if present and self.auto_manage and not self.is_running:
            self.camera_label.clear()
            self.start_camera()

    def show_camera_off_message(self):
        self.camera_label.clear()
        html = (
            "<div style='text-align:center;'>"
            "<div style='font-size:64px;'>&#128248;</div>"
            "<div style='font-size:22px; font-weight:bold; color:#89b4fa; "
            "margin-top:10px;'>Camera Off</div>"
            "<div style='font-size:14px; color:#a6adc8; margin-top:6px;'>"
            "No face detected</div>"
            "<div style='font-size:13px; color:#6c7086; margin-top:4px;'>"
            "Watching for a face to appear...</div>"
            "<div style='font-size:12px; color:#45475a; margin-top:18px;'>"
            "&#9679; Standby mode</div>"
            "</div>"
        )
        self.camera_label.setText(html)
        self.camera_label.setStyleSheet("""
            background-color: #1e1e2e;
            border: 2px dashed #45475a;
            border-radius: 12px;
        """)

    def stop_camera(self, auto=False):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.is_running = False
        self.last_face_time = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        if auto and self.auto_manage:
            self.start_watcher()
        else:
            self.status_label.setText("Stopped")
            self.camera_label.clear()
            self.camera_label.setText(
                "<div style='text-align:center;'>"
                "<div style='font-size:64px;'>&#128248;</div>"
                "<div style='font-size:22px; font-weight:bold; color:#a6adc8; "
                "margin-top:10px;'>Camera Stopped</div>"
                "<div style='font-size:14px; color:#6c7086; margin-top:6px;'>"
                "Press Start to begin</div>"
                "</div>"
            )
            self.camera_label.setStyleSheet("""
                background-color: #11111b;
                border-radius: 12px;
            """)

    def update_frame(self, frame):
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.camera_label.setPixmap(scaled)

    def on_recognized(self, name, sim):
        self.last_face_time = time.time()
        self.current_person = name
        self.last_person_label.setText(f"Detected: {name}")
        self.last_person_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #a6e3a1;
            font-size: 13px;
            font-weight: bold;
        """)
        self.sim_label.setText(f"Similarity: {sim:.4f}")

    def on_unknown(self):
        self.last_face_time = time.time()
        self.last_person_label.setText("Unknown face")
        self.last_person_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #f38ba8;
            font-size: 13px;
            font-weight: bold;
        """)
        self.sim_label.setText("")

    def on_no_face(self):
        self.last_person_label.setText("No face detected")
        self.last_person_label.setStyleSheet("""
            background-color: #45475a;
            border-radius: 6px;
            padding: 10px;
            color: #a6adc8;
            font-size: 13px;
            font-weight: bold;
        """)
        self.sim_label.setText("")

    def on_status(self, msg):
        self.status_label.setText(msg)

    def check_no_face(self):
        if not self.is_running:
            return
        if self.last_face_time is not None and (time.time() - self.last_face_time) > self.no_face_timeout:
            self.stop_camera(auto=True)

    def on_attendance_updated(self):
        self.refresh_attendance()

    def time_out_current(self):
        if not self.current_person:
            self.sim_label.setText("No person recognized to time out")
            return
        if check_out(self.current_person):
            self.sim_label.setText(f"Timed out: {self.current_person}")
            self.refresh_attendance()
        else:
            self.sim_label.setText(f"{self.current_person} is not checked in / already timed out")

    def refresh_attendance(self):
        records = get_today_records()
        self.log_table.setRowCount(len(records))
        for i, rec in enumerate(records):
            values = [rec["Name"], rec["TimeIn"], rec["TimeOut"], rec["Status"], rec["Duration"]]
            for j, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if j == 3:
                    if val == "Late":
                        item.setForeground(QColor("#f38ba8"))
                    elif val == "Present":
                        item.setForeground(QColor("#f9e2af"))
                    else:
                        item.setForeground(QColor("#a6e3a1"))
                self.log_table.setItem(i, j, item)

    def open_csv(self):
        import attendance
        log_file = os.path.abspath(attendance.ATTENDANCE_FILE)
        if os.path.exists(log_file):
            os.startfile(log_file)

    def closeEvent(self, event):
        self.stop_camera()
        if self.face_watcher and self.face_watcher.isRunning():
            self.face_watcher.stop()
            self.face_watcher = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    QTimer.singleShot(500, window.start_camera)
    sys.exit(app.exec())
