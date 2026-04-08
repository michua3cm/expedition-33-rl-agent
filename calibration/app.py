import threading
import time

import cv2
import mss
import numpy as np

import vision
from overlay_ui import OverlayWindow
from vision.engine import Detection

from .config import ASSETS_DIR, MONITOR_INDEX, SCREENSHOT_DIR, TARGETS
from .logger import CalibrationLogger

# Key Codes
VK_START = 0x78  # F9
VK_STOP  = 0x79  # F10
VK_EXIT  = 0x7A  # F11


class CalibrationApp:
    def __init__(self, engine="COMPOSITE"):
        self.engine_name = engine.upper()
        self.sct = mss.mss()
        self.monitor_config = self._setup_monitor()
        self.overlay = OverlayWindow()
        self.logger = CalibrationLogger()

        self.vision_engine = vision.registry.create(self.engine_name)
        self.vision_engine.load(TARGETS, ASSETS_DIR)

        self.running = True
        self._current_frame: np.ndarray | None = None
        self._last_detections: list[Detection] = []
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()

    def _setup_monitor(self):
        """Configure monitor cropping region."""
        raw = self.sct.monitors[MONITOR_INDEX]
        w, h = raw["width"], raw["height"]
        margin_w = int(w * 0.0)
        margin_h = int(h * 0.0)
        return {
            "top": raw["top"] + margin_h,
            "left": raw["left"] + margin_w,
            "width": w - (margin_w * 2),
            "height": h - (margin_h * 2),
            "mon": MONITOR_INDEX,
        }

    def _handle_input(self):
        """Check keyboard input for state changes."""
        import win32api  # type: ignore  # Windows-only; imported lazily so the module loads on Linux
        if win32api.GetAsyncKeyState(VK_START) & 0x8000:
            if not self.logger.get_record_status():
                self.logger.start_recording()
                sc = np.array(self.sct.grab(self.monitor_config))
                cv2.imwrite(f"{SCREENSHOT_DIR}/debug_capture.png", sc)
                time.sleep(0.3)

        if win32api.GetAsyncKeyState(VK_STOP) & 0x8000:
            if self.logger.get_record_status():
                self.logger.stop_recording()
                time.sleep(0.3)

        if win32api.GetAsyncKeyState(VK_EXIT) & 0x8000:
            self.running = False
            print("\n[Exit] Exiting program.")

    def _detection_loop(self) -> None:
        """Background thread: run vision detection as fast as the engine allows."""
        while self.running:
            with self._frame_lock:
                frame = self._current_frame
            if frame is not None:
                dets = self.vision_engine.detect(frame)
                with self._result_lock:
                    self._last_detections = dets

    def run(self) -> None:
        print("=== Calibration App Started ===")
        print("F9: Start | F10: Stop | F11: Exit")

        detect_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name="detection"
        )
        detect_thread.start()

        loop_time = time.time()
        try:
            while self.running:
                curr_time = time.time()
                fps = 1 / (curr_time - loop_time) if (curr_time - loop_time) > 0 else 0
                loop_time = curr_time

                # 1. Input
                self._handle_input()

                # 2. Capture frame and hand it to the detection thread
                screenshot = np.array(self.sct.grab(self.monitor_config))
                if self.vision_engine.needs_color:
                    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                else:
                    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)

                with self._frame_lock:
                    self._current_frame = frame

                # 3. Read latest detections (non-blocking)
                with self._result_lock:
                    detections = list(self._last_detections)

                # 4. Draw all detections on overlay and log them
                self.overlay.clear()
                off_x = self.monitor_config["left"]
                off_y = self.monitor_config["top"]
                status_parts = [f"FPS: {fps:.1f}"]

                for det in detections:
                    self.overlay.draw_box(
                        det.x + off_x, det.y + off_y,
                        det.w, det.h,
                        TARGETS[det.label]["color"], det.label,
                    )
                    self.logger.add_point(
                        det.x + off_x, det.y + off_y,
                        det.w, det.h, det.label,
                    )

                # 5. Status bar — best confidence per label
                best: dict[str, float] = {}
                for det in detections:
                    best[det.label] = max(best.get(det.label, 0.0), det.confidence)
                for label in TARGETS:
                    status_parts.append(f"{label}:{best.get(label, 0.0):.2f}")

                full_status = " | ".join(status_parts)
                if self.logger.get_record_status():
                    self.overlay.draw_status(f"● REC [{self.engine_name}] {full_status}", "red")
                else:
                    self.overlay.draw_status(f"○ IDLE [{self.engine_name}] {full_status}", "lime")

                self.overlay.update()
                time.sleep(0.016)  # ~60 FPS display cap

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.running = False  # signal detection thread to exit
            if self.logger.get_record_status():
                self.logger.save_to_csv()
            self.overlay.destroy()
