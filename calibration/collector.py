import time

import cv2
import mss
import numpy as np

from overlay_ui import OverlayWindow

from .config import MONITOR_INDEX, YOLO_RAW_DIR

# Key Codes
VK_CAPTURE = 0x78  # F9  — single screenshot
VK_AUTO    = 0x79  # F10 — toggle auto-capture
VK_EXIT    = 0x7A  # F11 — exit

AUTO_CAPTURE_INTERVAL = 1.0  # seconds between auto-captures


class ScreenshotCollector:
    """
    Captures and saves screenshots for YOLO training data collection.

    F9  — save a single screenshot now
    F10 — toggle auto-capture (one screenshot every second)
    F11 — exit
    """

    def __init__(self):
        self.sct = mss.mss()
        self.monitor_config = self._setup_monitor()
        self.overlay = OverlayWindow()
        self.running = True
        self.auto_capture = False
        self.count = 0
        self._last_auto_time = 0.0

    def _setup_monitor(self) -> dict:
        raw = self.sct.monitors[MONITOR_INDEX]
        w, h = raw["width"], raw["height"]
        margin_h = int(h * 0.2)
        return {
            "top": raw["top"] + margin_h,
            "left": raw["left"],
            "width": w,
            "height": h - (margin_h * 2),
            "mon": MONITOR_INDEX,
        }

    def _save_screenshot(self) -> str:
        timestamp = int(time.time() * 1000)
        filename = f"{YOLO_RAW_DIR}/frame_{timestamp}.png"
        screenshot = np.array(self.sct.grab(self.monitor_config))
        cv2.imwrite(filename, screenshot)
        self.count += 1
        return filename

    def _handle_input(self):
        import win32api  # type: ignore  # Windows-only; imported lazily so the module loads on Linux
        if win32api.GetAsyncKeyState(VK_CAPTURE) & 0x8000:
            self._save_screenshot()
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_AUTO) & 0x8000:
            self.auto_capture = not self.auto_capture
            state = "ON" if self.auto_capture else "OFF"
            print(f"[Collector] Auto-capture {state}")
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_EXIT) & 0x8000:
            self.running = False
            print(f"\n[Collector] Done. {self.count} screenshots saved to '{YOLO_RAW_DIR}'.")

    def run(self):
        print("=== Screenshot Collector ===")
        print(f"F9: Capture | F10: Toggle auto ({AUTO_CAPTURE_INTERVAL}s interval) | F11: Exit")
        print(f"Saving to: {YOLO_RAW_DIR}")

        loop_time = time.time()
        try:
            while self.running:
                curr_time = time.time()
                fps = 1 / (curr_time - loop_time) if (curr_time - loop_time) > 0 else 0
                loop_time = curr_time

                self._handle_input()

                # Auto-capture tick
                if self.auto_capture and (curr_time - self._last_auto_time) >= AUTO_CAPTURE_INTERVAL:
                    self._save_screenshot()
                    self._last_auto_time = curr_time

                # Overlay status
                self.overlay.clear()
                mode = "● AUTO" if self.auto_capture else "○ IDLE"
                self.overlay.draw_status(
                    f"{mode} | Captured: {self.count} | FPS: {fps:.1f}",
                    "lime" if not self.auto_capture else "red",
                )
                self.overlay.update()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.overlay.destroy()
