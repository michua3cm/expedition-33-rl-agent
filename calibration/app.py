import mss
import cv2
import time
import numpy as np
import win32api # type: ignore

# Import local modules
from .config import TARGETS, MONITOR_INDEX, SCREENSHOT_DIR
from .loader import load_templates
from .logger import CalibrationLogger
from overlay_ui import OverlayWindow

# Key Codes
VK_START = 0x78 # F9
VK_STOP  = 0x79 # F10
VK_EXIT  = 0x7A # F11

class CalibrationApp:
    def __init__(self, engine="PIXEL"):
        self.engine = engine.upper()
        self.sct = mss.mss()
        self.monitor_config = self._setup_monitor()
        self.overlay = OverlayWindow()
        self.logger = CalibrationLogger()
        self.templates = load_templates(TARGETS, engine=self.engine)
        self.running = True

        # Dynamic Engine Setup
        if self.engine == "SIFT":
            self.sift = cv2.SIFT_create()
            from .matcher.sift import match_target
            self.match_target = match_target
        else:
            self.sift = None
            from .matcher.pixel import match_target
            self.match_target = match_target

    def _setup_monitor(self):
        """Configure monitor cropping region."""
        raw = self.sct.monitors[MONITOR_INDEX]

        # Gather the width & height of the monitor
        w, h = raw["width"], raw["height"]
        
        # Padding
        margin_w = int(w * 0.0)
        margin_h = int(h * 0.2)
        
        return {
            "top": raw["top"] + margin_h,
            "left": raw["left"] + margin_w,
            "width": w - (margin_w * 2),
            "height": h - (margin_h * 2),
            "mon": MONITOR_INDEX
        }

    def _handle_input(self):
        """Check keyboard input for state changes."""
        # Check F9 (Start)
        if win32api.GetAsyncKeyState(VK_START) & 0x8000:
            if not self.logger.get_record_status():
                self.logger.start_recording()
                # Take debug screenshot
                sc = np.array(self.sct.grab(self.monitor_config))
                cv2.imwrite(f"{SCREENSHOT_DIR}/debug_capture.png", sc)
                time.sleep(0.3)

        # Check F10 (Stop & Save)
        if win32api.GetAsyncKeyState(VK_STOP) & 0x8000:
            if self.logger.get_record_status():
                self.logger.stop_recording()
                time.sleep(0.3)

        # Check F11 (Exit)
        if win32api.GetAsyncKeyState(VK_EXIT) & 0x8000:
            self.running = False
            print("\n[Exit] Exiting program.")

    def run(self):
        print("=== Calibration App Started ===")
        print("F9: Start | F10: Stop | F11: Exit")

        loop_time = time.time()
        try:
            while self.running:
                # FPS Calculation
                curr_time = time.time()
                fps = 1 / (curr_time - loop_time) if (curr_time - loop_time) > 0 else 0
                loop_time = curr_time

                # 1. Input
                self._handle_input()

                # 2. Vision Pipeline
                screenshot = np.array(self.sct.grab(self.monitor_config))
                img_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)    # Convert to grayscale
                
                self.overlay.clear()
                offset = (self.monitor_config["left"], self.monitor_config["top"])  # Offset for coordinate conversion
                status_parts = [f"FPS: {fps:.1f}"]  # Status string builder

                # 3. Pre-compute live frame features (FPS Optimization)
                frame_data = {}
                if self.engine == "SIFT":
                    live_kp, live_des = self.sift.detectAndCompute(img_gray, None)
                    frame_data = {"kp": live_kp, "des": live_des}

                # 4. Match Targets
                for name, data in self.templates.items():
                    conf = self.match_target(name, data, img_gray, offset, self.overlay, self.logger, frame_data)
                    # SIFT returns match count, PIXEL returns confidence %
                    val_str = f"{conf}" if self.engine == "SIFT" else f"{conf:.2f}"
                    status_parts.append(f"{name}:{val_str}")

                # 5. Update Status Bar
                full_status = " | ".join(status_parts)
                if self.logger.get_record_status():
                    self.overlay.draw_status(f"● REC [{self.engine}] {full_status}", "red")
                else:
                    self.overlay.draw_status(f"○ IDLE [{self.engine}] {full_status}", "lime")
                
                self.overlay.update()

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            pass
        finally:
            if self.logger.get_record_status():
                self.logger.save_to_csv()
            self.overlay.destroy()