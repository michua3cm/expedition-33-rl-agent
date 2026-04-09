import threading
import time

import cv2
import mss
import numpy as np

import vision
from overlay_ui import OverlayWindow
from vision.engine import Detection

from .config import (
    ASSETS_DIR,
    MONITOR_INDEX,
    TARGETS,
    YOLO_CLASSES,
    YOLO_LABELED_IMAGES_DIR,
    YOLO_LABELED_LABELS_DIR,
    YOLO_RAW_DIR,
)
from .roi_overlay import draw_roi_overlays

# Key Codes
VK_TRIGGER = 0x77  # F8  — toggle trigger mode (detection fires → save to labeled/)
VK_CAPTURE = 0x78  # F9  — manual: save one PNG to raw/
VK_AUTO    = 0x79  # F10 — toggle auto-capture at 4 FPS → raw/
VK_EXIT    = 0x7A  # F11 — exit
VK_ROI     = 0x09  # Tab — toggle ROI boundary overlay

AUTO_INTERVAL    = 0.25  # seconds between auto-captures (4 FPS)
TRIGGER_COOLDOWN = 0.4   # seconds between trigger saves per target


class SmartCollector:
    """
    Three-mode screenshot collector for YOLO training data.

    F8  — toggle trigger mode: each detection event saves PNG+TXT to labeled/
    F9  — manual: save one PNG to raw/ immediately
    F10 — toggle auto-capture: one PNG to raw/ every 0.25 s (4 FPS)
    Tab — toggle ROI boundary overlay
    F11 — exit

    Trigger cooldown: 0.4 s per target label to avoid redundant saves of the
    same static UI element.  GRADIENT_INCOMING is skipped in label files
    because it has no bounding box.
    """

    def __init__(self, engine: str = "COMPOSITE"):
        self.sct = mss.mss()
        self.monitor_config = self._setup_monitor()
        self.overlay = OverlayWindow()
        self.vision_engine = vision.registry.create(engine.upper())
        self.vision_engine.load(TARGETS, ASSETS_DIR)

        self.running = True
        self._trigger_mode = False
        self._auto_capture = False
        self._show_roi = True

        self._current_frame: np.ndarray | None = None
        self._last_detections: list[Detection] = []
        self._frame_lock = threading.Lock()
        self._result_lock = threading.Lock()

        # last save time per target label for trigger cooldown
        self._trigger_cooldowns: dict[str, float] = {}
        self._last_auto_time = 0.0

        self._raw_count = 0
        self._labeled_count = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_monitor(self) -> dict:
        raw = self.sct.monitors[MONITOR_INDEX]
        w, h = raw["width"], raw["height"]
        return {
            "top": raw["top"],
            "left": raw["left"],
            "width": w,
            "height": h,
            "mon": MONITOR_INDEX,
        }

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _write_yolo_label(
        self,
        path: str,
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Write a YOLO label file.  GRADIENT_INCOMING is skipped (no bbox)."""
        lines: list[str] = []
        for det in detections:
            if det.label == "GRADIENT_INCOMING":
                continue
            if det.label not in YOLO_CLASSES:
                continue
            class_id = YOLO_CLASSES.index(det.label)
            x_center = (det.x + det.w / 2) / frame_w
            y_center = (det.y + det.h / 2) / frame_h
            w_frac   = det.w / frame_w
            h_frac   = det.h / frame_h
            lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_frac:.6f} {h_frac:.6f}"
            )
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def _save_raw(self, frame_bgr: np.ndarray) -> None:
        timestamp = int(time.time() * 1000)
        img_path = f"{YOLO_RAW_DIR}/frame_{timestamp}.png"
        cv2.imwrite(img_path, frame_bgr)
        self._raw_count += 1

    def _save_labeled(
        self, frame_bgr: np.ndarray, detections: list[Detection]
    ) -> None:
        timestamp = int(time.time() * 1000)
        stem = f"frame_{timestamp}"
        img_path = f"{YOLO_LABELED_IMAGES_DIR}/{stem}.png"
        lbl_path = f"{YOLO_LABELED_LABELS_DIR}/{stem}.txt"
        cv2.imwrite(img_path, frame_bgr)
        h, w = frame_bgr.shape[:2]
        self._write_yolo_label(lbl_path, detections, w, h)
        self._labeled_count += 1

    # ------------------------------------------------------------------
    # Background detection thread
    # ------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """Background thread: run vision detection as fast as the engine allows."""
        while self.running:
            with self._frame_lock:
                frame = self._current_frame
            if frame is not None:
                dets = self.vision_engine.detect(frame)
                with self._result_lock:
                    self._last_detections = dets

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_input(self, frame_bgr: np.ndarray) -> None:
        import win32api  # type: ignore  # Windows-only; imported lazily so module loads on Linux

        if win32api.GetAsyncKeyState(VK_TRIGGER) & 0x8000:
            self._trigger_mode = not self._trigger_mode
            state = "ON" if self._trigger_mode else "OFF"
            print(f"[Collector] Trigger mode {state}")
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_CAPTURE) & 0x8000:
            self._save_raw(frame_bgr)
            print(f"[Collector] Manual save → raw/ ({self._raw_count} total)")
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_AUTO) & 0x8000:
            self._auto_capture = not self._auto_capture
            state = "ON" if self._auto_capture else "OFF"
            print(f"[Collector] Auto-capture {state}")
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_ROI) & 0x8000:
            self._show_roi = not self._show_roi
            state = "ON" if self._show_roi else "OFF"
            print(f"[Overlay] ROI boundaries {state}")
            time.sleep(0.3)  # debounce

        if win32api.GetAsyncKeyState(VK_EXIT) & 0x8000:
            self.running = False
            print(
                f"\n[Collector] Done.  raw={self._raw_count}  "
                f"labeled={self._labeled_count}"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("=== Smart Collector ===")
        print(
            "F8: Toggle trigger | F9: Manual capture | "
            "F10: Toggle auto (4 FPS) | Tab: ROI | F11: Exit"
        )
        print(f"Raw     → {YOLO_RAW_DIR}")
        print(f"Labeled → {YOLO_LABELED_IMAGES_DIR}")

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

                # 1. Capture frame
                screenshot = np.array(self.sct.grab(self.monitor_config))
                frame_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                if self.vision_engine.needs_color:
                    frame = frame_bgr
                else:
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                with self._frame_lock:
                    self._current_frame = frame

                # 2. Read latest detections (non-blocking)
                with self._result_lock:
                    detections = list(self._last_detections)

                # 3. Handle keyboard input
                self._handle_input(frame_bgr)

                # 4. Trigger mode: save labeled frame when any detection fires
                #    and its per-target cooldown has elapsed.
                if self._trigger_mode and detections:
                    for det in detections:
                        last = self._trigger_cooldowns.get(det.label, 0.0)
                        if curr_time - last >= TRIGGER_COOLDOWN:
                            self._save_labeled(frame_bgr, detections)
                            # Reset cooldowns for every label in this save so
                            # the same static scene is not saved again immediately.
                            for d in detections:
                                self._trigger_cooldowns[d.label] = curr_time
                            break  # one save per loop iteration

                # 5. Auto-capture tick
                if self._auto_capture and (curr_time - self._last_auto_time) >= AUTO_INTERVAL:
                    self._save_raw(frame_bgr)
                    self._last_auto_time = curr_time

                # 6. Overlay
                self.overlay.clear()
                off_x = self.monitor_config["left"]
                off_y = self.monitor_config["top"]

                if self._show_roi:
                    fh, fw = frame_bgr.shape[:2]
                    draw_roi_overlays(self.overlay, TARGETS, fw, fh, off_x, off_y)

                for det in detections:
                    self.overlay.draw_box(
                        det.x + off_x, det.y + off_y,
                        det.w, det.h,
                        TARGETS[det.label]["color"], det.label,
                    )

                # 7. Status bar
                trig = "●TRIG" if self._trigger_mode else "○TRIG"
                auto = "●AUTO" if self._auto_capture else "○AUTO"
                active = self._trigger_mode or self._auto_capture
                self.overlay.draw_status(
                    f"{trig} {auto} | raw={self._raw_count} "
                    f"labeled={self._labeled_count} | FPS:{fps:.1f}",
                    "red" if active else "lime",
                )
                self.overlay.update()
                time.sleep(0.016)  # ~60 FPS display cap

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.running = False
            self.overlay.destroy()
