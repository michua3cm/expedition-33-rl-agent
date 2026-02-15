import csv
import os
from datetime import datetime
from .config import LOG_DIR

class CalibrationLogger:
    def __init__(self):
        self._recording = False
        self.points = []
    
    def get_record_status(self):
        return self._recording

    def start_recording(self):
        self._recording = True
        self.points = []
        print("\n[Logger] Recording Started")

    def stop_recording(self):
        self._recording = False
        print("[Logger] Recording Stopped. Saving data...")
        self.save_to_csv()

    def add_point(self, x, y, w, h, label):
        if self._recording:
            self.points.append((x, y, w, h, label))

    def save_to_csv(self):
        if not self.points:
            print("[Logger] No data collected.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(LOG_DIR, f"calibration_{timestamp}.csv")

        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "w", "h", "type"])
                writer.writerows(self.points)
            print(f"[Success] Saved {len(self.points)} points to: {filepath}")
        except Exception as e:
            print(f"[Error] Failed to save CSV: {e}")