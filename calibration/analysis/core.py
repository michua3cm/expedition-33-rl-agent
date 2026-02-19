import os
import glob
import pandas as pd
import numpy as np
import mss
from ..config import LOG_DIR, MONITOR_INDEX

class LogAnalyzer:
    def __init__(self, padding=50):
        self.log_dir = LOG_DIR
        self.padding = padding
        self.screen_w = 0
        self.screen_h = 0
        self._get_screen_resolution()

    def _get_screen_resolution(self):
        """Internal helper to get screen size for clamping."""
        with mss.mss() as sct:
            mon = sct.monitors[MONITOR_INDEX]
            self.screen_w = mon['width']
            self.screen_h = mon['height']

    def load_and_merge_logs(self):
        """Step 1: Load all CSV files and merge them."""
        search_path = os.path.join(self.log_dir, "*.csv")
        all_files = glob.glob(search_path)  # Find all CSV files
        
        if not all_files:
            print(f"[Error] No log files found in {self.log_dir}")
            return None

        print(f"[Analyzer] Found {len(all_files)} log files.")
        
        # Load and Merge Data (DataFrame)
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                if not df.empty:
                    df_list.append(df)
            except Exception as e:
                print(f"[Warning] Skipping bad file {filename}: {e}")

        if not df_list:
            print("[Error] Logs exist but contain no valid data.")
            return None
        
        # Combine all dataframes
        return pd.concat(df_list, ignore_index=True)

    def calculate_roi(self, df):
        """Step 2: Analyze coordinates and apply padding."""
        if df is None or df.empty:
            return None

        # Extract coordinates
        # Columns: x, y, w, h, type
        xs = df['x'].values
        ys = df['y'].values
        ws = df['w'].values
        hs = df['h'].values

        # Find absolute boundaries
        min_x = np.min(xs)
        max_x = np.max(xs + ws)
        min_y = np.min(ys)
        max_y = np.max(ys + hs)

        # Apply padding and clamp to screen size
        final_left = max(0, int(min_x - self.padding))
        final_top = max(0, int(min_y - self.padding))
        
        final_right = min(self.screen_w, int(max_x + self.padding))
        final_bottom = min(self.screen_h, int(max_y + self.padding))

        # Calculate final width/height
        final_w = final_right - final_left
        final_h = final_bottom - final_top

        return {
            "top": final_top,
            "left": final_left,
            "width": final_w,
            "height": final_h,
            "mon": MONITOR_INDEX
        }

    def output_result(self, roi):
        """Step 3: Output the result clearly."""
        if not roi:
            return

        print("\n" + "="*50)
        print("OPTIMAL REGION OF INTEREST (ROI)")
        print("="*50)
        print(f"Padding Applied: {self.padding}px")
        print("-" * 50)
        print("Copy this dictionary into your RL Agent config:")
        print("")
        print(f"MONITOR_ROI = {roi}")
        print("")
        print("="*50)