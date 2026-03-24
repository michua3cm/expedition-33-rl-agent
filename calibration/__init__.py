from .app import CalibrationApp
from .collector import ScreenshotCollector
from .analysis.entry import run_analysis


def run_recorder(engine="PIXEL"):
    """Wrapper to instantiate and run the calibration recorder."""
    print(f">> [Recorder] Initializing Vision System using {engine.upper()}...")
    app = CalibrationApp(engine=engine.upper())
    app.run()


def run_collector():
    """Wrapper to instantiate and run the screenshot collector."""
    print(">> [Collector] Starting screenshot collection for YOLO training data...")
    collector = ScreenshotCollector()
    collector.run()