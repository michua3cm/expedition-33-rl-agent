from .analysis.entry import run_analysis as run_analysis
from .app import CalibrationApp
from .collector import SmartCollector


def run_recorder(engine="PIXEL"):
    """Wrapper to instantiate and run the calibration recorder."""
    print(f">> [Recorder] Initializing Vision System using {engine.upper()}...")
    app = CalibrationApp(engine=engine.upper())
    app.run()


def run_collector(engine="COMPOSITE"):
    """Wrapper to instantiate and run the smart screenshot collector."""
    print(">> [Collector] Starting smart collection for YOLO training data...")
    collector = SmartCollector(engine=engine)
    collector.run()
