from .app import CalibrationApp
from .analysis.entry import run_analysis

def run_recorder(engine="PIXEL"):
    """Wrapper to instantiate and run the recorder."""
    print(f">> [Recorder] Initializing Vision System using {engine.upper()}...")
    app = CalibrationApp(engine=engine.upper())
    app.run()