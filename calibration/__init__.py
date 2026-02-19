from .app import CalibrationApp
from .analysis.entry import run_analysis

def run_recorder():
    """Wrapper to instantiate and run the recorder."""
    print(">> [Recorder] Initializing Vision System...")
    app = CalibrationApp()
    app.run()