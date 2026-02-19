from .app import CalibrationApp

def run_recorder():
    """Wrapper to instantiate and run the recorder."""
    print(">> [Recorder] Initializing Vision System...")
    app = CalibrationApp()
    app.run()