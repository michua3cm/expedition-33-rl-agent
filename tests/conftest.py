"""
conftest.py — session-scoped stubs for platform-specific modules.

tkinter, pynput, and the win32 family are unavailable on the CI Linux
runner (no display, no Windows API layer).  Because these are imported
at the top level of application modules, any test that transitively
imports them would fail at collection time — before any mocking in the
test body can run.

Installing these stubs into sys.modules here, before the first test is
collected, lets every test in the suite import freely.  Tests that
exercise these modules directly use unittest.mock to replace them with
more specific fakes.
"""

import sys
import types


def _stub(name: str, **attrs) -> types.ModuleType:
    """Create a minimal stub module and register it in sys.modules."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# tkinter — graphical toolkit; not available on headless CI runners
# ---------------------------------------------------------------------------
if "tkinter" not in sys.modules:
    _tk = _stub("tkinter")
    _tk.Tk = type("Tk", (), {
        "title": lambda *a, **k: None,
        "winfo_screenwidth": lambda *a: 1920,
        "winfo_screenheight": lambda *a: 1080,
        "geometry": lambda *a, **k: None,
        "overrideredirect": lambda *a, **k: None,
        "attributes": lambda *a, **k: None,
        "config": lambda *a, **k: None,
        "after": lambda *a, **k: None,
        "update": lambda *a, **k: None,
        "destroy": lambda *a, **k: None,
    })
    _tk.Canvas = type("Canvas", (), {
        "__init__": lambda *a, **k: None,
        "pack": lambda *a, **k: None,
        "create_rectangle": lambda *a, **k: None,
        "create_text": lambda *a, **k: None,
        "delete": lambda *a, **k: None,
    })
    _stub("tkinter.ttk")
    _stub("tkinter.messagebox")


# ---------------------------------------------------------------------------
# pynput — keyboard/mouse listener; requires X11 / Windows API at import time
# ---------------------------------------------------------------------------
if "pynput" not in sys.modules:
    _pynput = _stub("pynput")

    _keyboard = _stub("pynput.keyboard")
    _keyboard.Key = type("Key", (), {"__getattr__": lambda self, name: name})()

    class _KeyCode:
        def __init__(self, char: str = "") -> None:
            self.char = char
        @classmethod
        def from_char(cls, char: str) -> "_KeyCode":
            return cls(char)
    _keyboard.KeyCode = _KeyCode

    _keyboard.Listener = type("Listener", (), {
        "__init__": lambda *a, **k: None,
        "start": lambda *a: None,
        "stop": lambda *a: None,
    })
    _keyboard.Controller = type("Controller", (), {
        "__init__": lambda *a, **k: None,
    })

    _mouse = _stub("pynput.mouse")
    _mouse.Button = type("Button", (), {"left": "left", "right": "right"})
    _mouse.Listener = type("Listener", (), {
        "__init__": lambda *a, **k: None,
        "start": lambda *a: None,
        "stop": lambda *a: None,
    })
    _mouse.Controller = type("Controller", (), {
        "__init__": lambda *a, **k: None,
    })

    _pynput.keyboard = _keyboard
    _pynput.mouse = _mouse


# ---------------------------------------------------------------------------
# win32 family — Windows API; not present on Linux
# ---------------------------------------------------------------------------
for _name in ("win32api", "win32con", "win32gui"):
    if _name not in sys.modules:
        _stub(_name)
