import tkinter as tk

# Re-assert HWND_TOPMOST every N milliseconds to prevent other windows
# (VS Code, video players, etc.) from covering the overlay.
_TOPMOST_INTERVAL_MS = 1000

class OverlayWindow:
    def __init__(self):
        self.root = tk.Tk()

        # Window setup: Title, Fullscreen, Borderless
        self.root.title("AI_Vision_Overlay")
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.overrideredirect(True)

        # Topmost (Always on top)
        self.root.attributes("-topmost", True)

        # Transparency Setup (Chroma Key method)
        # We set background to black, and tell Windows that "black is transparent"
        self.bg_color = "black"
        self.root.config(bg=self.bg_color)
        self.root.attributes("-transparentcolor", self.bg_color)

        # Canvas Setup
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h,
                                bg=self.bg_color, highlightthickness=0)
        self.canvas.pack()

        # Enable Click-through (via Windows API)
        # We use .after() to ensure window is fully created before applying styles
        self.root.after(10, self.set_click_through)

    def set_click_through(self):
        import ctypes  # type: ignore  # Windows-only; imported lazily so the module loads on Linux

        import win32con  # type: ignore
        import win32gui  # type: ignore
        # Get window handle (HWND)
        hwnd = win32gui.FindWindow(None, "AI_Vision_Overlay")
        if hwnd:
            # Get current extended style
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            # Add Layered and Transparent styles
            # WS_EX_TRANSPARENT makes mouse clicks pass through the window
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                   ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
            # SetWindowLong can silently reset the Z-order, so explicitly
            # re-assert HWND_TOPMOST after changing extended styles.
            # SWP_NOACTIVATE prevents the overlay from stealing keyboard focus.
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE,
            )
            # Exclude this window from screen capture (mss, OBS, PrintScreen, etc.)
            # so that overlay boxes and status text are never baked into saved screenshots.
            # WDA_EXCLUDEFROMCAPTURE = 0x11, available on Windows 10 build 19041+
            WDA_EXCLUDEFROMCAPTURE = 0x11
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        # Keep re-asserting topmost periodically so apps that temporarily
        # steal Z-order (video players, IDEs, etc.) don't cover the overlay.
        self.root.after(_TOPMOST_INTERVAL_MS, self.set_click_through)

    def draw_box(self, x, y, w, h, color, label):
        """Draws a bounding box and a label on the overlay."""
        self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=3)
        self.canvas.create_text(x, y-22, text=label, fill=color, font=("Arial", 14), anchor="nw")

    def draw_roi_rect(self, x: int, y: int, w: int, h: int, color: str, label: str) -> None:
        """Draws a dashed ROI boundary rectangle with a small label.

        Visually distinct from detection boxes (draw_box uses solid width-3
        lines; ROI rects use dashed width-1 lines) so the two are never
        confused at a glance.
        """
        self.canvas.create_rectangle(
            x, y, x + w, y + h,
            outline=color, width=1, dash=(6, 4),
        )
        self.canvas.create_text(
            x + 4, y + 4,
            text=f"[{label}]", fill=color,
            font=("Arial", 9), anchor="nw",
        )

    def draw_status(self, text, color="white"):
        """Displays the current program state in the top-left corner."""
        self.canvas.create_text(20, 20, text=text, fill=color, font=("Arial", 16, "bold"), anchor="nw")

    def clear(self):
        """Clears all drawings from the canvas."""
        self.canvas.delete("all")

    def update(self):
        """Updates the tkinter event loop."""
        self.root.update()

    def destroy(self):
        """Closes the window safely."""
        self.root.destroy()
