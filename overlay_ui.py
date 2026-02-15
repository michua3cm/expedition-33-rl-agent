import tkinter as tk
import win32gui # type: ignore
import win32con # type: ignore

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
        # Get window handle (HWND)
        hwnd = win32gui.FindWindow(None, "AI_Vision_Overlay")
        if hwnd:
            # Get current extended style
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            # Add Layered and Transparent styles
            # WS_EX_TRANSPARENT makes mouse clicks pass through the window
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                                   ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)

    def draw_box(self, x, y, w, h, color, label):
        """Draws a bounding box and a label on the overlay."""
        self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=3)
        self.canvas.create_text(x, y-15, text=label, fill=color, font=("Arial", 14, "bold"), anchor="nw")

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