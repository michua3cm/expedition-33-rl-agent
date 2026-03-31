import ctypes
import time
from ctypes import wintypes

import win32api
import win32con

# --- Low-Level Windows Structs for DirectInput ---
user32 = ctypes.windll.user32

# DirectInput Scan Codes (Hardware Mapping)
# These are universal for almost all PC games.
SCANCODES = {
    # Dual-purpose keys — behaviour depends on combat phase:
    #   Q  — DODGE (enemy turn, Phase 1)      | Open Gradient Menu (player turn, Phase 2)
    #   W  — GRADIENT_PARRY (enemy turn, P1)  | Open Item Menu (player turn, Phase 2)
    #   E  — PARRY (enemy turn, Phase 1)      | Open Skill Menu (player turn, Phase 2)
    "Q": 0x10,
    "W": 0x11,
    "E": 0x12,
    "R": 0x13,      # Switch skill pages
    "A": 0x1E,      # Choose left target
    "D": 0x20,      # Choose right target
    "F": 0x21,      # Basic attack / Confirm action

    # Special Actions
    "SPACE": 0x39,  # Jump
    "ESC": 0x01,    # Cancel / Back
    "ENTER": 0x1C,  # Confirm in some scenarios

    # Reserved
    "TAB": 0x0F,
    "SHIFT": 0x2A,  # Sprint
    "CTRL": 0x1D,
}

# CStructs for SendInput
class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_ulong)]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.DWORD),
                ("dwFlags", ctypes.DWORD),
                ("time", ctypes.DWORD),
                ("dwExtraInfo", ctypes.c_ulong)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.DWORD),
                ("ki", KEYBDINPUT), # Key Input
                ("mi", MOUSEINPUT), # Mouse Input
                ("hi", ctypes.c_void_p)] # Hardware Input

def _send_input(input_struct):
    user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

class GameController:
    """
    High-level interface for controlling Expedition 33.
    Uses DirectInput Scan Codes to ensure compatibility with the game engine.
    """

    def __init__(self):
        self.default_delay = 0.05  # 50ms hold time is usually safe for games

    def _press_key_scan(self, hexKeyCode):
        """Presses a key down."""
        x = INPUT(type=1, ki=KEYBDINPUT(wVk=0, wScan=hexKeyCode, dwFlags=0x0008, time=0, dwExtraInfo=0))
        _send_input(x)

    def _release_key_scan(self, hexKeyCode):
        """Releases a key."""
        x = INPUT(type=1, ki=KEYBDINPUT(wVk=0, wScan=hexKeyCode, dwFlags=0x0008 | 0x0002, time=0, dwExtraInfo=0))
        _send_input(x)

    def tap_key(self, key_name, duration=None):
        """
        Atomic Action: Press and Release a key.
        Args:
            key_name (str): 'W', 'A', 'SPACE', etc.
            duration (float): How long to hold the key.
        """
        if duration is None:
            duration = self.default_delay

        code = SCANCODES.get(key_name.upper())
        if code is None:
            print(f"[Warning] Key '{key_name}' not defined in SCANCODES.")
            return

        self._press_key_scan(code)
        time.sleep(duration)
        self._release_key_scan(code)
        # Small cooldown to prevent inputs merging
        time.sleep(0.02)

    def click_mouse(self, button="left"):
        """
        Atomic Action: Click mouse button.
        """
        if button == "left":
            down_flag = win32con.MOUSEEVENTF_LEFTDOWN
            up_flag = win32con.MOUSEEVENTF_LEFTUP
        elif button == "right":
            down_flag = win32con.MOUSEEVENTF_RIGHTDOWN
            up_flag = win32con.MOUSEEVENTF_RIGHTUP
        else:
            return

        win32api.mouse_event(down_flag, 0, 0, 0, 0)
        time.sleep(self.default_delay)
        win32api.mouse_event(up_flag, 0, 0, 0, 0)

    # --- PHASE 1: DEFENSIVE ACTIONS ---

    def dodge(self):
        """Press Q to Dodge."""
        self.tap_key("Q")

    def parry(self):
        """Press E to Parry."""
        self.tap_key("E")

    def gradient_parry(self):
        """Press W to Gradient Parry (grey screen / GRADIENT_INCOMING cue)."""
        self.tap_key("W")

    def jump(self):
        """Press SPACE to Jump."""
        self.tap_key("SPACE")

    def jump_attack(self):
        """Left click to counter-attack during the jump attack window (MOUSE cue)."""
        self.click_mouse("left")

    # --- PHASE 1: OFFENSIVE ACTIONS ---

    def attack(self):
        """Press F to initiate a basic attack."""
        self.tap_key("F")

    def normal_attack_init(self):
        """Alias for attack() — press F to initiate basic attack."""
        self.attack()

    # --- PHASE 2: MENU NAVIGATION (planned) ---

    def confirm_selection(self):
        """Press F to confirm target/action."""
        self.tap_key("F")

    def cancel_selection(self):
        """Press ESC to go back."""
        self.tap_key("ESC")

    def open_skill_menu(self):
        """Press E to open Skill Menu."""
        self.tap_key("E")

    def open_gradient_menu(self):
        """Press Q to open Gradient Menu."""
        self.tap_key("Q")

    def open_item_menu(self):
        """Press W to open Item Menu."""
        self.tap_key("W")

    def navigate_left(self):
        """Press A to choose target left."""
        self.tap_key("A")

    def navigate_right(self):
        """Press D to choose target right."""
        self.tap_key("D")

    def switch_skill_page(self):
        """Press R to switch skill pages."""
        self.tap_key("R")

    def select_slot(self, slot_num):
        """
        Selects item/skill slot 1, 2, or 3.
        slot_num: 1='Q', 2='W', 3='E'
        """
        mapping = {1: "Q", 2: "W", 3: "E"}
        key = mapping.get(slot_num)
        if key:
            self.tap_key(key)
