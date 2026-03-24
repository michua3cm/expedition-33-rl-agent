# Expedition 33 RL Agent 🤖⚔️

> **Project Goal:** To develop a Reinforcement Learning (RL) agent capable of playing _Expedition 33_ autonomously using computer vision and advanced control algorithms.

## 📖 Overview

This project bridges the gap between classic control theory and modern Deep Reinforcement Learning. The agent interacts with the game solely through visual inputs (screenshots) and keyboard simulation, mimicking a human player without accessing internal game memory or code injection.

**Current Phase: Vision System & Calibration**
We have built a highly modular "Eyes" module for the agent. This module is responsible for:

1. **Real-time Object Detection:** Identifying dynamic game states (Perfect, Dodge, Parry, UI Icons) regardless of window size or screen resolution.
2. **System Identification:** Calibrating the agent's focus area (Region of Interest) to maximize FPS.
3. **Data Collection:** Logging gameplay events to train the future Reward Model.

## 🚀 Key Features

- **Dual Vision Engines:** Switch between ultra-fast Pixel Template Matching and resolution-independent SIFT Feature Matching on the fly via CLI.

- **Non-Intrusive Capture:** Uses `mss` for high-speed screen capture (>60 FPS) without hooking into the game process.
- **Transparent Debug Overlay:** A custom Win32-based HUD that draws bounding boxes over the game in real-time for immediate visual feedback.
- **Data Logging Pipeline:** Automatically records coordinates and events to CSV for offline analysis and RL environment setup.
- **Modular Architecture:** Clean, OOP-driven separation between Configuration, Vision Logic, and UI.

## 📂 Project Structure

```text
expedition-33-rl-agent/
├── .python-version          # Anchors the Python runtime version
├── pyproject.toml           # Project metadata, dependencies, and tool configs
├── uv.lock                  # Deterministic dependency lockfile
├── assets/                  # Template images (png) for the Vision System
├── calibration/             # [Module] Vision & Data Collection
│   ├── app.py               # Main Application Logic (The Orchestrator)
│   ├── config.py            # Central Configuration (Targets & Thresholds)
│   ├── loader.py            # Asset Loading Logic
│   ├── logger.py            # CSV Logging Logic
│   ├── matcher/             # Core Computer Vision Algorithms
│   │   ├── pixel.py         # Standard TM_CCOEFF_NORMED matching
│   │   └── sift.py          # Scale-invariant feature matching
│   └── analysis/            # Offline ROI optimization tools
│       ├── core.py
│       └── entry.py
├── data/
│   ├── logs/                # Generated CSV training data
│   └── screenshots/         # Debug snapshots for template creation
├── overlay_ui.py            # Win32 Transparent Overlay Class
├── main.py                  # CLI Entry Point
└── README.md
```

## 🛠️ Installation

**Prerequisite:**

This project uses `uv` for package and environment management.

Install `uv` globally if you don't have it:

- **Git Bash (Windows) / macOS / Linux:**

  ```bash
  curl -sSL https://astral.sh/uv/install.sh | bash
  ```

- **VS Code Integrated Terminal (Windows PowerShell):**

  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

> ⚠️ **Note:** Restart your terminal (or VS Code) after installing to enable the `uv` command.

---

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd expedition-33-rl-agent
   ```

2. **Sync the environment**

   This single command automatically creates the isolated `.venv` folder and installs all exact, locked dependencies from the `uv.lock` file:

   ```bash
   uv sync
   ```

3. **Run the application**

   You do not need to manually activate the environment. `uv` automatically routes execution through the correct virtual environment:

   ```bash
   uv run main.py
   ```

## 🎮 Usage (Calibration Module)

**⚠️ Administrator Privileges Required:**
This program uses global hotkeys (`win32api`) and draws a topmost overlay. You must run your terminal or IDE as **Administrator**.

1. **Launch the Game:**  
   Ensure **_Clair Obscur: Expedition 33_** is running in **Windowed** or **Borderless Window** mode.

2. **Run the Agent's Vision System (Record Mode)**  
   You can choose which computer vision engine drives the agent using the `--engine` flag.

   ```bash
   # Use standard pixel matching (Best for fixed resolutions)
   uv run main.py record --engine pixel

   # Use scale-invariant feature matching (Best for dynamic resolutions)
   uv run main.py record --engine sift
   ```

3. **Run the Analysis Tool**
   After recording, calculate the optimal Region of Interest (ROI) from your logs.

   ```bash
   uv run main.py analyze
   ```

### Hotkeys & Controls

| Key     | Function            | Description                                                                               |
| :------ | :------------------ | :---------------------------------------------------------------------------------------- |
| **F9**  | **START Recording** | Begins logging data to CSV. Also saves a screenshot to `data/screenshots/` for debugging. |
| **F10** | **STOP & SAVE**     | Stops recording and commits the session data to disk (`data/logs/`).                      |
| **F11** | **TERMINATE**       | Closes the overlay and exits the program safely.                                          |

### Status Indicators

The overlay displays the system status in the top-left corner:

- **<span style="color:lime">○ IDLE (Green)</span>**: Vision is active, FPS is being calculated, but **no data** is being saved. Use this to test detection accuracy.
- **<span style="color:red">● REC (Red)</span>**: The agent is actively recording gameplay data for training.

## ⚙️ Configuration & Tuning

All settings are managed in `calibration/config.py`.

### 1. Adding New Targets

To teach the agent to recognize a new game element (e.g., a "Jump" prompt):

1. Run the program and press **F9** to capture a debug screenshot.
2. Crop the target element from `data/screenshots/`.
3. Save it to the `assets/` folder (e.g., `template_jump.png`).
4. Register it in `config.py`:

```python
TARGETS = {
    # ... other targets ...
    "JUMP": {
        "file": "template_jump.png",
        "color": "magenta",     # Overlay box color
        "threshold": 0.75,      # Used by PIXEL engine (0.0 - 1.0)
        "min_matches": 15       # Used by SIFT engine (Feature count)
    }
}
```

### 2. Tuning Thresholds

- **Text/Translucent UI:** Use lower thresholds (0.60 - 0.70).

- **Solid Icons/Buttons:** Use higher thresholds (0.85 - 0.95) to reduce false positives.

## 🛣️ Current State & Next Steps

The Vision and Calibration pipeline (Phase 1) is currently operational (still need some modifications).

Development will be shifting toward **Phase 2: Environment Wrapper**, where we will ingest the live CSV and coordinate data into a custom OpenAI Gym / Gymnasium environment to establish the `step()`, `reset()`, and `reward()` functions for the RL model.

---

_Project by Michael Tsai_
