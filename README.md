# Expedition 33 RL Agent 🤖⚔️

> **Project Goal:** To develop a Reinforcement Learning (RL) agent capable of playing _Expedition 33_ autonomously using computer vision and advanced control algorithms.

## 📖 Overview

This project aims to bridge the gap between classic control theory and modern Deep Reinforcement Learning. The agent interacts with the game solely through visual inputs (screenshots) and keyboard simulation, mimicking a human player without accessing internal game memory or code injection.

**Current Phase: Vision System & Calibration**
We are currently building the "Eyes" of the agent. This module is responsible for:

1.  **Real-time Object Detection:** Identifying game states (Perfect, Dodge, Parry, UI Icons).
2.  **System Identification:** Calibrating the agent's focus area (Region of Interest) to maximize FPS.
3.  **Data Collection:** logging gameplay events to train the future Reward Model.

## 🚀 Key Features

- **Non-Intrusive Vision:** Uses `mss` for high-speed screen capture (>60 FPS) without hooking into the game process.
- **Dynamic Template Matching:** A robust computer vision pipeline that detects UI elements with specific confidence thresholds.
- **Transparent Debug Overlay:** A custom Win32-based HUD that draws bounding boxes over the game in real-time for immediate visual feedback.
- **Data Logging Pipeline:** Automatically records coordinates and events to CSV for offline analysis and RL environment setup.
- **Modular Architecture:** Clean separation between Configuration, Vision Logic, and UI.

## 📂 Project Structure

```text
Expedition33/
├── assets/                  # Template images (png) for the Vision System
├── calibration/             # [Module] Vision & Data Collection
│   ├── __init__.py
│   ├── app.py               # Main Application Logic (The Orchestrator)
│   ├── config.py            # Central Configuration (Targets & Thresholds)
│   ├── loader.py            # Asset Loading Logic
│   ├── logger.py            # CSV Logging Logic
│   ├── matcher.py           # Core Computer Vision Algorithm
│   └── analysis/
│       ├── __init__.py
│       ├── core.py
│       ├── entry.py
│       └── __pycache__/
├── data/
│   ├── logs/                # Generated CSV training data
│   └── screenshots/         # Debug snapshots for template creation
├── overlay_ui.py            # Win32 Transparent Overlay Class
├── main.py                  # Entry Point
├── requirements.txt         # Project Dependencies
└── README.md
```

## 🛠️ Installation

1.  **Clone the repository**

    ```bash
    git clone <your-repo-url>
    cd Expedition33-RL
    ```

2.  **Set up Virtual Environment**

    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage (Calibration Module)

**⚠️ Administrator Privileges Required:**
This program uses global hotkeys (`win32api`) and draws a topmost overlay. You must run your terminal or IDE as **Administrator**.

1.  **Run the Agent's Vision System:**
    - Record the data

    ```bash
    python main.py record
    ```

    - Analyze the data

    ```bash
    python main.py analyze
    ```

2.  **Launch the Game:**
    Ensure _Expedition 33_ is running in **Windowed** or **Borderless Window** mode.

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

1.  Run the program and press **F9** to capture a debug screenshot.
2.  Crop the target element from `data/screenshots/`.
3.  Save it to the `assets/` folder (e.g., `template_jump.png`).
4.  Register it in `config.py`:

```python
TARGETS = {
    # ... other targets ...
    "JUMP": {
        "file": "template_jump.png",
        "color": "magenta",     # Overlay box color
        "threshold": 0.75       # Confidence threshold (0.0 - 1.0)
    }
}
```

### 2. Tuning Thresholds

- **Text/Translucent UI:** Use lower thresholds (0.60 - 0.70).

- **Solid Icons/Buttons:** Use higher thresholds (0.85 - 0.95) to reduce false positives.

## 🛣️ Project Roadmap

- [x] **Phase 1: Vision System & Calibration**
  - [x] High-speed screen capture.
  - [x] Template matching engine.
  - [x] Data collection pipeline.
- [ ] **Phase 2: Environment Wrapper**
  - [ ] Create OpenAI Gym / Gymnasium custom environment.
  - [ ] Implement `step()`, `reset()`, and `reward()` functions based on vision data.
- [ ] **Phase 3: Agent Training**
  - [ ] Implement PPO (Proximal Policy Optimization) or DQN.
  - [ ] Train agent on collected gameplay data.
- [ ] **Phase 4: Evaluation & Optimization**
  - [ ] Hyperparameter tuning and model deployment.

---

_Project by Michael Tsai_
