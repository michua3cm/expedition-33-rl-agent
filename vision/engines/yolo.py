import os

import numpy as np

from ..engine import Detection, VisionEngine
from ..registry import register

_DEFAULT_MODEL_PATH = os.path.join("data", "yolo_dataset", "train", "weights", "best.pt")
_DEFAULT_CONF       = 0.5


@register("YOLO")
class YOLOEngine(VisionEngine):
    """
    YOLOv8 object detection engine.

    Resolution-robust and scale-invariant by design.
    Requires a trained model at data/yolo_dataset/train/weights/best.pt
    (produced by 'uv run main.py train').

    Usage with non-default model:
        vision.registry.create("YOLO", model_path="path/to/custom.pt")
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        conf_threshold: float = _DEFAULT_CONF,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model = None

    @property
    def name(self) -> str:
        return "YOLO"

    @property
    def needs_color(self) -> bool:
        """YOLO was trained on colour screenshots; always request a BGR frame."""
        return True

    def load(self, targets: dict, assets_dir: str) -> None:
        if not os.path.exists(self._model_path):
            print(
                f"[YOLOEngine] Warning: model not found at '{self._model_path}'. "
                f"Run 'uv run main.py train' to produce it. "
                f"Engine will return no detections until a model is loaded."
            )
            return

        from ultralytics import YOLO  # deferred — only needed at runtime

        self._model = YOLO(self._model_path)
        print(f"[YOLOEngine] Loaded model from '{self._model_path}'")
        print(f"[YOLOEngine] Classes: {list(self._model.names.values())}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is None:
            return []

        results = self._model(frame, verbose=False, conf=self._conf_threshold)[0]

        detections: list[Detection] = []
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            label = self._model.names[int(box.cls[0])]
            detections.append(Detection(
                label=label,
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                confidence=conf,
            ))

        return detections
