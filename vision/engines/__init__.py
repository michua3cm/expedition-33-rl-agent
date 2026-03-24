# Importing each engine triggers its @register decorator.
# Add new engines here as they are implemented.
from .pixel import PixelEngine
from .sift import SIFTEngine
from .orb import ORBEngine
from .yolo import YOLOEngine

__all__ = ["PixelEngine", "SIFTEngine", "ORBEngine", "YOLOEngine"]
