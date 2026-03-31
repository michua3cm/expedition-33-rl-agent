# Importing each engine triggers its @register decorator.
# Add new engines here as they are implemented.
from .composite import CompositeEngine  # must be last — calls registry.create() at runtime
from .orb import ORBEngine
from .pixel import PixelEngine
from .sift import SIFTEngine
from .yolo import YOLOEngine

__all__ = ["PixelEngine", "SIFTEngine", "ORBEngine", "YOLOEngine", "CompositeEngine"]
