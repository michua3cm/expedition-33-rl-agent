# Importing each engine triggers its @register decorator.
# Add new engines here as they are implemented.
from .pixel import PixelEngine
from .sift import SIFTEngine
from .orb import ORBEngine
from .yolo import YOLOEngine
from .composite import CompositeEngine  # must be last — calls registry.create() at runtime

__all__ = ["PixelEngine", "SIFTEngine", "ORBEngine", "YOLOEngine", "CompositeEngine"]
