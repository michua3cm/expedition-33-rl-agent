# Importing each engine triggers its @register decorator.
# Add new engines here as they are implemented.
from .pixel import PixelEngine
from .sift import SIFTEngine

__all__ = ["PixelEngine", "SIFTEngine"]
