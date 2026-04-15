from .engine import VisionEngine

_REGISTRY: dict[str, type[VisionEngine]] = {}


def register(name: str):
    """Class decorator — registers a VisionEngine under the given name."""
    def decorator(cls: type[VisionEngine]) -> type[VisionEngine]:
        _REGISTRY[name.upper()] = cls
        return cls
    return decorator


def create(name: str, **kwargs) -> VisionEngine:
    """
    Instantiate a registered engine by name (case-insensitive).
    Extra kwargs are forwarded to the engine's __init__ (e.g. model_path for YOLO).
    """
    key = name.upper()
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys()) or "none loaded"
        raise ValueError(
            f"Unknown vision engine '{name}'. Available: {available}"
        )
    return _REGISTRY[key](**kwargs)


def available() -> list[str]:
    """Return names of all registered engines."""
    return list(_REGISTRY.keys())
