"""
Unit tests for vision/registry.py — register, create, and available.

Each test uses its own isolated registry copy to avoid polluting the
global _REGISTRY used by the real engines.
"""

import pytest
from unittest.mock import MagicMock, patch

import vision.registry as reg
from vision.engine import VisionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_engine_class(name_val: str):
    """Return a minimal concrete VisionEngine subclass."""
    class FakeEngine(VisionEngine):
        @property
        def name(self) -> str:
            return name_val
        def load(self, targets, assets_dir):
            pass
        def detect(self, frame):
            return []
    return FakeEngine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_adds_engine_under_uppercase_key(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            FakeEngine = _make_fake_engine_class("FAKE")

            # Act
            reg.register("fake")(FakeEngine)

            # Assert
            assert "FAKE" in reg._REGISTRY

    def test_register_returns_same_class(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            FakeEngine = _make_fake_engine_class("FAKE")

            # Act
            result = reg.register("FAKE")(FakeEngine)

            # Assert
            assert result is FakeEngine

    def test_register_overwrites_existing_entry(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            EngineA = _make_fake_engine_class("A")
            EngineB = _make_fake_engine_class("B")
            reg.register("DUPE")(EngineA)

            # Act
            reg.register("DUPE")(EngineB)

            # Assert
            assert reg._REGISTRY["DUPE"] is EngineB


class TestCreate:
    def test_create_instantiates_registered_engine(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            FakeEngine = _make_fake_engine_class("FAKE")
            reg._REGISTRY["FAKE"] = FakeEngine

            # Act
            instance = reg.create("FAKE")

            # Assert
            assert isinstance(instance, FakeEngine)

    def test_create_is_case_insensitive(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            FakeEngine = _make_fake_engine_class("FAKE")
            reg._REGISTRY["FAKE"] = FakeEngine

            # Act
            instance = reg.create("fake")

            # Assert
            assert isinstance(instance, FakeEngine)

    def test_create_raises_for_unknown_engine(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            # Act & Assert
            with pytest.raises(ValueError, match="Unknown vision engine"):
                reg.create("NONEXISTENT")

    def test_create_error_message_lists_available_engines(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            FakeEngine = _make_fake_engine_class("EXISTING")
            reg._REGISTRY["EXISTING"] = FakeEngine

            # Act & Assert
            with pytest.raises(ValueError, match="EXISTING"):
                reg.create("MISSING")


class TestAvailable:
    def test_available_returns_all_registered_names(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            reg._REGISTRY["ALPHA"] = _make_fake_engine_class("ALPHA")
            reg._REGISTRY["BETA"]  = _make_fake_engine_class("BETA")

            # Act
            names = reg.available()

            # Assert
            assert set(names) == {"ALPHA", "BETA"}

    def test_available_returns_empty_list_when_no_engines(self):
        # Arrange
        with patch.dict(reg._REGISTRY, {}, clear=True):
            # Act
            names = reg.available()

            # Assert
            assert names == []
