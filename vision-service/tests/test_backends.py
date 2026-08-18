"""Model-registry tests without downloading real weights.

The registry is a supply-chain boundary. These tests cover parsing, checksum verification, lazy
runtime degradation, and preprocessing contracts with tiny temporary files and arrays.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from openhup_vision.backends import (
    ModelRegistry,
    ModelSpec,
    ModelUnavailable,
    Session,
    SessionCache,
    fetch,
    sha256_of,
)

ROOT = Path(__file__).resolve().parents[2]


def test_shipped_registry_has_licences_and_expected_defaults() -> None:
    registry = ModelRegistry.load(ROOT / "models" / "registry.yaml")

    assert len(registry.models) >= 5
    assert all(spec.licence for spec in registry.models.values())
    assert registry.get("yolox-s").input_size == 640
    assert registry.get("clip-vit-b32").normalise == "clip"


def test_registry_rejects_unknown_models() -> None:
    registry = ModelRegistry(models={})

    with pytest.raises(ModelUnavailable, match="unknown model"):
        registry.get("not-a-model")


def test_model_spec_reports_verification_state_and_input_contract() -> None:
    verified = ModelSpec(
        id="verified",
        title="Verified",
        task="test",
        licence="Apache-2.0",
        file="model.onnx",
        sha256="a" * 64,
        input={"shape": [1, 3, 224, 224], "name": "pixels", "normalise": "imagenet"},
    )
    pending = ModelSpec(
        id="pending",
        title="Pending",
        task="test",
        licence="Apache-2.0",
        file="model.onnx",
    )

    assert verified.verified is True
    assert verified.input_name == "pixels"
    assert verified.input_size == 224
    assert verified.normalise == "imagenet"
    assert pending.verified is False


def test_session_preprocess_produces_nchw_float32() -> None:
    spec = ModelSpec(
        id="test",
        title="Test",
        task="test",
        licence="MIT",
        file="model.onnx",
        input={"shape": [1, 3, 32, 32], "normalise": "zero_one"},
    )
    session = Session(spec=spec, provider="CPUExecutionProvider")

    tensor = session.preprocess(np.full((16, 24, 3), 255, dtype=np.uint8))

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert tensor.max() == pytest.approx(1.0)


def test_session_run_requires_a_loaded_runtime_session() -> None:
    spec = ModelSpec("test", "Test", "test", "MIT", "model.onnx")
    session = Session(spec=spec, provider="CPUExecutionProvider")

    with pytest.raises(ModelUnavailable, match="not initialised"):
        session.run(np.zeros((1, 3, 8, 8), dtype=np.float32))


def test_session_cache_degrades_without_onnxruntime(tmp_path: Path) -> None:
    cache = SessionCache(ModelRegistry(models={}, directory=tmp_path))

    assert cache.available_providers() == ()
    assert cache.chosen_provider() == "CPUExecutionProvider"
    assert cache.try_get("missing") is None
    assert cache.loaded() == {}


def test_sha256_of_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"openhup test payload")

    assert sha256_of(path) == hashlib.sha256(b"openhup test payload").hexdigest()


def _write_registry(tmp_path: Path, *, checksum: str = "PENDING") -> tuple[Path, Path]:
    source = tmp_path / "source.onnx"
    source.write_bytes(b"small model")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "defaults": {"directory": str(tmp_path / "models")},
                "models": [
                    {
                        "id": "tiny",
                        "title": "Tiny",
                        "task": "test",
                        "licence": "MIT",
                        "url": source.as_uri(),
                        "file": "tiny.onnx",
                        "sha256": checksum,
                    }
                ],
            }
        )
    )
    return registry, source


def test_fetch_verifies_a_pinned_local_model(tmp_path: Path) -> None:
    expected = hashlib.sha256(b"small model").hexdigest()
    registry_path, _ = _write_registry(tmp_path, checksum=expected)
    registry = ModelRegistry.load(registry_path)

    fetched = fetch(registry)

    assert fetched == ["tiny.onnx"]
    assert (tmp_path / "models" / "tiny.onnx").read_bytes() == b"small model"


def test_fetch_refuses_unpinned_model_without_trust_flag(tmp_path: Path) -> None:
    registry_path, _ = _write_registry(tmp_path)
    registry = ModelRegistry.load(registry_path)

    with pytest.raises(ModelUnavailable, match="trust-first-use"):
        fetch(registry)

    assert not (tmp_path / "models" / "tiny.onnx").exists()


def test_fetch_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    registry_path, _ = _write_registry(tmp_path, checksum="0" * 64)
    registry = ModelRegistry.load(registry_path)

    with pytest.raises(ModelUnavailable, match="sha256 mismatch"):
        fetch(registry)

    assert not (tmp_path / "models" / "tiny.onnx").exists()


def test_fetch_can_pin_first_use_in_a_lock_file(tmp_path: Path) -> None:
    registry_path, _ = _write_registry(tmp_path)
    registry = ModelRegistry.load(registry_path)

    assert fetch(registry, trust_first_use=True) == ["tiny.onnx"]
    lock = yaml.safe_load((tmp_path / "models" / "registry.lock.yaml").read_text())

    assert lock["tiny.onnx"] == hashlib.sha256(b"small model").hexdigest()
