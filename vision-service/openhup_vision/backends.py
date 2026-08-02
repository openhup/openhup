"""Inference backends: ONNX Runtime session management, execution-provider selection, model fetch.

One code path covers four hardware tiers (CPU, Intel iGPU via OpenVINO, CUDA, TensorRT) because the
only thing that changes is the execution provider string (ADR-004). That is the entire reason this
project targets ONNX Runtime rather than a framework-specific runtime: the same install works on a
Raspberry Pi and on a used RTX 3060.

`onnxruntime` is imported lazily. The backend module must be importable - and the pure maths in
`fusion`, `roi`, and `sampler` must be testable - on a machine with no inference stack installed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)

#: Tried in order; the first one ONNX Runtime actually offers wins.
PROVIDER_PREFERENCE = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

#: Normalisation recipes. Getting this wrong produces a model that runs happily and detects nothing,
#: which is a genuinely unpleasant afternoon - so the recipe is declared per model in registry.yaml
#: rather than assumed.
NORMALISERS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),  # raw 0-255, e.g. YOLOX
    "zero_one": ((0.0, 0.0, 0.0), (255.0, 255.0, 255.0)),
    "imagenet": ((0.485 * 255, 0.456 * 255, 0.406 * 255), (0.229 * 255, 0.224 * 255, 0.225 * 255)),
    "clip": ((0.481 * 255, 0.457 * 255, 0.408 * 255), (0.268 * 255, 0.261 * 255, 0.275 * 255)),
}


class ModelUnavailable(RuntimeError):
    """A model is not present, not verified, or its runtime is not installed."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    title: str
    task: str
    licence: str
    file: str
    sha256: str = "PENDING"
    url: str | None = None
    optional: bool = False
    requires_explicit_consent: bool = False
    licence_warning: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    companions: list[dict[str, Any]] = field(default_factory=list)
    labels: str | None = None
    notes: str = ""

    @property
    def input_size(self) -> int:
        shape = self.input.get("shape") or [1, 3, 640, 640]
        return int(shape[-1])

    @property
    def normalise(self) -> str:
        return str(self.input.get("normalise", "zero_one"))

    @property
    def input_name(self) -> str:
        return str(self.input.get("name", "images"))

    @property
    def verified(self) -> bool:
        return self.sha256 not in {"", "PENDING", None}


@dataclass
class ModelRegistry:
    """Parsed models/registry.yaml."""

    models: dict[str, ModelSpec] = field(default_factory=dict)
    directory: Path = Path("./models")

    @classmethod
    def load(cls, path: str | Path | None = None) -> ModelRegistry:
        registry_path = Path(
            path or os.environ.get("OPENHUP_MODEL_REGISTRY") or _default_registry_path()
        )
        raw = yaml.safe_load(registry_path.read_text())
        directory = Path(
            os.environ.get("OPENHUP_MODEL_DIR")
            or (raw.get("defaults") or {}).get("directory")
            or registry_path.parent
        )
        models = {}
        for entry in raw.get("models", []):
            spec = ModelSpec(**{k: v for k, v in entry.items() if k in ModelSpec.__slots__})
            models[spec.id] = spec
        return cls(models=models, directory=directory)

    def get(self, model_id: str) -> ModelSpec:
        spec = self.models.get(model_id)
        if spec is None:
            raise ModelUnavailable(
                f"unknown model {model_id!r}; registry has {sorted(self.models)}"
            )
        return spec

    def path_for(self, spec: ModelSpec) -> Path:
        return self.directory / spec.file

    def present(self, model_id: str) -> bool:
        return self.path_for(self.get(model_id)).is_file()


def _default_registry_path() -> Path:
    for candidate in (
        Path("/etc/openhup/models/registry.yaml"),
        Path(__file__).resolve().parents[3] / "models" / "registry.yaml",
        Path("models/registry.yaml"),
    ):
        if candidate.is_file():
            return candidate
    raise ModelUnavailable("no models/registry.yaml found; set OPENHUP_MODEL_REGISTRY")


# --------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------


@dataclass
class Session:
    """A loaded model plus its preprocessing contract."""

    spec: ModelSpec
    provider: str
    _session: Any = field(repr=False, default=None)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR HxWx3 uint8 → NCHW float32, letterboxed and normalised per the model's recipe."""
        from .roi import resize_letterbox

        resized, _, _, _ = resize_letterbox(image, self.spec.input_size)
        rgb = resized[:, :, ::-1].astype(np.float32)
        mean, std = NORMALISERS.get(self.spec.normalise, NORMALISERS["zero_one"])
        rgb = (rgb - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self._session is None:
            raise ModelUnavailable(f"{self.spec.id} session not initialised")
        return self._session.run(None, {self.spec.input_name: tensor})

    def infer(self, image: np.ndarray) -> list[np.ndarray]:
        return self.run(self.preprocess(image))


class SessionCache:
    """Lazily loads and caches ONNX sessions, one per model id.

    Sessions are expensive to create and cheap to keep, and the working set is small - a typical
    install runs two or three models forever. Loading is deferred so an install that never enables
    `presence_absence` never pays for its weights.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        *,
        providers: tuple[str, ...] | None = None,
        allow_unverified: bool = False,
    ) -> None:
        self.registry = registry or ModelRegistry.load()
        self.preference = providers or PROVIDER_PREFERENCE
        self.allow_unverified = allow_unverified
        self._sessions: dict[str, Session] = {}
        self._available: tuple[str, ...] | None = None

    def available_providers(self) -> tuple[str, ...]:
        if self._available is None:
            try:
                import onnxruntime as ort
            except ImportError:
                self._available = ()
            else:
                self._available = tuple(ort.get_available_providers())
        return self._available

    def chosen_provider(self) -> str:
        available = self.available_providers()
        for candidate in self.preference:
            if candidate in available:
                return candidate
        return "CPUExecutionProvider"

    def get(self, model_id: str) -> Session:
        if model_id in self._sessions:
            return self._sessions[model_id]

        spec = self.registry.get(model_id)
        path = self.registry.path_for(spec)
        if not path.is_file():
            raise ModelUnavailable(
                f"{spec.id}: {path} is missing. Run "
                f"`python -m openhup_vision.backends --fetch --only {spec.id}`"
            )
        if not spec.verified and not self.allow_unverified:
            raise ModelUnavailable(
                f"{spec.id}: no sha256 recorded in the registry. Re-fetch with "
                f"--trust-first-use to pin the hash, or set allow_unverified."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailable(
                "onnxruntime is not installed. Install one execution-provider extra: "
                "`uv sync --extra cpu` (or --extra openvino / --extra cuda)."
            ) from exc

        provider = self.chosen_provider()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Home boxes run other things too; unbounded threads make the whole machine feel broken.
        options.intra_op_num_threads = int(os.environ.get("OPENHUP_ORT_THREADS", "2"))

        log.info("loading %s from %s on %s", spec.id, path, provider)
        session = Session(
            spec=spec,
            provider=provider,
            _session=ort.InferenceSession(str(path), options, providers=[provider]),
        )
        self._sessions[model_id] = session
        return session

    def try_get(self, model_id: str) -> Session | None:
        """Load, or return None. Detectors use this to degrade instead of crashing the loop."""
        try:
            return self.get(model_id)
        except ModelUnavailable as exc:
            log.warning("%s unavailable: %s", model_id, exc)
            return None

    def loaded(self) -> dict[str, str]:
        return {model_id: session.provider for model_id, session in self._sessions.items()}


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


def sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch(
    registry: ModelRegistry,
    *,
    only: str | None = None,
    include_optional: bool = False,
    trust_first_use: bool = False,
) -> list[str]:
    """Download and verify model weights.

    Refuses to accept a model whose registry entry has no checksum unless `trust_first_use` is
    passed, which pins whatever was downloaded into `registry.lock.yaml`. Silently accepting
    unverified weights would be a supply-chain hole in a project whose whole pitch is local trust.
    """
    registry.directory.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    pinned: dict[str, str] = {}

    for spec in registry.models.values():
        if only and spec.id != only:
            continue
        if spec.optional and not include_optional and not only:
            continue
        if spec.requires_explicit_consent and not only:
            log.warning(
                "skipping %s: %s", spec.id, spec.licence_warning.strip() or "requires consent"
            )
            continue
        if not spec.url:
            log.warning("%s has no URL - export it yourself; see registry notes", spec.id)
            continue

        targets = [
            (spec.file, spec.url, spec.sha256),
            *[(c["file"], c["url"], c.get("sha256", "PENDING")) for c in spec.companions],
        ]
        for filename, url, expected in targets:
            destination = registry.directory / filename
            if destination.is_file() and expected not in {"PENDING", ""}:
                if sha256_of(destination) == expected:
                    continue
                log.warning("%s checksum mismatch, re-downloading", filename)

            log.info("downloading %s -> %s", url, destination)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(url) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)

            actual = sha256_of(temporary)
            if expected in {"PENDING", ""}:
                if not trust_first_use:
                    temporary.unlink(missing_ok=True)
                    raise ModelUnavailable(
                        f"{filename}: registry has no sha256 and --trust-first-use was not given. "
                        f"Downloaded hash was {actual}; add it to models/registry.yaml to proceed."
                    )
                pinned[filename] = actual
            elif actual != expected:
                temporary.unlink(missing_ok=True)
                raise ModelUnavailable(
                    f"{filename}: sha256 mismatch. expected {expected}, got {actual}"
                )

            temporary.replace(destination)
            fetched.append(filename)

    if pinned:
        lock = registry.directory / "registry.lock.yaml"
        existing = yaml.safe_load(lock.read_text()) if lock.is_file() else {}
        lock.write_text(yaml.safe_dump({**(existing or {}), **pinned}, sort_keys=True))
        log.info("pinned %d checksum(s) into %s", len(pinned), lock)
    return fetched


def main(argv: list[str] | None = None) -> int:
    """`python -m openhup_vision.backends [--fetch] [--only ID] [--optional] [--info]`."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenHup model management")
    parser.add_argument("--fetch", action="store_true", help="download and verify weights")
    parser.add_argument("--only", help="a single model id")
    parser.add_argument("--optional", action="store_true", help="include opt-in models")
    parser.add_argument("--trust-first-use", action="store_true", help="pin unknown checksums")
    parser.add_argument("--info", action="store_true", help="show providers and model status")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    registry = ModelRegistry.load()
    cache = SessionCache(registry)

    if args.fetch:
        fetched = fetch(
            registry,
            only=args.only,
            include_optional=args.optional,
            trust_first_use=args.trust_first_use,
        )
        print(f"fetched {len(fetched)} file(s)")
        return 0

    print(f"model directory : {registry.directory}")
    print(
        f"ORT providers   : {', '.join(cache.available_providers()) or 'onnxruntime not installed'}"
    )
    print(f"would use       : {cache.chosen_provider()}")
    print()
    for spec in registry.models.values():
        state = "present" if registry.present(spec.id) else "missing"
        flags = []
        if spec.optional:
            flags.append("optional")
        if not spec.verified:
            flags.append("unverified")
        if spec.requires_explicit_consent:
            flags.append("consent-required")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {spec.id:24} {state:8} {spec.licence}{suffix}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "NORMALISERS",
    "PROVIDER_PREFERENCE",
    "ModelRegistry",
    "ModelSpec",
    "ModelUnavailable",
    "Session",
    "SessionCache",
    "fetch",
    "sha256_of",
]
