"""OpenHup vision service.

Decodes frames, runs detectors, emits observations. Holds no policy: thresholds, timing, and whether
anything deserves a task all live in the backend's skill engine (ADR-003).

The pure modules - `roi`, `sampler`, `fusion` - depend on numpy alone, so the parts that encode
judgement rather than plumbing are testable without ONNX Runtime, a GPU, or a camera.
"""

__version__ = "0.1.0"
