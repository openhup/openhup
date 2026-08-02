"""Region-of-interest geometry. Pure numpy, no I/O.

Anchors are polygons in normalised 0..1 coordinates so they survive a camera being swapped for one
with a different resolution (ADR-010). Everything downstream works on a masked crop, which matters
for three reasons: less compute, far fewer false positives from activity elsewhere in the frame,
and much less imagery on disk when a snapshot is attached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Frame = np.ndarray  # HxWx3, uint8, BGR


@dataclass(frozen=True, slots=True)
class Region:
    """A polygon in normalised coordinates, with its rasterised mask cached per frame shape."""

    id: str
    label: str
    points: tuple[tuple[float, float], ...]

    @property
    def is_full_frame(self) -> bool:
        return not self.points

    def pixel_points(self, height: int, width: int) -> np.ndarray:
        """Polygon in pixel space.

        Scaled by width/height rather than width-1/height-1, and paired with pixel-centre sampling
        in `mask`, so a polygon covering x in [0, 0.5] covers exactly the left half of the columns.
        Getting this wrong shows up as a missing bottom row in every mask - which silently biases
        every score by a fraction of a percent and is miserable to track down later.
        """
        return np.array([(x * width, y * height) for x, y in self.points], dtype=np.float64)

    def bbox(self, height: int, width: int) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) pixel bounds, clamped to the frame. Full frame when no polygon."""
        if self.is_full_frame:
            return 0, 0, width, height
        pts = self.pixel_points(height, width)
        x0 = max(int(np.floor(pts[:, 0].min())), 0)
        y0 = max(int(np.floor(pts[:, 1].min())), 0)
        x1 = min(int(np.ceil(pts[:, 0].max())), width)
        y1 = min(int(np.ceil(pts[:, 1].max())), height)
        # Degenerate polygons must not produce a zero-size crop that blows up downstream.
        return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)

    def mask(self, height: int, width: int) -> np.ndarray:
        """Boolean HxW mask. Even-odd ray casting on pixel centres - no OpenCV needed."""
        if self.is_full_frame:
            return np.ones((height, width), dtype=bool)

        pts = self.pixel_points(height, width)
        rows, cols = np.mgrid[0:height, 0:width]
        # Sample pixel centres, not corners: otherwise the row and column on the polygon's far
        # boundary fall outside it and every mask is short by one row.
        ys = rows + 0.5
        xs = cols + 0.5
        inside = np.zeros((height, width), dtype=bool)

        count = len(pts)
        for index in range(count):
            x_i, y_i = pts[index]
            x_j, y_j = pts[(index - 1) % count]
            # Does a horizontal ray from each pixel cross this edge?
            straddles = ((y_i > ys) != (y_j > ys)) & (y_j != y_i)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at_y = np.where(
                    straddles,
                    (x_j - x_i) * (ys - y_i) / np.where(y_j == y_i, 1, y_j - y_i) + x_i,
                    0,
                )
            inside ^= straddles & (xs < x_at_y)
        return inside

    def area_fraction(self, height: int, width: int) -> float:
        return float(self.mask(height, width).mean())


def crop(frame: Frame, region: Region, *, apply_mask: bool = True) -> Frame:
    """Crop to a region's bounding box, blacking out pixels outside the polygon.

    Masking matters as much as cropping: a countertop ROI that includes a slice of the doorway will
    otherwise light up every time somebody walks past, and the user will conclude clutter detection
    does not work.
    """
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = region.bbox(height, width)
    patch = frame[y0:y1, x0:x1]
    if not apply_mask or region.is_full_frame:
        return np.ascontiguousarray(patch)

    mask = region.mask(height, width)[y0:y1, x0:x1]
    out = patch.copy()
    out[~mask] = 0
    return out


def resize_letterbox(frame: Frame, size: int, *, fill: int = 114) -> tuple[Frame, float, int, int]:
    """Scale to a square `size` keeping aspect, padding the remainder.

    Returns (image, scale, pad_x, pad_y) so detections can be mapped back to region coordinates.
    Nearest-neighbour: this runs per detector invocation and the models are not sensitive enough to
    justify pulling in an interpolation dependency here.
    """
    height, width = frame.shape[:2]
    scale = min(size / max(height, 1), size / max(width, 1))
    new_h, new_w = max(round(height * scale), 1), max(round(width * scale), 1)

    row_idx = (np.arange(new_h) / scale).astype(np.int32).clip(0, height - 1)
    col_idx = (np.arange(new_w) / scale).astype(np.int32).clip(0, width - 1)
    resized = frame[row_idx][:, col_idx]

    canvas = np.full((size, size, frame.shape[2]), fill, dtype=frame.dtype)
    pad_y, pad_x = (size - new_h) // 2, (size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def to_region_coords(
    box: tuple[float, float, float, float],
    *,
    scale: float,
    pad_x: int,
    pad_y: int,
    region_w: int,
    region_h: int,
) -> tuple[float, float, float, float]:
    """Map a letterboxed model-space box back to normalised region coordinates."""
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_x) / scale / max(region_w, 1)
    x2 = (x2 - pad_x) / scale / max(region_w, 1)
    y1 = (y1 - pad_y) / scale / max(region_h, 1)
    y2 = (y2 - pad_y) / scale / max(region_h, 1)
    clamp = lambda v: float(min(max(v, 0.0), 1.0))  # noqa: E731
    return clamp(x1), clamp(y1), clamp(x2), clamp(y2)


def blur_boxes(
    frame: Frame,
    boxes: list[tuple[float, float, float, float]],
    *,
    blocks: int = 8,
) -> Frame:
    """Pixelate regions before a frame is written to disk.

    Downsample-and-repeat rather than a Gaussian: it is cheap, and it is irreversible, which is the
    property that actually matters for a redaction. Unredacted pixels never reach the filesystem,
    because this runs in `emitter.write_snapshot` before encoding rather than afterwards.

    `blocks` is the target resolution of the redacted patch - 8 means "reduce this face to an 8x8
    grid of flat colour", which stays destructive for a 30px box and a 300px one alike.
    Expressing it as a divisor would silently no-op on small boxes, i.e. on distant faces.
    """
    if not boxes:
        return frame
    height, width = frame.shape[:2]
    out = frame.copy()
    for x1, y1, x2, y2 in boxes:
        px1, py1 = int(x1 * width), int(y1 * height)
        px2, py2 = int(x2 * width), int(y2 * height)
        px1, py1 = max(px1, 0), max(py1, 0)
        px2, py2 = min(px2, width), min(py2, height)
        if px2 - px1 < 2 or py2 - py1 < 2:
            continue
        patch = out[py1:py2, px1:px2]
        block_h = max(patch.shape[0] // blocks, 1)
        block_w = max(patch.shape[1] // blocks, 1)
        small = patch[::block_h, ::block_w]
        rows = np.repeat(small, block_h, axis=0)[: patch.shape[0]]
        grown = np.repeat(rows, block_w, axis=1)[:, : patch.shape[1]]
        if grown.shape[:2] != patch.shape[:2]:
            grown = np.resize(grown, patch.shape)
        out[py1:py2, px1:px2] = grown
    return out


def region_from_anchor(anchor_id: str, label: str, polygon: list) -> Region:
    """Build a Region from an Anchor's polygon, accepting both point objects and [x, y] pairs."""
    points: list[tuple[float, float]] = []
    for point in polygon or []:
        if hasattr(point, "x"):
            points.append((float(point.x), float(point.y)))
        elif isinstance(point, dict):
            points.append((float(point["x"]), float(point["y"])))
        else:
            points.append((float(point[0]), float(point[1])))
    return Region(id=anchor_id, label=label, points=tuple(points))


__all__ = [
    "Frame",
    "Region",
    "blur_boxes",
    "crop",
    "region_from_anchor",
    "resize_letterbox",
    "to_region_coords",
]
