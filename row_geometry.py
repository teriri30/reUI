"""Metric, row-aligned geometry reconstruction for crop-band masks."""

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

MASK_REGULARIZATION_VERSION = 3


def odd_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 == 1 else size + 1


def meters_per_pixel(mask: np.ndarray, geo=None, state=None) -> float:
    if geo is not None and state is not None and geo.is_ready():
        ox = int(getattr(state, "mask_offset_x", 0))
        oy = int(getattr(state, "mask_offset_y", 0))
        h, w = mask.shape[:2]
        value = float(geo.meters_per_pixel(ox + w / 2, oy + h / 2))
        if 1e-5 <= value <= 5.0:
            return value
        raise ValueError(f"invalid metric scale: {value!r} m/px")
    return 0.01


def field_polygon_local(
    state,
    shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Return the selected field polygon in mask-local coordinates."""
    boundary = getattr(state, "field_boundary", None) if state is not None else None
    if not boundary or len(boundary) < 3:
        return None
    ox = float(getattr(state, "mask_offset_x", 0))
    oy = float(getattr(state, "mask_offset_y", 0))
    points = np.asarray(boundary, dtype=np.float64) - np.asarray((ox, oy))
    if not np.all(np.isfinite(points)):
        return None
    h, w = shape[:2]
    if (
        np.max(points[:, 0]) < 0
        or np.max(points[:, 1]) < 0
        or np.min(points[:, 0]) >= w
        or np.min(points[:, 1]) >= h
    ):
        return None
    return points


def clip_mask_to_polygon_stripes(
    mask: np.ndarray,
    polygon: Optional[np.ndarray],
    stripe_rows: int = 1024,
) -> np.ndarray:
    """Clip a large mask without allocating another full-resolution mask."""
    if polygon is None:
        return mask
    result = mask
    height, width = result.shape[:2]
    polygon = np.asarray(polygon, dtype=np.float64)
    for y0 in range(0, height, max(64, int(stripe_rows))):
        y1 = min(height, y0 + max(64, int(stripe_rows)))
        stripe_mask = np.zeros((y1 - y0, width), dtype=np.uint8)
        shifted = polygon.copy()
        shifted[:, 1] -= y0
        cv2.fillPoly(
            stripe_mask,
            [np.round(shifted).astype(np.int32)],
            255,
        )
        cv2.bitwise_and(
            result[y0:y1],
            stripe_mask,
            dst=result[y0:y1],
        )
    return result


def clip_band_polygon(
    polygon: np.ndarray,
    field_polygon: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Raster-clip one narrow band polygon to the selected field boundary."""
    polygon = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    if field_polygon is None or len(polygon) < 3:
        return polygon
    x, y, width, height = cv2.boundingRect(polygon)
    if width <= 0 or height <= 0:
        return None
    band_roi = np.zeros((height, width), dtype=np.uint8)
    field_roi = np.zeros_like(band_roi)
    cv2.fillPoly(band_roi, [polygon - (x, y)], 255)
    shifted_field = np.asarray(field_polygon, dtype=np.float64) - (x, y)
    cv2.fillPoly(
        field_roi,
        [np.round(shifted_field).astype(np.int32)],
        255,
    )
    clipped = cv2.bitwise_and(band_roi, field_roi)
    contours, _ = cv2.findContours(
        clipped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    return contour + (x, y)


def smooth_1d(values: np.ndarray, window: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    size = odd_size(window)
    if size > values.size:
        size = values.size if values.size % 2 == 1 else values.size - 1
    if size < 3:
        return values.copy()
    kernel = cv2.getGaussianKernel(size, max(size / 5.0, 0.8)).reshape(-1)
    padded = np.pad(values, (size // 2, size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    good = np.isfinite(values)
    if not np.any(good):
        return values.copy()
    indices = np.arange(values.size)
    return np.interp(indices, indices[good], values[good])


def _component_main_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size < 3:
        return 0.0
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    coords -= coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    return math.atan2(float(vec[1]), float(vec[0])) % math.pi


def _component_extent_along_angle(mask: np.ndarray, angle: float) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0
    direction = np.asarray((math.cos(float(angle)), math.sin(float(angle))), dtype=np.float64)
    projection = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
    return float(projection.max() - projection.min() + 1.0)


def _component_projection_interval(mask: np.ndarray, angle: float) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0, 0.0
    direction = np.asarray((math.cos(float(angle)), math.sin(float(angle))), dtype=np.float64)
    projection = xs.astype(np.float64) * direction[0] + ys.astype(np.float64) * direction[1]
    return float(projection.min()), float(projection.max())


def residual_mask_layers(
    raw_mask: np.ndarray,
    rebuilt_mask: np.ndarray,
    main_angle: float,
    angle_thresh_deg: float = 60.0,
    min_area_ratio: float = 0.001,
    short_parallel_ratio: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split raw-vs-rebuilt residuals into headland and row-body residual layers.

    Perpendicular residual components are headland. Row-parallel residuals are
    preserved as body support only when they are long enough to be plausible row
    bands and remain within the main body's row-direction extent. Detached
    parallel fragments stay in the light headland/edge layer instead of becoming
    work-line body.
    """
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    rebuilt = (np.asarray(rebuilt_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    residual = cv2.bitwise_and(raw, cv2.bitwise_not(rebuilt))
    headland = np.zeros_like(raw)
    body_residual = np.zeros_like(raw)
    if cv2.countNonZero(residual) == 0:
        return headland, body_residual

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    threshold = math.radians(float(angle_thresh_deg))
    total = max(1, int(cv2.countNonZero(raw)))
    min_area = max(3, int(round(total * float(min_area_ratio))))
    min_parallel_extent = max(8.0, float(short_parallel_ratio) * max(raw.shape[:2]))
    has_rebuilt_body = cv2.countNonZero(rebuilt) > 0
    rebuilt_start, rebuilt_end = _component_projection_interval(rebuilt, main_angle)
    body_extension = max(3.0, float(short_parallel_ratio) * max(raw.shape[:2]) * 0.15)

    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = (labels == label).astype(np.uint8) * 255
        if area < min_area:
            headland[labels == label] = residual[labels == label]
            continue

        comp_angle = _component_main_angle(component)
        diff = abs(comp_angle - float(main_angle)) % math.pi
        diff = min(diff, math.pi - diff)
        parallel_extent = _component_extent_along_angle(component, main_angle)
        comp_start, comp_end = _component_projection_interval(component, main_angle)
        overlaps_body_extent = (
            not has_rebuilt_body
            or (
                comp_end >= rebuilt_start - body_extension
                and comp_start <= rebuilt_end + body_extension
            )
        )
        if diff > threshold or parallel_extent < min_parallel_extent or not overlaps_body_extent:
            headland[labels == label] = residual[labels == label]
        else:
            body_residual[labels == label] = residual[labels == label]
    return headland, body_residual

def repair_row_gaps(
    mask: np.ndarray,
    main_angle: float,
    meters_per_px: float,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Close short gaps along crop-row direction without broad cross-row dilation."""
    cfg = config or {}
    src = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(src) == 0:
        return src
    gap_m = float(cfg.get("row_gap_close_m", 2.2))
    width_m = float(cfg.get("row_gap_close_width_m", 0.04))
    k_len = odd_size(gap_m / max(float(meters_per_px), 1e-9), minimum=3)
    k_w = odd_size(width_m / max(float(meters_per_px), 1e-9), minimum=1)
    # Bound the kernel so a bad GeoTIFF scale cannot bridge across the field.
    k_len = min(k_len, max(3, int(max(src.shape[:2]) * 0.20) | 1))
    k_w = min(k_w, max(1, int(max(src.shape[:2]) * 0.01) | 1))
    if k_len <= 3:
        return src
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, max(1, k_w)))
    center = (k_len // 2, max(1, k_w) // 2)
    matrix = cv2.getRotationMatrix2D(center, math.degrees(float(main_angle)), 1.0)
    rotated_kernel = cv2.warpAffine(
        kernel,
        matrix,
        (k_len, max(1, k_w)),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    rotated_kernel = (rotated_kernel > 0).astype(np.uint8)
    repaired = cv2.morphologyEx(src, cv2.MORPH_CLOSE, rotated_kernel, iterations=1)
    return repaired


def constrain_generated_body_to_raw_support(
    generated_mask: np.ndarray,
    raw_mask: np.ndarray,
    main_angle: float,
    meters_per_px: float,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Remove one-sided geometric protrusions not supported by raw mask evidence.

    New pixels are allowed only where the raw mask's own row-direction closing
    would bridge a gap. This preserves genuine short gap repairs while clipping
    geometry/closing artefacts that extend past a raw row endpoint.
    """
    cfg = dict(config or {})
    generated = (np.asarray(generated_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    raw_support = repair_row_gaps(raw, main_angle, meters_per_px, cfg)
    constrained = cv2.bitwise_and(generated, raw_support)
    return constrained


def residual_headland_mask(
    raw_mask: np.ndarray,
    rebuilt_mask: np.ndarray,
    main_angle: float,
    angle_thresh_deg: float = 60.0,
    min_area_ratio: float = 0.001,
) -> np.ndarray:
    """Return only headland-like residual pixels, not every missed row-band pixel."""
    headland, _body_residual = residual_mask_layers(
        raw_mask,
        rebuilt_mask,
        main_angle,
        angle_thresh_deg=angle_thresh_deg,
        min_area_ratio=min_area_ratio,
    )
    return headland


def repair_center_samples(
    centers: np.ndarray,
    widths: np.ndarray,
    observed: np.ndarray,
    target_width_px: float,
    sample_step: int,
    work_mpp: float,
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Bridge weak observations along the local row direction."""
    centers = np.asarray(centers, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)

    provisional_widths = fill_nan(widths)
    smooth_m = float(config.get("center_outlier_smooth_m", 1.2))
    smooth_n = max(
        3,
        int(round(smooth_m / max(work_mpp * sample_step, 1e-6))),
    )
    if smooth_n % 2 == 0:
        smooth_n += 1
    local_width_reference = smooth_1d(provisional_widths, smooth_n)

    min_width_ratio = float(config.get("gap_min_width_ratio", 0.72))
    local_width_ratio = float(config.get("gap_local_width_ratio", 0.86))
    trusted = (
        observed
        & np.isfinite(centers)
        & np.isfinite(widths)
        & (widths >= max(1.0, target_width_px * min_width_ratio))
        & (widths >= local_width_reference * local_width_ratio)
    )

    provisional = fill_nan(centers)
    trend = smooth_1d(provisional, smooth_n)
    outlier_px = float(config.get("center_outlier_m", 0.10)) / max(work_mpp, 1e-6)
    trusted &= np.abs(centers - trend) <= max(2.0, outlier_px)

    if np.count_nonzero(trusted) < 4:
        trusted = observed & np.isfinite(centers)

    repaired = fill_nan(np.where(trusted, centers, np.nan))
    context_m = float(config.get("gap_center_context_m", 0.9))
    context_n = max(
        3,
        int(round(context_m / max(work_mpp * sample_step, 1e-6))),
    )
    max_shift_px = float(config.get("gap_max_lateral_shift_m", 0.05)) / max(
        work_mpp,
        1e-6,
    )

    weak = ~trusted
    run_start = None
    for index in range(len(weak) + 1):
        is_weak = index < len(weak) and weak[index]
        if is_weak and run_start is None:
            run_start = index
            continue
        if is_weak or run_start is None:
            continue

        run_end = index
        left_idx = np.flatnonzero(trusted[:run_start])
        right_idx = np.flatnonzero(trusted[run_end:]) + run_end
        left_values = centers[left_idx[-context_n:]] if left_idx.size else np.array([])
        right_values = centers[right_idx[:context_n]] if right_idx.size else np.array([])

        if left_values.size and right_values.size:
            left_center = float(np.median(left_values))
            right_center = float(np.median(right_values))
            middle = 0.5 * (left_center + right_center)
            delta = float(np.clip(
                right_center - left_center,
                -max_shift_px,
                max_shift_px,
            ))
            repaired[run_start:run_end] = np.linspace(
                middle - 0.5 * delta,
                middle + 0.5 * delta,
                run_end - run_start + 2,
            )[1:-1]
        elif left_values.size:
            repaired[run_start:run_end] = float(np.median(left_values))
        elif right_values.size:
            repaired[run_start:run_end] = float(np.median(right_values))

        run_start = None

    return repaired, trusted


def rotate_bound(
    image: np.ndarray,
    angle_deg: float,
    interpolation: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate without clipping and return the affine matrix."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    out_w = int(math.ceil(h * sin_a + w * cos_a))
    out_h = int(math.ceil(h * cos_a + w * sin_a))
    matrix[0, 2] += out_w / 2.0 - center[0]
    matrix[1, 2] += out_h / 2.0 - center[1]
    rotated = cv2.warpAffine(
        image,
        matrix,
        (out_w, out_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, matrix


def estimate_row_angle(mask: np.ndarray) -> float:
    """Estimate crop-row direction with PCA followed by a Radon-like search."""
    ys, xs = np.where(mask > 0)
    if xs.size < 100:
        return 0.0

    step = max(1, xs.size // 400000)
    coords = np.column_stack([xs[::step], ys[::step]]).astype(np.float64)
    coords -= coords.mean(axis=0)
    covariance = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    coarse = math.atan2(vector[1], vector[0]) % math.pi

    best_angle = coarse
    best_score = -1.0
    for angle in np.linspace(coarse - math.radians(12), coarse + math.radians(12), 49):
        normal_x = -math.sin(angle)
        normal_y = math.cos(angle)
        projection = coords[:, 0] * normal_x + coords[:, 1] * normal_y
        bins = max(128, min(1600, int(np.ptp(projection)) + 1))
        histogram, _ = np.histogram(projection, bins=bins)
        trend = smooth_1d(histogram, max(5, bins // 80))
        high_frequency = histogram - trend
        score = float(np.std(high_frequency)) / max(float(np.mean(histogram)), 1.0)
        if score > best_score:
            best_score = score
            best_angle = angle % math.pi
    return best_angle


def _runs(binary: np.ndarray) -> List[Tuple[int, int]]:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return []
    changes = np.flatnonzero(np.diff(binary.astype(np.int8))) + 1
    bounds = np.concatenate(([0], changes, [binary.size]))
    return [
        (int(bounds[index]), int(bounds[index + 1]))
        for index in range(len(bounds) - 1)
        if binary[bounds[index]]
    ]


def _merge_close_runs(
    runs: List[Tuple[int, int]],
    max_gap: int,
) -> List[Tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(item) for item in merged]


def _longest_true_run(values: np.ndarray) -> Optional[Tuple[int, int]]:
    runs = _runs(values)
    return max(runs, key=lambda item: item[1] - item[0]) if runs else None


def close_boolean_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    values = np.asarray(values, dtype=bool)
    if values.size == 0 or max_gap <= 0:
        return values.copy()
    runs = _merge_close_runs(_runs(values), int(max_gap))
    result = np.zeros(values.size, dtype=bool)
    for start, end in runs:
        result[start:end] = True
    return result


def estimate_body_interval(
    foreground: np.ndarray,
    runs: List[Tuple[int, int]],
    work_mpp: float,
    config: Dict,
) -> Tuple[int, int, Dict]:
    """Find the longitudinal interval containing the repeated row lattice."""
    height, width = foreground.shape
    band_selector = np.zeros(height, dtype=bool)
    for start, end in runs:
        band_selector[start:end] = True
    gap_selector = np.zeros(height, dtype=bool)
    lattice_start = min(start for start, _ in runs)
    lattice_end = max(end for _, end in runs)
    gap_selector[lattice_start:lattice_end] = ~band_selector[lattice_start:lattice_end]
    if not np.any(band_selector) or not np.any(gap_selector):
        return 0, width, {"reason": "invalid_band_gap_selector"}

    band_signal = np.mean(foreground[band_selector], axis=0).astype(np.float64)
    gap_signal = np.mean(foreground[gap_selector], axis=0).astype(np.float64)
    continuity = np.zeros(width, dtype=np.float64)
    for start, end in runs:
        continuity += np.mean(foreground[start:end], axis=0) >= float(
            config.get("body_band_presence_threshold", 0.18)
        )
    continuity /= max(len(runs), 1)

    smooth_px = max(
        3,
        float(config.get("body_quality_smooth_m", 0.45)) / work_mpp,
    )
    band_signal = smooth_1d(band_signal, smooth_px)
    gap_signal = smooth_1d(gap_signal, smooth_px)
    continuity = smooth_1d(continuity, smooth_px)
    contrast = np.clip(band_signal - gap_signal, 0.0, 1.0)

    quality = continuity * contrast
    positive = quality[quality > 0]
    if positive.size == 0:
        return 0, width, {"reason": "no_positive_body_quality"}
    quality_threshold = max(
        float(config.get("body_quality_min", 0.04)),
        min(
            float(config.get("body_quality_max", 0.18)),
            float(np.quantile(positive, 0.15)) * 0.6,
        ),
    )
    continuity_threshold = float(config.get("body_continuity_min", 0.42))
    contrast_threshold = float(config.get("body_contrast_min", 0.08))
    body_candidate = (
        (quality >= quality_threshold)
        & (continuity >= continuity_threshold)
        & (contrast >= contrast_threshold)
    )

    close_px = odd_size(
        float(config.get("body_gap_close_m", 1.0)) / work_mpp,
        minimum=3,
    )
    candidate_u8 = body_candidate.astype(np.uint8).reshape(1, -1) * 255
    candidate_u8 = cv2.morphologyEx(
        candidate_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_px, 1)),
    )
    body_run = _longest_true_run(candidate_u8.reshape(-1) > 0)
    if body_run is None:
        return 0, width, {"reason": "no_body_run"}

    start, end = body_run

    # The high-quality body run is good for rejecting true headlands, but weak
    # end-row pixels can still be harvestable. Extend the interval through
    # row-supported tails that keep the repeated row lattice, instead of ending
    # the work lines as soon as the strict quality score drops.
    support_min_continuity = float(config.get(
        "body_support_extension_min_continuity",
        max(0.25, continuity_threshold * 0.60),
    ))
    support_min_contrast = float(config.get(
        "body_support_extension_min_contrast",
        max(0.03, contrast_threshold * 0.50),
    ))
    support_candidate = (
        (continuity >= support_min_continuity)
        & (contrast >= support_min_contrast)
    )
    support_gap_px = odd_size(
        float(config.get("body_support_extension_max_gap_m", 0.8))
        / max(work_mpp, 1e-6),
        minimum=3,
    )
    support_u8 = support_candidate.astype(np.uint8).reshape(1, -1) * 255
    support_u8 = cv2.morphologyEx(
        support_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (support_gap_px, 1)),
    )
    support_runs = _runs(support_u8.reshape(-1) > 0)
    overlapping = [
        item for item in support_runs
        if item[0] <= end and item[1] >= start
    ]
    if overlapping:
        support_start, support_end = max(
            overlapping,
            key=lambda item: (min(item[1], end) - max(item[0], start), item[1] - item[0]),
        )
        start = min(start, support_start)
        end = max(end, support_end)

    extension_px = int(round(
        float(config.get("body_end_extension_m", 0.20)) / work_mpp
    ))
    start = max(0, start - extension_px)
    end = min(width, end + extension_px)
    min_length_px = int(round(
        float(config.get("body_min_length_m", 8.0)) / work_mpp
    ))
    if end - start < min_length_px:
        return 0, width, {"reason": "body_run_too_short"}

    return start, end, {
        "quality_threshold": quality_threshold,
        "support_extension_min_continuity": support_min_continuity,
        "support_extension_min_contrast": support_min_contrast,
        "mean_continuity": float(np.mean(continuity[start:end])),
        "mean_contrast": float(np.mean(contrast[start:end])),
        "body_start_work_px": int(start),
        "body_end_work_px": int(end),
        "support_extended_start_px": int(start),
        "support_extended_end_px": int(end),
        "body_length_m": float((end - start) * work_mpp),
    }


def classify_band_widths(
    bands: List[Dict],
    fallback_threshold_m: float = 0.55,
    alternation_penalty: float = 0.35,
) -> Tuple[List[Dict], List[Dict], float]:
    """Classify ordered bands with width evidence and a soft sequence prior."""
    if not bands:
        return [], [], fallback_threshold_m
    if len(bands) < 6:
        wide = [item for item in bands if item["width_m"] >= fallback_threshold_m]
        narrow = [item for item in bands if item["width_m"] < fallback_threshold_m]
        for item in wide:
            item["band_type"] = "wide"
        for item in narrow:
            item["band_type"] = "narrow"
        return wide, narrow, fallback_threshold_m

    widths = np.asarray([
        item.get("classification_width_m", item["width_m"])
        for item in bands
    ], dtype=np.float64)
    center_a, center_b = np.quantile(widths, [0.25, 0.75])
    for _ in range(20):
        distance_a = np.abs(widths - center_a)
        distance_b = np.abs(widths - center_b)
        group_a = widths[distance_a <= distance_b]
        group_b = widths[distance_a > distance_b]
        if group_a.size == 0 or group_b.size == 0:
            break
        new_a = float(np.mean(group_a))
        new_b = float(np.mean(group_b))
        if abs(new_a - center_a) + abs(new_b - center_b) < 1e-6:
            break
        center_a, center_b = new_a, new_b

    low, high = sorted((float(center_a), float(center_b)))
    threshold = (low + high) / 2.0
    separation = (high - low) / max(low, 1e-6)
    low_count = int(np.count_nonzero(widths < threshold))
    high_count = int(np.count_nonzero(widths >= threshold))
    if separation < 0.16 or min(low_count, high_count) < 3:
        threshold = fallback_threshold_m
        states = (widths >= threshold).astype(np.int8)
    else:
        scale = max((high - low) * 0.55, 0.06)
        emission = np.column_stack([
            ((widths - low) / scale) ** 2,
            ((widths - high) / scale) ** 2,
        ])
        count = len(bands)
        cost = np.full((count, 2), np.inf, dtype=np.float64)
        previous = np.zeros((count, 2), dtype=np.int8)
        cost[0] = emission[0]
        for index in range(1, count):
            for state_value in (0, 1):
                options = cost[index - 1] + np.asarray([
                    alternation_penalty if prior == state_value else 0.0
                    for prior in (0, 1)
                ])
                prior_state = int(np.argmin(options))
                previous[index, state_value] = prior_state
                cost[index, state_value] = options[prior_state] + emission[index, state_value]
        states = np.zeros(count, dtype=np.int8)
        states[-1] = int(np.argmin(cost[-1]))
        for index in range(count - 1, 0, -1):
            states[index - 1] = previous[index, states[index]]

    for item, state_value in zip(bands, states):
        item["band_type"] = "wide" if state_value else "narrow"
        item["class_width_m"] = high if state_value else low
    wide = [item for item in bands if item["band_type"] == "wide"]
    narrow = [item for item in bands if item["band_type"] == "narrow"]
    return wide, narrow, threshold


def regularize_crop_mask(
    mask: np.ndarray,
    geo=None,
    state=None,
    config: Optional[Dict] = None,
    progress_callback=None,
) -> Dict:
    """Rebuild each crop band as a smooth polygon in a row-aligned frame."""
    cfg = config or {}
    cancel_callback = cfg.get("_cancel_callback")

    def abort_if_cancelled():
        if callable(cancel_callback) and cancel_callback():
            raise RuntimeError("task cancelled")

    abort_if_cancelled()
    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if np.count_nonzero(raw) < 100:
        return {
            "processed_mask": raw,
            "main_angle": 0.0,
            "wide_bands": [],
            "narrow_bands": [],
            "all_bands": [],
            "diagnostics": {"reason": "empty_mask"},
        }

    def report(value: float, stage: str):
        if progress_callback:
            try:
                progress_callback(float(value), stage)
            except Exception:
                pass

    h, w = raw.shape[:2]
    field_polygon = field_polygon_local(state, raw.shape)
    if field_polygon is not None:
        clip_mask_to_polygon_stripes(raw, field_polygon)
    mpp = meters_per_pixel(raw, geo, state)
    max_work_dim = int(cfg.get("max_work_dim", 5000))
    scale = min(1.0, float(max_work_dim) / max(h, w))
    work = cv2.resize(
        raw,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    work = (work >= int(cfg.get("resize_threshold", 96))).astype(np.uint8) * 255
    field_work_mask = None
    if field_polygon is not None:
        field_work_mask = np.zeros_like(work)
        cv2.fillPoly(
            field_work_mask,
            [np.round(field_polygon * scale).astype(np.int32)],
            255,
        )
        cv2.bitwise_and(work, field_work_mask, dst=work)
    work_mpp = mpp / scale

    report(0.08, "稻行方向估计")
    abort_if_cancelled()
    angle = estimate_row_angle(work)
    abort_if_cancelled()
    rotated, matrix = rotate_bound(work, math.degrees(angle), cv2.INTER_NEAREST)
    inverse = cv2.invertAffineTransform(matrix)
    foreground = rotated > 0

    report(0.18, "种植带投影分析")
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            odd_size(float(cfg.get("support_close_along_m", 1.2)) / work_mpp),
            odd_size(float(cfg.get("support_close_across_m", 1.6)) / work_mpp),
        ),
    )
    support = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, support_kernel) > 0
    support_counts = support.sum(axis=1).astype(np.float64)
    row_counts = foreground.sum(axis=1).astype(np.float64)
    profile = np.divide(
        row_counts,
        np.maximum(support_counts, 1.0),
        out=np.zeros_like(row_counts),
        where=support_counts > 0,
    )
    profile = smooth_1d(
        profile,
        float(cfg.get("profile_smooth_m", 0.06)) / work_mpp,
    )

    valid = support_counts > max(10, rotated.shape[1] * 0.08)
    if not np.any(valid):
        raise RuntimeError("无法建立田块支持区域")
    profile_u8 = np.clip(profile[valid] * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(
        profile_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = float(np.clip(
        otsu / 255.0,
        float(cfg.get("profile_threshold_min", 0.25)),
        float(cfg.get("profile_threshold_max", 0.72)),
    ))
    crop_rows = profile >= threshold
    runs = _runs(crop_rows)
    runs = _merge_close_runs(
        runs,
        max_gap=max(1, int(round(float(cfg.get("merge_gap_m", 0.08)) / work_mpp))),
    )
    min_band_px = max(
        3,
        int(round(float(cfg.get("min_band_width_m", 0.16)) / work_mpp)),
    )
    runs = [(start, end) for start, end in runs if end - start >= min_band_px]
    if len(runs) < 2:
        raise RuntimeError(f"仅检测到 {len(runs)} 条稳定种植带")

    report(0.26, "田头主体区间识别")
    abort_if_cancelled()
    body_start, body_end, body_diagnostics = estimate_body_interval(
        foreground,
        runs,
        work_mpp,
        cfg,
    )
    abort_if_cancelled()

    # Re-estimate the transverse row profile inside the body only.
    body_foreground = foreground[:, body_start:body_end]
    body_support = support[:, body_start:body_end]
    body_support_counts = body_support.sum(axis=1).astype(np.float64)
    body_row_counts = body_foreground.sum(axis=1).astype(np.float64)
    body_profile = np.divide(
        body_row_counts,
        np.maximum(body_support_counts, 1.0),
        out=np.zeros_like(body_row_counts),
        where=body_support_counts > 0,
    )
    body_profile = smooth_1d(
        body_profile,
        float(cfg.get("profile_smooth_m", 0.06)) / work_mpp,
    )
    body_valid = body_support_counts > max(10, (body_end - body_start) * 0.08)
    if np.any(body_valid):
        body_profile_u8 = np.clip(
            body_profile[body_valid] * 255.0,
            0,
            255,
        ).astype(np.uint8)
        body_otsu, _ = cv2.threshold(
            body_profile_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        body_threshold = float(np.clip(
            body_otsu / 255.0,
            float(cfg.get("profile_threshold_min", 0.25)),
            float(cfg.get("profile_threshold_max", 0.72)),
        ))
        body_runs = _runs(body_profile >= body_threshold)
        body_runs = _merge_close_runs(
            body_runs,
            max_gap=max(
                1,
                int(round(float(cfg.get("merge_gap_m", 0.08)) / work_mpp)),
            ),
        )
        body_runs = [
            (start, end)
            for start, end in body_runs
            if end - start >= min_band_px
        ]
        if len(body_runs) >= 2:
            runs = body_runs
            threshold = body_threshold
            body_start, body_end, refined_body_diagnostics = estimate_body_interval(
                foreground,
                runs,
                work_mpp,
                cfg,
            )
            body_diagnostics.update(refined_body_diagnostics)

    centers = np.asarray([(start + end - 1) * 0.5 for start, end in runs])
    center_spacing = float(np.median(np.diff(centers))) if len(centers) > 1 else 0.0
    gates = []
    for index in range(len(runs)):
        if index == 0:
            lower = int(round(centers[index] - center_spacing * 0.5))
        else:
            lower = int(round((centers[index - 1] + centers[index]) * 0.5))
        if index == len(runs) - 1:
            upper = int(round(centers[index] + center_spacing * 0.5))
        else:
            upper = int(round((centers[index] + centers[index + 1]) * 0.5))
        gates.append((max(0, lower), min(rotated.shape[0], upper)))

    profile_bands = [
        {
            "id": index + 1,
            "width_m": float((end - start) * work_mpp),
            "classification_width_m": float((end - start) * work_mpp),
        }
        for index, (start, end) in enumerate(runs)
    ]
    profile_wide, profile_narrow, width_threshold = classify_band_widths(
        profile_bands,
        fallback_threshold_m=float(cfg.get("band_width_threshold_m", 0.55)),
        alternation_penalty=float(cfg.get("band_alternation_penalty", 0.35)),
    )
    class_by_id = {
        item["id"]: item["band_type"]
        for item in profile_bands
    }
    class_widths = {}
    for band_type, items in (("wide", profile_wide), ("narrow", profile_narrow)):
        if items:
            class_widths[band_type] = float(np.median([
                item["width_m"] for item in items
            ]))

    sample_step = max(
        2,
        int(round(float(cfg.get("boundary_sample_m", 0.12)) / work_mpp)),
    )
    half_window = max(
        1,
        int(round(float(cfg.get("boundary_window_m", 0.08)) / work_mpp)),
    )
    xs = np.arange(body_start, body_end, sample_step, dtype=np.int32)
    rebuilt_rotated = np.zeros_like(rotated)
    bands = []
    skipped_bands = []
    filled_gap_lengths_m = []

    report(0.34, "逐带边界跟踪")
    for band_index, ((run_start, run_end), (gate_start, gate_end)) in enumerate(zip(runs, gates)):
        abort_if_cancelled()
        center_observed = np.full(xs.size, np.nan, dtype=np.float64)
        width_observed = np.full(xs.size, np.nan, dtype=np.float64)
        observed = np.zeros(xs.size, dtype=bool)

        for index, x_value in enumerate(xs):
            x0 = max(0, int(x_value) - half_window)
            x1 = min(rotated.shape[1], int(x_value) + half_window + 1)
            local_support = support[gate_start:gate_end, x0:x1]
            if local_support.size == 0 or np.mean(local_support) < 0.12:
                continue
            local = foreground[gate_start:gate_end, x0:x1]
            occupancy = local.mean(axis=1)
            local_y = np.flatnonzero(
                occupancy >= float(cfg.get("local_occupancy_threshold", 0.18))
            )
            if local_y.size < 2:
                continue
            center_observed[index] = gate_start + float(np.median(local_y))
            low = float(np.quantile(local_y, 0.04))
            high = float(np.quantile(local_y, 0.96))
            width_observed[index] = max(2.0, high - low + 1.0)
            observed[index] = True

        observed_count = int(np.count_nonzero(observed))
        observed_indices = np.flatnonzero(observed)
        observed_span_ratio = (
            float((observed_indices[-1] - observed_indices[0] + 1) / max(xs.size, 1))
            if observed_count else 0.0
        )
        if (
            observed_count < int(cfg.get("min_band_observations", 4))
            or observed_span_ratio < float(cfg.get("min_band_span_ratio", 0.35))
        ):
            skipped_bands.append({
                "id": band_index + 1,
                "observations": observed_count,
                "span_ratio": observed_span_ratio,
                "profile_width_m": float((run_end - run_start) * work_mpp),
            })
            continue

        missing_inside = ~observed.copy()
        missing_inside[:observed_indices[0]] = False
        missing_inside[observed_indices[-1] + 1:] = False
        missing_runs = _runs(missing_inside)
        longest_gap_m = max(
            ((end - start) * sample_step * work_mpp for start, end in missing_runs),
            default=0.0,
        )
        filled_gap_lengths_m.append(float(longest_gap_m))

        band_id = band_index + 1
        band_type = class_by_id.get(band_id, "wide")
        profile_width_px = float(run_end - run_start)
        class_width_px = (
            class_widths.get(band_type, profile_width_px * work_mpp) / work_mpp
        )
        class_prior = float(cfg.get("class_width_prior_blend", 0.45))
        target_width = (
            (1.0 - class_prior) * profile_width_px
            + class_prior * class_width_px
        )

        centers_i, center_trusted = repair_center_samples(
            center_observed,
            width_observed,
            observed,
            target_width,
            sample_step,
            work_mpp,
            cfg,
        )
        widths_i = fill_nan(np.where(center_trusted, width_observed, np.nan))
        center_detail = float(cfg.get("center_detail_blend", 0.65))
        trusted_indices = np.flatnonzero(
            center_trusted & np.isfinite(center_observed)
        )
        if trusted_indices.size >= 2:
            coefficients = np.polyfit(
                xs[trusted_indices].astype(np.float64),
                center_observed[trusted_indices],
                1,
            )
            row_trend = np.polyval(coefficients, xs.astype(np.float64))
            centers_i = row_trend + center_detail * (
                centers_i - row_trend
            )

        width_detail = float(cfg.get("width_detail_blend", 0.20))
        widths_i = target_width + width_detail * (
            widths_i - float(np.nanmedian(widths_i))
        )
        width_clip = float(cfg.get("width_clip_ratio", 0.28))
        widths_i = np.clip(
            widths_i,
            target_width * (1.0 - width_clip),
            target_width * (1.0 + width_clip),
        )
        centers_i = smooth_1d(
            centers_i,
            float(cfg.get("center_smooth_m", 0.55)) / (sample_step * work_mpp),
        )
        widths_i = smooth_1d(
            widths_i,
            float(cfg.get("width_smooth_m", 0.45)) / (sample_step * work_mpp),
        )

        sample_m = max(sample_step * work_mpp, 1e-6)
        internal_gap_samples = int(
            round(float(cfg.get("band_internal_gap_close_m", 6.0)) / sample_m)
        )
        extent_seed = observed & np.isfinite(center_observed)
        trusted_seed = center_trusted & np.isfinite(center_observed)
        if np.count_nonzero(trusted_seed) >= int(cfg.get("min_band_observations", 4)):
            extent_seed = trusted_seed
        extent_mask = close_boolean_gaps(extent_seed, internal_gap_samples)
        extent_runs = _runs(extent_mask)
        if not extent_runs:
            skipped_bands.append({
                "id": band_id,
                "reason": "no_longitudinal_support",
            })
            continue
        extent_start, extent_end = max(
            extent_runs,
            key=lambda item: (
                item[1] - item[0],
                int(np.count_nonzero(trusted_seed[item[0]:item[1]])),
            ),
        )
        trim_samples = int(round(float(cfg.get("band_end_trim_m", 0.10)) / sample_m))
        extent_start = min(extent_end, extent_start + max(0, trim_samples))
        extent_end = max(extent_start, extent_end - max(0, trim_samples))
        min_extent_samples = max(
            4,
            int(round(float(cfg.get("min_band_extent_m", 1.2)) / sample_m)),
        )
        if extent_end - extent_start < min_extent_samples:
            skipped_bands.append({
                "id": band_id,
                "reason": "short_longitudinal_support",
                "extent_m": float((extent_end - extent_start) * sample_m),
            })
            continue

        margin = max(1.0, float(cfg.get("inter_band_margin_m", 0.03)) / work_mpp)
        half_width = widths_i * 0.5
        lower = np.maximum(centers_i - half_width, gate_start + margin)
        upper = np.minimum(centers_i + half_width, gate_end - margin)
        centers_i = (lower + upper) * 0.5
        widths_i = np.maximum(2.0, upper - lower)

        x_i = xs[extent_start:extent_end].astype(np.float64)
        lower_i = lower[extent_start:extent_end]
        upper_i = upper[extent_start:extent_end]
        centers_i = centers_i[extent_start:extent_end]
        widths_i = widths_i[extent_start:extent_end]
        polygon_rotated = np.vstack([
            np.column_stack([x_i, lower_i]),
            np.column_stack([x_i[::-1], upper_i[::-1]]),
        ])

        centerline_rotated = np.column_stack([x_i, centers_i])
        centerline_work = cv2.transform(
            centerline_rotated.reshape(-1, 1, 2).astype(np.float64),
            inverse,
        ).reshape(-1, 2)
        centerline_full = centerline_work / scale

        interval_px = max(1.0, float(cfg.get("centerline_interval_m", 0.5)) / mpp)
        sampled = [centerline_full[0]]
        last_point = centerline_full[0]
        for point in centerline_full[1:]:
            if np.linalg.norm(point - last_point) >= interval_px:
                sampled.append(point)
                last_point = point
        if np.linalg.norm(centerline_full[-1] - sampled[-1]) > 1.0:
            sampled.append(centerline_full[-1])

        polygon_work = cv2.transform(
            polygon_rotated.reshape(-1, 1, 2).astype(np.float64),
            inverse,
        ).reshape(-1, 2)
        polygon_full = polygon_work / scale
        polygon_int = np.round(polygon_full).astype(np.int32)
        polygon_int = clip_band_polygon(polygon_int, field_polygon)
        if polygon_int is None or len(polygon_int) < 3:
            skipped_bands.append({
                "id": band_id,
                "reason": "outside_field_boundary",
            })
            continue
        if field_polygon is not None:
            sampled = [
                point
                for point in sampled
                if cv2.pointPolygonTest(
                    field_polygon.astype(np.float32),
                    point,
                    False,
                ) >= 0
            ]
            if len(sampled) < 2:
                skipped_bands.append({
                    "id": band_id,
                    "reason": "centerline_outside_field_boundary",
                })
                continue
        cv2.fillPoly(
            rebuilt_rotated,
            [np.round(polygon_rotated).astype(np.int32)],
            255,
        )
        bx, by, bw, bh = cv2.boundingRect(polygon_int)
        area_px = float(abs(cv2.contourArea(polygon_full.astype(np.float32))))
        centroid = np.mean(centerline_full, axis=0)

        bands.append({
            "id": band_id,
            "centroid": (float(centroid[0]), float(centroid[1])),
            "width_px": float(np.median(widths_i) / scale),
            "width_m": float(np.median(widths_i) * work_mpp),
            "profile_width_m": float(profile_width_px * work_mpp),
            "band_type": band_type,
            "area": area_px,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "centerline": [(float(x), float(y)) for x, y in sampled],
            "contour": polygon_int,
            "observed_ratio": float(np.mean(observed)),
            "observed_span_ratio": observed_span_ratio,
            "longitudinal_extent_ratio": float((extent_end - extent_start) / max(xs.size, 1)),
            "longitudinal_extent_m": float((extent_end - extent_start) * sample_m),
            "largest_filled_gap_m": float(longest_gap_m),
        })
        report(
            0.34 + 0.44 * (band_index + 1) / max(len(runs), 1),
            f"逐带重构 {band_index + 1}/{len(runs)}",
        )

    if len(bands) < 2:
        raise RuntimeError("种植带重构失败")

    report(0.82, "几何回写")
    abort_if_cancelled()
    rebuilt_work = cv2.warpAffine(
        rebuilt_rotated,
        inverse,
        (work.shape[1], work.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rebuilt_work = (rebuilt_work >= 127).astype(np.uint8) * 255
    report(0.89, "恢复原始分辨率")
    abort_if_cancelled()
    rebuilt = cv2.resize(rebuilt_work, (w, h), interpolation=cv2.INTER_LINEAR)
    rebuilt = (rebuilt >= 127).astype(np.uint8) * 255
    clip_mask_to_polygon_stripes(rebuilt, field_polygon)

    report(0.94, "种植带分类")
    wide = [item for item in bands if item.get("band_type") == "wide"]
    narrow = [item for item in bands if item.get("band_type") == "narrow"]
    raw_bool = raw > 0
    rebuilt_bool = rebuilt > 0
    headland_mask, body_residual_mask = residual_mask_layers(
        raw,
        rebuilt,
        angle,
        angle_thresh_deg=float(cfg.get("headland_angle_thresh_deg", 60.0)),
        min_area_ratio=float(cfg.get("headland_min_area_ratio", 0.001)),
    )
    visible_body = cv2.bitwise_or(rebuilt, body_residual_mask)
    visible_body = repair_row_gaps(visible_body, angle, mpp, cfg)
    visible_body = constrain_generated_body_to_raw_support(visible_body, raw, angle, mpp, cfg)
    visible_body_bool = visible_body > 0
    headland_bool = headland_mask > 0
    planning_support_mask = cv2.bitwise_or(visible_body, headland_mask)
    intersection = int(np.count_nonzero(raw_bool & visible_body_bool))
    union = int(np.count_nonzero(raw_bool | visible_body_bool))
    diagnostics = {
        "algorithm_version": MASK_REGULARIZATION_VERSION,
        "meters_per_pixel": float(mpp),
        "work_scale": float(scale),
        "work_meters_per_pixel": float(work_mpp),
        "profile_threshold": float(threshold),
        "band_width_threshold_m": float(width_threshold),
        "raw_area_px": int(np.count_nonzero(raw_bool)),
        "processed_area_px": int(np.count_nonzero(visible_body_bool)),
        "area_ratio": float(np.count_nonzero(visible_body_bool) / max(np.count_nonzero(raw_bool), 1)),
        "precision_to_raw": float(intersection / max(np.count_nonzero(visible_body_bool), 1)),
        "recall_to_raw": float(intersection / max(np.count_nonzero(raw_bool), 1)),
        "iou_to_raw": float(intersection / max(union, 1)),
        "headland_removed_px": int(np.count_nonzero(headland_bool)),
        "body_interval": body_diagnostics,
        "candidate_band_count": len(runs),
        "rebuilt_band_count": len(bands),
        "skipped_bands": skipped_bands,
        "wide_profile_width_m": class_widths.get("wide"),
        "narrow_profile_width_m": class_widths.get("narrow"),
        "max_filled_gap_m": max(filled_gap_lengths_m, default=0.0),
        "field_boundary_clipped": bool(field_polygon is not None),
        "center_trend_preserved": True,
    }
    report(0.98, "结果统计")
    report(1.0, "完成")
    return {
        "processed_mask": visible_body,
        "regularized_body_mask": rebuilt,
        "body_residual_mask": body_residual_mask,
        "headland_mask": headland_mask,
        "planning_support_mask": planning_support_mask,
        "main_angle": float(angle),
        "wide_bands": wide,
        "narrow_bands": narrow,
        "all_bands": bands,
        "diagnostics": diagnostics,
    }
