"""
workers.py — PySide6 后台工作线程

处理推理、路径规划等耗时操作，不阻塞 UI。
"""
from typing import Optional, Dict, List, Tuple, Callable
import traceback
import os
import time

import numpy as np

from PySide6.QtCore import QThread, Signal, QObject


def display_affine(source_affine, full_width, full_height, display_width, display_height):
    """Map display pixel centres to the exact source raster footprint."""
    from affine import Affine

    if min(full_width, full_height, display_width, display_height) <= 0:
        raise ValueError("raster dimensions must be positive")
    return source_affine * Affine.scale(
        float(full_width) / float(display_width),
        float(full_height) / float(display_height),
    )


def require_metric_scale(geo, state, shape) -> float:
    """Return a verified metres-per-pixel value or stop metric processing."""
    if geo is None or not geo.is_ready():
        raise RuntimeError("影像没有有效地理配准，不能执行米制掩膜处理或路径规划")
    if state is None or shape is None or len(shape) < 2:
        raise RuntimeError("缺少影像状态或掩膜尺寸，不能验证地面分辨率")
    height, width = int(shape[0]), int(shape[1])
    offset_x = float(getattr(state, "mask_offset_x", 0) or 0)
    offset_y = float(getattr(state, "mask_offset_y", 0) or 0)
    value = float(
        geo.meters_per_pixel(offset_x + width / 2.0, offset_y + height / 2.0)
    )
    if not np.isfinite(value) or not 1e-5 <= value <= 5.0:
        raise RuntimeError(f"影像地面分辨率无效: {value!r} m/px")
    return value


class TifLoadWorker(QObject):
    """Load/downsample GeoTIFF data outside the GUI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(5, "正在打开影像...")
            import rasterio
            from rasterio.enums import Resampling
            from pyproj import Transformer
            with rasterio.open(self.path) as src:
                crs = src.crs
                transform = src.transform
                transformer = None
                geo_error = ""
                determinant = float(transform.a * transform.e - transform.b * transform.d)
                has_valid_affine = not bool(transform.is_identity) and abs(determinant) > 1e-18
                if crs and has_valid_affine:
                    try:
                        # Every geographic display/export is WGS84 longitude,
                        # latitude. Never substitute a guessed projected CRS.
                        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    except Exception as exc:
                        geo_error = f"CRS transformation unavailable: {exc}"
                elif crs:
                    geo_error = "GeoTIFF has a CRS but no valid affine geotransform"
                else:
                    geo_error = "GeoTIFF has no CRS"
                bands = min(3, src.count)
                band_idx = list(range(1, bands + 1))
                full_h, full_w = src.height, src.width
                # Display should not load the full 15k×21k raster into a QPixmap.
                # Keep the longest display side bounded; processing uses masks/caches,
                # not this RGB preview.
                max_display_dim = 6000
                downsample = max(1, int(np.ceil(max(full_h, full_w) / max_display_dim)))
                out_h = max(1, full_h // downsample)
                out_w = max(1, full_w // downsample)
                self.progress.emit(25, f"正在读取影像预览 {out_w}×{out_h}...")
                img = src.read(
                    band_idx,
                    out_shape=(bands, out_h, out_w),
                    resampling=Resampling.average,
                )
            if self._cancelled:
                return
            self.progress.emit(75, "正在转换显示图像...")
            if img.shape[0] >= 3:
                rgb = np.ascontiguousarray(np.moveaxis(img[:3], 0, -1))
            else:
                rgb = np.stack([img[0]] * 3, axis=-1)
            if rgb.max() > 255:
                rgb = (rgb / max(float(rgb.max()), 1.0) * 255).astype(np.uint8)
            elif rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8, copy=False)
            self.finished.emit({
                "path": self.path,
                "rgb": rgb,
                "full_h": full_h,
                "full_w": full_w,
                "display_h": out_h,
                "display_w": out_w,
                "downsample": downsample,
                "crs": str(crs) if crs else "",
                "transform": display_affine(
                    transform, full_w, full_h, out_w, out_h
                ),
                "source_transform": transform,
                "transformer": transformer,
                "geo_error": geo_error,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class CacheRestoreWorker(QObject):
    """Load project JSON and large mask caches outside the GUI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            self.progress.emit(10, "正在读取项目缓存...")
            from cache import load_project_state, load_mask
            from row_geometry import MASK_REGULARIZATION_VERSION
            saved = load_project_state(self.path)
            if self._cancelled or not saved:
                self.finished.emit({"path": self.path, "saved": {}})
                return
            mask_result_summary = saved.get("mask_result") or {}
            diagnostics = (
                mask_result_summary.get("diagnostics", {})
                if isinstance(mask_result_summary, dict)
                else {}
            )
            from config import Config
            current_strength = str(Config()._raw.get("mask_processing", {}).get("strength", "standard"))
            cached_strength = str(diagnostics.get("strength", "standard"))
            processed_cache_valid = (
                bool(saved.get('mask_processed'))
                and diagnostics.get("algorithm_version") == MASK_REGULARIZATION_VERSION
                and cached_strength == current_strength
            )
            self.progress.emit(35, "正在读取 AI 掩膜缓存...")
            raw_mask, raw_ox, raw_oy = load_mask(self.path, "") if saved.get('inference_done') else (None, 0, 0)
            self.progress.emit(65, "正在读取掩膜处理缓存...")
            processed_mask, proc_ox, proc_oy = load_mask(self.path, "_processed") if processed_cache_valid else (None, 0, 0)
            if self._cancelled:
                return
            self.finished.emit({
                "path": self.path,
                "saved": saved,
                "processed_cache_valid": processed_cache_valid,
                "raw_mask": raw_mask,
                "raw_ox": raw_ox,
                "raw_oy": raw_oy,
                "processed_mask": processed_mask,
                "proc_ox": proc_ox,
                "proc_oy": proc_oy,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class PipelineWorker(QObject):
    """完整管线的后台工作线程。

    工作流程:
      1. 模型推理 (TiledInference)
      2. 掩膜处理 (process_mask)
      3. 行带分析 (classify_bands + extract_centerlines)
      4. 路径规划 (plan_path)
    """
    progress = Signal(int, str)   # (百分比, 消息)
    step_result = Signal(str, object)  # (步骤名, 数据)
    finished = Signal(dict)       # 全部完成
    error = Signal(str)           # 异常

    def __init__(self, state, geo, model_engine, image: np.ndarray,
                 params: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self.geo = geo
        self.model_engine = model_engine
        self.image = image
        self.params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit_plan_progress(self, progress, message):
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        self.progress.emit(min(99, 80 + int(value * 0.15)), message)

    def run(self):
        try:
            result = {}

            # ── Step 1: 推理 ──
            self.progress.emit(5, "正在执行 Tiled 推理...")
            from model import TiledInference
            inference = TiledInference(self.model_engine)
            raw_mask = inference.run(
                self.image,
                capture_sz=self.params.get("slice_size", 1280),
                overlap=self.params.get("overlap", 0.2),
            )
            if self._cancelled:
                return
            if raw_mask is None:
                raise RuntimeError("AI 推理没有返回有效掩膜")
            self.progress.emit(30, "推理完成，处理掩膜...")
            result["mask"] = raw_mask
            # 通知 UI：AI 识别完成
            self.step_result.emit("segment_done", raw_mask)

            # ── Step 2: 掩膜处理 ──
            self.progress.emit(40, "掩膜处理中...")
            from mask_processor import process_mask, extract_centerlines
            from config import Config
            mask_config = dict(Config()._raw.get("mask_processing", {}))
            require_metric_scale(self.geo, self.state, raw_mask.shape)
            processed = process_mask(
                raw_mask,
                geo=self.geo,
                state=self.state,
                config=mask_config,
            )
            if processed is None:
                raise RuntimeError("掩膜处理失败：未能提取有效行带")

            # process_mask 返回 dict:
            #   processed_mask, main_angle, wide_bands, narrow_bands, all_bands
            processed_mask = processed.get("processed_mask", raw_mask)
            main_angle = processed.get("main_angle", 0.0)
            wide_bands = processed.get("wide_bands", [])
            result["processed"] = processed
            result["main_angle"] = main_angle

            # Notify UI with the complete processing result; it needs bands and metadata too.
            self.step_result.emit("process_done", processed)
            self.progress.emit(60, "提取中心线...")

            # ── Step 3: 提取中心线 ──
            centerlines = extract_centerlines(
                wide_bands, main_angle,
                full_mask=processed_mask,
                geo=self.geo,
                state=self.state,
            )
            result["centerlines"] = centerlines
            result["bands"] = wide_bands
            self.progress.emit(70, "中心线提取完成")

            # ── Step 4: 路径规划 ──
            self.progress.emit(80, "路径规划...")
            from path_planner import build_band_mask, plan_path
            planning_bands = wide_bands  # 只对4行带规划

            planning_mask = build_band_mask(processed_mask.shape, planning_bands)
            if not np.any(planning_mask):
                planning_mask = processed_mask

            path_config = {
                "min_turn_radius_m": self.params.get("turn_radius", 2.5),
                "turn_strategy": self.params.get("turn_strategy", "auto"),
                "headland_mask": processed.get("headland_mask"),
            }
            path_result = plan_path(
                planning_bands, processed_mask, main_angle,
                geo=self.geo,
                state=self.state,
                config=path_config,
                progress_callback=self._emit_plan_progress,
                detected_mask=self.state.inference_original_mask if hasattr(self.state, 'inference_original_mask') else None,
                planned_mask=planning_mask,
            )
            result["path"] = path_result

            self.progress.emit(100, "处理完成")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _extract_mask(self, results, img_shape) -> np.ndarray:
        """从 ultralytics 结果提取二值掩膜。"""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if results is None:
            return mask
        import cv2
        for r in results if hasattr(results, '__iter__') else [results]:
            if hasattr(r, 'masks') and r.masks is not None:
                for i in range(len(r.masks)):
                    m = r.masks[i].data.cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    m = np.where(m > 0.5, 1, 0).astype(np.uint8)
                    m = cv2.resize(m, (w, h), interpolation=0)
                    mask = np.maximum(mask, m)
        return mask

class MaskProcessWorker(QObject):
    """Run mask processing outside the UI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, raw_mask: np.ndarray, geo, state, config: dict, parent=None):
        super().__init__(parent)
        self.raw_mask = raw_mask
        self.geo = geo
        self.state = state
        self.config = dict(config or {})
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit_mask_progress(self, progress, message):
        """Forward mask-processing progress and yield to the GUI event loop.

        For 15000×21000 GeoTIFF masks, OpenCV kernels plus Python row loops can
        keep CPU cores saturated long enough that Windows labels the main window
        as not responding.  Regular queued progress signals and sleep(0) keep
        the UI message pump getting time slices while the worker continues.
        """
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        pct = min(94, 35 + int(value * 0.58))
        self.progress.emit(pct, str(message or "掩膜处理中..."))
        time.sleep(0)

    def run(self):
        previous_cv_threads = None
        try:
            if self._cancelled:
                return
            try:
                import cv2
                previous_cv_threads = cv2.getNumThreads()
                cv2.setNumThreads(max(1, min(2, (os.cpu_count() or 2) - 1)))
            except Exception:
                previous_cv_threads = None
            self.progress.emit(15, "正在读取 AI 掩膜...")
            from mask_processor import process_mask
            self.progress.emit(35, "正在提取水稻行带...")
            config = dict(self.config)
            config["_cancel_callback"] = lambda: self._cancelled
            require_metric_scale(self.geo, self.state, self.raw_mask.shape)
            processed = process_mask(
                self.raw_mask,
                geo=self.geo,
                state=self.state,
                config=config,
                progress_callback=self._emit_mask_progress,
            )
            if self._cancelled:
                return
            if processed is None:
                raise RuntimeError("掩膜处理没有生成有效结果")
            self.progress.emit(95, "掩膜处理完成，正在刷新界面...")
            self.finished.emit(processed)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        finally:
            if previous_cv_threads is not None:
                try:
                    import cv2
                    cv2.setNumThreads(previous_cv_threads)
                except Exception:
                    pass


class PlanWorker(QObject):
    """Run path planning outside the UI thread."""
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, mask_result: dict, geo, state, config: dict, detected_mask=None, parent=None):
        super().__init__(parent)
        self.mask_result = dict(mask_result or {})
        self.geo = geo
        self.state = state
        self.config = dict(config or {})
        self.detected_mask = detected_mask
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def _emit_plan_progress(self, progress, message):
        try:
            value = float(progress)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 1.0:
            value *= 100.0
        value = max(0.0, min(100.0, value))
        self.progress.emit(min(99, 20 + int(value * 0.75)), str(message or "路径规划中..."))

    def run(self):
        try:
            if self._cancelled:
                return
            processed_mask = self.mask_result.get("processed_mask")
            if processed_mask is None:
                raise RuntimeError("缺少处理后的掩膜")
            require_metric_scale(self.geo, self.state, processed_mask.shape)
            main_angle = self.mask_result.get("main_angle", 0.0)
            wide_bands = self.mask_result.get("wide_bands", [])
            self.progress.emit(20, "正在构建可作业区域...")
            from path_planner import build_band_mask, plan_path
            planning_mask = build_band_mask(processed_mask.shape, wide_bands)
            if not np.any(planning_mask):
                planning_mask = processed_mask
            if self._cancelled:
                return
            self.progress.emit(35, "正在计算作业线与转弯段...")
            config = dict(self.config)
            config.setdefault("headland_mask", self.mask_result.get("headland_mask"))
            config["_cancel_callback"] = lambda: self._cancelled
            path_result = plan_path(
                wide_bands, processed_mask, main_angle,
                geo=self.geo,
                state=self.state,
                config=config,
                progress_callback=self._emit_plan_progress,
                detected_mask=self.detected_mask,
                planned_mask=planning_mask,
            )
            if self._cancelled:
                return
            self.progress.emit(98, "路径规划完成，正在刷新界面...")
            self.finished.emit({
                "path": path_result,
                "processed": self.mask_result,
                "bands": wide_bands,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

