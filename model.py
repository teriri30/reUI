"""模型加载 + 分块推理 + 后台推理运行器"""
import os, math, threading, traceback
from typing import Optional, List, Tuple, Any, Dict
import numpy as np
import cv2

from config import Config, AppLogger
from state import AppState
from geo import GeoUtils
# 掩膜处理和路径规划由独立步骤负责，不在推理运行器中调用



# ─── ModelEngine ────────────────────────────────────────────
class ModelEngine:
    def __init__(self):
        self._model = None
        self._model_path = ""
        self._loaded = False
        self.log = AppLogger()

    def load(self, model_path: str) -> bool:
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        if not os.path.exists(model_path):
            self.log.error(f"模型文件不存在: {model_path}")
            return False
        try:
            from ultralytics import YOLO
            self.log.info(f"正在加载模型: {model_path}")
            self._model = YOLO(model_path)
            self._model_path = model_path
            self._loaded = True
            self.log.info("模型加载成功")
            return True
        except Exception as e:
            self.log.error(f"模型加载失败: {e}")
            self._loaded = False
            return False

    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def unload(self):
        """Release the active model when project history restores 'no model'."""
        self._model = None
        self._model_path = ""
        self._loaded = False

    def predict(self, image: np.ndarray, conf: float = 0.5, iou: float = 0.5) -> Optional[Any]:
        if not self.is_loaded(): return None
        try:
            return self._model.predict(image, imgsz=640, conf=conf, iou=iou, retina_masks=True, verbose=False)
        except Exception as e:
            self.log.error(f"推理失败: {e}")
            return None

    def get_model_name(self) -> str:
        return os.path.basename(self._model_path) if self._model_path else "unknown"

# ─── 模型目录扫描 ──────────────────────────────────────────
def scan_model_dir(directory: str) -> List[Dict]:
    """扫描目录下的 .pt 模型文件，返回模型列表。
    支持扫描 yoloduibi/runs/ 下的训练权重。
    """
    models = []
    if not os.path.isdir(directory):
        return models

    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith('.pt'):
                full = os.path.join(root, f)
                # 从路径推断模型名
                rel = os.path.relpath(full, directory)
                name = rel.replace('\\', '/')
                models.append({"name": name, "path": full})
    return models


# ─── TiledInference ─────────────────────────────────────────
class TiledInference:
    INFER_SIZE = 640
    DEFAULT_CAPTURE_SIZE = 640
    DEFAULT_OVERLAP = 0.40          # 默认重叠率 0.40，用于降低切片边缘漏检
    DEFAULT_CONF = 0.25             # 默认模型置信度阈值 0.25
    DEFAULT_IOU = 0.45              # 默认 NMS IoU 阈值 0.45

    def __init__(self, engine: ModelEngine):
        self.engine = engine
        self.log = AppLogger()
        self._gauss_cache = {}  # 高斯权重缓存

    def _get_gauss_weight(self, sz: int) -> np.ndarray:
        """获取或生成高斯权重图：中心 1.0，边缘渐近 0，用于无缝融合"""
        if sz in self._gauss_cache:
            return self._gauss_cache[sz]
        gy, gx = np.ogrid[:sz, :sz]
        c = sz / 2.0
        s = sz / 4.5  # sigma更大 → 边缘降权更柔和
        w = np.exp(-((gx - c)**2 + (gy - c)**2) / (2 * s * s))
        # 边缘权重降至 ~0.05，中心保持 1.0
        w = (w - w.min()) / max(w.max() - w.min(), 1e-10)
        w = w * 0.95 + 0.05
        self._gauss_cache[sz] = w.astype(np.float32)
        return self._gauss_cache[sz]

    def run(self, full_image: np.ndarray, progress_cb=None,
            capture_sz=DEFAULT_CAPTURE_SIZE, overlap=DEFAULT_OVERLAP,
            erode=False, conf=DEFAULT_CONF, iou=DEFAULT_IOU,
            batch_size: int = 4) -> Optional[np.ndarray]:
        """SAHI 三步：滑窗切块 -> 单块概率推理 -> 高斯加权融合
        Args:
            capture_sz: 从原图裁切的块尺寸，默认固定 640，避免额外缩放损失
            overlap: 重叠率（0~1），默认 0.40，用于降低切片边缘漏检
            erode: 是否执行防粘连腐蚀（细线建议 False）
            conf: 模型置信度阈值，默认 0.25
            iou: NMS 阈值，默认 0.45
        """

        if not self.engine.is_loaded(): return None

        H, W = full_image.shape[:2]
        capture_sz = int(capture_sz)
        overlap = min(max(float(overlap), 0.0), 0.95)
        infer_size = self.INFER_SIZE
        native = (capture_sz == infer_size)

        capture_stride = max(1, int(round(capture_sz * (1 - overlap))))
        n_cols = max(1, math.ceil((W - capture_sz) / capture_stride) + 1)
        n_rows = max(1, math.ceil((H - capture_sz) / capture_stride) + 1)
        total = n_rows * n_cols

        self.log.info(f"最大精度分块推理: {n_rows}x{n_cols}={total}块, "
                      f"{'原生640' if native else str(capture_sz)+'px'}, "
                      f"重叠率 {overlap*100:.0f}%, conf={conf}, iou={iou}")

        # 概率加权累加器（float32）
        prob_acc = np.zeros((H, W), dtype=np.float32)
        weight_acc = np.zeros((H, W), dtype=np.float32)
        gauss_w = self._get_gauss_weight(capture_sz)
        processed = 0
        batch_images = []
        batch_meta = []

        def flush_batch():
            nonlocal processed, batch_images, batch_meta
            if not batch_images:
                return
            results = self.engine.predict(batch_images, conf=conf, iou=iou)
            if results is None:
                # Retry one-by-one after a batch failure, but never convert a
                # failed tile into valid background data.
                results = []
                for image, (x0, y0, _ah, _aw) in zip(batch_images, batch_meta):
                    single = self.engine.predict(image, conf=conf, iou=iou)
                    if not single:
                        raise RuntimeError(
                            f"tile inference failed at pixel window ({x0}, {y0})"
                        )
                    results.append(single[0])
            results = list(results)
            if len(results) != len(batch_meta):
                raise RuntimeError(
                    "tile inference failed: model result count does not match input tiles"
                )
            for result, (x0, y0, ah, aw) in zip(results, batch_meta):
                if result is None:
                    raise RuntimeError(
                        f"tile inference failed at pixel window ({x0}, {y0})"
                    )
                tile_prob = np.zeros((capture_sz, capture_sz), dtype=np.float32)
                if result is not None and result.masks is not None:
                    md = result.masks.data.cpu().numpy()
                    if len(md) > 0:
                        probability = np.max(md, axis=0)
                        if probability.shape[:2] != (capture_sz, capture_sz):
                            probability = cv2.resize(
                                probability,
                                (capture_sz, capture_sz),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        tile_prob = probability.astype(np.float32)
                roi_w = gauss_w[:ah, :aw]
                prob_acc[y0:y0+ah, x0:x0+aw] += tile_prob[:ah, :aw] * roi_w
                weight_acc[y0:y0+ah, x0:x0+aw] += roi_w
                processed += 1
            if progress_cb:
                progress_cb(processed / total)
            batch_images = []
            batch_meta = []

        for row in range(n_rows):
            for col in range(n_cols):
                x0 = min(col * capture_stride, W - capture_sz)
                y0 = min(row * capture_stride, H - capture_sz)
                x0, y0 = max(0, x0), max(0, y0)

                tile_raw = full_image[y0:y0 + capture_sz, x0:x0 + capture_sz]
                if tile_raw.shape[0] < capture_sz or tile_raw.shape[1] < capture_sz:
                    tile_raw = cv2.copyMakeBorder(tile_raw, 0, capture_sz - tile_raw.shape[0],
                                                  0, capture_sz - tile_raw.shape[1],
                                                  cv2.BORDER_REFLECT)

                # 原生尺寸直接推理，无需缩放
                if native:
                    tile_infer = tile_raw
                else:
                    tile_infer = cv2.resize(tile_raw, (infer_size, infer_size),
                                            interpolation=cv2.INTER_LINEAR)

                tile_rgb = cv2.cvtColor(tile_infer, cv2.COLOR_BGR2RGB)
                ah = min(capture_sz, H - y0)
                aw = min(capture_sz, W - x0)
                batch_images.append(tile_rgb)
                batch_meta.append((x0, y0, ah, aw))
                if len(batch_images) >= max(1, int(batch_size)):
                    flush_batch()
        flush_batch()

        # 归一化概率 → 自适应二值化
        np.maximum(weight_acc, 1e-10, out=weight_acc)
        np.divide(prob_acc, weight_acc, out=prob_acc)
        prob_uint8 = (np.clip(prob_acc, 0, 1) * 255).astype(np.uint8)
        otsu_th, mask = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.log.info(f"概率融合完成, Otsu阈值={otsu_th}, 非零像素: {np.count_nonzero(mask)}")

        # 可选轻度腐蚀（默认关闭）
        if erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=1)
            self.log.info(f"轻度腐蚀完成，剩余非零像素: {np.count_nonzero(mask)}")

        return mask

# ─── InferenceRunner ────────────────────────────────────────
class InferenceRunner:
    def __init__(self, state: AppState, engine: ModelEngine, geo: GeoUtils):
        self.state = state
        self.engine = engine
        self.geo = geo
        self.cfg = Config()
        self.log = AppLogger()
        self._thread: Optional[threading.Thread] = None

    def start(self, full_image: np.ndarray, field_boundary: Optional[List] = None):
        if self.state.inference_running: return
        self.state.safe_update(
            mask_raw=None,
            inference_original_mask=None,
            inference_done=False,
            inference_running=True,
            inference_progress=0.0,
            status_message="正在运行 AI 推理...",
        )
        self._thread = threading.Thread(target=self._run, args=(full_image, field_boundary), daemon=True)
        self._thread.start()

    def start_from_tif(self, tif_path: str, display_shape: Tuple[int, int],
                       field_boundary: Optional[List] = None, downsample: int = 1):
        """从原始 GeoTIFF 裁剪田块推理，而不是用降采样 UI 预览图推理。"""
        if self.state.inference_running:
            return
        self.state.safe_update(
            mask_raw=None,
            inference_original_mask=None,
            inference_done=False,
            inference_running=True,
            inference_progress=0.0,
            status_message="正在读取原始影像执行 AI 推理...",
        )
        self._thread = threading.Thread(
            target=self._run_from_tif,
            args=(tif_path, display_shape, field_boundary, max(1, int(downsample or 1))),
            daemon=True,
        )
        self._thread.start()

    def _run_from_tif(self, tif_path: str, display_shape: Tuple[int, int],
                      field_boundary: Optional[List] = None, downsample: int = 1):
        succeeded = False
        try:
            import rasterio
            from rasterio.windows import Window

            if not field_boundary or len(field_boundary) < 3:
                self.state.safe_update(status_message="推理失败：请先圈选田块", inference_running=False)
                return

            display_h, display_w = [max(1, int(v)) for v in display_shape[:2]]
            pts_display = np.asarray(field_boundary, dtype=np.float64)
            pad_display = 200
            x0_d = max(0, int(math.floor(float(np.min(pts_display[:, 0])))) - pad_display)
            y0_d = max(0, int(math.floor(float(np.min(pts_display[:, 1])))) - pad_display)
            x1_d = min(display_w, int(math.ceil(float(np.max(pts_display[:, 0])))) + 1 + pad_display)
            y1_d = min(display_h, int(math.ceil(float(np.max(pts_display[:, 1])))) + 1 + pad_display)

            self.state.safe_update(inference_progress=0.03, status_message="正在读取原始分辨率田块裁剪...")
            with rasterio.open(tif_path) as src:
                scale_x = float(src.width) / float(display_w)
                scale_y = float(src.height) / float(display_h)
                pts_source = np.rint(
                    pts_display * np.asarray([scale_x, scale_y], dtype=np.float64)
                ).astype(np.int32)
                x0_src = max(0, int(math.floor(x0_d * scale_x)))
                y0_src = max(0, int(math.floor(y0_d * scale_y)))
                x1_src = min(src.width, int(math.ceil(x1_d * scale_x)))
                y1_src = min(src.height, int(math.ceil(y1_d * scale_y)))
                if x1_src <= x0_src or y1_src <= y0_src:
                    self.state.safe_update(status_message="推理失败：田块范围无效", inference_running=False)
                    return
                crop_w = int(x1_src - x0_src)
                crop_h = int(y1_src - y0_src)
                crop_pixels = crop_w * crop_h
                max_crop_pixels = int(
                    Config()._raw.get("model", {}).get(
                        "max_source_crop_pixels",
                        36_000_000,
                    )
                )
                if max_crop_pixels > 0 and crop_pixels > max_crop_pixels:
                    message = (
                        f"推理失败：圈选区域过大 ({crop_w}x{crop_h}, "
                        f"{crop_pixels / 1_000_000:.1f}MP)，"
                        f"超过上限 {max_crop_pixels / 1_000_000:.1f}MP。"
                        "请缩小田块范围，或在 config.json 中调高 max_source_crop_pixels。"
                    )
                    self.log.warning(message)
                    self.state.safe_update(
                        status_message=message,
                        inference_running=False,
                    )
                    return
                img = src.read(
                    list(range(1, min(3, src.count) + 1)),
                    window=Window(x0_src, y0_src, x1_src - x0_src, y1_src - y0_src),
                )

            crop_img = np.ascontiguousarray(np.moveaxis(img[:3], 0, -1)) if img.shape[0] >= 3 else np.stack([img[0]] * 3, axis=-1)
            if crop_img.max() > 255:
                crop_img = (crop_img / max(float(crop_img.max()), 1.0) * 255).astype(np.uint8)
            elif crop_img.max() <= 1.0:
                crop_img = (crop_img * 255).astype(np.uint8)
            else:
                crop_img = crop_img.astype(np.uint8, copy=False)

            local_src = pts_source.copy()
            local_src[:, 0] -= x0_src
            local_src[:, 1] -= y0_src
            crop_field_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(crop_field_mask, [local_src], 255)
            self.log.info(
                f"原始分辨率田块裁剪尺寸: {crop_img.shape[1]}x{crop_img.shape[0]}, "
                f"display_scale=({scale_x:.9f},{scale_y:.9f})"
            )

            def progress_cb(pct: float):
                self.state.safe_update(
                    inference_progress=0.04 + pct * 0.84,
                    status_message=f"原始影像切片推理中... {int(pct * 100)}%",
                )

            crop_mask_src = TiledInference(self.engine).run(
                crop_img,
                progress_cb=progress_cb,
                capture_sz=int(self.cfg.TILE_CAPTURE_SIZE),
                overlap=float(self.cfg.TILE_OVERLAP),
                erode=False,
                conf=float(self.cfg.MODEL_CONF),
                iou=float(self.cfg.MODEL_IOU),
                batch_size=4,
            )
            if crop_mask_src is not None:
                self.state.safe_update(inference_progress=0.92, status_message="正在裁剪田块范围...")
                clipped = cv2.bitwise_and(crop_mask_src, crop_mask_src, mask=crop_field_mask)
                if np.count_nonzero(clipped) > 0:
                    crop_mask_src = clipped
            if crop_mask_src is None or np.count_nonzero(crop_mask_src) == 0:
                self.state.safe_update(status_message="推理失败：掩膜为空", inference_running=False)
                return

            out_w = max(1, x1_d - x0_d)
            out_h = max(1, y1_d - y0_d)
            mask = cv2.resize(crop_mask_src, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            boundary_meta = [[float(px), float(py)] for px, py in (self.state.field_boundary or [])]
            cache_meta = {
                "model": self.state.current_model_name or "",
                "model_path": self.state.current_model_path or "",
                "strategy": "source_tif_sahi_640_40p",
                "field_boundary": boundary_meta,
                "field_area_m2": float(self.state.field_area_m2 or 0.0),
                "source_crop_shape": [int(crop_mask_src.shape[0]), int(crop_mask_src.shape[1])],
                "display_downsample": int(downsample),
                "display_scale_x": float(scale_x),
                "display_scale_y": float(scale_y),
            }
            from cache import save_mask, save_project_state
            save_mask(self.state.tif_path, mask, x0_d, y0_d, meta=cache_meta)
            save_mask(self.state.tif_path, mask, x0_d, y0_d, suffix="_best", meta=cache_meta)
            non_zero = np.count_nonzero(mask)
            self.state.safe_update(
                mask_raw=mask,
                inference_original_mask=mask,
                mask_offset_x=x0_d,
                mask_offset_y=y0_d,
                _mask_overlay_dirty=True,
                inference_done=True,
                inference_progress=1.0,
                status_message=f"AI 识别完成，原始影像推理掩膜已保存（非零像素: {non_zero}）。请执行「掩膜处理」。",
            )
            save_project_state(self.state.tif_path, self.state, stage="inference")
            succeeded = True
            self.log.info(f"原始影像推理完成: source_mask={crop_mask_src.shape}, display_mask={mask.shape}, offset=({x0_d},{y0_d})")
        except Exception as e:
            self.log.error(f"原始影像推理异常: {traceback.format_exc()}")
            self.state.safe_update(status_message=f"推理失败: {e}")
        finally:
            self.state.safe_update(
                inference_running=False,
                inference_progress=1.0,
                inference_done=succeeded,
            )
            if succeeded:
                from workflow import WorkflowUpdater
                WorkflowUpdater.advance(self.state, "INFERENCE_DONE")

    def _run(self, image: np.ndarray, field_boundary: Optional[List] = None):
        succeeded = False
        try:
            def progress_cb(pct: float):
                self.state.safe_update(
                    inference_progress=0.04 + pct * 0.84,
                    status_message=f"切片推理中... {int(pct * 100)}%",
                )

            H, W = image.shape[:2]
            crop_mask = None
            x0 = y0 = 0

            if field_boundary and len(field_boundary) >= 3:
                pts = np.array(field_boundary, dtype=np.int32)
                pad = 200
                bx, by, bw, bh = cv2.boundingRect(pts)
                x0, y0 = bx, by
                x1, y1 = bx + bw, by + bh
                y0, y1 = max(0, y0 - pad), min(H, y1 + pad)
                x0, x1 = max(0, x0 - pad), min(W, x1 + pad)

                crop_img = image[y0:y1, x0:x1]
                crop_field_mask = np.zeros(crop_img.shape[:2], dtype=np.uint8)
                local_pts = pts.copy()
                local_pts[:, 0] -= x0
                local_pts[:, 1] -= y0
                cv2.fillPoly(crop_field_mask, [local_pts], 255)
                H_c, W_c = crop_img.shape[:2]
                self.log.info(f"田块裁剪尺寸: {W_c}x{H_c}")

                tiled = TiledInference(self.engine)
                crop_mask = tiled.run(
                    crop_img,
                    progress_cb=progress_cb,
                    capture_sz=int(self.cfg.TILE_CAPTURE_SIZE),
                    overlap=float(self.cfg.TILE_OVERLAP),
                    erode=False,
                    conf=float(self.cfg.MODEL_CONF),
                    iou=float(self.cfg.MODEL_IOU),
                    batch_size=4,
                )
                if crop_mask is not None:
                    self.state.safe_update(
                        inference_progress=0.92,
                        status_message="正在裁剪田块范围...",
                    )
                    clipped_mask = cv2.bitwise_and(crop_mask, crop_mask, mask=crop_field_mask)
                    if np.count_nonzero(clipped_mask) > 0:
                        crop_mask = clipped_mask
                    elif np.count_nonzero(crop_mask) > 0:
                        self.log.warning("裁剪后的田块掩膜为空，保留原始推理结果")

                if crop_mask is not None:
                    mask = crop_mask
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
            else:
                self.log.info("未圈选田块，跳过全图推理")
                mask = np.zeros((H, W), dtype=np.uint8)

            if mask is None or np.count_nonzero(mask) == 0:
                self.state.safe_update(
                    status_message="推理失败：掩膜为空",
                    inference_running=False,
                )
                return

            boundary_meta = [[float(px), float(py)] for px, py in (self.state.field_boundary or [])]
            self.state.safe_update(
                inference_progress=0.97,
                status_message="正在保存项目推理结果...",
            )
            cache_meta = {
                "model": self.state.current_model_name or "",
                "model_path": self.state.current_model_path or "",
                "strategy": "sahi_640_40p",
                "field_boundary": boundary_meta,
                "field_area_m2": float(self.state.field_area_m2 or 0.0),
            }

            from cache import save_mask, save_project_state
            save_mask(self.state.tif_path, mask, x0, y0, meta=cache_meta)
            save_mask(self.state.tif_path, mask, x0, y0, suffix="_best", meta=cache_meta)

            non_zero = np.count_nonzero(mask)
            self.state.safe_update(
                mask_raw=mask,
                inference_original_mask=mask,
                mask_offset_x=x0,
                mask_offset_y=y0,
                _mask_overlay_dirty=True,
                inference_done=True,
                inference_progress=1.0,
                status_message=f"AI 识别完成，原始掩膜已保存（非零像素: {non_zero}）。请执行「掩膜处理」。",
            )
            save_project_state(self.state.tif_path, self.state, stage="inference")
            succeeded = True
            self.log.info(f"原始掩膜推理完成: mask shape={mask.shape}, non_zero={non_zero}, offset=({x0},{y0})")

        except Exception as e:
            self.log.error(f"推理异常: {traceback.format_exc()}")
            self.state.safe_update(status_message=f"推理失败: {e}")
        finally:
            self.state.safe_update(
                inference_running=False,
                inference_progress=1.0,
                inference_done=succeeded,
            )
            if succeeded:
                from workflow import WorkflowUpdater
                WorkflowUpdater.advance(self.state, "INFERENCE_DONE")
