"""Configuration management and logging for the PySide6 application."""
import os, json, logging, datetime, tempfile, glob as _glob, threading
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional


class ConfigValidationError(ValueError):
    """Configuration is syntactically valid JSON but scientifically unsafe."""


def _validated_number(section: dict, key: str, *, minimum=None, maximum=None, strict_min=False):
    if key not in section:
        return
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValidationError(f"{key} must be a number")
    value = float(value)
    if minimum is not None and (value < minimum or (strict_min and value == minimum)):
        op = ">" if strict_min else ">="
        raise ConfigValidationError(f"{key} must be {op} {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigValidationError(f"{key} must be <= {maximum}")


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ConfigValidationError("configuration root must be an object")
    schema_version = str(raw.get("_schema_version", "1.1"))
    if schema_version not in {"1.1", "1.2"}:
        raise ConfigValidationError(f"unsupported _schema_version: {schema_version}")
    for name in ("model", "harvester", "path_planning", "mask_processing", "raster_preprocessing", "geo"):
        if name in raw and not isinstance(raw[name], dict):
            raise ConfigValidationError(f"{name} must be an object")

    model = raw.get("model", {})
    _validated_number(model, "conf_threshold", minimum=0.0, maximum=1.0)
    _validated_number(model, "iou_threshold", minimum=0.0, maximum=1.0)
    _validated_number(model, "tile_overlap", minimum=0.0, maximum=0.95)
    _validated_number(model, "tile_capture_size", minimum=64, maximum=8192)
    _validated_number(model, "max_source_crop_pixels", minimum=1)

    harvester = raw.get("harvester", {})
    for key in (
        "cutter_width_m", "track_width_m", "track_gauge_m", "wheelbase_m",
        "track_length_m", "turn_radius_m",
    ):
        _validated_number(harvester, key, minimum=0.0, strict_min=True)
    if "track_width_m" in harvester and "track_gauge_m" in harvester:
        if float(harvester["track_width_m"]) >= float(harvester["track_gauge_m"]):
            raise ConfigValidationError("track_width_m must be smaller than track_gauge_m")

    planning = raw.get("path_planning", {})
    _validated_number(planning, "min_turn_radius_m", minimum=0.0, strict_min=True)
    _validated_number(planning, "planning_max_dim", minimum=256)
    _validated_number(planning, "validation_max_dim", minimum=256)
    for key in (
        "max_track_core_overlap_pct", "min_harvest_coverage_pct",
        "max_track_outside_field_pct",
    ):
        _validated_number(planning, key, minimum=0.0, maximum=100.0)

    mask = raw.get("mask_processing", {})
    if "strength" in mask and str(mask["strength"]) not in {
        "light", "standard", "strong", "very_strong",
    }:
        raise ConfigValidationError("strength must be light, standard, strong, or very_strong")
    _validated_number(mask, "max_work_dim", minimum=256)
    _validated_number(mask, "band_width_threshold_m", minimum=0.0, strict_min=True)

    raster = raw.get("raster_preprocessing", {})
    if "mode" in raster and str(raster["mode"]) not in {
        "percentile_per_band", "uint8_identity",
    }:
        raise ConfigValidationError("raster_preprocessing.mode is unsupported")
    _validated_number(raster, "lower_percentile", minimum=0.0, maximum=100.0)
    _validated_number(raster, "upper_percentile", minimum=0.0, maximum=100.0)
    lower = float(raster.get("lower_percentile", 2.0))
    upper = float(raster.get("upper_percentile", 98.0))
    if lower >= upper:
        raise ConfigValidationError("lower_percentile must be smaller than upper_percentile")

    geo = raw.get("geo", {})
    _validated_number(geo, "max_pixel_anisotropy_ratio", minimum=1.0)
    _validated_number(geo, "max_gsd_variation_ratio", minimum=0.0, maximum=1.0)
    return raw

def save_json_atomic(path: str, data: dict):
    """原子写入 JSON 文件：先写临时文件再替换，防止写入中断导致文件损坏。
    写入前进行 JSON 序列化校验，确保数据可序列化。"""
    content = json.dumps(data, indent=2, ensure_ascii=False)
    dir_name = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(tmp_path, path)
        try:
            from project_log import ProjectJournal
            ProjectJournal().change(
                "JSON configuration saved",
                module="config",
                details={"path": os.path.abspath(path)},
            )
        except Exception:
            pass
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


class Config:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._loaded = False
                cls._instance._raw = {}
                cls._instance._lock = threading.RLock()
        return cls._instance

    def load(self, path: str = "") -> bool:
        if not path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, "config.json")
        if not os.path.exists(path):
            example_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.example.json")
            if os.path.exists(example_path):
                path = example_path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                validated = validate_config(raw)
                with self._lock:
                    self._raw = validated
                    self._loaded = True
                return True
            except ConfigValidationError:
                with self._lock:
                    self._raw = {}
                    self._loaded = False
                raise
            except Exception as e:
                print(f"[Config] 加载配置文件失败: {e}")
        with self._lock:
            self._raw = {}
            self._loaded = False
        return False

    def _get(self, *keys: str, default: Any = None) -> Any:
        with self._lock:
            val = self._raw
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k)
                    if val is None:
                        return default
                else:
                    return default
            return val if val is not None else default

    def section(self, name: str) -> dict:
        with self._lock:
            value = self._raw.get(str(name), {})
            return deepcopy(value) if isinstance(value, dict) else {}

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy(self._raw)

    def replace(self, raw: dict) -> dict:
        candidate = validate_config(deepcopy(raw))
        with self._lock:
            self._raw = candidate
            self._loaded = True
            return deepcopy(self._raw)

    def update_section(self, name: str, updates: dict) -> dict:
        with self._lock:
            candidate = deepcopy(self._raw)
            section = candidate.setdefault(str(name), {})
            if not isinstance(section, dict):
                raise ConfigValidationError(f"{name} must be an object")
            section.update(deepcopy(updates or {}))
            validate_config(candidate)
            self._raw = candidate
            return deepcopy(self._raw)

    @contextmanager
    def temporary_section(self, name: str, updates: dict):
        with self._lock:
            original = deepcopy(self._raw)
            candidate = deepcopy(self._raw)
            candidate.setdefault(str(name), {}).update(deepcopy(updates or {}))
            validate_config(candidate)
            self._raw = candidate
            try:
                yield
            finally:
                self._raw = original

    @property
    def WIN_W(self) -> int: return self._get("window", "width", default=1500)
    @property
    def WIN_H(self) -> int: return self._get("window", "height", default=900)
    @property
    def UI_HEIGHT(self) -> int: return 96
    @property
    def PANEL_W(self) -> int: return 320
    @property
    def CANVAS_TOOLBAR_H(self) -> int: return 76
    @property
    def CANVAS_W(self) -> int: return self.WIN_W - self.PANEL_W
    @property
    def CANVAS_H(self) -> int:
        return self.WIN_H - self.UI_HEIGHT - self.CANVAS_TOOLBAR_H
    @property
    def INFO_BAR_H(self) -> int: return 120   # 左侧面板底部信息枢纽高度
    @property
    def TITLE(self) -> str: return self._get("window", "title", default="智能农机规划系统")

    @property
    def CV2_WIN_ID(self) -> str:
        """cv2 内部窗口标识（必须为 ASCII，用于 namedWindow/setMouseCallback/imshow 匹配）。
        中文标题会导致 Windows 上 OpenCV 回调注册失败，因此内部用英文 ID，
        创建窗口后通过 Win32 SetWindowTextW 修改标题栏显示为中文。"""
        return "HarvesterPlanner"

    @property
    def MODEL_CONF(self) -> float: return self._get("model", "conf_threshold", default=0.25)
    @property
    def MODEL_IOU(self) -> float: return self._get("model", "iou_threshold", default=0.45)
    @property
    def TILE_SIZE(self) -> int: return self._get("model", "tile_size", default=640)
    @property
    def TILE_OVERLAP(self) -> float: return self._get("model", "tile_overlap", default=0.4)
    @property
    def TILE_CAPTURE_SIZE(self) -> int:
        return self._get("model", "tile_capture_size", default=640)

    @property
    def ERODE_KERNEL_SIZE(self) -> int:
        return self._get("model", "erode_kernel_size", default=7)

    @property
    def ERODE_ITERATIONS(self) -> int:
        return self._get("model", "erode_iterations", default=1)

    @property
    def MIN_BAND_AREA_PX(self) -> int: return self._get("path_planning", "min_band_area_px", default=100)
    @property
    def BAND_ASPECT_RATIO_MIN(self) -> float: return self._get("path_planning", "band_aspect_ratio_min", default=3.0)
    @property
    def MORPH_CLOSE_KERNEL(self) -> tuple:
        w = self._get("path_planning", "morph_close_kernel_w", default=25)
        h = self._get("path_planning", "morph_close_kernel_h", default=5)
        return (w, h)
    @property
    def PATH_POINT_INTERVAL_M(self) -> float: return self._get("path_planning", "path_point_interval_m", default=0.5)
    @property
    def HEADLAND_BUFFER_M(self) -> float: return self._get("path_planning", "headland_buffer_m", default=3.0)
    @property
    def MIN_CORRIDOR_WIDTH_M(self) -> float: return self._get("path_planning", "min_corridor_width_m", default=0.8)
    @property
    def MIN_TURN_RADIUS_M(self) -> float: return self._get("path_planning", "min_turn_radius_m", default=1.0)

    @property
    def HEADLAND_ANGLE_THRESH_DEG(self) -> float: return self._get("mask_processing", "headland_angle_thresh_deg", default=60.0)
    @property
    def HEADLAND_MIN_AREA_RATIO(self) -> float: return self._get("mask_processing", "headland_min_area_ratio", default=0.005)
    @property
    def AGGREGATE_CLOSE_KERNEL(self) -> int: return self._get("mask_processing", "aggregate_close_kernel", default=15)
    @property
    def ADHESION_THIN_KERNEL(self) -> int: return self._get("mask_processing", "adhesion_thin_kernel", default=12)
    @property
    def ADHESION_USE_DIST_TRANSFORM(self) -> bool: return self._get("mask_processing", "adhesion_use_distance_transform", default=True)
    @property
    def ADHESION_DIST_SPLIT_RATIO(self) -> float: return self._get("mask_processing", "adhesion_distance_split_ratio", default=0.3)
    @property
    def BAND_WIDTH_THRESHOLD_M(self) -> float: return self._get("mask_processing", "band_width_threshold_m", default=0.55)
    @property
    def BAND_AUTO_CALIBRATE(self) -> bool: return self._get("mask_processing", "band_auto_calibrate", default=True)
    @property
    def MASK_PROCESSING_STRENGTH(self) -> str: return self._get("mask_processing", "strength", default="standard")

    @property
    def CUTTER_WIDTH_M(self) -> float: return self._get("harvester", "cutter_width_m", default=2.0)
    @property
    def TRACK_WIDTH_M(self) -> float: return self._get("harvester", "track_width_m", default=0.35)
    @property
    def TRACK_GAUGE_M(self) -> float: return self._get("harvester", "track_gauge_m", default=1.7)
    @property
    def WHEELBASE_M(self) -> float: return self._get("harvester", "wheelbase_m", default=2.5)
    @property
    def TRACK_LENGTH_M(self) -> float: return self._get("harvester", "track_length_m", default=1.5)
    @property
    def TURN_RADIUS_M(self) -> float: return self._get("harvester", "turn_radius_m", default=2.0)

    @property
    def DEFAULT_EPSG(self) -> str: return self._get("geo", "default_epsg", default="32650")
    @property
    def MASK_OPACITY(self) -> float: return self._get("ui", "mask_opacity", default=0.3)

    @property
    def ROUTE_ID(self) -> int: return self._get("export", "route_id", default=0)
    @property
    def OFFSET_X(self) -> str: return self._get("export", "offset_x", default="0.00")
    @property
    def OFFSET_Y(self) -> str: return self._get("export", "offset_y", default="0.00")
    @property
    def INFM_LINE(self) -> str: return self._get("export", "infoline", default="$INFM,4,0.7,0.7,0.7,0.7,1,15,254*")


class AppLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance._status_callback = None
        return cls._instance

    def init(self, log_dir: str = ""):
        if self._initialized:
            return
        if not log_dir:
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._cleanup_old_logs(log_dir, max_keep=5)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"app_log_{ts}.log")

        self._logger = logging.getLogger("HarvesterPro")
        self._logger.setLevel(logging.DEBUG)

        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024,
                                 backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self._logger.addHandler(ch)

        self._status_callback = None
        self._initialized = True
        self.info(f"日志文件: {log_path}")

    @staticmethod
    def _cleanup_old_logs(log_dir: str, max_keep: int = 5):
        """清理旧日志文件，仅保留最近 max_keep 个。"""
        pattern = os.path.join(log_dir, "v*_log_*.log")
        log_files = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for old_file in log_files[max_keep:]:
            try:
                os.remove(old_file)
            except OSError:
                pass

    def set_status_callback(self, cb):
        self._status_callback = cb

    def _log(self, level: int, msg: str, event_type: str = "operation"):
        try:
            if self._initialized:
                self._logger.log(level, msg)
            else:
                print(msg)
        except Exception:
            pass  # 日志写入失败不应影响主流程
        try:
            from project_log import ProjectJournal
            if level >= logging.ERROR:
                ProjectJournal().error(msg, module="application")
            else:
                ProjectJournal().write(
                    logging.getLevelName(level),
                    event_type,
                    msg,
                    module="application",
                )
        except Exception:
            pass
        if self._status_callback and level >= logging.WARNING:
            self._status_callback(msg)

    def debug(self, msg: str): self._log(logging.DEBUG, msg)
    def info(self, msg: str): self._log(logging.INFO, msg)
    def change(self, msg: str): self._log(logging.INFO, msg, event_type="change")
    def warning(self, msg: str): self._log(logging.WARNING, msg)
    def error(self, msg: str): self._log(logging.ERROR, msg)
