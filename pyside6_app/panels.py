import math
import time
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QScrollArea, QFrame, QTextEdit, QToolButton
)
from pyside6_app.styles import COLORS
from pyside6_app.task_state import TASK_ORDER

UI_FONT = '"Microsoft YaHei UI","Microsoft YaHei","SimHei","Segoe UI",sans-serif'


TASK_DEFS = [
    {"id": "import_img", "label": "影像导入", "hint": "加载正射影像 GeoTIFF 文件。", "color": "#6FA37F", "action": "导入"},
    {"id": "draw_field", "label": "田块圈选", "hint": "在影像上圈选田块范围。", "color": "#5EAF6E", "action": "圈选"},
    {"id": "segment", "label": "AI识别", "hint": "使用模型分割作物行带。", "color": "#649C7E", "action": "运行"},
    {"id": "process", "label": "掩膜处理", "hint": "优化 AI 识别掩膜并提取作物行带。", "color": "#648C82", "action": "处理"},
    {"id": "params", "label": "农机参数", "hint": "设置农机参数。", "color": "#AC7E56", "action": "设置"},
    {"id": "entry", "label": "起点", "hint": "布设或确认机具入田起点。", "color": "#589C94", "action": "布设"},
    {"id": "exit", "label": "终点", "hint": "布设或确认机具离田终点。", "color": "#C76B6B", "action": "布设"},
    {"id": "unload", "label": "卸粮点", "hint": "按需要添加一个或多个卸粮点，可为空。", "color": "#C9A64A", "action": "布设"},
    {"id": "plan", "label": "路径规划", "hint": "生成作业路线。", "color": "#589862", "action": "生成"},
    {"id": "simulate", "label": "模拟演示", "hint": "沿规划路径模拟农机运动。", "color": "#4C8C7C", "action": "演示"},
    {"id": "export", "label": "路径导出", "hint": "导出路径结果。", "color": "#4484BE", "action": "导出"},
]

STEP_GROUPS = [
    {"label": "影像导入", "tasks": ["import_img"]},
    {"label": "田块圈选", "tasks": ["draw_field"]},
    {"label": "AI识别", "tasks": ["segment"]},
    {"label": "掩膜处理", "tasks": ["process"]},
    {"label": "参数配置", "tasks": ["params", "entry", "exit", "unload"]},
    {"label": "路径规划", "tasks": ["plan"]},
    {"label": "模拟演示", "tasks": ["simulate"]},
    {"label": "路径导出", "tasks": ["export"]},
]


def _task_def(tid):
    return next((item for item in TASK_DEFS if item["id"] == tid), None)


class TaskRow(QWidget):
    clicked = Signal(str)
    delete_clicked = Signal(str)

    def __init__(self, task_def, parent=None):
        super().__init__(parent)
        self.setObjectName("taskRow")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(48)
        self.setStyleSheet("#taskRow{background:transparent;}")
        self.tid = task_def["id"]
        self._color = task_def["color"]
        self._done = False
        self._status_text = "待处理"
        lo = QHBoxLayout(self)
        lo.setContentsMargins(8, 6, 8, 6)
        lo.setSpacing(8)
        self._badge = QLabel("")
        self._badge.setFixedSize(14, 14)
        self._badge.setStyleSheet(f"background:{task_def['color']};border-radius:3px;")
        lo.addWidget(self._badge)
        self._label = QLabel(task_def["label"])
        self._label.setFont(QFont("Microsoft YaHei UI", 10))
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        lo.addWidget(self._label, 1)
        self._status = QLabel(self._status_text)
        self._status.setFont(QFont("Microsoft YaHei UI", 9))
        self._status.setStyleSheet(f"color:{COLORS['text_dim']};font-size:11px;font-family:{UI_FONT};")
        lo.addWidget(self._status)
        self._action_btn = QPushButton(task_def["action"])
        self._action_btn.setFixedSize(48, 36)
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{task_def['color']};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
        )
        self._action_btn.clicked.connect(lambda: self.clicked.emit(self.tid))
        lo.addWidget(self._action_btn)
        self._delete_btn = QPushButton("回退")
        self._delete_btn.setFixedSize(48, 36)
        self._delete_btn.setToolTip("回退到此步骤之前，并清除后续结果")
        self._delete_btn.setStyleSheet(
            "QPushButton{background:#7A4D3B;color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            "QPushButton:hover{background:#9A654D;}"
        )
        self._delete_btn.clicked.connect(lambda: self.delete_clicked.emit(self.tid))
        lo.addWidget(self._delete_btn)

    def set_status(self, done=False, enabled=True, in_progress=False, progress_pct=-1, text=None, status="pending"):
        self._done = bool(done)
        self._in_progress = bool(in_progress)
        self._status_key = status
        self.setEnabled(bool(enabled))
        self._action_btn.setEnabled(bool(enabled))
        if text:
            display_text = str(text)
        elif done:
            display_text = "已完成"
        elif in_progress:
            display_text = f"处理中 {progress_pct}%" if progress_pct >= 0 else "处理中"
        elif status == "available":
            display_text = "可操作"
        elif status == "failed":
            display_text = "失败"
        else:
            display_text = "待处理"
        if done:
            color = COLORS.get('green', '#64D26A')
        elif in_progress or status == "available":
            color = COLORS.get('accent', '#58A6FF')
        elif status == "failed":
            color = COLORS.get('red', '#E85D63')
        else:
            color = COLORS.get('text_dim', '#8B909A')
        self._status.setText(display_text)
        self._status.setStyleSheet(f"color:{color};font-size:11px;font-family:{UI_FONT};")
        opacity = "1.0" if enabled else "0.55"
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{self._color};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self.setStyleSheet(f"#taskRow{{background:transparent;opacity:{opacity};}}")

    def refresh_theme(self):
        status = getattr(self, "_status_key", "pending")
        in_progress = getattr(self, "_in_progress", False)
        if self._done:
            color = COLORS.get('green', '#64D26A')
        elif in_progress or status == "available":
            color = COLORS.get('accent', '#58A6FF')
        elif status == "failed":
            color = COLORS.get('red', '#E85D63')
        else:
            color = COLORS.get('text_dim', '#8B909A')
        opacity = "1.0" if self.isEnabled() else "0.55"
        self._status.setStyleSheet(f"color:{color};font-size:11px;font-family:{UI_FONT};")
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-family:{UI_FONT};")
        self._action_btn.setStyleSheet(
            f"QPushButton{{background:{self._color};color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            f"QPushButton:disabled{{background:{COLORS.get('bg_light', '#3A3A44')};color:{COLORS.get('text_dim', '#8B909A')};}}"
        )
        self._delete_btn.setStyleSheet(
            "QPushButton{background:#7A4D3B;color:white;border:none;"
            f"border-radius:6px;font-size:12px;font-weight:600;padding:0;font-family:{UI_FONT};}}"
            "QPushButton:hover{background:#9A654D;}"
        )
        self.setStyleSheet(f"#taskRow{{background:transparent;opacity:{opacity};}}")


class StepCard(QWidget):
    toggled = Signal()

    def __init__(self, title, rows, parent=None):
        super().__init__(parent)
        self.setObjectName("stepCard")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setMinimumHeight(38)
        self._rows = rows
        self._expanded = False
        lo = QVBoxLayout(self)
        lo.setContentsMargins(7, 6, 7, 6)
        lo.setSpacing(5)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self._toggle = QToolButton(); self._toggle.setText("▸"); self._toggle.setFixedSize(24, 28)
        self._title = QLabel(title)
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._title.setFont(QFont("Microsoft YaHei UI", 10))
        self._title.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-weight:600;font-family:{UI_FONT};")
        self._mark = QLabel("")
        self._mark.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._mark.setFixedSize(24, 28)
        self._mark.setAlignment(Qt.AlignCenter)
        head.addWidget(self._toggle); head.addWidget(self._title, 1); head.addWidget(self._mark)
        lo.addLayout(head)
        self._content = QWidget()
        self._content.setAttribute(Qt.WA_StyledBackground, True)
        self._content.setStyleSheet("background:transparent;")
        cl = QVBoxLayout(self._content); cl.setContentsMargins(10, 2, 0, 0); cl.setSpacing(2)
        for row in rows: cl.addWidget(row)
        lo.addWidget(self._content)
        self._toggle.clicked.connect(self._toggle_expanded)
        self.set_expanded(False)
        self.refresh_state()

    def _toggle_expanded(self):
        self.set_expanded(not self._expanded)
        self.toggled.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().y() <= 40:
            self._toggle_expanded()
            event.accept()
            return
        super().mousePressEvent(event)

    def set_expanded(self, expanded):
        self._expanded = bool(expanded)
        self._content.setVisible(self._expanded)
        self._toggle.setText("▾" if self._expanded else "▸")

    def refresh_state(self):
        task_rows = [row for row in self._rows if hasattr(row, "_done")]
        done = bool(task_rows) and all(row._done for row in task_rows)
        self._title.setStyleSheet(f"color:{COLORS['text']};font-size:13px;font-weight:600;font-family:{UI_FONT};")
        if done:
            bg = COLORS.get("success_bg", "#1A3A1A")
            border = COLORS.get("green", "#6CCB6E")
            self._mark.setText("✓")
            self._mark.setStyleSheet(f"color:{border};font-weight:bold;font-size:14px;")
        else:
            bg = COLORS.get("bg_card", "#2A2A35")
            border = COLORS.get("border", "#383848")
            self._mark.setText("")
        self.setStyleSheet(
            f"#stepCard{{background:{bg};border:1px solid {border};"
            "border-radius:7px;margin:1px 0;}}"
            f"#stepCard QLabel{{font-family:{UI_FONT};}}"
            f"QToolButton{{background:transparent;color:{COLORS['text_dim']};border:none;font-family:{UI_FONT};}}"
            f"QToolButton:hover{{background:{COLORS['bg_hover']};border-radius:4px;}}"
        )


class TaskPanel(QWidget):
    load_tif_requested = Signal()
    load_model_requested = Signal()
    run_requested = Signal()
    segment_requested = Signal()
    process_requested = Signal()
    plan_requested = Signal()
    simulate_requested = Signal()
    params_requested = Signal()
    entry_exit_toggle = Signal()
    export_geojson_requested = Signal()
    export_csv_requested = Signal()
    export_img_requested = Signal()
    field_drawing_requested = Signal()
    model_changed = Signal(str)
    mask_strength_changed = Signal(str)
    turn_strategy_changed = Signal(str)
    route_edit_requested = Signal()
    step_deleted = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task_widgets = {}
        self._step_cards = []
        self._current_step = 0
        self._manual_step = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        lo = QVBoxLayout(self); lo.setContentsMargins(6, 8, 6, 8); lo.setSpacing(6)
        self._title = QLabel("任务清单")
        self._title.setFont(QFont("Microsoft YaHei UI", 12, QFont.Weight.DemiBold))
        self._title.setStyleSheet(f"font-size:16px;font-weight:600;color:{COLORS['text_bright']};font-family:{UI_FONT};")
        lo.addWidget(self._title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self._scroll = scroll
        body = QWidget(); bl = QVBoxLayout(body); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(4)
        body.setStyleSheet("background:transparent;")
        for idx, group in enumerate(STEP_GROUPS, start=1):
            rows = []
            for tid in group["tasks"]:
                td = _task_def(tid); row = TaskRow(td); self._task_widgets[tid] = row
                row.clicked.connect(self._on_task_clicked); row.delete_clicked.connect(self.step_deleted.emit)
                rows.append(row)
                if tid == "process":
                    self._mask_strength_label = QLabel("处理强度")
                    self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};padding-left:8px;")
                    self.mask_strength_combo = QComboBox()
                    self.mask_strength_combo.setToolTip("根据作物断行、粘连和噪声情况调节掩膜几何重建强度")
                    self.mask_strength_combo.setMinimumHeight(30)
                    self.mask_strength_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
                    )
                    for label, key in [
                        ("轻量/快速", "light"),
                        ("标准", "standard"),
                        ("增强", "strong"),
                        ("强力", "very_strong"),
                    ]:
                        self.mask_strength_combo.addItem(label, key)
                    self.mask_strength_combo.currentIndexChanged.connect(self._on_mask_strength_changed)
                    rows.append(self._mask_strength_label)
                    rows.append(self.mask_strength_combo)
                if tid == "segment":
                    self.model_combo = QComboBox(); self.model_combo.currentTextChanged.connect(self.model_changed.emit)
                    self.model_combo.setMinimumHeight(30)
                    self.model_combo.setToolTip("选择 AI 识别模型，或从文件夹导入其他 .pt 权重")
                    self.model_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
                    )
                    rows.append(self.model_combo)
                if tid == "plan":
                    turn_label = QLabel("调头方式")
                    turn_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};padding-left:8px;")
                    self.turn_combo = QComboBox()
                    self.turn_combo.setToolTip("选择路径规划使用的调头/转弯方式，可重算对比不同效果")
                    self.turn_combo.setMinimumHeight(30)
                    self.turn_combo.setStyleSheet(
                        f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
                    )
                    for label, key in [
                        ("自动推荐", "auto"), ("弓形调头", "bow"), ("半圆调头", "semicircle"),
                        ("梨形调头", "pear"), ("鱼尾折返", "fishtail"), ("紧凑 Alpha", "alpha")
                    ]:
                        self.turn_combo.addItem(label, key)
                    self.turn_combo.currentIndexChanged.connect(self._on_turn_combo_changed)
                    rows.append(turn_label)
                    rows.append(self.turn_combo)
                    self.route_edit_btn = QPushButton("编辑路线")
                    self.route_edit_btn.setToolTip("进入人工路线编辑：Alt+左键在线段插点，拖动控制点，Backspace 删除选中点")
                    self.route_edit_btn.setMinimumHeight(32)
                    self.route_edit_btn.setStyleSheet(
                        f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                        f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;font-family:{UI_FONT};}}"
                        f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                        f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
                    )
                    self.route_edit_btn.clicked.connect(self.route_edit_requested.emit)
                    rows.append(self.route_edit_btn)
            card = StepCard(f"{idx}. {group['label']}", rows)
            card.toggled.connect(lambda _checked=False, i=idx - 1: self._on_card_toggled(i))
            self._step_cards.append(card); bl.addWidget(card)
        bl.addStretch(); scroll.setWidget(body); lo.addWidget(scroll, 1)
        self._refresh_cards()

    def _on_task_clicked(self, tid):
        mapping = {
            "import_img": self.load_tif_requested,
            "draw_field": self.field_drawing_requested,
            "segment": self.segment_requested,
            "process": self.process_requested,
            "params": self.params_requested,
            "entry": self.entry_exit_toggle,
            "exit": self.entry_exit_toggle,
            "unload": self.entry_exit_toggle,
            "plan": self.plan_requested,
            "simulate": self.simulate_requested,
            "export": self.export_geojson_requested,
        }
        sig = mapping.get(tid)
        if sig: sig.emit()

    def _refresh_cards(self):
        first = 0
        for i, group in enumerate(STEP_GROUPS):
            if any(not self._task_widgets[tid]._done for tid in group["tasks"]):
                first = i; break
        expanded = self._manual_step if self._manual_step is not None else first
        for i, card in enumerate(self._step_cards):
            card.refresh_state()
            # Accordion behaviour: only one card owns vertical space at a time.
            # Keeping both the current workflow card and a manually opened card
            # expanded can make the two content areas collide/cover each other in
            # the narrow left splitter, especially after project restore.
            card.set_expanded(i == expanded)
        self._current_step = first

    def _on_card_toggled(self, index):
        card = self._step_cards[index]
        if card._expanded:
            self._manual_step = None if index == self._current_step else index
            for i, other in enumerate(self._step_cards):
                if i != index:
                    other.set_expanded(False)
            self._ensure_card_visible(index)
        else:
            self._manual_step = None
            self._refresh_cards()

    def _ensure_card_visible(self, index):
        """Scroll the selected accordion card into view after expansion."""
        scroll = getattr(self, "_scroll", None)
        if not scroll or not (0 <= index < len(self._step_cards)):
            return
        scroll.ensureWidgetVisible(self._step_cards[index], 0, 8)

    def set_task_status(self, tid, status, progress_pct=-1, text=None):
        row = self._task_widgets.get(tid)
        if not row: return
        row.set_status(
            done=status == "done",
            enabled=status in ("available", "in_progress", "done", "failed"),
            in_progress=status == "in_progress",
            progress_pct=progress_pct,
            text=text,
            status=status,
        )
        self._refresh_cards()

    def apply_task_statuses(self, statuses):
        for tid in TASK_ORDER:
            st = statuses.get(tid)
            if not st:
                continue
            self.set_task_status(tid, st.status, text=st.text)
            if st.action:
                self.set_task_action(tid, st.action)
            else:
                task_def = _task_def(tid)
                if task_def:
                    self.set_task_action(tid, task_def["action"])

    def set_task_action(self, tid, text):
        row = self._task_widgets.get(tid)
        if row: row._action_btn.setText(str(text))


    def _on_turn_combo_changed(self, index):
        if not hasattr(self, 'turn_combo'):
            return
        key = self.turn_combo.itemData(index) or "auto"
        self.turn_strategy_changed.emit(str(key))

    def set_turn_strategy(self, key):
        if not hasattr(self, 'turn_combo'):
            return
        for i in range(self.turn_combo.count()):
            if self.turn_combo.itemData(i) == key:
                self.turn_combo.blockSignals(True)
                self.turn_combo.setCurrentIndex(i)
                self.turn_combo.blockSignals(False)
                return

    def _on_mask_strength_changed(self, index):
        if not hasattr(self, 'mask_strength_combo'):
            return
        key = self.mask_strength_combo.itemData(index) or "standard"
        self.mask_strength_changed.emit(str(key))

    def set_mask_strength(self, key):
        if not hasattr(self, 'mask_strength_combo'):
            return
        key = str(key or "standard")
        for i in range(self.mask_strength_combo.count()):
            if self.mask_strength_combo.itemData(i) == key:
                self.mask_strength_combo.blockSignals(True)
                self.mask_strength_combo.setCurrentIndex(i)
                self.mask_strength_combo.blockSignals(False)
                return

    def set_model_options(self, models, current=""):
        if not hasattr(self, 'model_combo'): return
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models or []:
            name = model.get("name") or model.get("path") or str(model)
            path = model.get("path", name) if isinstance(model, dict) else str(model)
            self.model_combo.addItem(name, path)
            if path == current or name == current:
                self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self.model_combo.addItem("导入模型文件夹...", "__IMPORT_MODEL_FOLDER__")
        self.model_combo.blockSignals(False)

    def populate_models(self, models, current=""):
        normalized = [{"name": str(item), "path": str(item)} for item in (models or [])]
        self.set_model_options(normalized, current)

    def set_model_path(self, name):
        if hasattr(self, 'model_combo'):
            idx = self.model_combo.findText(name)
            if idx >= 0: self.model_combo.setCurrentIndex(idx)

    def refresh_theme(self):
        self.setStyleSheet(f"background:{COLORS['bg']};")
        self._title.setStyleSheet(f"font-size:16px;font-weight:600;color:{COLORS['text_bright']};font-family:{UI_FONT};")
        combo_style = (
            f"QComboBox{{background:{COLORS['bg_light']};color:{COLORS['text']};"
            f"border:1px solid {COLORS['border']};border-radius:4px;padding:4px 8px;font-family:{UI_FONT};}}"
        )
        if hasattr(self, "_mask_strength_label"):
            self._mask_strength_label.setStyleSheet(f"color:{COLORS['text_dim']};font-size:12px;font-family:{UI_FONT};padding-left:8px;")
        for combo_name in ("model_combo", "turn_combo", "mask_strength_combo"):
            combo = getattr(self, combo_name, None)
            if combo:
                combo.setStyleSheet(combo_style)
        if hasattr(self, "route_edit_btn"):
            self.route_edit_btn.setStyleSheet(
                f"QPushButton{{background:{COLORS['bg_light']};color:{COLORS['text']};"
                f"border:1px solid {COLORS['border']};border-radius:5px;padding:4px 8px;font-family:{UI_FONT};}}"
                f"QPushButton:hover{{background:{COLORS['bg_hover']};}}"
                f"QPushButton:disabled{{color:{COLORS.get('text_dim', '#8B909A')};}}"
            )
        for row in self.findChildren(TaskRow):
            row.refresh_theme()
        for card in self._step_cards:
            card.refresh_state()


class RouteSegmentRow(QWidget):
    clicked = Signal(int)
    def __init__(self, index, label, distance_m, is_work=True, parent=None):
        super().__init__(parent); self._index = index; self._is_work = is_work
        lo = QHBoxLayout(self); lo.setContentsMargins(8, 3, 8, 3); lo.setSpacing(8)
        self._badge = QLabel("线" if is_work else "转"); self._badge.setFixedSize(24, 18); self._badge.setAlignment(Qt.AlignCenter)
        self._label = QLabel(f"{label} {index + 1:02d}")
        self._dist = QLabel(f"{distance_m:.1f} m")
        lo.addWidget(self._badge)
        lo.addWidget(self._label, 1)
        lo.addWidget(self._dist)
        self.refresh_theme()
    def refresh_theme(self):
        self._badge.setStyleSheet(f"background:{'#529C62' if self._is_work else '#BA9648'};color:white;border-radius:2px;font-size:10px;")
        self._label.setStyleSheet(f"color:{COLORS['text']};font-size:12px;")
        self._dist.setStyleSheet(f"color:{COLORS['text_dim']};font-family:Consolas;")
    def mousePressEvent(self, event):
        self.clicked.emit(self._index); super().mousePressEvent(event)


class RouteInfoPanel(QWidget):
    segment_selected = Signal(int)
    service_point_selected = Signal(str, int)
    def __init__(self, parent=None):
        super().__init__(parent); self._segments = []; self._build_ui()
    def _build_ui(self):
        lo = QVBoxLayout(self); lo.setContentsMargins(0,0,0,0); lo.setSpacing(0)
        self._title_hdr = QLabel("路径结果"); self._title_hdr.setStyleSheet(f"font-size:13px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary = QLabel("作业线 0 / 转弯段 0")
        self._summary.setWordWrap(True)
        self._summary.setMinimumWidth(0)
        self._summary.setStyleSheet(f"font-size:11px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        lo.addWidget(self._title_hdr); lo.addWidget(self._summary)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame)
        self._list_container = QWidget(); self._list_layout = QVBoxLayout(self._list_container); self._list_layout.setContentsMargins(8,0,8,0); self._list_layout.setSpacing(2)
        self._empty_label = QLabel("尚无路径结果。"); self._empty_label.setStyleSheet(f"font-size:12px;color:{COLORS['text_dimmer']};padding:8px;")
        self._list_layout.addWidget(self._empty_label); self._list_layout.addStretch(); sc.setWidget(self._list_container); lo.addWidget(sc,1)
    def _clear_list(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0); w = item.widget()
            if w and w != self._empty_label: w.deleteLater()
        self._segments.clear(); self._list_layout.addStretch()
    def _add_row(self, row):
        self._list_layout.insertWidget(max(0, self._list_layout.count()-1), row)
    def update_route(self, data, validation=None):
        data = data or {}
        tracks = data.get("tracks", []); turns = data.get("turns", [])
        validation = validation or data.get("validation", {}) or {}
        self._clear_list(); self._empty_label.setVisible(not (tracks or turns))
        idx = 0
        for t in tracks:
            row = RouteSegmentRow(idx, "作业线", float(t.get("length", self._calc_distance(t.get("points", []))) or 0.0), True); row.clicked.connect(self.segment_selected.emit); self._add_row(row); idx += 1
        for t in turns:
            row = RouteSegmentRow(idx, "转弯段", float(t.get("length", self._calc_distance(t.get("points", []))) or 0.0), False); row.clicked.connect(self.segment_selected.emit); self._add_row(row); idx += 1
        parts = [f"作业线 {len(tracks)}", f"转弯段 {len(turns)}"]
        total = float(validation.get("total_length_m", data.get("total_length_m", 0.0)) or 0.0)
        work = float(validation.get("work_length_m", data.get("work_length_m", 0.0)) or 0.0)
        turn = float(validation.get("turn_length_m", data.get("turn_length_m", 0.0)) or 0.0)
        coverage = validation.get("harvest_coverage_pct", validation.get("planned_target_coverage_pct"))
        rolling = validation.get("track_core_overlap_pct", validation.get("rolling_crop_pct"))
        efficiency = validation.get("field_efficiency_pct", validation.get("field_efficiency"))
        if total > 0: parts.append(f"总长 {total:.1f}m")
        if work > 0: parts.append(f"作业 {work:.1f}m")
        if turn > 0: parts.append(f"转弯 {turn:.1f}m")
        if coverage is not None: parts.append(f"覆盖 {float(coverage):.1f}%")
        if rolling is not None: parts.append(f"碾压 {float(rolling):.1f}%")
        if efficiency is not None: parts.append(f"效率 {float(efficiency):.1f}%")
        issues = validation.get("issues") or []
        if issues: parts.append(f"风险 {len(issues)}处")
        self._summary.setToolTip(" / ".join(parts))
        if len(parts) > 4:
            self._summary.setText(" / ".join(parts[:4]) + f" / … 共{len(parts)}项")
        else:
            self._summary.setText(" / ".join(parts))
    def update_from_auto_path(self, auto_path):
        self._clear_list(); self._empty_label.setVisible(not bool(auto_path))
        work = turn = 0; total = work_len = turn_len = 0.0
        for idx, seg in enumerate(auto_path or []):
            typ = getattr(seg, 'segment_type', 'work'); is_work = typ == 'work'; work += int(is_work); turn += int(not is_work)
            length = float(getattr(seg, 'length_m', 0.0) or 0.0); total += length
            if is_work: work_len += length
            else: turn_len += length
            row = RouteSegmentRow(idx, "作业线" if is_work else "转弯段", length, is_work)
            row.clicked.connect(self.segment_selected.emit); self._add_row(row)
        parts = [f"作业线 {work}", f"转弯段 {turn}"]
        if total > 0: parts.append(f"总长 {total:.1f}m")
        if work_len > 0: parts.append(f"作业 {work_len:.1f}m")
        if turn_len > 0: parts.append(f"转弯 {turn_len:.1f}m")
        self._summary.setText(" / ".join(parts))
    def update_service_points(self, entry_point=None, exit_point=None, unload_points=None):
        self._clear_list(); items=[]
        if entry_point: items.append(("entry",0,"起点",True))
        if exit_point: items.append(("exit",0,"终点",False))
        for i,_ in enumerate(unload_points or []): items.append(("unload",i,f"卸粮点 {i+1}",False))
        self._empty_label.setVisible(not items)
        for row_idx,(kind,point_idx,label,is_work) in enumerate(items):
            row=RouteSegmentRow(row_idx,label,0.0,is_work); row.clicked.connect(lambda _r,k=kind,i=point_idx: self.service_point_selected.emit(k,i)); self._add_row(row)
        self._summary.setText(f"起终点与卸粮点 {len(items)}")
    def clear_info(self):
        self._clear_list(); self._empty_label.setVisible(True); self._summary.setText("作业线 0 / 转弯段 0")
    def refresh_theme(self):
        self._title_hdr.setStyleSheet(f"font-size:13px;font-weight:600;color:{COLORS['text']};padding:8px 14px 2px;")
        self._summary.setStyleSheet(f"font-size:11px;color:{COLORS['text_dim']};padding:2px 14px 4px;")
        self._empty_label.setStyleSheet(f"font-size:12px;color:{COLORS['text_dimmer']};padding:8px;")
        for row in self.findChildren(RouteSegmentRow):
            row.refresh_theme()
    @staticmethod
    def _calc_distance(points):
        pts = [(float(p[0]), float(p[1])) for p in points or [] if len(p) >= 2]
        return sum(math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) for i in range(1,len(pts)))


class LogPanel(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent); self.setReadOnly(True); self.setMinimumHeight(120)
    def _append(self, level, msg):
        ts = time.strftime("%H:%M:%S")
        self.append(f"[{ts}] {msg}")
    def info(self, m): self._append("INFO", m)
    def success(self, m): self._append("SUCCESS", m)
    def warn(self, m): self._append("WARN", m)
    def error(self, m): self._append("ERROR", m)


class ParamPanel(QWidget):
    params_changed = Signal(); load_tif_requested = Signal(); load_model_requested = Signal(); run_requested = Signal()
    export_geojson_requested = Signal(); export_csv_requested = Signal(); export_img_requested = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tif_path = ""
        self.model_path = ""
    def set_tif_path(self, p):
        self.tif_path = str(p or "")
    def set_model_path(self, p):
        self.model_path = str(p or "")


class ToolPanel(QWidget):
    tool_changed = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = ""
    def set_tool(self, tool):
        self.current_tool = str(tool or "")
        self.tool_changed.emit(self.current_tool)


class StatusSteps(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
    def set_current(self, idx):
        self.current_index = int(idx or 0)
