import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from affine import Affine
from PySide6.QtWidgets import QApplication

from pyside6_app.main_window import MainWindow
from pyside6_app.image_view import MetricScaleBar


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._tif_rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    return window


def test_path_result_is_normalised_and_stored_for_current_ui():
    window = _window()
    path_result = {
        "full_path": [
            [(0, 0), (10, 0)],
            [(10, 0), (10, 5)],
            [(10, 5), (0, 5)],
        ],
        "segment_types": ["work", "turn", "work"],
        "geo_points": [{"segment_index": 0, "point_index": 0, "x": 0, "y": 0}],
        "validation": {"valid": True, "total_length_m": 15},
        "is_valid": True,
        "description": "ok",
    }

    normalised = window._store_path_result(path_result)

    assert len(normalised["tracks"]) == 2
    assert len(normalised["turns"]) == 1
    assert window.state.auto_path_planned is True
    assert len(window.state.auto_path) == 3
    assert len(window.state.path_points) == 4
    assert window._pipeline_result["path"]["tracks"]
    assert all(hasattr(p, "pixel_x") for p in window.state.auto_path[0].points)
    assert window._build_anim_segments(normalised, window.state.auto_path_segments) is window.state.auto_path_segments


def test_pipeline_step_results_update_state():
    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    processed = {
        "processed_mask": raw,
        "main_angle": 0.0,
        "wide_bands": [{"centerline": [(0, 0), (1, 1)]}],
    }

    window._on_step_result("segment_done", raw)
    assert window.state.inference_done is True
    assert window.state.mask_raw is raw

    window._on_step_result("process_done", processed)
    assert window.state.mask_processed is True
    assert window.state.mask_result is processed


def test_pending_segment_completion_enables_mask_process_action():
    window = _window()
    raw = np.ones((20, 20), dtype=np.uint8) * 255
    window._pending_segment_display = True
    window.state.safe_update(
        tif_path="D:/zhlonly/project/reUI/data/tif/result.tif",
        field_boundary=[(1, 1), (10, 1), (10, 10)],
        mask_raw=raw,
        inference_original_mask=raw,
        inference_done=True,
        inference_running=False,
    )

    window._refresh_state()

    process_row = window.task_panel._task_widgets["process"]
    assert process_row.isEnabled()
    assert process_row._action_btn.isEnabled()
    assert process_row._status.text() == "可操作"
    assert process_row._action_btn.text() == "处理"


def test_field_overlay_uses_viewer_instance():
    window = _window()
    window._field_pts = [(1, 1), (10, 1)]
    window._redraw_field_overlay(mouse_pos=(10, 10))

    window.state.safe_update(field_boundary=[(1, 1), (10, 1), (10, 10)])
    window._draw_field_boundary()


def test_auto_start_does_not_load_bundled_tif_without_user_choice(monkeypatch):
    window = _window()
    loaded = []

    class RejectingLauncher:
        file_selected = ""

        def __init__(self, parent=None):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr("pyside6_app.main_window.LauncherDialog", RejectingLauncher)
    monkeypatch.setattr("sys.argv", ["pyside6_main.py"])
    monkeypatch.setattr(window, "_load_tif_path", lambda path: loaded.append(path))

    window._auto_start()

    assert loaded == []


def test_auto_start_loads_launcher_selected_path(monkeypatch):
    window = _window()
    selected = r"D:\zhlonly\project\reUI\data\tif\result.tif"
    loaded = []

    class AcceptingLauncher:
        file_selected = object()

        def __init__(self, parent=None):
            self.selected_path = selected

        def exec(self):
            return 1

    monkeypatch.setattr("pyside6_app.main_window.LauncherDialog", AcceptingLauncher)
    monkeypatch.setattr("sys.argv", ["pyside6_main.py"])
    monkeypatch.setattr("pyside6_app.main_window.QTimer.singleShot", lambda _ms, fn: fn())
    monkeypatch.setattr(window, "_load_tif_path", lambda path: loaded.append(path))

    window._auto_start()

    assert loaded == [selected]


def test_scale_bar_uses_continuous_distance():
    bar = MetricScaleBar()
    bar.update_scale(ppm=10.0, view_scale=1.0)
    first = bar._scale_m
    bar.update_scale(ppm=10.0, view_scale=1.1)
    second = bar._scale_m

    assert first == 10.0
    assert 9.0 < second < 9.2
    assert MetricScaleBar._format_distance(second).endswith("m")


def test_render_animation_frame_accepts_tuple_pose(monkeypatch):
    window = _window()
    window._pipeline_result = {"path": {"tracks": [], "turns": []}}

    class TuplePoseEngine:
        def get_interpolated_position(self):
            return (5.0, 6.0), 0.25

    window._anim_engine = TuplePoseEngine()

    monkeypatch.setattr(window.image_view, "set_image", lambda image: (_ for _ in ()).throw(AssertionError("set_image should not run per animation frame")))

    window._render_anim_frame()

    assert window._anim_trail[-1] == (5, 6)


def test_animation_setup_falls_back_when_geo_distance_is_nan():
    from state import AutoPathSegment, PathPoint, AppState
    from workflow import AnimationEngine

    class NanGeo:
        def pixel_distance_m(self, _p1, _p2):
            return float("nan")

    state = AppState()
    engine = AnimationEngine(state, NanGeo())
    segment = AutoPathSegment(
        points=[PathPoint(0, 0), PathPoint(3, 4)],
        status=1,
        segment_type="work",
    )

    engine.setup([segment])

    assert engine._seg_dists == [5.0]
    assert engine._cum_dists == [0.0, 5.0]
    assert state.anim_total_frames == 2


def test_geo_projected_distance_uses_dataset_xy_not_lonlat_geod():
    from geo import GeoUtils

    class ProjectedDataset:
        crs = object()
        def xy(self, row, col):
            return float(col) * 0.0035, float(row) * 0.0035

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    geo = GeoUtils()
    geo.set_dataset(ProjectedDataset(), IdentityTransformer())

    assert abs(geo.pixel_distance_m((0, 0), (3, 4)) - 0.0175) < 1e-9


def test_geo_display_affine_scales_downsampled_preview_pixels_to_source_metres():
    from geo import GeoUtils

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    geo = GeoUtils()
    source_transform = Affine.scale(0.0035, 0.0035)
    display_transform = source_transform * Affine.scale(4, 4)
    geo.set_affine(display_transform, IdentityTransformer())

    assert abs(geo.meters_per_pixel(10, 10) - 0.014) < 1e-12
    assert abs(geo.pixels_per_meter(10, 10) - (1.0 / 0.014)) < 1e-9
    assert abs(geo.pixel_distance_m((0, 0), (3, 4)) - 0.07) < 1e-12


def test_field_area_uses_display_geotransform_and_is_persisted():
    from geo import GeoUtils

    class IdentityTransformer:
        def transform(self, x, y):
            return x, y

    window = _window()
    window.geo = GeoUtils()
    source_transform = Affine.scale(0.0035, 0.0035)
    window.geo.set_affine(source_transform * Affine.scale(4, 4), IdentityTransformer())
    points = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]

    area_m2 = window._field_area_m2(points)
    area_info = window._estimate_field_area(points)

    assert abs(area_m2 - 196.0) < 1e-9
    assert area_info == "196.0 m2"


def test_segment_uses_source_geotiff_instead_of_downsampled_preview(monkeypatch):
    window = _window()
    window._tif_path = "D:/zhlonly/project/reUI/data/tif/result.tif"
    window.state.tif_path = window._tif_path
    window._tif_rgb = np.zeros((100, 150, 3), dtype=np.uint8)
    window.state.source_img_h = 400
    window.state.source_img_w = 600
    window.state.downsample_factor = 4
    window.state.field_boundary = [(10, 20), (50, 20), (50, 60), (10, 60)]
    window.model_engine._loaded = True
    window.model_engine._model = object()
    calls = []

    class FakeRunner:
        def __init__(self, state, model_engine, geo):
            pass

        def start_from_tif(self, tif_path, display_shape, field_boundary, downsample):
            calls.append((tif_path, display_shape, field_boundary, downsample))

    monkeypatch.setattr("model.InferenceRunner", FakeRunner)

    window._on_segment()

    assert calls == [(
        window._tif_path,
        (100, 150),
        window.state.field_boundary,
        4,
    )]


def test_default_harvester_params_are_complete():
    window = _window()
    window.state.harvester_params = {}

    params = window._effective_harvester_params()

    assert params["cutter_width_m"] > 0
    assert params["track_width_m"] > 0
    assert params["track_gauge_m"] > 0
    assert params["turn_radius_m"] > 0


def test_restore_visual_prefers_planned_path_over_boundary_only(monkeypatch):
    window = _window()
    called = []
    window._pipeline_result = {"path": {"full_path": [[(0, 0), (5, 0)]], "segment_types": ["work"]}}
    window.state.auto_path_planned = True
    monkeypatch.setattr(window, "_show_path", lambda path: called.append(("path", path)))

    window._restore_visual_from_state()

    assert called and called[0][0] == "path"


def test_animation_render_does_not_replace_large_base_image(monkeypatch):
    window = _window()
    window._pipeline_result = {"path": {"tracks": [], "turns": []}}

    class TuplePoseEngine:
        def get_interpolated_position(self):
            return (5.0, 6.0), 0.25

    window._anim_engine = TuplePoseEngine()
    monkeypatch.setattr(window.image_view, "set_image", lambda image: (_ for _ in ()).throw(AssertionError("set_image should not run per animation frame")))

    window._render_anim_frame()

    assert window._anim_trail[-1] == (5, 6)


def test_cached_auto_path_rebuilds_pipeline_result_for_restore():
    window = _window()
    saved = {
        "auto_path_planned": True,
        "auto_path": [
            {"segment_type": "work", "status": 1, "corridor_id": 0, "length_m": 5.0, "points": [[0, 0], [5, 0]]},
            {"segment_type": "turn", "status": 0, "corridor_id": 1, "length_m": 3.0, "points": [[5, 0], [5, 3]]},
        ],
        "path_status": [1, 0],
        "auto_path_valid": True,
        "auto_path_desc": "cached",
    }

    window._restore_cached_path_geometry(saved)

    assert len(window.state.auto_path_segments) == 2
    assert window.state.auto_path_planned is True
    assert window._pipeline_result["path"]["full_path"] == [[(0.0, 0.0), (5.0, 0.0)], [(5.0, 0.0), (5.0, 3.0)]]
    assert window._pipeline_result["path"]["segment_types"] == ["work", "turn"]


def test_task_panel_accordion_keeps_only_one_card_expanded():
    window = _window()
    panel = window.task_panel

    panel._step_cards[0].set_expanded(True)
    panel._on_card_toggled(0)
    panel._step_cards[1].set_expanded(True)
    panel._on_card_toggled(1)

    expanded = [i for i, card in enumerate(panel._step_cards) if card._expanded]
    assert expanded == [1]


def test_task_actions_reset_after_rollback_to_pending():
    from state import AppState
    from pyside6_app.task_state import derive_task_statuses

    state = AppState()
    panel = _window().task_panel

    state.tif_path = "dummy.tif"
    state.field_boundary = [(0, 0), (1, 0), (1, 1)]
    raw = np.ones((3, 3), dtype=np.uint8)
    state.mask_raw = raw
    state.inference_done = True
    state.mask_processed = True
    state.mask_result = {"processed_mask": raw}
    panel.apply_task_statuses(derive_task_statuses(state, has_image=True))
    assert panel._task_widgets["process"]._action_btn.text() == "重做"

    state.mask_processed = False
    state.mask_result = None
    panel.apply_task_statuses(derive_task_statuses(state, has_image=True))

    row = panel._task_widgets["process"]
    assert row._action_btn.text() == "处理"
    assert row._action_btn.isEnabled()


def test_undo_redo_buttons_follow_history_stack_state():
    from state import AppState

    window = _window()
    state = AppState()
    window.top_toolbar.refresh_actions(state)
    assert not window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert not window.top_toolbar._history_buttons["REDO"].isEnabled()

    state.save_undo("change")
    window.top_toolbar.refresh_actions(state)
    assert window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert not window.top_toolbar._history_buttons["REDO"].isEnabled()

    state.undo()
    window.top_toolbar.refresh_actions(state)
    assert not window.top_toolbar._history_buttons["UNDO"].isEnabled()
    assert window.top_toolbar._history_buttons["REDO"].isEnabled()


def test_connect_work_lines_uses_global_service_points_with_local_work_lines():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (110, 220)
    state.exit_point = (150, 260)

    full_path, segment_types = _connect_work_lines(
        [[(10, 20), (10, 60)], [(20, 60), (20, 20)]],
        state=state,
        min_turn_radius_m=1.0,
        turn_strategy="auto",
    )

    assert segment_types[0] == "entry"
    assert full_path[0][0] == (10.0, 20.0)
    assert full_path[0][-1] == full_path[1][0]
    assert segment_types[-1] == "exit"
    assert full_path[-1][-1] == (50.0, 60.0)


def test_manual_bow_falls_back_when_spacing_is_less_than_two_radius():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 20)], [(3, 20), (3, 0)]],
        state=state,
        min_turn_radius_m=2.0,
        turn_strategy="bow",
        return_diagnostics=True,
    )

    assert "turn" in segment_types
    assert diagnostics["used_strategies"] == ["semicircle"]
    assert diagnostics["requested_strategy"] == "bow"


def test_path_segments_are_globalised_for_display_when_mask_is_cropped():
    from path_planner import _path_to_global_coords
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200

    assert _path_to_global_coords([[(1, 2), (3, 4)]], state) == [[(101.0, 202.0), (103.0, 204.0)]]


def test_bow_fallback_removes_requested_bow_hard_error():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    _path, _types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 20)], [(3, 20), (3, 0)]],
        state=state,
        min_turn_radius_m=2.0,
        turn_strategy="bow",
        return_diagnostics=True,
    )

    assert diagnostics["used_strategies"] == ["semicircle"]
    assert diagnostics["fallback_reasons"]
    assert not any("弓形转弯" in reason for reason in diagnostics["hard_reasons"])


def test_turn_strategy_selector_matrix():
    from path_planner import _select_turn_strategy

    assert _select_turn_strategy("auto", 5.0, 2.0)["strategy"] == "bow"
    assert _select_turn_strategy("auto", 4.02, 2.0)["strategy"] == "semicircle"
    assert _select_turn_strategy("auto", 3.0, 2.0)["strategy"] == "semicircle"
    decision = _select_turn_strategy("bow", 3.0, 2.0)
    assert decision["strategy"] == "semicircle"
    assert decision["fallback_from"] == "bow"


def test_unload_point_is_explicit_service_segment_before_exit():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.zeros((100, 100), dtype=np.uint8)
    state.entry_point = (0.0, 0.0)
    state.exit_point = (80.0, 80.0)
    state.unload_points = [(60.0, 70.0), (10.0, 90.0)]

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(0, 0), (0, 40)], [(20, 40), (20, 0)]],
        state=state,
        min_turn_radius_m=1.0,
        turn_strategy="auto",
        return_diagnostics=True,
    )

    assert "unload" in segment_types
    unload_index = segment_types.index("unload")
    assert unload_index < segment_types.index("exit")
    assert full_path[unload_index][-1] == (60.0, 70.0)
    assert full_path[unload_index + 1][0] == (60.0, 70.0)
    assert diagnostics["service_points"]["unload_visit_count"] == 1


def test_planning_factor_report_records_core_decision_inputs():
    from path_planner import _build_planning_factor_report
    from state import AppState

    state = AppState()
    state.mask_offset_x = 100
    state.mask_offset_y = 200
    state.mask_raw = np.zeros((50, 60), dtype=np.uint8)
    state.entry_point = (110.0, 220.0)
    state.exit_point = (150.0, 240.0)
    state.unload_points = [(130.0, 230.0)]
    state.harvester_params = {
        "cutter_width_m": 2.2,
        "track_width_m": 0.45,
        "track_gauge_m": 1.65,
        "turn_radius_m": 2.1,
    }

    report = _build_planning_factor_report(
        state=state,
        work_lines=[[(0, 0), (0, 10)], [(10, 10), (10, 0)]],
        layout={"generated_pass_count": 2},
        turn_assessment={"strategy": "fishtail", "fallback_reasons": ["narrow headland"]},
        validation={"harvest_coverage_pct": 91.0, "track_core_overlap_pct": 1.2},
    )

    assert report["harvester"]["turn_radius_m"] == 2.1
    assert report["mask"]["offset_xy"] == [100, 200]
    assert report["service_points"]["entry_local"] == [10.0, 20.0]
    assert report["service_points"]["unload_count"] == 1
    assert report["route_structure"]["work_line_count"] == 2
    assert report["decision_chain"][0] == "processed_mask_and_field_support"


def test_candidate_scoring_prefers_safe_route_over_short_invalid_route():
    from path_planner import _rank_path_candidates

    candidates = [
        {
            "strategy": "bow",
            "validation": {
                "valid": False,
                "total_length_m": 80.0,
                "track_core_overlap_pct": 12.0,
                "track_outside_field_pct": 4.0,
                "harvest_coverage_pct": 96.0,
            },
            "turn_assessment": {"hard_reasons": ["越界"], "turn_self_crossing_count": 1},
            "segment_types": ["work", "turn", "work"],
        },
        {
            "strategy": "fishtail",
            "validation": {
                "valid": True,
                "total_length_m": 95.0,
                "track_core_overlap_pct": 1.0,
                "track_outside_field_pct": 0.2,
                "harvest_coverage_pct": 93.0,
            },
            "turn_assessment": {"hard_reasons": [], "turn_self_crossing_count": 0},
            "segment_types": ["work", "turn_reverse", "turn_aux", "work", "unload", "exit"],
        },
    ]

    ranked = _rank_path_candidates(candidates)

    assert ranked[0]["strategy"] == "fishtail"
    assert ranked[0]["candidate_score"] < ranked[1]["candidate_score"]
    assert ranked[0]["candidate_metrics"]["unload_visits"] == 1


def test_auto_candidate_strategy_list_is_expanded_for_multi_strategy_planning():
    from path_planner import _candidate_strategies_for_request

    assert _candidate_strategies_for_request("auto") == ["semicircle", "bow", "pear", "fishtail", "alpha"]
    assert _candidate_strategies_for_request("bow") == ["bow"]


def test_remove_headlands_can_preserve_headland_layer():
    from mask_processor import remove_headlands

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[35:45, 10:110] = 255  # horizontal field-body row band
    mask[10:70, 2:12] = 255    # vertical headland-like component

    body, headland = remove_headlands(mask, 0.0, angle_thresh_deg=45.0, min_area_ratio=0.01, return_headland=True)

    assert body[40, 60] == 255
    assert body[40, 5] == 0
    assert headland[40, 5] == 255
    assert np.count_nonzero(headland & body) == 0


def test_validate_footprints_reports_headland_overlap_separately():
    from footprint_planner import validate_footprints

    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 20:100] = 255
    headland = np.zeros_like(body)
    headland[10:70, 2:14] = 255
    path = [[(8.0, 15.0), (8.0, 65.0)]]

    metrics = validate_footprints(
        path,
        body,
        harvester={"cutter_width_m": 2.0, "track_width_m": 1.0, "track_gauge_m": 0.0},
        config={"headland_mask": headland},
        segment_types=["turn"],
    )

    assert metrics["track_headland_overlap_pct"] > 0
    assert metrics["track_core_overlap_pct"] == 0


def test_mask_overlay_draws_headland_as_separate_light_layer(monkeypatch):
    window = _window()
    window._tif_rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    window.image_view.set_image(window._tif_rgb)
    body = np.zeros((80, 120), dtype=np.uint8)
    body[35:45, 20:100] = 255
    headland = np.zeros_like(body)
    headland[10:70, 2:14] = 255
    window.state.mask_result = {"processed_mask": body, "headland_mask": headland}
    calls = []

    def fake_draw(mask, x=0, y=0, color=(120, 255, 140), max_dim=2200):
        calls.append((mask, color))

    monkeypatch.setattr(window.image_view.viewer, "draw_mask_overlay", fake_draw)

    window._show_mask_overlay(body)

    assert calls[0][0] is headland
    assert calls[0][1] != calls[1][1]
    assert calls[1][0] is body
