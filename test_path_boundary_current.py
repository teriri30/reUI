import numpy as np


def test_field_support_uses_field_boundary_not_convex_hull():
    from footprint_planner import _field_support
    from state import AppState

    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10:70, 10:110] = 255
    state = AppState()
    state.field_boundary = [(30, 10), (110, 10), (110, 70), (30, 70)]

    support = _field_support(mask, state)

    assert support[40, 20] == 0
    assert support[40, 40] == 255


def test_connect_work_lines_turn_points_stay_inside_field_boundary_when_outward_arc_would_leave():
    from path_planner import _connect_work_lines
    from state import AppState

    state = AppState()
    state.mask_raw = np.ones((120, 120), dtype=np.uint8) * 255
    state.field_boundary = [(10, 10), (110, 10), (110, 110), (10, 110)]

    full_path, segment_types, diagnostics = _connect_work_lines(
        [[(80.0, 12.0), (12.0, 12.0)], [(12.0, 24.0), (80.0, 24.0)]],
        state=state,
        min_turn_radius_m=1.0,
        turn_strategy="semicircle",
        config={"turn_outward_clearance_m": 0.6, "turn_safety_margin_m": 0.1},
        return_diagnostics=True,
    )

    turn = full_path[segment_types.index("turn")]
    assert min(x for x, _ in turn) >= 10.0
    assert max(y for _, y in turn) <= 110.0
    assert diagnostics["boundary_constraints"]["field_boundary_present"] is True
    assert diagnostics["boundary_constraints"]["turns_adjusted"] >= 1


def test_prepare_band_centerlines_does_not_trim_harvestable_row_ends_by_default():
    from path_planner import _smooth_band_centerline

    points = [(float(x), 20.0) for x in range(10, 111, 10)]
    line = _smooth_band_centerline(
        points,
        main_angle=0.0,
        mpp=0.1,
        config={"centerline_interval_m": 1.0},
    )

    assert line[0][0] == 10.0
    assert line[-1][0] == 110.0
