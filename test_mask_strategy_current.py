import numpy as np


def test_regularize_output_preserves_raw_parallel_residual_in_processed_mask(monkeypatch):
    import row_geometry

    raw = np.zeros((100, 160), dtype=np.uint8)
    raw[30:36, 5:150] = 255      # row-parallel residual that must remain visible
    raw[5:85, 152:158] = 255     # crosswise headland residual
    rebuilt = np.zeros_like(raw)
    rebuilt[50:56, 5:150] = 255  # clean regularized body row

    def fake_residual_layers(raw_mask, rebuilt_mask, angle, angle_thresh_deg=60.0, min_area_ratio=0.001):
        headland = np.zeros_like(raw_mask)
        headland[5:85, 152:158] = raw_mask[5:85, 152:158]
        body_residual = np.zeros_like(raw_mask)
        body_residual[30:36, 5:150] = raw_mask[30:36, 5:150]
        return headland, body_residual

    monkeypatch.setattr(row_geometry, "residual_mask_layers", fake_residual_layers)

    visible_body = np.bitwise_or(rebuilt, fake_residual_layers(raw, rebuilt, 0.0)[1])
    planning_support = np.bitwise_or(visible_body, fake_residual_layers(raw, rebuilt, 0.0)[0])

    assert visible_body[32, 30] == 255
    assert visible_body[40, 155] == 0
    assert planning_support[40, 155] == 255


def test_detached_parallel_fragment_outside_body_extent_does_not_enter_body_residual():
    from row_geometry import residual_mask_layers

    raw = np.zeros((120, 220), dtype=np.uint8)
    rebuilt = np.zeros_like(raw)
    rebuilt[40:46, 20:150] = 255
    raw[40:46, 20:150] = 255
    raw[85:91, 166:216] = 255

    headland, body_residual = residual_mask_layers(
        raw,
        rebuilt,
        main_angle=0.0,
        min_area_ratio=0.001,
    )

    assert body_residual[87, 190] == 0
    assert headland[87, 190] == 255


def test_generated_body_keeps_internal_gap_fill_but_removes_isolated_fragment():
    from row_geometry import constrain_generated_body_to_raw_support

    raw = np.zeros((80, 180), dtype=np.uint8)
    raw[35:41, 20:80] = 255
    raw[35:41, 98:150] = 255
    generated = raw.copy()
    generated[35:41, 80:98] = 255
    generated[60:66, 160:168] = 255

    constrained = constrain_generated_body_to_raw_support(
        generated,
        raw,
        main_angle=0.0,
        meters_per_px=0.1,
        config={"row_gap_close_m": 2.2, "row_gap_close_width_m": 0.2},
    )

    assert constrained[37, 88] == 255
    assert constrained[62, 164] == 0


def test_default_row_gap_repair_bridges_visible_internal_breaks():
    from row_geometry import repair_row_gaps

    raw = np.zeros((80, 180), dtype=np.uint8)
    raw[35:41, 20:80] = 255
    raw[35:41, 98:150] = 255

    repaired = repair_row_gaps(raw, main_angle=0.0, meters_per_px=0.1, config={})

    assert repaired[37, 88] == 255
