"""Fail-closed readiness assessment for future machine route exporters."""

import math
from typing import Dict, Optional


MACHINE_READINESS_SCHEMA = "reui.machine-readiness.v1"


def _verified_report(report: object) -> bool:
    return isinstance(report, dict) and bool(report.get("verified"))


def assess_machine_readiness(
    path_result: Dict,
    execution_config: Optional[Dict] = None,
) -> Dict:
    """Report whether a validated route may enter a machine-adapter pipeline.

    This function does not make current GIS exports machine executable. It
    only records missing evidence and remains fail-closed by default.
    """
    result = dict(path_result or {})
    cfg = dict(execution_config or {})
    validation = dict(result.get("validation") or {})
    layout = dict(result.get("layout") or {})
    turning = dict(result.get("turn_assessment") or {})
    blockers = []

    if not bool(result.get("is_valid", validation.get("valid", False))):
        blockers.append("path_validation_failed")
    if turning.get("hard_reasons"):
        blockers.append("turn_hard_errors_present")
    if layout.get("work_line_mode") != "footprint_optimized":
        blockers.append("footprint_optimized_work_lines_not_used")
    if not bool(validation.get("field_boundary_present")):
        blockers.append("confirmed_field_boundary_missing")
    if not bool(validation.get("semantic_support_present")):
        blockers.append("semantic_support_missing")
    if not (
        bool(validation.get("forbidden_mask_present"))
        and bool(cfg.get("forbidden_mask_confirmed"))
    ):
        blockers.append("confirmed_forbidden_map_missing")

    geo_report = cfg.get("geo_accuracy_report")
    geo_rmse = geo_report.get("rmse_m") if isinstance(geo_report, dict) else None
    if not (
        _verified_report(geo_report)
        and isinstance(geo_rmse, (int, float))
        and math.isfinite(float(geo_rmse))
        and bool(geo_report.get("within_tolerance"))
    ):
        blockers.append("external_geo_accuracy_not_verified")

    reference_point = str(cfg.get("vehicle_reference_point", "")).strip()
    gnss_offset = cfg.get("gnss_offset_m")
    if not reference_point or not (
        isinstance(gnss_offset, (list, tuple))
        and len(gnss_offset) == 3
        and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in gnss_offset)
    ):
        blockers.append("vehicle_reference_geometry_incomplete")
    if not bool(cfg.get("kinematic_model_validated")):
        blockers.append("full_vehicle_kinematics_not_validated")
    if not (
        str(cfg.get("terminal_adapter", "")).strip()
        and bool(cfg.get("terminal_adapter_validated"))
    ):
        blockers.append("terminal_adapter_not_validated")
    if not _verified_report(cfg.get("field_tracking_report")):
        blockers.append("field_tracking_validation_missing")
    if not bool(cfg.get("manual_review_signed")):
        blockers.append("manual_review_not_signed")

    blockers = list(dict.fromkeys(blockers))
    return {
        "schema": MACHINE_READINESS_SCHEMA,
        "eligible_for_machine_export": not blockers,
        "current_route_classification": (
            "machine_export_candidate" if not blockers else "research_manual_review"
        ),
        "blockers": blockers,
        "manual_review_required": bool(blockers),
    }
