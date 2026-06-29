"""导航文件导出"""
from typing import List, Tuple
from config import Config, AppLogger
from state import AppState, AutoPathSegment
from geo import GeoUtils


class ExportEngine:
    def __init__(self, geo: GeoUtils, state: AppState):
        self.geo = geo
        self.state = state
        self.cfg = Config()
        self.log = AppLogger()

    def export_to_file(self, save_path: str):
        pts, st = self.state.path_points, self.state.path_status
        if len(pts) < 2:
            raise ValueError("路径点不足")
        if len(st) not in (len(pts) - 1, len(pts)):
            raise ValueError(
                f"路径状态数量与显示路径不一致: status={len(st)}, points={len(pts)}"
            )
        if not st:
            raise ValueError("路径状态为空，已停止导出")

        from path_planner import global_path_to_geo, export_path_format

        geo_points = global_path_to_geo([pts], self.geo, ["work"])
        for index, point in enumerate(geo_points):
            status_index = min(index, len(st) - 1)
            point["status"] = int(st[status_index])
        export_path_format(geo_points, save_path)
        self.log.info(f"导出成功: {save_path} ({len(pts)}路径点)")

    def export_auto_path_to_file(self, save_path: str):
        """导出自动路径到文件。
        直接使用已有的 path_points/path_status（已由 _action_plan 同步），
        不再破坏性覆写，保留段间过渡状态。
        """
        if not self.state.auto_path:
            raise ValueError("自动路径为空")
        # path_points 和 path_status 已在 _action_plan 中同步完成
        if len(self.state.path_points) < 2:
            raise ValueError("路径点不足")
        self.export_to_file(save_path)
