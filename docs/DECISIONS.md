# reUI 关键设计决策

本文件记录不能被随意删除、绕过或以“简化”为理由弱化的系统级决策。修改决策相关代码时，必须同步核对代码锚点和不变量测试。

## DECISION-001：正式路径导出必须失败关闭

**结论**：GeoJSON、CSV、KML、JSON 和 `$PATH` 必须先经过 `_geo_export_payload()`；输出成功后必须生成 manifest。

**原因**：路径可能进入受控田间试验。可用性不能优先于坐标、参数和路径安全。

**不变量**：

- 无有效地理配准、路径未验证或 provenance 不一致时禁止导出。
- 农机参数、调头策略与规划时不一致时禁止导出。
- 路径点超出影像范围时禁止仿射外推。
- manifest 写入失败时不得保留正式路径文件。
- 旧 `ExportEngine` 不得成为旁路。

**代码锚点**：`MainWindow._geo_export_payload`、`MainWindow._write_export_manifest`、`integrity_check.verify_manifest`。

**测试锚点**：`test_invalid_or_manually_edited_path_is_blocked_from_export`、`test_offline_manifest_checker_detects_route_tampering`、`test_legacy_export_engine_is_fail_closed`。

## DECISION-002：缓存恢复必须验证完整阶段证据

**结论**：缓存必须绑定源影像、模型、配置、上游指纹和产物哈希；缺失或不一致时逐级失效。

**原因**：同名模型、同尺寸影像或同一路径不代表内容相同，错误恢复会污染全部下游结果。

**不变量**：

- 模型内容变化后旧推理掩膜失效。
- 不按图像尺寸跨源复用缓存。
- 推理失效必须使掩膜和路径失效；掩膜失效必须使路径失效。
- 旧模式缓存失败关闭，不猜测兼容。

**代码锚点**：`CacheRestoreWorker`、`cache.load_project_state`、`provenance.verify_stage_record`。

**测试锚点**：`test_cache_restore_rejects_old_mask_when_model_file_changes`、`test_cache_loader_never_reuses_same_shape_mask_from_another_source`、`test_stage_record_rejects_changed_model_or_artifact`。

## DECISION-003：无可靠地理配准时禁止米制处理

**结论**：掩膜正则化、路径规划和地理导出必须建立在有效 CRS、仿射和可接受 GSD 上。

**原因**：履带宽度、转弯半径、覆盖率和碾压率都是米制量，像素假尺度会直接产生错误结论。

**不变量**：

- 无地理配准时 `require_metric_scale()` 必须失败。
- 非有限、超范围、横纵尺度差异过大或图幅内 GSD 变化过大时必须失败。
- 不允许回退到猜测的 `1 m/px` 或 `0.01 m/px`。

**代码锚点**：`require_metric_scale`、`GeoUtils`、`TifLoadWorker`。

**测试锚点**：`test_metric_processing_requires_valid_georeference`、`test_metric_processing_rejects_anisotropic_raster_pixels`、`test_tif_loader_rejects_crs_without_valid_affine_geotransform`。

## DECISION-004：正式推理必须读取原始 GeoTIFF

**结论**：UI 预览图只用于显示；正式识别从原始 GeoTIFF 田块窗口读取。超大田块使用原始分辨率流式切片，不整体缩图。

**原因**：预览降采样或整体缩图会丢失细行带信息，改变分割结果和后续路径。

**不变量**：

- 模型看到的 tile 保持配置的原始像素尺寸，默认 `640 x 640`。
- 大裁剪不得因为 36MP 阈值直接失败，也不得先整体下采样。
- 任一 tile 推理失败时整次推理失败。
- 田块窗口、切片参数和预处理信息进入 provenance。

**代码锚点**：`InferenceRunner.start_from_tif`、`InferenceRunner._run_source_tif_tiles`、`InferenceRunner._run_from_tif`。

**测试锚点**：`test_segment_uses_source_geotiff_instead_of_downsampled_preview`、`test_source_tif_inference_streams_oversized_crop_at_source_tile_size`、`test_failed_inference_tile_aborts_instead_of_becoming_background`。

## DECISION-005：栅格预处理必须确定且可追溯

**结论**：RGB 波段选择、NoData、位深和归一化必须通过统一模块处理并记录元数据。

**原因**：不同 GeoTIFF 的波段解释和位深不同，隐式转换会让同一模型得到不可比较的输入。

**不变量**：

- 优先依据 `colorinterp` 选择 RGB；缺失时使用明确回退。
- NoData 不得进入有效像素统计。
- 原生 8 位数据保持字节值；非 8 位处理规则必须版本化。
- 非有限元数据写入 provenance 时规范为 `null`。

**代码锚点**：`raster_preprocessing.normalise_raster_bands`、`raster_preprocessing.read_rgb_raster`。

**测试锚点**：`test_uint16_preprocessing_is_per_band_deterministic_and_masks_nodata`、`test_provenance_canonical_json_normalises_nonfinite_metadata`。

## DECISION-006：模型权重必须按信任和哈希管理

**结论**：内置模型必须匹配注册表 SHA-256；外部 `.pt` 必须经过用户信任确认，推理时记录模型哈希。

**原因**：同路径权重可能被替换，且 PyTorch `.pt` 不是纯数据格式。

**不变量**：

- 内置模型哈希不匹配时拒绝加载。
- 外部模型默认不信任。
- 模型内容变化后禁止复用旧推理结果和导出旧路径。
- manifest 必须记录模型哈希和注册信息。

**代码锚点**：`MainWindow._on_load_model_at`、`data/models/model_registry.json`、推理 provenance。

**测试锚点**：`test_builtin_model_registry_matches_present_weight_files`、`test_export_rejects_model_content_change_after_inference`、`test_cache_restore_rejects_old_mask_when_model_file_changes`。

## DECISION-007：科学输入变化必须使下游结果失效

**结论**：模型、掩膜参数、农机参数、调头策略、起终点和卸粮点变化后，不能继续使用旧下游结果。

**原因**：路径由完整输入集合决定；只更新界面参数但保留旧路径会产生显示与计算不一致。

**不变量**：

- 失效顺序固定为 inference -> mask -> path -> simulation/export。
- 重新推理不保留旧掩膜作为成功结果。
- 手工编辑路线后路径状态必须为未验证，禁止正式导出。

**代码锚点**：`MainWindow._invalidate_analysis_from`、`_on_params_dialog`、`_on_turn_strategy_changed`、`_place_entry_exit_point`。

**测试锚点**：`test_input_changes_invalidate_stale_downstream_results`、`test_export_blocks_when_harvester_parameters_changed_after_planning`、`test_export_rejects_service_point_change_after_path_planning`。

## DECISION-008：必须保留科研原型的能力边界

**结论**：项目只能描述为科研原型、方法验证或辅助决策平台，不能宣称已经能够直接驱动农机长期生产作业。

**原因**：当前缺少 GCP/RTK 绝对精度、目标终端协议、真实轨迹 RMSE、作业后碾压和异常工况闭环验证。

**不变量**：

- README 必须保留“不适用未经人工复核直接驱动农机”的声明。
- 文档必须区分内部一致性与外场有效性。
- 启发式农机候选不得描述成地形安全结论。

**文档锚点**：`README.md`、`docs/SCIENTIFIC_VALIDATION.md`、`AI_RULES.md`。

**测试锚点**：`test_readme_preserves_research_prototype_boundary`、`test_machine_parameter_output_is_explicitly_unvalidated_heuristic`。

## 修改决策的方式

决策不是永久禁止演进。需要改变时，必须在同一个变更中：

1. 说明旧决策为什么不再成立及新增证据。
2. 更新本文件中的结论和不变量。
3. 更新所有相关代码锚点。
4. 修改或新增行为测试。
5. 在交付说明中列出风险变化和仍未验证条件。
