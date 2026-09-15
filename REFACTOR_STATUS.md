# 重構短進度

更新：2026-09-15。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析／資料集入口遷移至第十七階段（Jenqwei 資料集）。
- 當前：[第十八階段品質 scorer](docs/refactor/TASKS.md#stage-18) 進行中；核心、共用 raw、TFLite、entropy、Goertzel 與 R6 quality_check 增量完成，剩餘 direct helper 待完成。
- 本輪：[R6 quality_check 增量](docs/refactor/STAGE18_QUALITY_CHECK_REPORT_2026-09-15.md)：NaN 異常明確 flagged，samples raw／filtered 與 anomalies raw 診斷、CSV／sidecar／來源 reader、品質圖完成。完整檢查 345 tests／0 skipped、188 Python／bundle 通過。
- 數值：五份完整 Jenqwei、真實 James samples 與 Hardy anomalies，加合成邊界共 20 案例及污染 helper；974 陣列精確相同、17 來源 reader、970 表格列、八圖目視。只有 3 個 NaN 異常的原因／severity 有意修正；有限分數及選取不變。
- 下一步：剩餘 direct helper（event 舊 `compute_quality_windowed`）及第十八階段整體收斂驗收；保留 legacy_overall 與既有門檻，Goertzel rolling median 分段平滑另待方法校準。
- 其後：baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 單一 raw channel，無 BP／模型／品質 scorer／baseline；保留 `int(win_sec*fs)`、原始全域格點與整數微秒，無 timestamp helper 保留舊語義。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- TFLite baseline 品質用 filtered 500 Hz 四通道；Before 用 filtered_resampled 200 Hz 來源 ch1/2，After 用 model_output 200 Hz；reader 不重跑模型，模型／baseline 數值須獨立驗證。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY、qEEG CLI、眼開閉、Jenqwei 分析／資料集已完成，勿重寫。
- 眼開閉只做 float32 bandpass，不混入 notch／bandstop；模型輸出依序為來源 ch1/2/5/6，before 比較取真正來源 ch1/2/5/6。
- Jenqwei 分析 Before 保留全重採樣尾端，After 才按 400 點段內裁尾；`max_samples` 在完整來源段濾波後截斷，raw／Before／After 使用各自索引與明確映射。
- Jenqwei 顯示 ch3/4 僅 Before，After 明示不存在；PSD 分別畫各來源段、不串接或平均。分段 main 可接受缺口，舊 `run_pipeline` guard 保留。
- 資料集每來源段先等分再裁窗口，不是 train/test split、不執行模型；保留連續數值／舊欄位，新增真實窗口數、`fs_out` 及 raw／resampled 索引；短段排除需 `--allow-short-drop`，全排除仍失敗。
- 不覆寫原始錄製、模型或舊圖。bundle 改主版後執行 `python build_bundles.py`，再 `--check`。
- Git：核心 `a0a7ba6`、raw／TFLite 程式 `40d2a23` 與 73 個舊產物補提交 `0dd9ff3` 保留；R1–R6 的程式、測試、文件及最終驗收證據整理於本分支。重複開發期 run 與正式錄製副本不納入 Git；局部 ignore 只放行必要產物，R6 可提交／正式來源核對範圍分開記錄。

## 驗證入口

- 本輪：[完整檢查](docs/refactor/validation/stage18/quality_check_r6_release_checks/summary.json) 345 tests／0 skipped、188 Python；[數值](docs/refactor/validation/stage18/quality_check_r6_release/analysis.json)／[產物 evidence](docs/refactor/validation/stage18/quality_check_r6_release_evidence/summary.json)／[目視](docs/refactor/validation/stage18/quality_check_r6_release/visual_review.json)。Git 可提交清單另有 [142 項核對](docs/refactor/validation/stage18/quality_check_r6_repository_evidence/summary.json)；本機正式來源版 [156 項核對](docs/refactor/validation/stage18/quality_check_r6_source_evidence/summary.json)。
- 前增量：[Goertzel 完整檢查](docs/refactor/validation/stage18/goertzel_quality_checks/summary.json) 為當時 336 tests；354 陣列／36 reader／七圖見 [R5 報告](docs/refactor/STAGE18_GOERTZEL_QUALITY_REPORT_2026-09-15.md)。
- 前輪：[entropy 完整檢查](docs/refactor/validation/stage18/entropy_diagnostics_checks/summary.json) 為當時 327 tests；35 案例／714 陣列／77 reader 及五圖見 [entropy 報告](docs/refactor/STAGE18_ENTROPY_DIAGNOSTICS_REPORT_2026-09-15.md)。
- 前次：[驗收檢查與反例](docs/refactor/STAGE18_REVIEW_REPORT_2026-09-11.md) 是修正前的 40 tests 與 R1–R6 發現；重現腳本以當時缺陷為成功條件，不可當修復後回歸測試。
- 前增量：[TFLite 完整檢查](docs/refactor/validation/stage18/tflite_callers/checks.json) 為當時 317 tests／0 skipped；當前 joint-MI 工作區已變更，不能直接視為此版本全套證據。
- [數值摘要](docs/refactor/validation/stage18/tflite_callers/analysis.json)／[evidence](docs/refactor/validation/stage18/tflite_callers/evidence_checks.json)：145 checks＝118 hash＋18 表重讀＋9 NPZ 比較；持久 NPZ 另有 [27 checks](docs/refactor/validation/stage18/tflite_callers/numeric_evidence_checks.json)。
- [目視紀錄](docs/refactor/validation/stage18/tflite_callers/visual_review.json)／[4 產物 hash](docs/refactor/validation/stage18/tflite_callers/visual_evidence_checks.json)：真實 talk、缺口、有限 fallback。半秒 baseline 新舊皆明確無可用基線，診斷可追蹤；開發期失敗已修正並留 log。
- 前增量：[raw 呼叫端完整檢查](docs/refactor/validation/stage18/raw_callers/checks.json) 為當時 310 tests／33 表／10 NPZ 比較及三圖證據；共用 helper 後續修改已納入本輪完整回歸。
- 核心：[當時完整檢查](docs/refactor/validation/stage18/core/checks.json) 301 tests／0 skipped，四 presets／1,950 陣列精確比較與核心圖完成。
- 前階段：[第十七階段完整檢查](docs/refactor/validation/stage17/final/checks.json) 291 tests／0 skipped，為當時版本證據。
- 開發中依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [TFLite 呼叫端增量報告](docs/refactor/STAGE18_TFLITE_CALLERS_REPORT_2026-09-11.md)／[品質盤點與相容契約](docs/refactor/STAGE18_QUALITY_INVENTORY.md)／[實跑腳本](tools/validate_tflite_quality_stage18.py)。
- 本輪實跑：`tools/validate_quality_check_stage18.py`；證據在 `docs/refactor/validation/stage18/quality_check_r6_release/`。[R3 恢復報告](docs/refactor/STAGE18_ARTIFACT_RESTORE_REPORT_2026-09-15.md) 確認舊 CSV／圖已在 Git，舊 TFLite 重讀仍依賴暫存四通道來源。
- [raw 呼叫端增量報告](docs/refactor/STAGE18_RAW_CALLERS_REPORT_2026-09-11.md) 保留 event／zoom／comparison 驗收範圍。
- [核心增量報告](docs/refactor/STAGE18_CORE_REPORT_2026-09-10.md) 與 `tests/fixtures/quality_stage18_*` 保留原 scorer 相容基準。
- [第十七階段驗收報告](docs/refactor/STAGE17_REPORT_2026-09-10.md)／[資料集契約](docs/refactor/STAGE17_DATASET_CONTRACT.md)；資料集 `dataset_jenqwei_*` 與分析 `jenqwei_*` 基準不可混用。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
