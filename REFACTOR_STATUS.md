# 重構短進度

更新：2026-09-11。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析／資料集入口遷移至第十七階段（Jenqwei 資料集）。
- 當前：[第十八階段品質 scorer](docs/refactor/TASKS.md#stage-18) 進行中；核心、共用 raw 及 TFLite baseline／summary 呼叫端增量完成，其餘直接／filtered 呼叫端待完成。
- 本輪：TFLite baseline／Before／After 分別保存 filtered／filtered_resampled／model_output 診斷，CSV／audit／來源 reader 及前後品質圖完成。317 tests／0 skipped、176 Python 靜態與 bundle 通過。
- 數值：五真實來源四通道副本，另加缺口／半秒／fallback／threshold=0 對照，共九案例、107 qEEG 窗口／271 模型窗口／510 子窗口、399 陣列；最大誤差 0、NaN 位置與選取結果一致，18 新舊表重讀及三圖目視通過。
- 下一步：entropy clean／state／MI，再接 Goertzel、quality_check 及剩餘直接 helper；逐入口保存 stage／diagnostics 並驗收。保留 legacy_overall，不暗改門檻或採用 usable_overall。
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
- Git：第十六階段後半及第十七階段 `ef9eada`／產物 `23aa2df`；第十八階段核心 `a0a7ba6`。raw 與 TFLite 呼叫端增量尚未提交，未 push；提交驗收 CSV／PNG 時須明確處理 ignore 規則。

## 驗證入口

- 本輪：[TFLite 呼叫端完整檢查](docs/refactor/validation/stage18/tflite_callers/checks.json)：317 tests／0 skipped，176 Python 編譯／Pyflakes／bundle／diff 通過；不代表整階段已完成。
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
- 本輪證據在 `docs/refactor/validation/stage18/tflite_callers/`；完整表重讀仍依賴報告記載的暫存四通道來源，可用腳本重建；持久數值／圖 manifest 可獨立重核。
- [raw 呼叫端增量報告](docs/refactor/STAGE18_RAW_CALLERS_REPORT_2026-09-11.md) 保留 event／zoom／comparison 驗收範圍。
- [核心增量報告](docs/refactor/STAGE18_CORE_REPORT_2026-09-10.md) 與 `tests/fixtures/quality_stage18_*` 保留原 scorer 相容基準。
- [第十七階段驗收報告](docs/refactor/STAGE17_REPORT_2026-09-10.md)／[資料集契約](docs/refactor/STAGE17_DATASET_CONTRACT.md)；資料集 `dataset_jenqwei_*` 與分析 `jenqwei_*` 基準不可混用。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
