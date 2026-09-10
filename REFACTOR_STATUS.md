# 重構短進度

更新：2026-09-10。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析／資料集入口遷移至第十七階段（Jenqwei 資料集）。
- 當前：[第十八階段品質 scorer](docs/refactor/TASKS.md#stage-18) 進行中；核心診斷增量完成，呼叫端／audit／圖形遷移尚未完成。
- 本輪：已提交第十六階段後半及第十七階段；新增 scorer 有效性／fallback 原因／preset／stage context，保留舊 overall／detail。301 tests／0 skipped、168 Python 靜態與 20 evidence 通過。
- 數值：五真實來源 raw／BP 118 窗口，加邊界共 130 輸入、四 presets、520 評分／1,950 陣列；最大誤差 0，NaN 位置一致，一張核心診斷圖目視通過。
- 下一步：逐呼叫端傳遞 raw／filtered stage 與 diagnostics 到 audit／reader／圖形；明示是否採用 usable_overall，核對接受／排除窗口差異，不能暗改門檻。
- 其後：baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 單一 raw channel，無 BP／模型／品質 scorer／baseline；保留 `int(win_sec*fs)`、原始全域格點與整數微秒，無 timestamp helper 保留舊語義。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY、qEEG CLI、眼開閉、Jenqwei 分析／資料集已完成，勿重寫。
- 眼開閉只做 float32 bandpass，不混入 notch／bandstop；模型輸出依序為來源 ch1/2/5/6，before 比較取真正來源 ch1/2/5/6。
- Jenqwei 分析 Before 保留全重採樣尾端，After 才按 400 點段內裁尾；`max_samples` 在完整來源段濾波後截斷，raw／Before／After 使用各自索引與明確映射。
- Jenqwei 顯示 ch3/4 僅 Before，After 明示不存在；PSD 分別畫各來源段、不串接或平均。分段 main 可接受缺口，舊 `run_pipeline` guard 保留。
- 資料集每來源段先等分再裁窗口，不是 train/test split、不執行模型；保留連續數值／舊欄位，新增真實窗口數、`fs_out` 及 raw／resampled 索引；短段排除需 `--allow-short-drop`，全排除仍失敗。
- 不覆寫原始錄製、模型或舊圖。bundle 改主版後執行 `python build_bundles.py`，再 `--check`。
- Git：第十六階段後半及第十七階段已提交 `ef9eada`，代表圖／manifest 補提交 `23aa2df`。本輪第十八階段核心增量尚未提交；未 push。

## 驗證入口

- 本輪：[第十八階段核心完整檢查](docs/refactor/validation/stage18/core/checks.json)：301 tests／0 skipped，168 Python 編譯／Pyflakes／bundle／diff 通過；不代表整階段已完成。
- [數值摘要](docs/refactor/validation/stage18/core/analysis.json)／[evidence](docs/refactor/validation/stage18/core/evidence_checks.json)：1,950 陣列精確比較及來源／舊碼／fixtures／產物 hash，共 20 checks。
- [目視紀錄](docs/refactor/validation/stage18/core/visual_review.json)：100／101 點 legacy flat 窗口邊界與無效標記一致。全套遇到的 pandas SHA-256 型別猜測崩潰已改用正式 reader 解決，log 留存。
- 前階段：[第十七階段完整檢查](docs/refactor/validation/stage17/final/checks.json) 291 tests／0 skipped，為當時版本證據。
- 開發中依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [第十八階段核心增量報告](docs/refactor/STAGE18_CORE_REPORT_2026-09-10.md)／[品質盤點與相容契約](docs/refactor/STAGE18_QUALITY_INVENTORY.md)／[實跑腳本](tools/validate_quality_stage18.py)。
- 品質舊基準 `tests/fixtures/quality_stage18_*`；本輪 NPZ／diagnostics／摘要／log／圖在 `docs/refactor/validation/stage18/core/`，可用其中 evidence manifest 重讀。
- [第十七階段驗收報告](docs/refactor/STAGE17_REPORT_2026-09-10.md)／[資料集契約](docs/refactor/STAGE17_DATASET_CONTRACT.md)；資料集 `dataset_jenqwei_*` 與分析 `jenqwei_*` 基準不可混用。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
