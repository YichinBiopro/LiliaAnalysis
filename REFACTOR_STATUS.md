# 重構短進度

更新：2026-09-10。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析／資料集入口遷移至第十七階段（Jenqwei 資料集）。
- 當前：第十七階段完成；下一工作單元為 [TASKS](docs/refactor/TASKS.md) 的品質 scorer，待盤點、尚未編號。
- 本輪：分段資料集輸出／manifest／reader／audit 完成；291 tests／0 skipped、165 Python 靜態、545 evidence、17 舊基準及 4 目視 hash 通過。
- 數值：五份真實來源兩模式 170 CSV，加上合成／199.5 Hz／真實訊號分段共 202 CSV；16 組比較最大誤差 0，int64 微秒精確一致，三張診斷圖目視通過。
- 下一步：盤點品質 scorer 的短窗、非有限與例外 fallback；保留既有數值，明示 invalid reason、preset／stage，先保存必要舊基準。
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
- Git：第十五階段完整驗收 `141bde2`；第十六階段基準／分段處理 `e8ebe18`。第十六階段後半及第十七階段均未提交；未 push，後續推送由使用者負責。

## 驗證入口

- 本輪：[第十七階段完整檢查](docs/refactor/validation/stage17/final/checks.json)：291 tests／0 skipped，165 Python 編譯／Pyflakes／bundle／diff 通過。
- [數值摘要](docs/refactor/validation/stage17/final/analysis.json)／[evidence](docs/refactor/validation/stage17/final/evidence_checks.json)：202 表重讀、16 組 NPZ 比較與 327 次 hash。
- [目視紀錄](docs/refactor/validation/stage17/final/visual_review.json)：真實、分段、199.5 Hz 三張診斷圖；四項持久 hash 通過，無未解驗收失敗。
- 前階段：[第十六階段完整檢查](docs/refactor/validation/stage16/final/checks.json) 269 tests／0 skipped，為當時版本證據；本輪完整檢查已包含既有回歸。
- 開發中依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [第十七階段驗收報告](docs/refactor/STAGE17_REPORT_2026-09-10.md)／[資料集契約](docs/refactor/STAGE17_DATASET_CONTRACT.md)／[實跑腳本](tools/validate_jenqwei_dataset_stage17.py)。
- 舊資料集基準 `tests/fixtures/dataset_jenqwei_*_reference.*`；摘要／完整 log／audit／三張圖在 `docs/refactor/validation/stage17/final/`，大型產物 `/tmp/lilia-stage17-final-v2/` 不保證永久存在。
- [第十六階段驗收報告](docs/refactor/STAGE16_ACCEPTANCE_REPORT_2026-09-10.md)；舊模型／PSD／STFT 基準維持 `tests/fixtures/jenqwei_*_reference.*`，不要與資料集混用。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
