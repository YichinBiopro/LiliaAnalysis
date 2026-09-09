# 重構短進度

更新：2026-09-09。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析入口遷移至第十五階段（眼開閉）。
- 當前：[任務 16](docs/refactor/TASKS.md#stage-16) Jenqwei 分析，待盤點，尚未開始實作。
- 本輪：眼開閉分段 TD／STFT、elapsed 軸及 ch5/6 映射驗收完成；246 tests／0 skipped、靜態、88 項 evidence 與 9 項持久目視產物 hash 通過，CLI guard 解除已確認。
- 數值：真實連續 18,109 列、真實訊號分段副本 1,403 列；合成連續／小缺口 602／800 列；模型與 STFT dB 最大誤差 0，整數時間精確一致。
- 下一步：盤點 `analyze_jenqwei_pipeline.py` callers、通道／BP／重採樣／TFLite 與 TD／PSD／STFT 契約，先凍結真實連續模型基準。
- 順序：16 Jenqwei 分析 → 17 Jenqwei 資料集；細節於接手盤點。
- 其後：品質短窗／fallback、baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 直接分析單一 raw channel；無 BP、模型、品質 scorer 或 baseline。無 timestamp 共用 helper 保留舊語義。
- qEEG 保留 `int(win_sec*fs)` 與原始全域格點；段首／短尾明示排除，圖為 elapsed，CSV 保留原始微秒。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY、qEEG CLI、眼開閉已完成，勿重寫。
- 眼開閉只做 float32 bandpass，不混入 notch／bandstop；模型輸出依序為來源 ch1/2/5/6，before 比較圖取來源 ch1/2/5/6；STFT 數值保留原法、逐段繪於 elapsed 軸。
- 不覆寫原始錄製、模型或舊圖。bundle 改主版後執行 `python build_bundles.py`，再 `--check`。
- Git：第十四階段 `96fee9d`；第十五階段基準與 CLI／IO `b84029e`。繪圖、測試、完整驗收工具與文件未提交；本輪未 push，後續推送由使用者負責。

## 驗證入口

- 本輪：[第十五階段完整檢查](docs/refactor/validation/stage15/final/checks.json)：246 tests／0 skipped，151 Python 編譯／Pyflakes／bundle／diff 通過。
- [數值摘要](docs/refactor/validation/stage15/final/analysis.json)／[產物驗證](docs/refactor/validation/stage15/final/evidence_checks.json)：5 表重讀、74 hashes、9 組 NPZ 比較；[目視紀錄](docs/refactor/validation/stage15/final/visual_review.json)／[持久目視 hash](docs/refactor/validation/stage15/final/visual_evidence_checks.json)。
- 直接檢視六張正式 CLI 圖、兩張細節圖；小缺口與短前段放大確認，無未解驗收失敗。初次補充視圖版面失敗及修正另留檔。
- 前階段：[第十四階段完整檢查](docs/refactor/validation/stage14/checks.json)，221 tests／0 skipped，為當時版本證據；完整 Hardy 6 段、845 有限窗口、5 個跨缺口格點排除。
- 開發中依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [當前與後續任務單](docs/refactor/TASKS.md)／[報告模板](docs/refactor/REPORT_TEMPLATE.md)。
- [第十五階段驗收報告](docs/refactor/STAGE15_ACCEPTANCE_REPORT_2026-09-09.md)／[實跑驗證腳本](tools/validate_eye_stage15.py)。
- 眼開閉凍結基準在 `tests/fixtures/eye_*reference.*`；本輪大型產物在 `/tmp/lilia-stage15-acceptance-v1/`，暫存不保證永久存在；摘要、完整 log、audit 及八張目視圖保存在 `docs/refactor/validation/stage15/final/`。
- [起步基準](docs/refactor/STAGE15_REPORT_2026-09-09.md)／[CLI／IO 歷史增量](docs/refactor/STAGE15_IO_REPORT_2026-09-09.md) 的未完成描述不是目前狀態。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；其中歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
