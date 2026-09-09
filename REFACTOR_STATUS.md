# 重構短進度

更新：2026-09-09。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析入口遷移至第十四階段（qEEG CLI）。
- 當前：第十五階段眼開閉進行中；main 已接入分段函式，CSV sidecar／來源模型索引重讀／失敗 audit 完成，guard 保留。
- 本輪：53 tests／0 skipped、靜態與 38 項 evidence 通過；真實 CLI 18,109 列、真實訊號分段 IO 1,403 列，舊模型最大誤差 0。
- 下一步：[任務 15](docs/refactor/TASKS.md#stage-15)，分段 TD／STFT 與 elapsed 軸；修正 ch5/6 圖形映射，完成分段 CLI、目視及完整階段驗收。
- 順序：15 眼開閉 → 16 Jenqwei 分析 → 17 Jenqwei 資料集；15–17 細節於接手盤點。
- 其後：品質短窗／fallback、baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 直接分析單一 raw channel；無 BP、模型、品質 scorer 或 baseline。無 timestamp 共用 helper 保留舊語義。
- qEEG 保留 `int(win_sec*fs)` 與原始全域格點；段首／短尾明示排除，圖為 elapsed，CSV 保留原始微秒。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY、qEEG CLI 已完成，勿重寫。
- Git：第十四階段已提交 `96fee9d`；第十五階段基準與 CLI／IO 已提交 `b84029e`，繪圖與完整驗收待續。後續推送由使用者負責，不再代為 push。
- 眼開閉只做 float32 bandpass，不能混入共用 denoise 的 notch／bandstop；模型輸出依序為來源 ch1/2/5/6。
- 不覆寫原始錄製、模型或舊圖。bundle 改主版後執行 `python build_bundles.py`，再 `--check`。

## 驗證入口

- 本輪：[第十五階段 CLI／IO 測試](docs/refactor/validation/stage15/cli_io/checks.json) 53 tests／0 skipped；[靜態](docs/refactor/validation/stage15/cli_io/static.json) 149 Python／Pyflakes／bundle／diff 通過；尚未做完整階段驗收。
- 前階段：[第十四階段完整檢查](docs/refactor/validation/stage14/checks.json)，221 tests、0 skipped；為當時版本證據。
- [數值摘要](docs/refactor/validation/stage14/analysis.json)／[產物驗證](docs/refactor/validation/stage14/evidence_checks.json)：12 表重讀、75 hashes、7 組數值對照；[最終目視紀錄](docs/refactor/validation/stage14/visual_review.json)。
- 完整 Hardy：6 段、845 有限窗口、5 個跨缺口格點排除；最後中心 elapsed 4788.420551 秒。主版／相容／bundle 表完全相同。
- 開發中：依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [當前與後續任務單](docs/refactor/TASKS.md)／[報告模板](docs/refactor/REPORT_TEMPLATE.md)。
- [第十五階段 CLI／IO 報告](docs/refactor/STAGE15_IO_REPORT_2026-09-09.md)／[數值及失敗案例](docs/refactor/validation/stage15/cli_io/analysis.json)；ch5/6 舊圖仍有已知錯誤，目視未驗收。
- [起步與凍結基準](docs/refactor/STAGE15_REPORT_2026-09-09.md) 在 `tests/fixtures/eye_*reference.*`；本次大型產物在 `/tmp/lilia-stage15-cli-io-v1/`。
- [第十四階段增量報告](docs/refactor/STAGE14_REPORT_2026-09-08.md)／[實跑驗證腳本](tools/validate_qeeg_stage14.py)。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；其中歷史下一步、Git 描述與指紋不代表當前版本。
- `/tmp/lilia-stage14-review/` 保存最終產物；暫存可能消失，持久基準在 `tests/fixtures/qeeg_*reference.*`，摘要／完整 log／代表圖在 `docs/refactor/validation/stage14/`。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
