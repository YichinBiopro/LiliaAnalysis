# 重構短進度

更新：2026-09-08。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析入口遷移至第十三階段（TYY meditation）。
- 當前：第十四階段 qEEG CLI，尚未開始實作。
- 本輪：落實精簡交接、任務單、驗證腳本、按變更驗證與增量報告。
- 下一步：[任務 14](docs/refactor/TASKS.md#stage-14)，先保存既有 CLI 連續數值／時間基準。
- 順序：14 qEEG → 15 眼開閉 → 16 Jenqwei 分析 → 17 Jenqwei 資料集；15–17 為待接手盤點的任務拆分。
- 其後：品質短窗／fallback、baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 直接分析單一 raw channel；無 BP、模型、品質 scorer 或 baseline。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY 已完成，勿重寫。
- 本次提交範圍：第九至十三階段及流程工具；實際提交／工作區狀態以 `git log -1`、`git status --short` 為準。
- 使用者於 2026-09-08 明確要求本次 commit／push；歷史報告的「未提交」為撰寫當時狀態，不代表本次發佈結果。
- 不覆寫原始錄製、模型或舊圖；後續發佈依當次授權與審核結果執行。
- bundle 改主版後執行 `python build_bundles.py`，再 `--check`。

## 驗證入口

- 最新：[流程工具驗證](docs/refactor/validation/workflow_checks.json)，206 tests、0 skipped、138 Python 編譯、Pyflakes／bundle／diff 通過。
- 第十三階段歷史基準為 197 tests；新增 9 項工具測試，科學分析程式本輪未改。
- 開發中：`python tools/refactor_check.py check --tests tests/test_signal_contract_regression.py tests/test_time_utils_regression.py`
- 階段完成：`python tools/refactor_check.py check --full`
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。工具輸出 log 路徑與短摘要。

## 按需參考

- [當前與後續任務單](docs/refactor/TASKS.md)／[報告模板](docs/refactor/REPORT_TEMPLATE.md)。
- [本輪流程改善報告](docs/refactor/WORKFLOW_REPORT_2026-09-08.md)；未開始第十四階段實作。
- [第十三階段報告](PYTHON_FIX_REPORT_2026-09-08_STAGE13.md)／[數值證據](PYTHON_FIX_VALIDATION_2026-09-08_STAGE13.json)。
- [完整歷史交接](REFACTOR_HANDOFF.md)／[原始審查](PYTHON_REVIEW_2026-09-06.md)；其中歷史行號與指紋不代表當前版本。
- `/tmp/lilia-stage13/` 保存當時真實／合成产物；暫存可能消失，持久基準在 `tests/fixtures/`。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
