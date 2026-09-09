# 第十六階段：Jenqwei 基準與分段處理起步

日期：2026-09-09。狀態：進行中，本次完成盤點／凍結基準／分段函式；main、IO、圖形及整階段驗收待續。

## 本階段變更

- 第十五階段已依使用者授權提交 `141bde2`（`Complete eye pipeline segmented plotting and stage 15 acceptance`），含八張目視 PNG；未 push。暫存區檢查發現原始測試 log 的空白行尾，保留原 log，其他檔案 whitespace 通過後提交。
- 盤點 [Jenqwei 入口](../../analyze_jenqwei_pipeline.py)：repo 無外部 `run_pipeline`／`analyze_csv` callers；main 對同一 CSV 的每個顯示通道重跑整套濾波／模型。實際 CLI 為 `--channels`，預設 0、1，非檔頭範例的單數 `--channel`。
- [五份真實來源](validation/stage16/inventory.json) 均連續、四個選用輸入通道有限：move head 29,984、talk 30,648、vibration level1 30,976、level2 32,312、level3 30,648 raw samples；只讀原始檔，未改動研究輸出。
- 由 `141bde2` 凍結 [舊入口](validation/stage16/baseline_legacy_entry.py.txt)、共享依賴／TFLite hash、五份完整真實與兩個合成基準；[捕捉工具](../../tools/capture_jenqwei_baseline_stage16.py) 只寫新目錄，持久 fixtures 在 `tests/fixtures/jenqwei_*_reference.*`。
- 新增 `process_segments`，尚未取代 `run_pipeline`：逐來源段 float32 bandpass／重採樣／400 點不重疊 TFLite 推論，分開保存 raw／Before／After 索引、原始 fractional index、model-to-before 映射及來源段 metadata。
- Before 保留各保留段的全部重採樣尾樣本；After 才裁至完整 400 點窗口。共用 `run_tflite_recording` 會同時裁 Before，不能直接套用。短於模型窗的來源段從三個 packed 陣列排除，仍保留原始 segment id／原因，全部無模型窗明確失敗。
- 保留 `max_samples` 的「先濾波完整來源段，再截原始前綴、重採樣」順序，metadata 另記完整 filter context。所有輸入非有限值（含短段／截斷後資料）明確失敗；模型錯誤含 segment／raw 範圍，不回傳部分成功結果。
- 原 main／TD／PSD／STFT 及 continuity guard 保留。既有 ch=2/3（0-based）會把 After clamp 到模型 ch=1，Before 卻用來源 ch=2/3，不能當作有效同通道比較；基準保存舊值並明示此問題，後續繪圖需明確處理來源映射。

## 驗證

- [專屬與共享測試](validation/stage16/checks.json)：**41 tests、0 skipped**，含 8 項新 Jenqwei 測試；覆蓋五份真實與合成凍結值、跨段隔離、短段／污染、模型第二段失敗、大 epoch、小缺口、尾端、max_samples 濾波順序與原 guard。
- [靜態](validation/stage16/static.json)：**154 Python** 編譯、Pyflakes、bundle、diff 通過；`python build_bundles.py` 為 0 copies updated。本次為處理增量，未跑階段 `--full`。
- 先跑上述測試，再新增 [數值驗證腳本](../../tools/validate_jenqwei_adapter_stage16.py) 並跑最終靜態／實跑 evidence；測試時的入口／測試／fixtures 未再改動，不將早期測試指紋宣稱為包含後加工具。
- [數值摘要](validation/stage16/adapter_analysis.json)：七個連續案例 filtered／Before／After **最大誤差 0**、整數時間精確相同。五個真實 After 行數依序 **11,600／12,000／12,000／12,800／12,000**；Before 為 **11,994／12,260／12,391／12,925／12,260**。
- 真實 move head 訊號副本切成 10／1,503／2,001／997 raw 點四段；兩短段排除，Before **1,403 × 4**、After **1,200 × 2**，兩保留段模型裁尾 **202／1** 點。與凍結舊入口逐段真實推論比較，數值／整數時間／來源映射最大誤差 **0**，epoch `9000000000000001` 未浮點化。
- 連續 PSD／STFT 以舊繪圖函式實際計算陣列凍結，再由新處理輸出重算，測試要求逐值精確一致。PSD 為 Welch density、最多 800 點、預設 Hann／半重疊／constant detrend；STFT 先截最多 60 秒、Hann 256／overlap 128／zero padding／`20*log10(abs(z)+1e-8)`。
- [持久基準 evidence](validation/stage16/baseline_fixture_evidence.json)／[結果](validation/stage16/baseline_fixture_evidence_checks.json)：**28 hashes**；[adapter evidence](validation/stage16/adapter_evidence.json)／[結果](validation/stage16/adapter_evidence_checks.json)：**55 checks = 47 hashes＋8 組 NPZ 比較**，浮點 tolerance `rtol=atol=1e-6`、整數精確比較。
- 本次未生成新 CLI 輸出表／研究 PNG，**未做來源驗證表重讀、分段繪圖或目視驗收**；NPZ 比較不替代這些後續工作。無測試失敗；TensorFlow 既有 interpreter deprecation warning 保留於 log，未變更 backend。
- [完整 log](validation/stage16/logs/)、[基準 log](validation/stage16/baseline.log)／[數值 log](validation/stage16/adapter.log) 已保存。大型產物在 `/tmp/lilia-stage16-baseline-v1/` 與 `/tmp/lilia-stage16-adapter-v1/`；fixtures／重要摘要／manifest 持久保存，暫存不保證永久存在。
- 可重現：`MPLCONFIGDIR=/tmp/lilia-mpl python tools/validate_jenqwei_adapter_stage16.py --out /tmp/lilia-jenqwei-new-run`，再 `python tools/refactor_check.py evidence /tmp/lilia-jenqwei-new-run/evidence.json`。

## 未解問題與下一步

- 接續 [任務 16](TASKS.md#stage-16)：main 每份來源只推論一次，共用結果畫各通道；新增可來源驗證的 Before／After 輸出、sidecar／失敗 audit。不可用同一 packed 索引推定 Before／After 對齊。
- TD／STFT 逐段、elapsed 與段尾／裁尾明示；PSD 不可對跨缺口串接陣列做 Welch，需保留逐段方法並明示呈現政策，避免暗加跨段平均。處理 ch3/4 比較誤映射及 `--max-sec` 的 elapsed／截窗語義，維持連續基準。
- 完成真實／合成分段 CLI、表重讀／hash／目視及 `--full` 後才能解除 guard。第十六階段新增程式、tests、fixtures、證據與文件隨本次提交收錄；未 push，第十七階段未開始。
