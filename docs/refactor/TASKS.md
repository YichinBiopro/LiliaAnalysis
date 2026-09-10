# 階段任務單

只讀當前卡片。14–17 已完成；下一工作單元為品質 scorer，待盤點、尚未編號。
所有任務共用 [WORKFLOW](WORKFLOW.md) 的驗證與收尾規則。方法校準另立任務，勿混入等價搬移。

<a id="stage-14"></a>
## 14：qEEG CLI — 已完成（2026-09-08）

- 成果：221 tests／0 skipped、完整 Hardy 845 窗口、三入口表相同；表重讀／hash／數值／目視已完成，CLI guard 已解除。[增量報告](STAGE14_REPORT_2026-09-08.md)。

- 目標：`lilia/qeeg.py`、`qeeg_indices.py`、必要的新 IO／測試與生成 bundle。
- 先讀：上述兩檔、`lilia/windowing.py`；用 `rg` 盤點 `compute_qeeg_indices_windowed` callers。
- 保留：單一 raw channel 的 theta／alpha／beta 相對功率、四指標、連續窗口中心、summary／PNG；無 BP／TFLite／品質／baseline。無 timestamp 的共用 helper 保留既有語義。
- 實作：保留整數微秒；WindowGrid 原始索引／segment／真實時間；不跨缺口窗口；短尾、非有限與無合法窗口有 audit。CSV/meta 核對來源／channel／設定／全部窗口。兩面板圖斷線且孤立窗口可見。
- 完成條件：舊連續數值基準通過；缺口、短段、非有限、錯誤 channel、邊界案例通過；全短／全排除明確失敗；完整有缺口 Hardy 真實分析；主版、相容 CLI 與 bundle 核對；表重讀、產物 hash、目視完成後才解除 guard。
- 起始測試：`python tools/refactor_check.py check --tests tests/test_signal_contract_regression.py tests/test_time_utils_regression.py tests/test_entropy_windows_regression.py tests/test_qeeg_cli_regression.py`
- 已新增入口專屬測試並加入命令；階段完成依據另含真實資料、產物與目視證據。
- 收尾：完整檢查＋實跑 evidence manifest＋增量報告；更新短進度至任務 15。

<a id="stage-15"></a>
## 15：眼開閉 — 已完成（2026-09-09）

- 成果：[完整驗收報告](STAGE15_ACCEPTANCE_REPORT_2026-09-09.md)。分段 TD／STFT、elapsed 軸、ch5/6 來源映射與標題完成；CLI guard 的解除已通過驗收。
- 驗證：246 tests／0 skipped、151 Python 靜態／bundle／diff；88 項數值／產物 evidence、9 項持久目視產物 hash、六張正式圖與兩張細節圖目視通過。
- 真實連續 18,109 列、真實訊號分段副本 1,403 列、合成連續 602 列及小缺口 800 列；模型與 STFT dB 最大誤差 0，整數時間精確一致。
- 保留：八通道映射、float32 bandpass、兩組四通道 PyTorch 輸入與 ch1/2/5/6 輸出、鏡射 OLA、完整重採樣尾樣本；不引入 notch／bandstop 或品質 scorer。
- 專屬測試：`tests/test_eye_open_close_regression.py`、`tests/test_eye_plot_regression.py`；實跑：`tools/validate_eye_stage15.py --out /path/to/new-directory`。
- 歷史：[起步基準](STAGE15_REPORT_2026-09-09.md)／[CLI／IO 增量](STAGE15_IO_REPORT_2026-09-09.md)。其中 guard 保留、繪圖未完成均為當時狀態。
- Git：基準與 CLI／IO 已提交 `b84029e`；繪圖／測試／完整驗收證據已提交 `141bde2`，未 push。

<a id="stage-16"></a>
## 16：Jenqwei 分析 — 已完成（2026-09-10）

- 成果：[完整驗收報告](STAGE16_ACCEPTANCE_REPORT_2026-09-10.md)。main 每來源只處理一次；Before／After 專屬 CSV／sidecar／reader／audit、分段 TD／PSD／STFT、elapsed 與裁尾完成。
- 驗證：269 tests／0 skipped、160 Python 靜態／bundle／diff；217 項 evidence（18 表重讀／25 組 NPZ）、13 項目視 hash；五份真實與合成／分段案例數值最大誤差 0，十二張圖目視通過。
- 保留：四通道 float32 bandpass、500→200 Hz、400 點不重疊 TFLite；Before 全尾端／After 段內裁尾、max_samples 先濾完整段再截斷、raw／Before／After 各自索引。PSD 逐段顯示，無跨缺口串接或新增平均。
- 通道：顯示索引 0/1 比較真正來源 ch1/2；2/3 僅顯示來源 ch3/4 Before，After 明示不存在。STFT 顯示 ≤128 點時明示省略，未改方法。
- Guard：分段 main 已驗收可接受缺口，舊 `run_pipeline` 的 continuity guard 保留；不影響其他入口 guard。
- 專屬測試：`tests/test_jenqwei_pipeline_regression.py`、`tests/test_jenqwei_output_regression.py`；完整實跑 `tools/validate_jenqwei_stage16.py --out /path/to/new-directory`。
- 歷史：[起步報告](STAGE16_REPORT_2026-09-09.md) 已提交 `e8ebe18`；本次 CLI／IO／繪圖／驗收修改未提交，未 push。

<a id="stage-17"></a>
## 17：Jenqwei 資料集 — 已完成（2026-09-10）

- 成果：[驗收報告](STAGE17_REPORT_2026-09-10.md)／[資料集契約](STAGE17_DATASET_CONTRACT.md)。先等分再裁窗口，逐來源段獨立處理；來源／原始與重採樣索引／窗口／裁尾與排除原因均可回讀。
- 盤點：訊號準備，不執行模型或 train/test split；專案內未找到其他下游讀取者。保留連續 CSV、檔名與舊欄位，另明示真實窗口數及實際採樣率。
- 驗證：291 tests／0 skipped、165 Python 靜態／bundle／diff；545 evidence checks（327 hash、202 表重讀、16 組 NPZ），另 17 項舊基準及 4 項持久目視 hash。
- 數值：五份真實資料兩模式共 170 CSV；連同合成、199.5 Hz 與真實訊號分段共 202 CSV，float32 值與 int64 微秒最大誤差 0。
- 邊界：22 項專屬測試，含窗口錨點、短段／padlen、污染、同名來源、上採樣、manifest 缺漏／錯序及來源／產物篡改；全排除仍失敗。三張診斷圖目視通過。
- Guard：入口已完成分段遷移與驗收，可接受缺口；不改其他入口 guard。`--allow-short-drop` 才允許排除；既有輸出不覆寫。
- 實跑：`python tools/validate_jenqwei_dataset_stage17.py --out /path/to/new-directory`；專屬測試 `tests/test_jenqwei_dataset_regression.py`。
- Git：本階段與第十六階段後半修改尚未提交，未 push。下一工作單元為下表品質 scorer，尚未開始。

## 入口遷移後的待辦（尚未編號）

| 工作單元 | 接手範圍／完成依據 |
| --- | --- |
| 品質 scorer | 短窗、非有限與例外 fallback；invalid reason、preset／stage 可追蹤與數值案例。 |
| 方法校準 | baseline policies、PSD profiles、Goertzel 門檻／normalization、MI estimator／surrogates；保留舊 profile 及比較證據。 |
| 架構與 F19 | 大入口拆分、移除計算對繪圖私有函式的依賴、明示 deprecated 參數的相容策略。 |
| 批次與產物 | 批次失敗報告、來源／設定追蹤、CSV／sidecar／圖形成套原子發佈。 |
| 整體驗收 | 所有入口、真實基準、必要方法差異、bundle、文件與發佈狀態；提交／推送依當時授權。 |
