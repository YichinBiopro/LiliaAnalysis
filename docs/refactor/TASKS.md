# 階段任務單

只讀當前卡片。14–15 已完成；當前為 16，待盤點。16–17 細節於接手時確認。
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
- Git：基準與 CLI／IO 已提交 `b84029e`；繪圖／測試／本次驗收工具及證據未提交，未 push。

<a id="stage-16"></a>
## 16：Jenqwei 分析 — 待盤點

- 目標：`analyze_jenqwei_pipeline.py`、必要的共享 TFLite／IO。
- 保留：現有通道選擇、BP／重採樣／TFLite 及 TD／PSD／STFT 定義；先確認 callers 與真實模型基準。
- 實作：逐段模型流程、來源索引與圖形時間、短尾／失敗 audit；盤點重複推論是否可共用結果。
- 起始測試：`python tools/refactor_check.py check --tests tests/test_neural_timeline_regression.py tests/test_signal_contract_regression.py tests/test_pipeline_output_regression.py`
- 完成條件：專屬連續／缺口／短段／污染／模型失敗測試、真實 Jenqwei 模型數值對照、產物重讀／hash／目視及完整檢查；完成後才解除 guard。

<a id="stage-17"></a>
## 17：Jenqwei 資料集 — 待盤點

- 目標：`build_jenqwei_tflite_dataset.py`、切片輸出／manifest。
- 保留：現有資料集切片與模型窗口契約；先盤點切分單位及下游讀取者，不猜測它是 train/test split。
- 實作：每個片段與模型窗口都在同一來源段，保留 source／segment／原始及重採樣索引、裁尾和排除原因。
- 起始測試：`python tools/refactor_check.py check --tests tests/test_neural_timeline_regression.py tests/test_csv_contract_regression.py tests/test_segment_sampling_regression.py`
- 完成條件：專屬窗口邊界／短段／污染案例、真實資料集生成與逐片段回讀、manifest 對齊、完整檢查；完成後才解除 guard。

## 入口遷移後的待辦（尚未編號）

| 工作單元 | 接手範圍／完成依據 |
| --- | --- |
| 品質 scorer | 短窗、非有限與例外 fallback；invalid reason、preset／stage 可追蹤與數值案例。 |
| 方法校準 | baseline policies、PSD profiles、Goertzel 門檻／normalization、MI estimator／surrogates；保留舊 profile 及比較證據。 |
| 架構與 F19 | 大入口拆分、移除計算對繪圖私有函式的依賴、明示 deprecated 參數的相容策略。 |
| 批次與產物 | 批次失敗報告、來源／設定追蹤、CSV／sidecar／圖形成套原子發佈。 |
| 整體驗收 | 所有入口、真實基準、必要方法差異、bundle、文件與發佈狀態；提交／推送依當時授權。 |
