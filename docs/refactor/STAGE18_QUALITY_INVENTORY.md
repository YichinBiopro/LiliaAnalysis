# 第十八階段品質 scorer 盤點與相容契約

更新：2026-09-11。核心、共用 raw 及 TFLite baseline／summary 增量完成；其餘直接／filtered 呼叫端待完成。

## 現況與範圍

- 唯一主版：`lilia/quality.py`；`eeg_quality_v2.py` 是相容 shim。`build_bundles.py` 生成 `plot_index_vs_raw_bundle/lilia/quality.py` 與 `signal_quality_package/quality.py`。
- 四種現有 presets：default、mean_abs_corr、flat_spectrum_only、ibrain_device；本次不改權重、門檻、PSD slope、濾波或任何校準。
- 直接呼叫者：`plot_event_markers.py`、`plot_goertzel_vs_raw.py`、`plot_tflite_summary.py`、`spectral_entropy.py`、`quality_check.py`；`lilia/event_qeeg.py` 接受注入 scorer，供 event／zoom／subject comparison／summary 使用。`lilia/__init__.py` 另有公開 re-export。
- event qEEG 的品質輸入是映射回原始時間區間的 raw channels；entropy clean 的品質輸入是 filtered channels。不能看到 preset 就推斷 raw／BP，也不能把不同入口的 stage 混用。
- `quality_check.py` 同時有 raw／BP 對照及異常窗口評分；Goertzel 使用 BP 訊號、另以 raw 做 hard-artifact features。抽樣／舊 helper 的缺口政策不在本核心增量重寫。

## 已確認的邊界

- flat activity 使用 `range(0, max(n_samples-window_size, 0), step)`，stop 不含最後邊界；恰好 0.5 秒沒有 activity subwindow。新增包含末端的窗口會改數值，本次保留舊 loop。
- spectrum 少於兩個 fit bins 或 Welch／polyfit 例外回傳 0.5；非有限 slope 另可能得到 NaN。0.5 是舊替代值，不保證評分成功。
- kurtosis 非有限時回傳設定 floor，例外回傳 0.5；常數訊號在啟用 kurtosis 時會走 floor。flat-only 的常數訊號則可得到有效低分，不能把「低品質」與「無法評分」合併。
- correlation 例外改用 identity matrix；非有限或常數通道可能讓其他通道的相關評分也無效。單通道固定 1.0 是原設計，另在 context 明示。
- 零權重 component 不計算，也不因該 component 的例外或不可用而使結果無效。受污染輸入本身始終標為無效。

## 新增的相容診斷

- `overall`／`detail` 保留舊分數與 fallback；新欄位不自動改變既有呼叫端的接受／排除結果。
- `valid`、`invalid_reasons` 是每通道有效性／原因；`component_valid`、`component_reasons` 保留各啟用 component 的細節，例外記錄型別。
- `usable_overall` 將無效通道設為 NaN，供後續明確選擇採用；它不取代 `overall`，更不是品質門檻。
- `context` 包含 profile、推斷自有效完整參數的 preset 名稱、完整 parameters、fs、active components、樣本／通道數、相容 fallback／窗口政策及 config hash。
- 新增 keyword-only `stage`；未提供時明示 `unspecified`，不猜測處理階段。fs 非有限／非正、零通道、空 stage 拒絕；設定 metadata 要求可序列化的有限浮點參數。
- deprecated parameters 仍只作相容 metadata；不因新增 diagnostics 啟用它們。

## 呼叫端完成範圍

| 路徑 | 品質輸入／目前狀態 |
| --- | --- |
| event qEEG → event markers／zoom／subject comparison | raw 映射切片；逐窗診斷、CSV／sidecar／audit／來源 reader 完成，event 品質圖標記與實際 preset 標題完成。 |
| TFLite baseline／summary | baseline filtered 500 Hz 四通道；Before filtered_resampled 200 Hz ch1/2；After model_output 200 Hz。診斷／CSV／audit／來源 reader／品質圖完成，legacy sampler 診斷亦保存。 |
| entropy clean／state／MI | filtered 品質路徑，待串接。 |
| Goertzel | BP 品質與 raw hard-artifact features，待串接。 |
| quality_check／舊直接 helper | raw／BP 對照與抽樣路徑，待逐一串接；event 舊 `compute_quality_windowed` 不等同已驗收的 main 路徑。 |

- [raw 呼叫端增量](STAGE18_RAW_CALLERS_REPORT_2026-09-11.md) 固定 `legacy_overall`，診斷有效性與分數門檻分別保存；短窗、非有限與 scorer 例外有原因，外部舊 scorer 明示 unavailable。
- `lilia/quality_audit.py` 在呼叫端綁定實際 raw stage，保留注入 scorer 舊簽名；新表宣告 `quality_diagnostics_version=1`，四個 CSV 診斷欄位與 sidecar 逐窗資料必須一致。舊表缺少整組診斷時仍可讀，不推定有效。
- 來源 reader 重算 raw slice 的一般 diagnostics；歷史例外僅核對 fallback 與其餘可重算部分，無法由產物證明當時例外必然再發生。310 tests／33 表／10 新舊數值組／三圖完成，不代表其他 caller 已驗收。
- [TFLite 呼叫端增量](STAGE18_TFLITE_CALLERS_REPORT_2026-09-11.md) 完成 317 tests／18 新舊表／9 數值組（399 陣列）／三圖；同樣保留 `legacy_overall`，半秒無可用 baseline 的失敗結果與有限 fallback 接受結果皆不變。
- TFLite source reader 重建 filtered／resampled 輸入、raw 削波與 baseline 短路順序及 seed 選取；After 不重跑模型，只核對模型身分與診斷一致性。模型及 baseline qEEG 數值由獨立凍結舊碼實跑比較，未以 reader 代替。

## 剩餘增量完成條件

- 逐呼叫端保存 raw／filtered stage 與診斷，保留注入 scorer／舊 API 相容性；各種 CSV／audit reader 要能核對新欄位，不能只在 scorer 回傳後丟棄。
- 對照既有接受／排除窗口，明示是否採用 `usable_overall`；任何改變都須記錄具體窗口與原因，不在 metadata 重構中暗改篩選政策。
- 確認品質圖能區分無效／低分及例外；真實資料、污染／短窗／強制例外與完整檢查完成後，才把第十八階段標為完成。
