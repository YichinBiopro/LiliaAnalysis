# Python 第四階段修正報告 — 2026-09-07

本輪完成 **非 denoise 的獨立 `spectral_entropy.py --joint-mi` 分段遷移**：兩 channels、品質與 CSV 共用 WindowGrid；逐段 front-end；保存可核對來源／窗口的 metadata；series 與 peri-event 圖不跨錄製缺口或無效品質區間；whole-population surrogate 在各連續段內独立位移。只解除這條已驗證路徑的 continuity guard，神經網路和 baseline guard 仍在。

開始時 HEAD 為 `c76bc43`，第三階段修改尚未提交。本輪保留這些修改並繼續工作，沒有 commit、fetch 或 push。第三階段驗證 JSON 的 Python 指紋已成為歷史；目前版本指紋見 `PYTHON_FIX_VALIDATION_2026-09-07_STAGE4.json`。

## 1. 同一組窗口與品質遮罩

`compute_joint_mi_windowed` 接收 `WindowGrid`，沿原始樣本格點排除跨缺口窗口，中心時間取真實 timestamps。兩 channels 長度不同會失敗，不再默默截短。含非有限值的窗口回傳 NaN，保留窗口位置；CLI 的原始 CSV loader 本身仍拒絕非有限 EEG 樣本。

非 denoise CLI 在主流程建立同一組 grid，逐連續段濾波，再將相同 grid 傳給 joint MI 與 `compute_quality_windowed_aligned`。只有無法提供任何完整窗口的短段會略過濾波，原索引不壓縮。

品質沿用既有 **所選兩 channels 的 median**，沒有換評分方法。`~isfinite(quality) | quality < threshold` 都會遮罩兩個 MI 欄位；評分模組缺失或結果數量不符會明確失敗。只有 `--no-quality-mask` 才明確停用。新增 `quality_state`、`quality_valid` 與 `signal_valid`，保存評分狀態及數值可用性。

停用品質時 `quality_valid=True` 表示未被本次品質政策排除，不能解讀成已測得品質合格，必須同時查看 `quality_state=disabled`。

## 2. 可核對的 joint-MI metadata

`lilia/entropy_io.py` 將既有寫入／讀取驗證抽成共用內部函式；entropy 公共介面和既有來源／窗口核對規則保留。新增：

- `write_joint_mi_table`：輸出 `*_timeseries.csv` 與 `.csv.meta.json`，`kind=joint_mi`。
- `load_joint_mi_table(path, raw_csv, channels)`：核對 CSV、sidecar、來源內容、ordered channel pair，並以來源 timestamps 重建每個窗口索引、時間與 segment。

metadata 保存 fs、window/step、兩 channels、bandpass、品質參數與 reduction、MI bins/binning、程式指紋。joint-MI 和 entropy 的 kind 不同，不能互相冒用；刪掉 sidecar、交換 channels、混入同長度其他 raw，或改動索引後重算 table hash 都會被測試拒絕。

這是來源與設定一致性驗證，不是防偽簽章。沒有指定 `raw_csv` 時只能驗證表格／sidecar 自身；要核對來源與實際窗口，必須傳入 raw。CSV 與 sidecar 仍不是多檔案交易式發佈，中断可能留下不一致對，讀取端會拒絕，需重新產生。

## 3. 整體分布與 surrogate 的分段語義

品質遮罩只套用 windowed MI 時序。原本整體 histogram 與 pre/onset histogram 就沒有使用該窗口品質遮罩，本輪保留這一點，並在輸出、圖標題及 CLI 明示 `quality unmasked`／`population_quality_state=disabled`，避免將它們當成清潔資料統計。

非 denoise 的整體 histogram 使用 **有貢獻合法 grid 窗口的連續段內，所有成對有限樣本，每個原始樣本只算一次**；包含這些段的末端餘樣本，不重複計入重疊窗口。不參與分析的短段排除，summary 保存 `population_scope`、`population_runs`、`excluded_samples`。

`compute_joint_mi_significance(segment_ids=...)` 將 Y 在每個連續 run 中獨立 circular shift，然後以所有成對樣本估計原本的 pooled histogram MI。非有限值導致的 run 分界亦不跨越。這個 null 保留各段的邊際分布與樣本數，與跨整段錄製隨意 roll 的 null 不同；它是以段內結構為條件的對照，不宣稱所有形式的相依性都已被打散。

單個連續 run 的 RNG 抽樣次序完全保留，與改前 surrogate 數值一致。若任一 run 少於 16 個樣本，保留 observed MI，但停用 surrogate 並記錄 `null_state=segment_too_short`；使用者要求 0 次時記錄 `disabled`，正常則是 `computed`。不會把短 run 拼到另一段湊長度。

合成兩段（60/80 樣本）的固定比較：輸入 seed=9、8 quantile bins、7 次 surrogate、surrogate seed=5。

| 方法 | surrogate mean (bits) | std (bits) |
| --- | ---: | ---: |
| 舊的整體 circular shift | 0.3198232676547152 | 0.04990311992902855 |
| 各段獨立 circular shift | 0.24727320594067223 | 0.05029027299117802 |

完整 draws 與重現參數保存在驗證 JSON。這個對照展示方法確實改變，並非主張新舊 null 在有缺口資料上數值應相同。每次段內 shift 與手動逐段 `np.roll` 結果精確相等。

## 4. series 與 peri-event 繪圖

- 時序線插入繪圖用 NaN 切斷 segment，不改 CSV，不丟缺口後首個有效窗口。
- 平滑使用 `transform_runs`，遇到錄製分段或 NaN 品質就分開計算。
- peri-event 插值只在同一連續有效 run 中進行，不跨缺口或無效品質內插出假的 MI。
- static 與 interactive 共用相同的相對時間格點及 event mean/std 計算；加入 run 邊界和缺口中點作為繪圖格點，因此即使缺口比設定 step 小，也有明確斷點。單點 run 保留於其實際相對時間。
- Plotly 設定 `connectgaps=False`，animation frame ID 使用唯一索引，避免新增非整數時間點的四捨五入名稱碰撞。

新增格點只用於展示，不增加分析窗口或統計樣本。本輪沒有改變活動參與名單與既有 pre/onset 完整區間政策。互動 HTML 成功產生；未執行瀏覽器自動化，HTML 延用既有 Plotly CDN 載入方式。

## 5. 驗證

- **87/87 tests 通過**；新增 `tests/test_joint_mi_windows_regression.py` 的 11 個測試。
- 修改前基準保存於 `tests/fixtures/joint_mi_continuous_reference.json`，記錄第三階段未提交版本的 `spectral_entropy.py` SHA-256。seed=427、4000×2、200 Hz、2 秒窗口／1 秒 step、8 quantile bins、12 次 surrogate。
- joint histogram 全部欄位、windowed MI、quality、每次 surrogate 與摘要統計，和基準在 `rtol=atol=1e-12` 內一致。
- 非格點缺口與 timestamp jitter 的窗口索引、真實中心時間，逐個和手動切片比較。
- 品質 NaN/低分保留原窗口列；mock scorer 收到的樣本與 loader 的 float32 原始樣本精確相等。
- 缺品質模組、品質列數不符、錯來源／channel pair／kind、缺 sidecar、重新計算 hash 後的錯窗口，都有拒絕測試。
- 段內 surrogate、短 run 停用、非有限窗口、短段不濾波、缺口後首點與不跨段平滑／插值皆有回歸案例。
- 原本針對尚未遷移 joint-MI 的 guard 測試改為 `--joint-mi --denoise`；另驗證 baseline guard 仍在。
- Pyflakes、**91 個 Python 編譯**、bundle 同步與 whitespace 檢查通過。

## 6. 完整 Hardy

輸入 `iBrainCenter/Hardy(SN036)/merged.csv`，2,126,712 樣本、6 段；fs=500、win=step=2 秒、ch1/ch2、quality threshold=0.5、16 quantile bins。

| 項目 | 結果 |
| --- | ---: |
| 保存合法窗口 | 2,121 |
| 品質通過／遮罩 | 1,975／146 |
| 最後中心真實時間 | 4,791.920551 秒 |
| 整體 histogram 樣本 | 2,126,712 |
| histogram MI | 0.1325492871326368 bits |
| pre/onset 活動比較 | 8 個 |

品質數量與普通 entropy 第二階段的 1,944 不同：joint-MI 沿用所選兩 channels median，普通 entropy 當時評分使用所有輸入 channels，不能將不同評分集合的數字當成回歸失敗。

成功產生 distribution、excess、timeseries、events、peri-event 的 PNG/SVG、interactive HTML、summary CSV、events CSV/audit、timeseries CSV/metadata；重新用 `load_joint_mi_table(..., source, [1,2])` 完成來源與全部窗口核對。目視檢查時序圖確認大型錄製缺口保留。

真實資料本輪用 **5 次 surrogate 作流程驗證**，不是用這組 p/z 作研究判讀，程式預設次數未改。

## 重現與下一步

```bash
MPLCONFIGDIR=/tmp/lilia-stage4-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
python build_bundles.py --check
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
MPLCONFIGDIR=/tmp/lilia-stage4-mpl MPLBACKEND=Agg python spectral_entropy.py \
  --csv 'iBrainCenter/Hardy(SN036)/merged.csv' --joint-mi --ibrain-events \
  --peri-event-html --mi-surrogates 5 --out /tmp/lilia-stage4/hardy
```

驗證紀錄：`PYTHON_FIX_VALIDATION_2026-09-07_STAGE4.json`；暫存產物：`/tmp/lilia-stage4/hardy/`；log：`/tmp/lilia-stage4-all-tests.log`、`/tmp/lilia-stage4-hardy.log`。正式錄製與舊研究產物未覆寫。

下一步是神經網路路徑的實際輸出時間軸與分段推論。先讀 `denoise_channels`、`lilia/tflite.py`、`data_analysis.py` 的 filters/resampling/model outputs，明確保存原始 segment 與推論窗口／輸出對應，禁止僅用輸出列數與名目 fs 猜測缺口時間。完成連續基準與缺口驗證後，才解除 `--joint-mi --denoise` guard。

baseline／clean 的不相鄰 1 秒 epochs 拼接問題仍待續；品質 scorer 內部短窗/fallback、band-event 品質政策、大型模組拆分與 artifact 原子發佈亦未在本輪處理。
