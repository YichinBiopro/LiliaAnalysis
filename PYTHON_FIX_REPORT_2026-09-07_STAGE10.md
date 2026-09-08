# Python 修正報告：第十階段

日期：2026-09-07。範圍：`plot_event_markers.py --hardy2-analysis` 特殊入口。

本階段已完成。第九階段修改完整保留；第九、十階段仍在工作區，本階段沒有 commit／push。整體重構尚有其他入口及工程收尾。

## 問題與修正

Hardy_2 特殊分支原本整段 BP 濾波、按樣本數起窗，再跨整份指標陣列平滑；繪圖僅按中心時間的大缺口補 NaN。時段摘要用窗口中心分類，因此跨越 Wear Device／Light On／End 邊界的窗口會混入其中一個時段。

新增 `lilia.hardy2`，沿用共享 `WindowGrid`，讓每個來源連續段重新起完整窗口，保存原始索引、整數微秒時間、原 segment ID、短段與未滿窗口的尾端。BP 只在有候選窗口的段內執行，平滑只使用同一段內相鄰且有限的指標。

保留此分支的原始算法：每通道 Welch 積分 delta 1–4、theta 4–8、alpha 8–13、beta 13–30、gamma 30–45 Hz，以五頻帶功率和正規化，再取通道 median；四個 qEEG 指標仍獨立沿用既有三頻帶公式。沒有改用一般 event markers 的品質／TFLite pipeline。

## 明示的窗口與時段政策

- 預設 500 Hz、5 秒非重疊窗口，段末不足窗口不運算。即使缺口很小，原始 segment ID 也會讓窗口、平滑和圖形保持分離。
- BP 模式中，來源段只要含非有限 raw sample，該段所有候選窗口均記為 `nonfinite_segment`；filter error 同樣保存原因。未濾波的 helper 模式逐窗口檢查有限值。指標失敗保留 NaN 列與原因，不以剩餘通道補成有效窗口。
- 本分支原本沒有品質評分，現在 CSV、audit 與圖上明示 `quality_state=disabled`。`metric_valid` 只表示成功產生有限數值，不等於品質合格。
- 四個時段使用半開區間，只有完整落在該時段內的窗口才計入平均。跨事件邊界的窗口仍留在時序圖及 CSV，但列入 `boundary_crossing_rows`，不算入任何時段。
- 時段平均沿用「先取每窗口的通道 median，再對未平滑窗口取 mean」。5 窗口中心平滑只供顯示；同一有效連續段內仍可跨事件邊界平滑，不用平滑值計算時段平均。
- 時段底色只涵蓋相鄰、有效、完整納入的窗口範圍；沒有資料、被排除窗口與跨事件窗口保持未著色。孤立有效窗口保留可見點。
- 圖表以原始 UTC timestamps 轉 UTC+8 顯示，不再次加入 header offset。X 軸限於來源範圍，範圍外事件不擴張畫面。

## 輸出與使用

保留五張頻帶圖與一張四指標圖，各輸出 PNG／SVG，共 12 張圖。新增：

- `Hardy_2_SN036_metrics.csv` 與 `.csv.meta.json`：全部候選窗口，包括被排除列；每通道數值、通道 median、平滑值、有效狀態、原始索引與時間、source／config identity。
- `Hardy_2_SN036_analysis.json`：成功／失敗狀態、來源與程式 hash、設定、每段短尾／排除、逐窗口原因、四時段候選／接受／排除 rows、繪圖 spans、時段 means 及產物 hashes。

`lilia.hardy2_io.load_hardy2_table(table, raw_csv)` 核對 CSV／設定／analysis／raw 指紋，從 raw 重建分段窗口及短尾，核對各欄映射、通道 median、有效 run 平滑，以及完整時段 rows／spans／mean。這是來源、輸出與摘要的一致性驗證；不會重跑 BP／Welch 來重新認證每個通道指標值。

沒有完整窗口時只保存失敗 audit；有窗口但全數無效時保留 CSV/meta 與 audit，CLI 非零退出。來源檔案不存在也保存失敗 audit。

預設路徑保持相容；新增 `--hardy2-csv` 可明確指定對應 Hardy_2 session 的來源，bundle 不必自帶錄製資料：

```bash
MPLCONFIGDIR=/tmp/lilia-stage10-mpl MPLBACKEND=Agg \
python plot_event_markers.py --hardy2-analysis \
  --hardy2-outdir /tmp/lilia-stage10/real

MPLCONFIGDIR=/tmp/lilia-stage10-mpl MPLBACKEND=Agg \
python plot_index_vs_raw_bundle/plot_event_markers.py --hardy2-analysis \
  --hardy2-csv 'iBrainCenter/Hardy_2(SN036)/merged.csv' \
  --hardy2-outdir /tmp/lilia-stage10/bundle_real
```

## 驗證結果

**155 項測試通過，0 skipped；121 個 Python 檔案編譯、Pyflakes、bundle 一致性、diff whitespace 全通過。** 新增 13 項測試，涵蓋連續 raw／BP 基準、缺口與段間獨立性、非有限值／filter／metric 失敗、事件邊界、平滑與底色、CSV 重讀及重算 hash 後的錯誤映射拒絕、全排除／缺來源的 CLI audit。

修改前保存 `tests/fixtures/hardy2_continuous_reference.json`，seed 1010、20003×4、8 個完整窗口，含舊程式 hash。raw 與 BP 兩模式的五頻帶、四指標、平滑、相容時間及同政策時段平均，最大絕對誤差均為 **0**。

真實 `iBrainCenter/Hardy_2(SN036)/merged.csv` 共 **958384 樣本、2 段**，兩段间無資料時間 **313.290063 秒**。新流程產生 **383 個有限窗口**，最後中心為來源起點後 **2225.790063 秒**；尾端 **884 樣本**未滿窗口，明示不分析。

| 時段 | 舊中心分類窗口數 | 新完整窗口數 | 舊 alpha mean | 新 alpha mean |
| --- | ---: | ---: | ---: | ---: |
| Pre-16:18 | 149 | 149 | 0.155308942 | 0.155117237 |
| 16:18–16:21 | 36 | 35 | 0.297593943 | 0.302639521 |
| 16:21–16:36 | 180 | 179 | 0.363100186 | 0.362176003 |
| Post-16:36 | 18 | 17 | 0.333081271 | 0.327851110 |

跨邊界 rows 為 **149、185、365**，其餘 380 窗口納入時段平均。這份來源的第一段恰為 108 個完整窗口，新舊窗口時間一致；逐段 BP 改變了缺口邊界的數值，因此即使窗口數相同，也不宣稱所有真實數值相等。最大五頻帶差異是 delta **0.071473713**；四指標最大差異是 calm **0.000104225**。所有指標新舊最大差異及時段均值均已持久保存。

另以合成四段資料驗證短前綴、小缺口、污染段與大缺口：9 個候選窗口、7 個有限窗口，污染段的兩窗口明示排除。CSV/meta 重讀及圖形目視完成，沒有跨排除區域連線、平滑或底色。

主版與 bundle 均完整跑過真實資料，重讀後 metrics 表完全相同。真實、bundle、合成三個成功案例各核對 **14 個產物 hash**（12 圖 + CSV/meta），另存 analysis audit。輸出及 log 在 `/tmp/lilia-stage10/`；持久摘要見 [PYTHON_FIX_VALIDATION_2026-09-07_STAGE10.json](PYTHON_FIX_VALIDATION_2026-09-07_STAGE10.json)。

## 限制與下一步

事件日期及三個 Hardy_2 時點仍沿用既有 session 設定；`--hardy2-csv` 不會自動辨認不同受試者／日期。品質評分、ADC／飽和判斷及五頻帶研究方法校準尚未加入本分支。

舊中心分類的 period helpers 與其他相容 helper 保留，但這個 CLI 已不再使用舊摘要／平滑流程。不能據本入口完成推論所有 legacy callers 都已支援缺口。

整批產物尚非原子發佈；失敗或重用 outdir 可能留下先前／部分檔案，需以本次 audit 狀態及登錄的產物清單判讀。

下一階段處理 `compare_subjects.py`：保存 iBrain 事件及 YoGa session baseline 的連續基準，再遷移 BP／TFLite 分段窗口、raw 品質時間映射、baseline 與缺失結果呈現，驗證後解除其 continuity guard。其後仍有 meditation、qEEG CLI、眼開閉、Jenqwei、品質 scorer、方法校準與整庫工程收尾。
