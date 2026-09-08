# 第十一階段：跨受試者 BP／TFLite 比較

日期：2026-09-08。接手時發現工作區已有第十一階段程式、連續 fixture 與 13 項測試，交接文件仍停在第十階段。本輪接續審查，補上窗口計算失敗處理、不同模型裁尾窗口數與全缺值圖形測試，完成全部真實受試者驗證及文件。本輪沒有 commit／push，保留第九、十階段既有修改。

## 原因與修改

`compare_subjects.py` 舊流程整段 BP／重採樣，以列數截齊 raw 品質與模型指標，未配對尾列預設為品質良好；品質 NaN 也可能因單純 `< threshold` 而被接受。有缺口的來源原由 continuity guard 阻擋。圖表將缺值畫成零高柱，無法區分未參與、無結果與真正零變化。

現改用 `lilia.event_qeeg.analyze_recording` 的逐來源段 BP 及 `lilia.tflite` 完整模型窗口 timeline。兩分支各建 5 秒完整指標窗口，BP 品質使用同一原始樣本範圍，TFLite 品質用實際微秒起訖搜尋 raw 範圍，不再用列位置配對。保留來源 segment ID、短段、BP 尾樣本、模型逐段裁尾、品質與指標逐窗口狀態。

新增 `lilia.subject_comparison` 負責 baseline 選擇與差值，`lilia.subject_comparison_io` 保存並重讀指標表。`extract_subject_deltas` 保留 dict-of-tuples 相容介面，改走分段流程；主 CLI 在單一受試者或模型失敗後繼續其餘受試者／群組，保留 BP 部分成果，最終彙總非零失敗。

共享 `score_branch` 增加單窗口 metric 例外與非有限輸出審核：部分通道計算後失敗時清除該窗口所有指標，保留候選列及 `metric_error` 原因，繼續後續窗口。品質合格不會覆蓋指標失敗。一般 event markers 也使用此函式；既有連續基準及回歸測試均保留。

## baseline 與缺值政策

- **iBrain**：第一個實際參與活動之前的完整窗口作共同 baseline；事件也只納入完整位於半開活動區間內的窗口。沒有 baseline 不 fallback；未參與活動明示 `not_participating`，不改變 baseline 起點。
- **YoGa**：先按該分支全部候選指標列取前 `max(1, N//5)` 列，再套品質遮罩；其餘候選列中合格者作比較。這是窗口數的前五分之一，不是 elapsed 時間的 20%，也不是 30 秒 heatmap baseline。模型裁尾可能使兩分支 baseline 實際時間不同，表中明列各自選窗。
- 每通道先各取 baseline／comparison 的窗口平均並相減，再對通道 delta 算 mean 與 population SD。BP 使用全部來源通道，模型使用兩個輸出通道；channel SD 不是跨受試者變異或推論信賴區間。
- 品質 NaN／Inf／錯誤 shape／例外／低分與指標非有限值均不能成為有效窗口。缺值以 `NA`、未參與以 `NP`、`--no-tflite` 以 `OFF` 標示，不建立假的零柱。單分支及合併圖固定受試者軸範圍，全缺值或首位未參與時標示仍在正確位置。
- 有參與項目但所有 baseline／comparison 配對皆失敗，該受試者報錯；有其他成功配對時仍保存所有缺資料事件狀態，不因單一缺事件宣稱整份錄製失敗。

## 產物與重讀

每群組仍產生 BP、TFLite、合併比較三類 PNG／SVG，另有 `comparison_analysis.json` 登錄群組狀態、每受試者 audit hash、圖形 hashes。

每受試者資料夾保存 `bp_metrics.csv`、`tflite_metrics.csv` 及各自 `.csv.meta.json`，另有 `analysis.json`。sidecar 的 kind 為 `subject_comparison_bp`／`subject_comparison_tflite`，保存 raw／模型／設定／程式指紋、所有原始或模型窗口、raw 品質索引、每窗口狀態、候選／接受／排除的 baseline 與 event rows、逐通道及群體摘要。

`load_comparison_table(table, raw_csv, model_path)` 核對表與設定 hashes、raw／可選模型內容，從來源重建分段窗口及模型裁尾映射，核對 raw 品質範圍，重算有效性及所有 baseline／delta 摘要。它驗證來源映射與表內一致性，不會重跑濾波、品質 scorer 或模型來認證每個數值。

完整執行命令（輸出到暫存，不覆蓋正式研究圖）：

```bash
MPLCONFIGDIR=/tmp/lilia-stage11-mpl MPLBACKEND=Agg \
python compare_subjects.py \
  --ibrain-outdir /tmp/lilia-stage11/real/iBrainCenter \
  --yoga-outdir /tmp/lilia-stage11/real/YoGa
```

## 數值與案例驗證

**171 項測試通過，0 skipped；126 個 Python 檔案編譯、Pyflakes、bundle 一致性與 diff whitespace 通過。** 第十一階段共 16 項新增測試，涵蓋連續真實模型基準、不同窗口數的品質映射、短段／缺口、品質與 metric 失敗、兩種 baseline、模型缺失／停用、重讀及重算 hash 後的錯誤映射拒絕、缺值圖形與 CLI 部分失敗續跑。

`tests/fixtures/subject_comparison_continuous_reference.json` 保存修改前程式 hash、真實模型 hash、seed 1111、30003×4 連續輸入、raw 品質、BP／TFLite 指標及 iBrain／YoGa 兩種 delta。兩分支各 12 窗口，品質、四指標、事件與 session delta 最大絕對差異全部 **0**。這份事件 fixture 邊界對齊窗口；不宣稱新的完整窗口政策與舊中心選窗政策普遍等價。

全部真實受試者均成功，BP 與 TFLite 在這八份來源恰有相同窗口數及品質合格數：

| 群組／受試者 | 原始樣本 | 來源段數 | 每分支候選窗口 | 每分支合格窗口 |
| --- | ---: | ---: | ---: | ---: |
| iBrain／Ann | 1965633 | 1 | 786 | 648 |
| iBrain／Hsin | 1978896 | 1 | 791 | 748 |
| iBrain／Hardy | 2126712 | 6 | 848 | 683 |
| iBrain／TYY | 2009328 | 1 | 803 | 735 |
| iBrain／James | 1906544 | 4 | 759 | 621 |
| YoGa／James | 2459472 | 1 | 983 | 790 |
| YoGa／Jammie | 2820000 | 3 | 1128 | 954 |
| YoGa／TYY | 2879360 | 1 | 1151 | 1027 |

Hardy 來源缺資料合計 **540.920551 秒**，最後窗口中心為 **4791.844551 秒**；iBrain James 缺資料 **176.751874 秒**；YoGa Jammie 缺資料 **51.28 秒**。所有窗口保留各自真實 elapsed 時間。

額外以同一份新分段 scores／quality 分別套用舊中心分類及新完整窗口分類，隔離事件選窗政策的數值影響；逐事件候選數與新舊 delta 保存在驗證 JSON。此對照沒有執行舊跨缺口濾波，因此不能解讀成完整舊、新 pipeline 的真實錄製數值差異。

合成四段資料含短前綴、5 秒來源段、8 ms 小缺口與大缺口，搭配真實模型和 scorer。BP 有 **9** 個合格窗口，TFLite 有 **8** 個：5 秒段只能保留 4 秒模型輸出，不得產生 5 秒模型指標，後續品質對應仍正確。事件案例完整 baseline 分別 **3／2** 窗口、comparison 各 **4** 窗口；未參與與資料範圍外事件顯示 NP／NA。Session baseline 各取第一候選列，比較分別 **8／7** 列。兩案例均完成表重讀與合併圖目視。

真實案例 **16 表重讀、32 表／sidecar hashes、8 個受試者 audit hashes、12 個圖形 hashes** 核對完成；合成案例另有 4 表重讀、8 表／sidecar hashes 與 4 圖 hashes。詳細檢查及當前 Python 指紋見 [第十一階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE11.json)。

## 限制與下一步

500 Hz 原始取樣率、5 秒指標、四通道模型與既有受試者／日期 registry 沿用目前設定。品質與 baseline 的研究方法校準尚未完成；來源缺口以共享 `continuous_slices` 的三個名目取樣週期門檻判定。TFLite 保留段含污染值或模型推論失敗會使該模型分支失敗，BP 可保留有效窗口；本階段不更改整段模型失敗政策。

整批產物尚未原子發佈，重用輸出目錄時可能留有歷史檔案，應依本次 audit 登錄判讀。bundle 已同步新增模組及共享失敗處理；既有 bundle manifest 不包含 `compare_subjects.py` CLI，未額外擴大其發行入口。

下一階段先遷移 `plot_meditation_zoom.py`：應與一般 event markers 的 pre-event-rest 結果一致，重用分段 BP／品質及真正 30 秒 bin 範圍，移除空選區 fallback 全部 bins 的行為；完成 Hsin 真實 meditation、缺 baseline／短事件／缺口對照後才解除 guard。再處理 `plot_tyy_meditation.py`，先保存其獨立 baseline／品質政策，不能直接與 zoom 或 YoGa 政策混用。qEEG CLI、眼開閉、Jenqwei、品質 scorer 及工程收尾仍待處理。
