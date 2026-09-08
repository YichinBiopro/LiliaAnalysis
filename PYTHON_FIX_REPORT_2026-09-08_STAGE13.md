# 第十三階段：TYY meditation BP／TFLite

日期：2026-09-08。本階段完成 `plot_tyy_meditation.py` 的分段裁切／BP／TFLite／固定 baseline 遷移，接續第十二階段後執行；第九至十三階段工作區修改保留，沒有 commit／push。

## 接手時確認的政策更正

前一份交接依舊註解寫「BP 只用 Ch1／Ch2」，但實際舊 `plot_tyy_meditation()` 將完整四通道 `data_filt` 傳入 `_compute_qeeg_windowed`，heatmap 再取四通道 median。**只有 raw 波形圖顯示 Ch1／Ch2，BP 指標實際使用四通道；模型使用四通道輸入、兩通道輸出。** 本階段執行舊主函式並保存陣列 shape，依實際行為保留四通道 BP，沒有把它縮成兩通道。第十二階段報告的下一步描述已加上更正。

品質 scorer 原本未啟用，本輪仍明示 `quality_state=disabled`；有限指標不代表經過品質／artifact 篩選。baseline 為固定 14:24 前，不是 meditation 前休息，也不是 YoGa session 的前五分之一。

## 修改內容

新增 `lilia.meditation`：

- 分析區間使用原始 UTC 微秒的半開裁切，預設 14:10 至 meditation 結束後 2 分鐘（15:19）。來源原始 segment ID 不因裁切重新編號；BP 指標索引保持完整 raw 的索引空間，elapsed 保持完整 raw 第一樣本為起點。
- 每個來源段與裁切區間交集獨立 BP；短段、段末不足、非有限段、filter／metric 失敗均保存 audit。分析範圍外樣本不參與濾波。單窗口 metric 例外不留下部分通道結果。
- TFLite 共用 `run_tflite_recording` 的逐段 polyphase／400 點 RMS 模型窗口及逐段裁尾。adapter 原以 cropped input 為索引，本輪另明確轉回完整 raw 的 segment ID、raw 索引、epoch／source_samples；模型 output 索引仍屬 `retained_tflite_output`，不當作 raw 索引。
- 每段獨立組六個 5 秒指標窗口為 30 秒 bin，不跨段拼接；不足六窗的尾列保留紀錄。每窗口先取通道 median，再對 bin 的有限窗口取 median。baseline 只取完整位於固定邊界之前的 bins；無 baseline 時輸出 NaN，不再 fallback 零。
- raw 顯示沿用 meditation 前 6 分鐘（15:00）至 15:19。各來源顯示 run 保留首尾並分別畫線，meditation 底色也只涵蓋實際來源 run。heatmap 每 bin 使用真正起訖，缺口／尾端無完整 bin 留白。獨立 colorbar 欄保證 raw 與 BP／TFLite 的時間位置對齊；短顯示區間使用秒刻度。

新增 `lilia.meditation_io`，輸出 distinct `meditation_bp`／`meditation_tflite` kinds。CSV／sidecar 保存 source／model／config／code hashes、全部候選窗口與有限性、逐通道指標、crop／段／模型映射、baseline／heatmap／顯示 raw 索引及所有排除。`load_meditation_table(table, raw_csv, model_path)` 重建來源裁切與原始 ID、模型裁尾、指標 grid、顯示 crop、有效性、整份 baseline／heatmap 及 audit。一致性驗證不重跑濾波與模型。

保留成功時原有 PNG／SVG 命名、`--ds`、`--no-tflite` 與主函式入口；新增 `--csv`／`--model` 指定來源或模型，仍使用固定 TYY session schedule。要求的模型缺失、推論失敗、缺 baseline 或空 crop 皆保存可取得的 BP 表／圖及失敗 audit，最後非零退出。模型缺失與明確 `--no-tflite` 分開記錄，失敗 audit 包含原本要求的 model path／hash。

## 驗證結果

**197 項測試通過、0 skipped；136 個 Python 編譯、Pyflakes、bundle 及 diff whitespace 通過。** 新增 13 項測試，涵蓋實際四通道 BP／真實模型連續數值、全來源索引與 segment ID、半開／非整齊裁切邊界、來源段濾波獨立性、無 baseline、污染段、disabled 品質、模型缺失／推論失敗、全短／空 crop、重讀與篡改拒絕、raw／heatmap 軸寬及時間刻度。

修改前實際執行舊主函式，保存 `tests/fixtures/tyy_meditation_continuous_reference.json/.npz`：seed 1313、60003×4、程式／模型 hashes、BP／TFLite 指標、delta、窗口／bin 時間及模型輸出。兩分支各 **24 窗口**，四指標與同政策 delta 最大誤差 **0**；真實模型 output 與舊主函式保存陣列逐值相同。舊 bin 中心取第四窗口中心，較真實矩形中心偏右 **2.5 秒**，新版有意改正。

真實 TYY 實跑命令：

```bash
MPLCONFIGDIR=/tmp/lilia-stage13-mpl MPLBACKEND=Agg \
python plot_tyy_meditation.py --ds 10 --outdir /tmp/lilia-stage13/real
```

來源共 **2009328 樣本、1 連續段**，實際起點為 14:10:27.626459、終點支撐至 15:17:26.282459，本次固定裁切涵蓋全部來源。BP、TFLite 各 **803 候選／803 有限窗口、133 bins、27 baseline bins、34 顯示 bins**；最後窗口中心為原始 epoch 後 **4012.5 秒**。TFLite 保留 **803600 output samples／2009 模型窗口**，逐段裁掉 132 個重採樣尾樣本。每分支剩餘 5 個完整指標窗口不足構成下一個 30 秒 bin，保持明示尾端。

此真實錄製的新舊 baseline 選窗恰好相同，absolute heatmap 與四指標 delta 最大差異都為 **0**；每窗口指標經 CSV round-trip 後差異不超過 `1.12e-16`。不據此宣稱非整齊事件／裁切邊界政策普遍等價。

| 分支 | 舊／新 Focus baseline | 舊／新 Flow baseline | 舊／新 Calm baseline | 舊／新 Relaxation baseline |
| --- | ---: | ---: | ---: | ---: |
| BP 四通道 | -0.486068520 | 0.174563734 | -0.119664002 | 0.205487964 |
| TFLite 兩通道 | -0.035613886 | -0.156819077 | -0.245699706 | 0.379964974 |

四個合成案例使用真實模型：8 ms 小缺口、20.008 秒大缺口、無 baseline、缺模型。來源先有兩段在 crop 外，crop 從原始索引 **6100**、原始 segment **2** 中途開始，之後仍保留 segment **3**。BP／TFLite 各 **29 有限窗口**；成功案例有一個完整 baseline bin。缺 baseline 保留兩分支表及灰色圖，缺模型保留 BP 表並顯示 TFLite unavailable，兩者均明確失敗。

真實案例與四個合成案例共 **9 份表重讀、28 個產物 hash** 核對（4 個完整 BP＋TFLite 案例各 6 個，模型缺失案例 4 個），另有每例 analysis audit。真實、大／小缺口、無 baseline、模型缺失圖完成目視；不跨缺口連線或補色。數值與當前指紋見 [第十三階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE13.json)。

## 限制與下一步

500 Hz／200 Hz、五秒指標窗、30 秒 bin、固定 session 時段及未啟用品質遮罩均沿用本入口。raw 顯示仍保留 ±150 µV 的舊圖軸限，超出範圍的 raw 值可能被圖框裁去；數據與指標表不受此顯示限制影響。新 `--csv` 不會判斷任意錄製是否屬於 TYY 2026-05-12 session。

污染的保留模型段仍使整個模型分支失敗，沒有默默跳過或重新串接其他段。baseline 聚合／品質方法／qEEG 公式的研究校準、整批 artifact 原子發佈等尚未完成。bundle manifest 不包含 TYY CLI，本階段同步新增共享模組，不擴大發行入口。

下一階段處理 `python -m lilia.qeeg`／相容入口 `qeeg_indices.py`。它目前直接分析指定單一 raw channel，沒有 BP、品質 scorer 或 baseline；先保存三個相對頻帶／四指標與 CLI 時間基準，再接上原始微秒 WindowGrid、缺口／非有限窗口政策、表／audit 重讀及不跨段圖形。不要直接套用本階段的多通道／BP／TFLite／baseline 政策。
