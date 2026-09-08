# 第十二階段：單事件 meditation zoom

日期：2026-09-08。本階段完成 `plot_meditation_zoom.py`，保留尚未提交的第九至十一階段修改；沒有 commit／push。`plot_tyy_meditation.py` 留待下一階段獨立遷移。

## 原因與修正

舊 zoom 雖聲稱與一般 event markers 的 pre-event-rest 圖一致，實際仍整段濾波、按列數對應品質、以中心時間選 baseline／事件。每六個 5 秒窗口組成 30 秒 bin 時，取第四個窗口中心作 bin 中心，造成 **2.5 秒向右偏移**。事件沒有 bin 時會 fallback 顯示全部 bins；colorbar 只縮窄 heatmap，使它與 raw 面板雖共用 x 軸數值，實際時間位置仍錯開。

現改用 `lilia.event_qeeg.analyze_recording` 的逐來源段 BP、共同 5 秒指標／raw 品質窗口，並直接使用 `summarize_branch(..., 'pre-event-rest')`。只解除本入口的 continuity guard，不改寫一般 event markers 的研究政策。

新增 `lilia.event_zoom`：從整份錄製分析結果選目標活動，保存來源段內的 raw 顯示索引、半開事件區間、缺資料範圍、完整 baseline／event rows、品質排除、跨邊界窗口，以及顯示／完整／有效／跨邊界 bins。raw 先按事件裁切，再於各來源段獨立降採樣，保留每個顯示 run 的第一與最後樣本；每段各畫一條線，底色也不跨缺口。

每個 30 秒 bin 用真正起訖畫獨立矩形。僅完整位於目標活動內且有有效 baseline／品質數值的 bin 顯示 delta；與事件邊界重疊但不完整的 bin 保留灰色。無完整 bin 的來源短尾或缺口留白，不借用其他時段補圖。colorbar 改置獨立欄，所有 raw／heatmap 面板具有完全相同的左右邊界；短事件刻度顯示秒，避免多個同名分鐘刻度。

## 保留的方法與失敗政策

- 500 Hz 原始取樣率、0.5–45 Hz BP、5 秒不重疊指標窗，以及每六窗一個 30 秒 heatmap bin 沿用一般入口。
- 品質使用全部 raw 通道；NaN／錯誤／低分與非有限指標不接受。沒有品質合格窗口的 bin 保留缺值。
- heatmap 先取每窗口的通道 median，再對 bin 內有效窗口取 median；baseline 是「目標活動開始前、上一個活動結束後」完整 bins 的 median。上一活動即使未參與，也保留其時間邊界。沒有事前休息不 fallback。
- `computed` 才成功。未參與、無 raw、無完整 5 秒窗口、無完整事件 bin、缺 baseline、事件 bin 全排除皆保存明確狀態；可讀來源仍產生 raw／缺值 PNG、SVG 與 audit，CLI 非零退出。短到沒有指標窗口時沒有 metrics 表，但保留段 audit 及兩張圖。
- 本入口只分析 BP；沒有加入 TFLite 或改用 TYY 的前第一活動 baseline。bin 可由部分品質合格窗口計算，其 coverage 政策與一般 event markers 相同，尚未加入最低合格比例。

## CSV 重讀時額外找到的錯誤

大缺口合成來源的 SHA-256 為 `68e6869767633b42663ae25cd2d009dec1cf0d89a139e438ea1784b4cd028241`。pandas 2.3.3 的自動數值推斷在處理這個字串時造成 **segmentation fault**：整合驗證及序列重試均重現，獨立單欄 CSV 也以 signal 11 結束，faulthandler 指向 CSV parser。Python parser engine 同樣會在後續型別轉換失敗，並非單純更換 parser 即可解決。

修正共享 `lilia.entropy_io._load_window_table`，明確將 `source_id`／`config_id` 讀為文字。相同來源表立即可重讀，亦避免全數字 hash 的前導零被移除。新增獨立程序測試，重現該 hash 前綴及 64 個零的識別碼；即使未來 parser 回歸，子程序失敗也不會中止整個測試 runner。bundle 同步此修正。此處記錄的是可重現的輸入與修復效果，沒有宣稱已確認 pandas 底層所有根因。

## 產物與使用

保持預設 subject Hsin、event Mindfulness Meditation、ds 10、原 PNG／SVG 命名。新增 `--csv` 可指定外部來源，但仍沿用所選 subject 的 session event schedule，不自動推斷日期。

```bash
MPLCONFIGDIR=/tmp/lilia-stage12-mpl MPLBACKEND=Agg \
python plot_meditation_zoom.py --outdir /tmp/lilia-stage12/real

# 真實沒有事前休息的活動：預期保存缺值圖與 audit，並非零退出。
MPLCONFIGDIR=/tmp/lilia-stage12-mpl MPLBACKEND=Agg \
python plot_meditation_zoom.py --event 'Color Agility Ladder' \
  --outdir /tmp/lilia-stage12/real_no_baseline
```

新增 `*_bp_metrics.csv`／`.csv.meta.json`、`*_analysis.json`。metrics 沿用 `event_marker_bp` kind，加上 `scope=event_zoom`，可由一般讀表介面讀取；sidecar 額外保存 zoom selection／segments 及 hash。包含整份錄製候選窗口與所有活動摘要，使事前 baseline 可核對。品質排除列保留索引與品質，但指標值遮成 NaN。

`lilia.event_zoom_io.load_zoom_table(table, raw_csv)` 核對來源及所有指紋，重建 source grid、raw 品質索引、有效性、整份 summary、目標選區／顯示索引／缺資料區間、段短尾與 window audit。一致性核對不會重跑 BP／Welch／quality scorer 來重新認證每個值。產物仍非原子整批發布；重用目錄時應依本次 audit 登錄的 hashes 判讀。

## 驗證結果

**184 項測試、0 skipped；131 個 Python 編譯、Pyflakes、bundle 與 diff whitespace 檢查通過。** 本階段新增 12 項 zoom 測試與 1 項共享 hash parser 測試。

修改前實際執行舊 `main()`，保存 `tests/fixtures/meditation_zoom_continuous_reference.json`：seed 1212、60003×4、24 指標窗口、4 bins、raw 品質、BP 指標、heatmap／繪圖邊界及程式 hashes。事件邊界對齊的基準中，新舊品質、四指標、absolute／delta heatmap 最大差異皆 **0**；bin 中心與繪圖邊界有意改正 **2.5 秒**，不宣稱座標完全相等。

真實 Hsin 共 **1978896 樣本、1 來源段、791 候選窗口、748 品質合格、131 bins**。新版 zoom 與另一程序實際執行的一般 `plot_subject(..., use_tflite=False, baseline_mode='pre-event-rest')`，除了設定識別碼之外整份 metrics 表相同，整份 summary 完全相同。

Meditation 選到 **20 個完整且有效 event bins**、**5 個完整 baseline bins**；顯示還保留 1 個跨事件起點的灰色 bin。錄製提早結束及不滿完整 bin 的尾段保持空白。與舊版比較：原始品質／BP 指標的 CSV round-trip 差異不超過 `1.12e-16`，absolute heatmap 差異 0；baseline 選窗改變，因此 delta 不宣稱等價：

| 指標 | 舊 meditation baseline | 新完整 baseline |
| --- | ---: | ---: |
| Focus | 0.563625463 | 0.538094737 |
| Flow | -0.709751455 | -0.685982436 |
| Calm | -0.670913394 | -0.650588767 |
| Relaxation | 0.059599309 | 0.081188226 |

完整 session delta 的各指標有限 bins 由 121 變成 108；不同活動邊界與 baseline 政策造成的逐指標最大差異另存驗證 JSON。真實 Color Agility Ladder 的前一活動恰於其開始時結束，新版明示 `missing_baseline`，保存灰色 heatmap、raw 與表／audit，CLI exit 1，沒有改用較早 baseline。

合成案例使用真實品質 scorer：短前綴、8 ms 小缺口、20.008 秒大缺口、短事件及無事前休息。每份來源 **32 指標窗口／32 合格／5 bins**；成功目標各有 2 完整 event bins。大小缺口均保留來源分段，20 秒大缺口圖中 raw／底色斷開，heatmap 不跨段補色。短事件只保留其重疊灰色 bin，沒有 fallback 其他 bins。

兩個真實 zoom 案例與四個合成案例各完成 CSV 重讀及 **4 個產物 hash**（PNG、SVG、CSV、sidecar），另保存各自 analysis audit；一般完整圖的表也重讀比對。Hsin、真實缺 baseline、大小缺口及短事件圖完成目視，包含 raw／heatmap 軸寬一致性檢查。詳細數值、驗證事故與當前指紋見 [第十二階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE12.json)。

## 下一步

第十三階段實跑更正：以下原依註解寫 BP 只用 Ch1／Ch2；實際舊主函式使用四通道 BP，只有 raw 圖顯示前兩通道。後續以第十三階段保存的實跑基準及報告為準。

遷移 `plot_tyy_meditation.py`。先保存既有 BP 只用 Ch1／Ch2、模型四通道輸入／兩通道輸出、未啟用品質遮罩、固定 14:10 起算到 meditation 結束後 2 分鐘的載入區間、14:24 前 baseline 及前 6 分鐘顯示政策。以原始索引／segment ID 保留裁切映射，逐段 BP／重採樣／模型裁尾，不得直接換成 zoom 的 pre-event-rest baseline 或默默加入品質遮罩。

其後仍有 qEEG CLI、眼開閉、Jenqwei、品質方法校準與整庫工程收尾。bundle manifest 目前不包含 zoom CLI，本階段僅同步共享模組，沒有擴大發行入口。
