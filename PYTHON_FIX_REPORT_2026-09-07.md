# Python 第二階段修正報告 — 2026-09-07

## 本輪完成範圍

依照 [接手狀態](REFACTOR_HANDOFF.md)，本輪完成第一個具體後續任務：**讓 spectral_entropy 的普通 band-entropy 流程、附加的 lagged-MI synchrony，以及其下游指標圖使用共同的分段視窗與真實時間 metadata。**

這次有實際解除已遷移路徑的 continuity guard，並以完整 Hardy 錄製驗證。獨立 `--joint-mi`、`--band-event-mi`、`--baseline/--event` 與特殊 TFLite 基線流程仍保留 guard；沒有把普通 entropy 的成功誤當成所有 MI／baseline 路徑均已完成。

驗證成果：

- **66/66 測試通過**，包含本輪新增的 12 個回歸測試。
- 連續資料與上一版 `8c07401` 保存的基準比較：band energies、band proportions、entropy、lagged-MI 與品質分數均在 `rtol=atol=1e-12` 內一致。
- 完整 Hardy 錄製成功產生 entropy、composition、ternary、focus/relax 四種圖與 CSV／metadata，再由下游生成指定活動的 delta 圖、absolute 圖與 peri-event summary。
- 合成缺口案例同時執行 entropy、品質與 synchrony，保留 12 個一致的窗口；根目錄與獨立 bundle 的 session 圖均成功使用新 metadata。
- Pyflakes、87 個 Python 檔案編譯、bundle 一致性及 `git diff --check` 通過。修改的兩個主入口 help 檢查通過。

本輪輸出放在 `/tmp/lilia-stage2/`，未覆寫正式資料、既有研究產物或模型。數值摘要與來源指紋另外保存於 [PYTHON_FIX_VALIDATION_2026-09-07.json](PYTHON_FIX_VALIDATION_2026-09-07.json)。第一階段報告與驗證紀錄保留為歷史證據，不將其舊指紋冒充目前版本。

## 1. 共用分析視窗

**修改原因：** 原本 entropy、quality、synchrony 各自執行 `range(start, end, step)`；即使只在一個地方移除跨缺口視窗，其他結果仍可能按列號錯位。以名目採樣率產生中心時間，也會壓縮錄製缺口。

**修正方法：** 在 `lilia/windowing.py` 新增 `WindowGrid` 與 `build_window_grid`。

- 由整數微秒 timestamps 建立同一組窗口；三個計算函式接收同一個 WindowGrid。
- 維持原始全錄製的樣本格點，不在每個缺口之後任意重設 step 起點。
- 只保留完整位於同一連續區段的視窗；缺口規則仍為相鄰 timestamp 差大於 3 個名目採樣週期。
- 中心時間取原始中心樣本的實際 timestamp，不使用「第幾列 × step」推算。
- 驗證資料長度、fs、window 與 step；窗口設定不匹配、沒有任何完整窗口、timestamp 重複／逆序等情況明確報錯。
- 舊公共函式未傳 windows 時維持原有連續陣列介面；它們無法憑只有 signal 的輸入自行辨識錄製缺口。

**修改後成果：** 非格點位置的缺口與 ±30 µs jitter 測試能精確追溯每個窗口到原始索引與中心時間。品質分數也以識別樣本位置的替身驗證，確實與相同窗口一一對應，而不是只比對陣列長度。

## 2. 分段濾波與 guard 遷移

**修改原因：** 第一階段以 guard 防止錯誤分析，但完整有缺口錄製因而無法執行。直接刪掉 guard 會重新引入跨缺口濾波。

**修正方法：** 普通 entropy CLI 先建立 WindowGrid，再逐連續區段 bandpass。完全無法貢獻合法窗口的短區段不進入濾波與指標計算；其占據的原始樣本索引仍保留，不能把它刪除後壓縮後面的位置。

entropy、quality 與 synchrony 都只切取 WindowGrid 中的有效區間。普通 entropy 的 `--sync-pair` 在已分段濾波後計算，不會跨缺口估計 lagged-MI。公共 synchrony API 若使用 WindowGrid，要求先按 timestamps 濾波並設定 `apply_bandpass=False`，避免只有視窗 metadata 卻再次整段濾波。

**修改後成果：** 含 10 個樣本的過短區段不再令後續可分析的長區段一起失敗；有效窗口的數值與獨立濾波後按原索引計算相符。沒有完整窗口的錄製仍明確失敗。若使用極短分析窗、使有窗口的區段本身仍不足零相位濾波長度，會報錯，不偷偷改成另一種濾波方法。

尚未遷移的 `--joint-mi`、`--band-event-mi` 與 `--baseline/--event` 仍要求連續錄製，並有回歸測試確認 guard 沒有被意外拿掉。

## 3. CSV metadata 與來源驗證

**修改原因：** 舊 entropy CSV 只有相對時間與指標值，繪圖端必須根據列數猜測來源窗口；檔案混用、刪列或參數不一致可能產生看似正常但對不上的圖。

**修正方法：** 新增 `lilia/entropy_io.py`，由 `write_entropy_table` 匯出表格與 `.csv.meta.json` sidecar。

| 欄位 | 定義 |
| --- | --- |
| window_start_idx / window_end_idx | 原始輸入中的半開樣本區間 `[start, end)` |
| window_start_us | 第一個樣本的微秒 timestamp |
| window_end_us | 最後一個樣本 timestamp 加一個名目採樣週期，作為 exclusive end |
| window_center_us | 實際中心樣本的微秒 timestamp |
| segment_id | 原始錄製的連續區段編號 |
| time_s | 中心 timestamp 減原始第一筆 timestamp，再換算秒 |
| source_id / config_id | 原始檔內容與分析設定的 SHA-256 |
| window_schema_version | 視窗資料格式版本，目前為 1 |
| quality_valid / quality_state | 是否通過本次品質篩選策略，以及 scored／disabled 狀態；disabled 不代表有實測品質分數 |

Sidecar 另保存來源檔案、來源樣本數與 epoch、fs/channel/window/step、bandpass、quality 參數、sync 設定、計算程式指紋與整張 CSV 的內容指紋。

讀取時：

1. 新版表格必須有完整 metadata 與 sidecar；部分新欄位、缺 sidecar 或表格被修改均明確拒絕。
2. 比對指定 raw 檔內容與來源指紋、指定 channel 與分析 channel。
3. 依 raw timestamps 和保存設定重新建立窗口，逐欄核對 indexes、timestamps、區段編號與 time_s。
4. 新版 CSV 的設定與時間是繪圖依據，session CLI 的舊映射預設值不覆蓋它們。

**修改後成果：** 同長度但不同內容的 raw 檔、錯誤 channel、缺 metadata，以及竄改索引後重新計算表格 hash 的測試均被拒絕。檔名、列數或單一 checksum 不再是唯一對齊證據。

表格先寫暫存檔再替換，sidecar 隨後輸出；這不是兩個檔案的交易式原子發佈。若中途中斷，讀取端會因缺少／不一致 metadata 而拒絕使用，需重新產生該結果。

## 4. 品質無效值與資料保留

**修改原因：** 原先 `quality < threshold` 會讓 NaN 比較結果為 False，因而漏掉無效品質窗口。

**修正方法：** 普通 entropy 路徑改為 `~isfinite(quality) | (quality < threshold)`。指標值設為 NaN，但原始窗口列、整數索引、timestamps、segment ID 都保留；synchrony 僅遮罩其數值欄位，不遮罩區段 metadata。

需要品質篩選卻無法載入 scorer 時會報錯；只有明確使用 `--no-quality-mask` 才略過評分，metadata 記錄 disabled。下游若遇到未評分的新版表格，須明確使用 `--quality-threshold -1` 才能畫未經品質篩選的資料，圖標題會標示 quality mask disabled。

**修改後成果：** 六個窗口中同時包含 NaN 品質與低分窗口的測試，確認這兩列的 entropy／proportions／synchrony 皆被遮罩，時間仍為 `[1, 2, 3, 21, 22, 23]` 秒且所有索引存在。

本輪未改動 quality scorer 內部的裝置校準、例外 fallback 或短窗末端處理；這些仍是後续獨立任務。

## 5. 繪圖不跨缺口連線或平滑

**修改原因：** 即使 time_s 修正，兩個相鄰表格列可能相隔數十秒；一般 line plot 仍會連線，rolling mean／median 也可能把缺口兩側混在一起。把缺口後第一列改成 NaN 又會丟失真實有效窗口。

**修正方法：**

- 新增 `plot_breaks`，只在繪圖陣列插入額外 NaN，不刪除 CSV 的任何窗口或缺口後第一個有效點。
- 新增 `finite_runs`／`transform_runs`；普通 entropy 與 metadata 指標圖按 segment 和有限值區段平滑，NaN 品質區段也不被跨越。
- entropy 線圖、同步曲線、標準差帶、composition stack 及 focus/relax 軌跡都切斷缺口；散點仍使用真實中心時間著色。
- raw 波形在顯示降採樣時保留每段首尾樣本，再加入斷線點。
- 指標圖的品質陰影使用保存的 start/end timestamps，避免以相鄰時間的中位數推算缺口附近陰影寬度。
- session、event delta、absolute/peri-event summary、single-signal renderer 都先核對新版表格與 raw 檔，才允許讀取有缺口的錄製。

**修改後成果：** `[0, 0]` 與缺口後 `[10, 10]` 的五窗口平滑結果仍分別為 0 與 10；繪圖資料為 `[0, 0, NaN, 10, 10]`，不是丟掉第一個 10。200 Hz 新版 CSV 即使 session caller 留著 500 Hz／5 秒舊預設，也會按已核實的 metadata 畫出正確時間。

## 6. 完整 Hardy 驗證

來源：`iBrainCenter/Hardy(SN036)/merged.csv`，fs=500 Hz、window=2 秒、step=2 秒、ch1，quality threshold=0.5。

| 項目 | 實際結果 |
| --- | ---: |
| 原始樣本數 | 2,126,712 |
| 連續區段數 | 6 |
| 單純按樣本長度可形成的窗口 | 2,126 |
| 排除跨缺口窗口 | 5 |
| 保存窗口 | 2,121 |
| 品質合格／不合格 | 1,944／177 |
| 錄製真實跨度 | 4,794.342551 秒 |
| 最後保存窗口的真實中心時間 | 4,791.920551 秒 |
| 同一窗口若按樣本數計算的中心時間 | 4,251 秒 |
| 兩者差異 | **540.920551 秒** |

此案例確認本輪確實解決了接手文件中的時間壓縮問題。下游以 `--event 'Mindfulness Meditation'` 實際產出主圖、absolute 圖及一列 peri-event summary，讀取相同的 2,121 個窗口，正確保留 177 個品質不合格窗口的遮罩。

這不代表舊的完整 Hardy 圖與新圖數值應完全相同：舊流程曾跨缺口濾波並納入跨缺口窗口，這些本來就是本輪要移除的運算。等價性檢查使用的是沒有缺口的上一版數值基準。

## 7. 新增測試與使用方式

新增 `tests/test_entropy_windows_regression.py` 共 12 個測試，基準檔為 `tests/fixtures/entropy_continuous_reference.json`。原有 54 個測試亦全部通過。

普通 entropy + synchrony：

```bash
python spectral_entropy.py --csv /path/to/merged.csv --win 2 --step 1 \
  --sync-pair 1 2 --tau-ms 10 20 --mi-bins 8 --out /path/to/output
```

由新版 CSV 畫 session 指標圖：

```bash
python plot_index_vs_raw.py --session-baseline \
  --be-csv /path/to/output/merged_band_entropy_ch1.csv \
  --raw-csv /path/to/merged.csv --label Session --out /path/to/session.png
```

使用 `--sync-pair` 時輸出 CSV 名稱帶 `_sync_ch1_ch2` 後綴，請選用實際生成的檔案。搬移分析 CSV 時，需一併保存其 `.meta.json` sidecar；raw 可移動到另一條路徑，但內容須相同。

舊 CSV 沒有 metadata：連續錄製的相容路徑仍存在；有缺口的舊檔仍拒絕猜測對齊，應用新流程重算。這輪沒有自動改寫任何舊分析 CSV。

## 8. 接續工作與 Git 狀態

下一個具體任務改為遷移 **spectral_entropy 的獨立 MI／baseline-event 模式與 joint_mi**；需要以真實 timestamps 選事件區間、在區段內估計頻帶包絡，並避免把不相鄰 epochs 拼進同一個模型或 PSD 窗口。其餘大型模組拆分、品質評分內部邊界與方法有效性評估仍保留在接手清單。

開始本輪時工作區乾淨，上一輪 commit 為 `8c07401`，本機比 origin 同分支領先一個 commit。上一輪 push 被自動審核攔下，仍待明確 payload／destination 授權；本輪沒有將「繼續修正」視為該 push 確認，也沒有重試推送。本輪修改尚未 commit。
