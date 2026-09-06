# Python 程式修正報告 — 2026-09-06

> 本文件保留第一階段歷史狀態。2026-09-07 已完成普通 entropy 分段與窗口 metadata 遷移，最新成果見 [第二階段報告](PYTHON_FIX_REPORT_2026-09-07.md)。

## 1. 本輪成果與範圍

本輪依據 [完整審查報告](PYTHON_REVIEW_2026-09-06.md) 的 F01–F19，實際修改共用模組與呼叫端，加入回歸測試，並建立 bundle 產生流程。原始盤點為 72 個 Python 檔案；新增共用模組、建置入口與測試後為 **84 個**。本文件說明目前程式碼；原審查報告保留修正前的證據與行號。

**這是完成驗證的第一階段修正，不代表全庫大型重構或研究方法驗證已完成。** 尤其，錄製缺口已在核心濾波、重取樣、TFLite extraction、Goertzel 和 custom marker 流程處理；其他尚未具備分段語義的基線／事件流程，現在會明確拒絕有缺口的錄製。這項行為變更避免繼續產生時間錯位的結果，也意味部分過去可以執行的完整錄製會暫時報錯，不能把它當成已支援全部分段分析。

驗證重點：

- **54/54 單元與回歸測試通過**：原有 28 個，加上 26 個。修正一個原先期待保留 hard artifact 的測試規格。
- **84 個 Python 檔案編譯通過；35 個 CLI／模組入口 help 檢查通過**。套件內 qEEG 使用 `python -m lilia.qeeg --help`；不直接以檔案路徑執行 `lilia/qeeg.py`，避免同目錄的 `signal.py` 遮蔽 Python 標準函式庫。
- Pyflakes、`git diff --check`、`python build_bundles.py --check` 通過。
- 真實 Hardy 前 10 秒：5000×4 → 2000×4 → 2000×2，5 個 TFLite 視窗；**模型輸出逐元素等於修正前的同一資料結果**。
- 合成兩段各 7 秒、中間中斷的錄製：7000×4 → 2800×4 → 2400×2；共 6 個模型視窗，各段尾端各捨棄 200 個 200 Hz 樣本，缺口後的輸出從真實第 20 秒開始。
- 本機合成市場資料完成 2 組 RNN 小型網格訓練，每組 1 epoch；模型與標準化資訊重新載入後，可重現 41×3 預測，最大絕對差異約 **8.3521e-6**。這驗證匯出與還原流程，不是預測品質評估。

機器可讀紀錄見 [PYTHON_FIX_VALIDATION_2026-09-06.json](PYTHON_FIX_VALIDATION_2026-09-06.json)。接手任務見 [REFACTOR_HANDOFF.md](REFACTOR_HANDOFF.md)。

## 2. 修改前既有內容的保護

開始修正時，工作區已存在 README.md、merge_subject_csvs.py 的修改，以及未追蹤的 plot_index_vs_raw_bundle/、requirement.txt、twse_index_lstm_rnn.py、twse_tracker_output/。本輪保留這些內容，沒有整體還原工作區。

- README.md 保持開始本輪時的版本；不能把其既有新增內容算成本輪成果。
- merge_subject_csvs.py 保留使用者原有的「只刪完全相同整列，保留相同 timestamp 的不同樣本」語義，並加上衝突提醒及回歸測試。
- bundle 原有 custom markers、marker date、基線 fallback、summary 功能先移回根目錄主程式，再同步副本；沒有因為兩者相似就刪掉獨有功能。
- 所有實際訓練與推論驗證輸出置於 `/tmp/lilia-fix-integration/`。沒有覆寫正式錄製、既有研究圖表、模型權重或 twse_tracker_output/。
- 本輪未建立 commit。所有修改可透過工作區 diff 與新增檔案檢查。

## 3. 逐項修正：原因、方法、成果

### F01：秒／微秒混用 — 已修正

**原因：** data_analysis.downsample_data 傳入秒，原共用函式把它當成微秒整數，導致秒的小數部分消失；眼開閉處理也沿用此路徑。

**方法：** `lilia.signal.resample_with_time` 明確要求整數微秒；新增 `resample_seconds` 作為秒數介面，data_analysis 改接此介面。重取樣時以每段相對整數微秒插值，再加回原始 epoch，避免直接對大型 UTC 數值插值；不足整數的最後樣本位置依名目採樣間隔外插，避免重複尾端 timestamp。

**成果：** 500 Hz 的 1000 個樣本重取樣後，時間精確對應 `0, 0.005, …, 1.995`，共有 400 個不同時間點。微秒介面若誤收浮點秒會立即報錯。眼開閉匯出測試驗證 400 列與 5000 µs 間隔。

### F02：TFLite summary 缺少 import — 已修正

**原因：** `_resample_500_to_200` 呼叫未匯入的共用函式，只有執行基線分支才會發生 NameError，help 測試抓不到。

**方法：** 補入 `resample_polyphase`，加入實際呼叫該 helper 的測試。

**成果：** 1000×4 輸入正常產生 400×4 輸出。這項修正處理 NameError；隨機基線拼接的方法問題仍見第 7 節。

### F03、F04：真實時間錯位與跨缺口運算 — 已分層處理，完整遷移待續

**原因：** 有缺口的錄製中，`elapsed_seconds * fs` 不等於原陣列位置；整段濾波、重取樣、模型窗口會把中斷兩側連在一起。Hardy 的真實資料已有此情況。

**方法：**

1. 新增 `lilia.windowing`，以相鄰時間差大於 **3 個名目採樣週期** 判定缺口。timestamp 必須嚴格遞增；重複或逆序明確報錯，避免偷偷刪除可能有意義的樣本。
2. 共用 bandpass、notch、bandstop、apply_filters 可接受 `time_us`，分段運算；resample_with_time 也逐段重取樣。
3. 新增 `apply_tflite_with_time`：每段各自保留完整模型窗，再送進同一個推論迴圈；回傳實際保留的 timestamps。extraction 報告記錄各段尾端合計捨棄數。
4. Goertzel 保留原樣本視窗格點，略過跨缺口的視窗；CSV 記錄起訖樣本索引與起訖／中心微秒時間。抽样波形優先使用原索引，舊檔 fallback 以 timestamp 搜尋物理時間區間。
5. custom marker 先逐段濾波，再將跨缺口的分析窗設為無效。
6. 特殊基線／事件流程加入 `require_continuous`；舊 entropy CSV 的 session 繪圖另檢查視窗數，不再把超出範圍的中心索引硬夾到最後一筆。

**成果：** 合成模型案例在缺口後保留第 20 秒時間；Goertzel 的中心時間為 `[2.5, 3.5, 4.5, 22.5, 23.5, 24.5]`，不會壓縮成連續的第 9.5 秒。分段濾波數值與分別呼叫兩段濾波完全一致。連續 Hardy 前 10 秒模型數值不變。

**限制：** 三週期是本輪明確採用的缺口規則，不是對裝置 jitter 的完整校準。太短而無法零相位濾波的區段會報錯。只傳訊號、沒有傳 timestamps 的共用函式仍視輸入為連續資料；不能藉由新增模組就宣稱所有外部呼叫已具備時間驗證。

目前要求連續錄製的入口包括 analyze_jenqwei_pipeline、build_jenqwei_tflite_dataset、compare_subjects、data_analysis、joint_mi、plot_event_markers、plot_meditation_zoom、plot_tflite_summary、plot_tyy_meditation、process_lilia_eye_open_close、spectral_entropy、qEEG CLI，以及 plot_index_vs_raw 的舊 entropy CSV 繪圖模式。custom marker 模式不受此舊 CSV 限制。

### F05：CSV 多餘欄位造成靜默錯位 — 已修正主要讀取入口

**原因：** 某些原始匯出宣告 5 欄，資料列卻有 9 欄；pandas 原本可能把前幾欄推斷為索引，導致時間與 EEG 通道錯位。

**方法：** 新增 `read_lilia_frame`，固定讀取四列 metadata 與第五列宣告欄位，使用明確 names/usecols，重複欄名穩定改名。未宣告尾欄會警告並忽略；timestamp 非數值／非整數、空資料明確報錯。合併工具可明確選擇丟棄無效 timestamp 並發出警告。通用 loader 不自行猜測相對時間或加 UTC offset。

**成果：** `[0,1,2,99,98]` 在宣告三欄時仍讀成時間 0、通道 1/2；各原始資料入口與 qEEG loader 已接共用讀取器。一般分析 CSV 仍使用 pandas 的一般表格讀法，不混入 Lilia 原始檔解析器。

### F06：合併輸入污染／重跑重複加 offset — 已修正

**原因：** 只排除字面上的 merged.csv，會把自訂輸出或分析表再次合併。

**方法：** 探索來源時排除實際輸出路徑、merged.csv、一般分析表、明示已處理檔與 UTC 檔。新合併檔標示 `Time Basis,UTC`；對舊版明顯已是 absolute timestamp 的檔案採取跳過與警告。寫入前檢查通道／採樣率 schema，一致後才以暫存檔與原子替換輸出。

**成果：** 自訂 outname 連跑兩次結果逐 byte 相同；重複整列去除、不同樣本但相同 timestamp 保留，並提示先解決碰撞再分析。schema 不符時既有輸出不會被覆寫。

**限制：** 舊檔辨識含保守 heuristic；metadata 格式差異也可能被當作 schema 差異拒絕。這比默默混合不同採樣率安全，但未等於完整的裝置格式版本管理。

### F07：多步預測 label 跨資料切分 — 已修正

**原因：** 原先只看第一個預測目標的位置；horizon > 1 時後續 label 可能落進 validation 或 test。

**方法：** 新增 `sequence_split_masks`，要求完整 target horizon 位於同一 split；一般訓練與 grid search 共用此邏輯。跨邊界的序列不分配到任一 split，歷史輸入可使用此前已觀測資料。

**成果：** horizon=5 的測試在兩個邊界共剔除 8 個跨界序列，train/validation/test 的 label 集合完全隔離。小型 horizon=3 實際訓練依選定 lookback=10 得到 train 184、validation 40、test 41 個序列。

### F08：眼開閉輸出仍宣告舊通道與採樣率 — 已修正

**方法：** 重建輸出 header 為來源通道 1/2/5/6、200 Hz，使用 ch1/ch2/ch5/ch6 欄名並標示 TinyUNetV4 處理資訊。

**成果：** 整合測試實際跑讀取、濾波、重取樣與 CSV 匯出，確認 header 與 400×4 資料一致；此測試以替身替代 PyTorch 模型及繪圖，並非聲稱已重新驗證眼開閉模型品質。

### F09：1–2 秒 qEEG 視窗崩潰 — 已修正

**方法：** Welch 的 nperseg 上限改為實際輸入長度，noverlap 限制於可用範圍。拒絕不足 8 個樣本、非有限值與無效窗口步長。保留原有四秒 PSD profile 在長輸入的行為；淘汰 `np.trapz` 呼叫，改用 `np.trapezoid`。

**成果：** 1、2、5 秒輸入均產生有限值；5 秒輸入與原先 nperseg=2000／noverlap=1000 的相對功率計算相符。短窗可執行不代表與長窗具有相同頻率解析度。

### F10：Goertzel 品質規則不一致 — 已修正

**方法：** 新增 `valid_goertzel_rows`，供全域／單受試者抽樣、分布、hard artifact 匯出／timeline 與 Goertzel renderer 使用。預設採 quality_final，要求有限的 quality/db/time，嚴格 `quality > threshold`，排除 hard artifact；需要功率時再要求有限且非負。明確關閉 hard artifact 排除時改用原 quality，以符合該選項的意義。

**成果：** 原案例從錯誤保留 3 列改成 2 列；有效樣本仍保有 window 索引。NaN／Inf／hard artifact 測試覆蓋；原測試期待的 hard artifact 不再作為正確答案。

### F11：MI 事件未檢查參與者 — 已修正數值分析路徑

**方法：** joint_mi 與 spectral_entropy 的事件解析加入 participant 篩選。spectral_entropy 新增 --subject，CLI 可從已知受試者資料夾推得名稱；使用內建事件但無法辨識時要求明確 subject。公共 pre/onset helper 未提供 subject 時僅使用開放給所有人的事件。

**成果：** 測試中只有 Hardy 參與的活動不會進入 Ann 的事件 pool；所有人參與的活動保留。事件背景圖仍可標示整場活動時序，不應把圖上標線數量當成實際納入估計的事件數。

### F12：custom marker 非 500 Hz 品質篩選被跳過 — 已修正

**方法：** quality window helper 接受 fs，custom marker 明確傳入 fs；品質陣列形狀不符時報錯，不再略過篩選。

**成果：** 200 Hz、30 秒資料得到 6 個對齊的五秒窗口；測試把品質設低後，baseline/post 合格數均為 0，delta 為 NaN，不會因長度不符而把所有資料視為合格。

### F13：固定 Y 軸隱藏負 delta／超界值 — 已修正相關圖表

**方法：** entropy delta、custom marker、absolute/single-signal 與 qEEG 指標面板改依實際資料自動縮放；相對頻帶功率仍保留有意義的 0–1 範圍。沒有為配合圖軸而改寫指標公式或截斷數值。

**成果：** 回歸案例 entropy 從 0.8 降到 0.1 時，−0.7 的 delta 在可見範圍內。qEEG 指標公式本身的理論範圍與文件一致性仍屬後續方法整理。

### F14：CLI 參數未生效 — 已修正已知案例

- `--event` 在主圖、absolute/peri-event summary 與 split 圖一致生效。先算全事件基線關係，再篩選要展示的活動，保留前一活動影響 baseline anchor 的語義。
- 非整數採樣率以 Fraction 比率重取樣，不再截成整數；500.9 → 200 Hz、5009 個輸入得到 2000 個輸出。
- `--power-ymin` 單獨指定也能套用，不必同時指定 ymax。
- 眼開閉 STFT fmax 的 help 與 50 Hz default 一致。

### F15：缺 Volume 路徑／模型匯出不完整 — 已修正並做小型訓練驗證

**方法：** 前一日 Volume 缺失或非正時，VolChg 明確編碼為 0；保留原 Volume 特徵，清除其他運算產生的 Inf。新增 `preprocessing.json`，保存特徵順序、平均／標準差、target scaler、選定超參數、horizon 與 split 語義；metrics 的 train/validation 數量按最終選定 lookback 計算。

**成果：** 220 列全零 Volume 輸入在 rolling features 後保留 201 列，而非全數消失。兩組實際小型 RNN 訓練完成，使用匯出的 model.keras 與 preprocessing.json 還原預測通過。沒有下載市場資料，也沒有把一個 epoch 的測試當作正式模型訓練成果。

### F16：輸出碰撞／舊快取缺乏設定驗證 — 已修正已知三個位置

**方法與成果：**

- Goertzel summary 檔名含頻率、threshold 與設定摘要；CSV 帶 config_id／threshold／頻率，另存設定 JSON。不同頻率或 bin 設定不再只因 threshold 相同而共用檔名。
- Dataset 資料夾名稱加入來源絕對路徑、來源內容與處理設定的摘要。兩个不同資料夾中的 same.csv 實際匯出到不同目錄。
- Goertzel 每個 metrics CSV 旁產生 `.meta.json`，記錄來源／輸出内容 SHA-256、相關程式 fingerprint 與數值設定。`--reuse-csv` 驗證這些欄位；舊檔沒有 metadata 或來源／表格／設定／程式改變時，明確要求重新計算。圖形外觀如 ymin 不納入數值快取設定，可重畫。

**限制：** 本輪不是全庫 artifact schema 重構；其他大型報告的設定識別、批次原子發佈及歷史結果版本管理尚未全部統一。舊快取不能靠補一個空 sidecar 就視為已驗證。

### F17：有足夠容量卻抽不到片段 — 已修正

**方法：** 以可行片段數與剩餘空間直接建構非重疊起點，取代 greedy 重試。`max_attempts` 保留呼叫相容，但新演算法不使用重試。

**成果：** n_total=200、seg_len=100、n_segs=2，在十個種子下均得到 `[0,100]`；既有可重現性與非重疊測試通過。此函式只保證樣本索引非重疊，錄製連續性仍由上游判斷。

### F18：驗證失敗仍回報成功 — 已修正

**方法：** shell 啟用 pipefail，compile/help/test 任一失敗皆 exit 1；從腳本所在位置執行，log 預設移到暫存位置並可由 LILIA_TEST_LOG 指定。

**成果：** 使用替身 Python 分別令 compile、help、unit-test 失敗，三種情況均返回 1；實際整個 shell 驗證亦成功，4 個 compile、3 個 help、54 個測試通過。

### F19：保留了不影響分數的參數 — 已明示棄用，未移除公開介面

**方法：** 在 canonical quality 模組與獨立包文件列出 `DEPRECATED_QUALITY_PARAMETERS`：target_score 與 10 個 alpha／nonalpha DPR、低頻及 line-noise preset key。明確說明它們不參與目前計分。

**成果：** 使用者不再只能依參數名稱猜測功能。保留 preset keys 與 target_score 呼叫相容，也沒有為了使參數「看起來生效」而改變已校準的分數；真正移除需另做相容性遷移。

## 4. 共用化與可移除程式碼的處理

| 共用位置 | 本輪責任 | 已接入範圍 |
| --- | --- | --- |
| lilia/io.py | Lilia schema、timestamp 讀取；bandpass 相容 wrapper | 原始檔讀取入口、merge、qEEG、bundle |
| lilia/signal.py | 濾波、比率重取樣、秒／微秒介面 | data_analysis、extraction、analyze_jenqwei、dataset 等 |
| lilia/windowing.py | 連續區段、合法窗口、物理時間搜尋與舊流程 guard | extraction、Goertzel、取樣、圖表與 legacy guard |
| lilia/tflite.py | 模型輸入檢查、每段完整視窗推論 | extraction；既有 windowed API 保留 |
| lilia/quality_policy.py | Goertzel 有效列規則 | sampling、distribution、artifact 與 renderer |
| lilia/provenance.py | 來源／程式／輸出指紋與快取驗證 | Goertzel；dataset 使用來源 hash helper |
| build_bundles.py | 從 canonical source 產生與檢查副本 | bundle 的 Python 檔與 standalone quality.py |

移除了已被共用 resampler 取代的 GCD／插值實作、重複 bandpass 實作，以及確定不再使用的 imports。舊相容入口（例如 eeg_utils、qeeg_indices、plot_index_vs_raw_session）和 BASD 等公共函式保留；沒有單憑「repo 內沒有引用」就宣稱對外無人使用。

分發副本仍存在，因此不是磁碟上只有一份程式碼；維護來源已集中，`--check` 能找出不一致。bundle README 與 requirements 保留，避免建置工具覆蓋手寫說明或資料。

## 5. 測試與驗證證據

新增測試檔：

- tests/test_signal_contract_regression.py：秒／微秒、分數採樣率、分段濾波／重取樣、完整模型窗、短 Welch、基線 helper、緊密抽樣。
- tests/test_csv_contract_regression.py：多餘欄位、無效／空時間、自訂輸出重跑、timestamp 碰撞、schema 失敗保護。
- tests/test_pipeline_output_regression.py：眼開閉輸出 header、200 Hz marker 品質對齊、event 篩選與負 entropy delta。
- tests/test_twse_contract_regression.py：完整 label horizon 隔離、零 Volume 與除零後有限特徵。
- tests/test_safety_regression.py：快取 stale 檢查、品質政策、Goertzel 缺口時間、MI 參與者、shell 三階段失敗退出碼。

人工／整合檢查另包含 dataset 同名來源區隔、qEEG 讀取器、舊 entropy CSV 視窗數／缺口拒絕、真實 TFLite 推論、小型 RNN grid search 及模型重新載入。編譯與 help 僅證明語法與入口可用，不作為數值正確的替代證據。

可重跑的核心命令：

```bash
MPLCONFIGDIR=/tmp/lilia-audit-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
python build_bundles.py --check
git diff --check
bash validate_signal_processing.sh
```

執行紀錄暫存於 `/tmp/lilia-fix-tests.log` 與 `/tmp/lilia-fix-integration/`；重要數字已寫入本報告與驗證 JSON，不依賴暫存檔跨時段保留。

## 6. 使用上的變更

1. 舊 entropy／MI 分析遇到錄製缺口會報錯；請先依接手計畫完成 timestamp-aware window metadata 遷移，不要直接刪掉 guard。
2. 原始 timestamp 衝突由 merge 保留並警告；需要判斷真實重複錄製或不同樣本的來源，再決定如何解決。
3. 舊 Goertzel 快取須不帶 --reuse-csv 重算一次，才能建立可信的 metadata。
4. Dataset 子目錄與 Goertzel summary 檔名已加入設定識別；依賴舊檔名的下游腳本須改讀新輸出／manifest。
5. 圖軸自動縮放可能改變視覺比例；數值本身未因此裁切或修改。
6. 原有研究結果檔案未重算；有受 F01/F03/F04/F05/F07 影響的產物，不能因為程式修好就視為舊結果也已修正。

## 7. 尚未完成與下一階段順序

| 優先順序 | 工作 | 驗收方式 |
| --- | --- | --- |
| 1 | 讓 spectral_entropy 及 legacy event/baseline 流程真正支援分段，統一包含 start/end index、timestamp、source/config 的 WindowResult | 用含缺口與 jitter 的資料驗證事件窗口對齊；每筆結果能追溯到原始樣本；有對應測試才解除 guard |
| 2 | 解決隨機 1 秒 baseline 拼接後進入 2 秒模型／PSD 的人工接縫 | 比較「原連續錄製先推論再選窗」與現行方法；明確記錄 baseline 統計單位與結果差異 |
| 3 | 統一品質無效狀態與有效範圍 | 處理 legacy NaN fallback、品質評分短窗末端與失敗時回 0.5 的語義，避免未驗證即更動裝置校準 |
| 4 | 拆分 spectral_entropy／plot_event_markers 的指標、session 設定、推論、繪圖與 CLI | 先保留公共 API／薄入口，建立現有連續資料結果基準，再抽模組；不可只比較 PNG 檔案 |
| 5 | 統一全庫 provenance、批次成功／失敗報告與 artifact 輸出策略 | 任一來源失敗不被總體 Done 蓋掉，任何圖／表均可追溯設定；完成 artifact-level 原子輸出測試 |
| 6 | 整理 PSD profile、qEEG 範圍、Goertzel 正規化與 MI surrogate 統計方法 | 明確區分研究方法改變與等價重構，使用數值敏感性案例驗證，不用單元測試通過代替統計有效性 |

本輪沒有重新跑全部受試者所有圖、完整 PyTorch→TF→TFLite 模型轉換或正式長期預測訓練。上述方法與架構工作仍是開放項目，已完整留下接手狀態。
