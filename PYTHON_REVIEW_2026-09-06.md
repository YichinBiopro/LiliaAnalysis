# Python 全庫分析與重構評估

分析日期：2026-09-06。範圍為目前工作目錄，包含尚未追蹤的 Python 檔案；不是只分析 Git HEAD。此次只新增本報告，未修改 Python、模型或原始資料。

## 1. 四項問題的結論

| 問題 | 結論 |
| --- | --- |
| 是否有可以共用的部分 | 有，而且已有 `lilia/` 基礎。主要缺口是 CSV 格式契約、連續資料分段、滑動視窗、session/event 設定、品質篩選、前處理與推論 orchestration、baseline 與繪圖資料準備。 |
| 是否有過時可移除程式碼 | 有：4 個未使用 import、實際不生效的 quality 參數、已抽取邏輯的舊入口、實體複本。但舊入口需要相容性遷移，bundle 有獨有功能，不能直接整份刪除。 |
| 是否有邏輯問題 | 有。已重現時間單位截斷、未匯入函式、CSV 欄位錯位、合併重跑不冪等、短視窗錯誤、品質規則不一致；另由程式碼確認事件定位、資料中斷、模型資料切分等問題。 |
| 是否能重構 | 可以，適合漸進重構。應先固定資料與數值契約、修復高優先錯誤，再搬移模組；不宜先大規模改檔名或把不同分析方法強制合成同一套。 |

## 2. 覆蓋範圍與驗證

共 **72 個 Python 檔案、20,474 行**，包含註解與空白：根目錄 33、`lilia/` 14、bundle 17、`signal_quality_package/` 2、`tests/` 6。逐檔清單見第 8 節。

執行項目：

- 全部 72 個檔案經 AST 盤點與記憶體內編譯，無語法錯誤；全庫跑 Pyflakes。
- 既有 `python -m unittest discover -s tests -p 'test_*.py' -v`：**28/28 通過**。
- 全庫有 ArgumentParser 的入口及 `qeeg_indices.py` 舊入口，共 **34/34 通過 `--help`**。未把無 CLI 的 library、模擬圖及轉檔程式冒充 CLI 測試。
- 實際執行 `extract_tflite_signal_pipeline.py`，以 Hardy 資料前 10 秒推論：5000×4 → 2000×4 → 2000×2；5 個 TFLite 視窗完成。
- 以臨時合成資料重現下列邊界問題；讀取 Hardy 的真實 timestamps，及 test02 的 CSV 欄位數確認格式問題。
- 模型推論與 CLI 檢查的臨時產物位於 `/tmp/lilia_python_audit_pipeline/`、`/tmp/lilia_python_audit_help.json`，不是正式研究結果。

限制：沒有重跑全部受試者的所有圖、完整模型轉換、股價模型訓練／網格搜尋，也沒有重新驗證研究方法與裝置校準的有效性。測試通過不代表所有分析結果正確；既有測試主要是匯入、help、抽樣、檔案探索及時區轉換。

優先級：**P1**＝正常或已存在資料可触發、阻斷流程或改變結果；**P2**＝特定參數／邊界／下游使用觸發；**P3**＝維護與介面清理。以下清楚區分執行重現、程式碼確認及方法風險。

## 3. 優先修復的邏輯問題

### F01 — P1：秒被當成微秒整數，降採樣後時間軸失真【已重現】

位置：[data_analysis.py:129](/home/bps-yichin/lilia_analysis/data_analysis.py:129)、[lilia/signal.py:137](/home/bps-yichin/lilia_analysis/lilia/signal.py:137)、[process_lilia_eye_open_close.py:260](/home/bps-yichin/lilia_analysis/process_lilia_eye_open_close.py:260)。

`load_file()` 回傳浮點秒，`downsample_data(time_s, ...)` 卻直接呼叫 `resample_with_time()`；後者按微秒契約把結果轉 `int64`。因此 200 Hz 的 `0, 0.005, 0.010, ...` 變成 `0, 0, 0, ...`。2 秒輸入的 400 筆輸出只剩 2 種時間值。眼開閉腳本還把這些整數秒乘回 1e6 寫入 CSV，形成大量重複 timestamps。

影響：時間圖、STFT 的定位及匯出 CSV；訊號值本身並不是因此變成 1 Hz，但時間資訊已損壞。修復應明確區分 `time_s: float64` 與 `time_us: int64`，在入口轉換、出口還原，或提供獨立 API。不能只移除共用工具的 `astype(int64)`，因為其他呼叫者確實要求微秒整數。

### F02 — P1：TFLite summary 基線路徑必然 NameError【已重現】

位置：[plot_tflite_summary.py:109](/home/bps-yichin/lilia_analysis/plot_tflite_summary.py:109)。

`_resample_500_to_200()` 呼叫 `resample_polyphase`，檔案未匯入該名稱。直接呼叫即得到 `NameError: name 'resample_polyphase' is not defined`。session 或 pre-event 模式成功選到基線後，`_tflite_qeeg_reference()` 都會走到這裡。`--help` 和 py_compile 不會執行此分支；Pyflakes 可以抓到。

修復：補正匯入並實際執行「選基線 → 降採樣 → 推論 → qEEG」的小型整合測試。

### F03 — P1：事件與抽樣把真實經過時間當成連續樣本時間【真實資料＋程式碼確認】

位置：[spectral_entropy.py:491](/home/bps-yichin/lilia_analysis/spectral_entropy.py:491)、[spectral_entropy.py:1587](/home/bps-yichin/lilia_analysis/spectral_entropy.py:1587)、[spectral_entropy.py:3273](/home/bps-yichin/lilia_analysis/spectral_entropy.py:3273)、[joint_mi.py:75](/home/bps-yichin/lilia_analysis/joint_mi.py:75)、[sample_segments_by_goertzel_db.py:32](/home/bps-yichin/lilia_analysis/sample_segments_by_goertzel_db.py:32)、[filter_and_plot_hard_artifacts.py:104](/home/bps-yichin/lilia_analysis/filter_and_plot_hard_artifacts.py:104)。

Hardy `merged.csv` 的 2,126,712 筆資料中，有 5 個相鄰樣本間隔 >1 秒；最大間隔 409.754551 秒。首末 timestamps 相差 **4794.342551 秒**，用 `(N-1)/500` 只有 **4253.422 秒**，相差 **540.920551 秒**。

具體錯誤鏈：

- spectral entropy 的時間用 `(start + win/2)/fs` 產生；CSV 雖讀了原始 timestamps，最後 offset 仍為 0，沒有補回中斷。事件圖及 `plot_index_vs_raw` 的一般事件模式因此錯位。
- `joint_mi.resolve_events()`、spectral 的 event onset／pre-onset 分析用 `(event_us - first_us) * fs / 1e6` 找 index，中斷後會選到錯誤區段，甚至將真實存在的事件判定為範圍外。
- Goertzel `time_s` 來自真實 timestamps；全域抽樣與 hard gallery 卻再用 `center_s * fs` 切原始資料。後者甚至會把過大的 index 夾到檔尾。
- session 版 `_real_window_times()` 是局部補救，但由「CSV 列数＋手動 win/step/fs」重建位置；CSV 曾篩列或參數不符時仍會錯，超界還會靜默 clip。

修復：結果 CSV 保存 `window_start_idx/end_idx`、起訖／中心 `time_us`、segment ID；事件用 timestamps 搜尋，抽樣用原始 index。不要在繪圖層猜回時間軸。

### F04 — P1：跨錄製中斷做濾波、重取樣及分析視窗【程式碼確認】

位置：[lilia/signal.py:137](/home/bps-yichin/lilia_analysis/lilia/signal.py:137)、[extract_tflite_signal_pipeline.py:59](/home/bps-yichin/lilia_analysis/extract_tflite_signal_pipeline.py:59)、[plot_event_markers.py:180](/home/bps-yichin/lilia_analysis/plot_event_markers.py:180)、[plot_event_markers.py:623](/home/bps-yichin/lilia_analysis/plot_event_markers.py:623)、[plot_goertzel_vs_raw.py:57](/home/bps-yichin/lilia_analysis/plot_goertzel_vs_raw.py:57)、[build_jenqwei_tflite_dataset.py:166](/home/bps-yichin/lilia_analysis/build_jenqwei_tflite_dataset.py:166)。

大多數流程把整個 merged 陣列當連續訊號，直接 `sosfiltfilt`／`resample_poly`，再固定樣本切窗。CSV 的 timestamp 中斷沒有阻止濾波、模型或 PSD 視窗橫跨两段錄製。畫圖插入 NaN 只能打斷線條，無法修正已算出的數值。bundle custom marker 雖會丟棄部分跨 gap 視窗，仍是先對整段濾波，且用整窗跨度 >7.5 秒判斷，會漏掉較小的中斷。

修復：讀檔後按 timestamp 相鄰差及 session 來源建立連續區段，逐段濾波、重取樣與切窗；將邊界不足長度及模型尾段的處置記錄在 metadata。

### F05 — P1：CSV 多出未宣告欄位時，時間與通道會靜默錯位【已重現，真實資料存在】

位置：[lilia/io.py:18](/home/bps-yichin/lilia_analysis/lilia/io.py:18)、[data_analysis.py:82](/home/bps-yichin/lilia_analysis/data_analysis.py:82)、[extract_tflite_signal_pipeline.py:46](/home/bps-yichin/lilia_analysis/extract_tflite_signal_pipeline.py:46)、[analyze_jenqwei_pipeline.py:93](/home/bps-yichin/lilia_analysis/analyze_jenqwei_pipeline.py:93)。

這些 loader 使用 `pd.read_csv(skiprows=4)`，再以 `iloc` 當時間與通道。若 header 宣告 5 欄、資料有 7 欄，pandas 會將前兩欄當 index；合成案例的原始時間 `[100,102]` 被 loader 讀成 `[2,12]`。

test02 的兩份真實 raw CSV 是 **header 5 欄、資料 9 欄**。目前未提交的 `merge_subject_csvs.load_csv_data()` 已用 `names/usecols` 處理，但這個修復沒有進到其他 loader。以正確合併後的 CSV 讀取不會觸發同一錯位；直接使用原始 CSV 的入口仍有風險。

修復：共用格式解析與欄位驗證，明確決定多餘欄位是忽略、保留為非 EEG metadata，或拒絕；區分 raw 的相對時間與 merged 的 UTC 絕對時間。當前眼開閉 raw 第一筆是 0，也證明「所有四列 header CSV 都是 UTC」不是成立的契約。

### F06 — P1：合併來源污染與自訂檔名重跑重複加 offset【已重現】

位置：[merge_subject_csvs.py:68](/home/bps-yichin/lilia_analysis/merge_subject_csvs.py:68)、[merge_subject_csvs.py:134](/home/bps-yichin/lilia_analysis/merge_subject_csvs.py:134)，bundle 同步存在。

`collect_csvs()` 收入資料夾內所有 CSV，只排除字面 `merged.csv`。目前 Hardy 找到 19 份來源，其中至少 7 份是 Goertzel、entropy 或 peri-event 產物；Ann 14 份中也有 7 份。這些 CSV 格式不是原始錄製格式，會被錯讀或造成失敗。

若使用 `--outname custom.csv`，下次重跑會把上次輸出當 raw，再次加 offset。合成案例第一次 `[1000000,1002000]`，第二次變成 `[1000000,1002000,2000000,2002000]`。

修復：依錄製 schema／manifest 選來源，明確排除實際輸出路徑；驗證相對／絕對時間、欄位及 sampling rate。現有「只刪整列完全相同資料、保留同 timestamp 不同值」是刻意保留資料的修改，不應改回只按 timestamp 去重；碰撞應另行報告、保留來源供處理。

### F07 — P1：TWSE 多步預測 label 跨越 train/validation/test 邊界【程式碼確認＋索引重現】

位置：[twse_index_lstm_rnn.py:204](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:204)、[twse_index_lstm_rnn.py:477](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:477)、[twse_index_lstm_rnn.py:532](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:532)。

每個 label 是 `target[i:i+horizon]`，split 卻只檢查起點 `i`。horizon=5、train_end=70 時，最後一筆 training label 包含到 index 73；validation 最後一筆 label 也伸進 test。模型訓練與 EarlyStopping 因而接觸下一區間的目標值。一般訓練及 grid search 都有同樣問題；horizon=1 不觸發此跨界問題。

修復：以完整 target 區間歸屬切分，例如 train 要求 `i+horizon <= train_end`，validation 要求 `i >= train_end and i+horizon <= val_end`。跨邊界的 target 視窗排除。輸入 lookback 使用較早歷史本身是合理的，不應連同合法歷史一起禁止。

### F08 — P2：處理後 CSV 的 header 仍宣告舊取樣率與通道【程式碼確認】

位置：[process_lilia_eye_open_close.py:280](/home/bps-yichin/lilia_analysis/process_lilia_eye_open_close.py:280)。

輸出已是 200 Hz、4 個欄位（來源 ch1/2/5/6），卻原樣複製 8 channels、500 Hz 的 header，欄名還都是 `value`。即使修完 F01，下游按 header 讀取仍會錯。應重建取樣率、通道對應、時間基準與處理 provenance。

### F09 — P2：qEEG 的 1–2 秒視窗直接崩潰【已重現】

位置：[lilia/qeeg.py:42](/home/bps-yichin/lilia_analysis/lilia/qeeg.py:42)。

Welch 固定 `nperseg=fs*4, noverlap=fs*2`。資料只有 1–2 秒時，SciPy 縮短 nperseg 後 noverlap 仍過大，回報 `noverlap must be less than nperseg`。以 500 Hz 正弦波測試：1 秒、2 秒失敗；3 秒、5 秒成功。`qeeg_indices.py --win 2` 是公開允許的參數。

修復：限制合法視窗，或讓 nperseg/noverlap 一起依段長調整；需記錄 PSD 設定，因為這會改變數值，不能只讓函式「不報錯」。

### F10 — P2：全域與逐受試者 Goertzel 抽樣的品質標準不一致【已重現】

位置：[lilia/goertzel_sampling.py:11](/home/bps-yichin/lilia_analysis/lilia/goertzel_sampling.py:11)、[lilia/goertzel_sampling.py:57](/home/bps-yichin/lilia_analysis/lilia/goertzel_sampling.py:57)、[lilia/goertzel_distribution.py:23](/home/bps-yichin/lilia_analysis/lilia/goertzel_distribution.py:23)。

全域用 `quality`；逐受試者與分布分析優先用 `quality_final` 並排除 hard clip。`quality=.9, quality_final=0, artifact_hard_clip=1` 的同一列，全域選入 1 列、逐受試者選入 0 列。現有 global 測試甚至包含這種差異，不能只把舊测试通過當成目標。

修復：抽出明確 `QualityPolicy`。如果 global 本來就要抽髒訊號供研究，應提供明示選項並標注輸出；預設與宣稱的 clean sampling 一致。統一有限值檢查與 `>`／`>=` 邊界；分布 collector 目前只驗證 quality finite，未排除 NaN/Inf 的 power/dB。

### F11 — P2：MI 事件分析忽略是否參與活動【程式碼確認】

位置：[joint_mi.py:75](/home/bps-yichin/lilia_analysis/joint_mi.py:75)、[spectral_entropy.py:1587](/home/bps-yichin/lilia_analysis/spectral_entropy.py:1587)、[spectral_entropy.py:3273](/home/bps-yichin/lilia_analysis/spectral_entropy.py:3273)。

EVENTS 已有 participants，但這些路徑將欄位丟棄，沒有 subject 參數。只要資料涵蓋該時間，就分析／pool 所有活動。例如 Ann 與 TYY 並未參與的部分 cycling，也會被當活動 onset。與 `compare_subjects` 和 `plot_index_vs_raw` 的規則不同。

修復：事件解析接收 subject/session，明確区分「觀察所有時點」與「參與的活動」。連續活動借用哪段 baseline 也應統一設定。

### F12 — P2：bundle custom marker 的非 500 Hz 品質篩選可能被整段跳過【程式碼確認】

位置：[plot_index_vs_raw_bundle/plot_index_vs_raw.py:337](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/plot_index_vs_raw.py:337)，特別是 398–402 行附近。

qEEG 依 CLI `--fs` 切窗，quality 呼叫的 `pem.compute_quality_windowed` 卻固定 FS=500。兩者視窗數不同時，`if len(quality)==len(score_us)` 不成立，初始為全 True 的 `score_good` 就保留下來，只剩 gap 檢查。應將相同 fs 傳到共用 scorer；資料形狀不符應明確報錯，不應等同品質全部合格。

### F13 — P2：delta 圖使用絕對值範圍，負 entropy delta 被藏起來【程式碼確認】

位置：[plot_index_vs_raw.py:597](/home/bps-yichin/lilia_analysis/plot_index_vs_raw.py:597) 及 bundle 同函式。

`plot_single_signal()` 在畫 `ys-pre_mean`、`ys-post_mean`，entropy 的 y 軸仍固定 `[0,1]`，因此下降的 entropy 全部落在圖外。qEEG delta 也可能超出目前固定的 `[-1.1,1.1]`。

另 `flow_index(0,0,1)≈-1.10`、`relaxation_index(0,0,1)≈-1.15`，因為 bounded ratio 後又減 penalty。這是現有公式的結果；不能未確認規格就 clamp 改公式，但絕對值圖／標示不應宣稱嚴格在 [-1,1] 而裁切資料。

### F14 — P2：部分 CLI 參數實際不生效【程式碼確認／重現】

- [plot_index_vs_raw.py:186](/home/bps-yichin/lilia_analysis/plot_index_vs_raw.py:186) 接收 `event_name`，函式內完全沒使用，CLI `--event` 不會篩選事件；bundle 複本亦同。
- [lilia/signal.py:110](/home/bps-yichin/lilia_analysis/lilia/signal.py:110) 接受 float fs 卻直接 `int(fs)`。500.9→200、5009 samples 應得 2000，實際是 2004。相同问题存在 `resample_with_time()`；dataset builder 反而會拒絕非整數率。
- [run_goertzel_single_csv.py:63](/home/bps-yichin/lilia_analysis/run_goertzel_single_csv.py:63) 的 `--power-ymin` 只有搭配 `--power-ymax` 才生效，help 未說明。

修復：生效的參數才保留；不支援的組合立即驗證。取樣率以有理數處理，或明確限制 integer-like，避免靜默截斷。

### F15 — P2：TWSE 缺 Volume 路徑不可用，匯出也不足以重現推論【前者已重現，後者程式碼確認】

位置：[twse_index_lstm_rnn.py:141](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:141)、[twse_index_lstm_rnn.py:174](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:174)、[twse_index_lstm_rnn.py:718](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py:718)。

loader 容許缺 Volume 並補 0，後面 `Volume.pct_change()` 得到整欄 NaN，再 `dropna()` 把整份資料清空；220 列合成資料最後為 0 列。若 Volume 由 0 變正數也可能得到 Inf，`dropna()` 不會移除 Inf。

此外只保存 model.keras 與報表，未保存 feature 順序、feat_mu/sigma、tar_mu/sigma；日後無法只靠已輸出模型穩定重現標準化及逆轉換。grid search 選到不同 lookback 時，metrics 的 n_train/n_val 仍使用初始 lookback 的長度。

修復：明確 missing/zero-volume 特徵策略、finite 驗證；保存完整 feature/scaler/config artifact，以及最佳 trial 的真實資料計數。

### F16 — P2：輸出／重用檔案缺少設定識別，會混用或覆寫分析【程式碼確認】

- [summarize_goertzel_distribution.py:155](/home/bps-yichin/lilia_analysis/summarize_goertzel_distribution.py:155)：輸出檔名只有 threshold，相同 outdir 先跑 10 Hz 再跑 60 Hz 會覆寫；表格也沒有每列標記 target_freq。
- [build_jenqwei_tflite_dataset.py:189](/home/bps-yichin/lilia_analysis/build_jenqwei_tflite_dataset.py:189)：以 `Path(csv_path).stem` 當 session 目錄，不同父目錄同名 CSV 會寫到相同目的地。
- [run_goertzel_single_csv.py:80](/home/bps-yichin/lilia_analysis/run_goertzel_single_csv.py:80)：`--reuse-csv` 只檢查檔案存在，沒有確認來源內容、fs、win/step 或處理設定。更改參數後可以重用舊指標並加上新圖表標示。

修復：输出包含 session ID 與設定／來源 fingerprint；重用前檢查 metadata；輸出以暫存檔寫完再替換。

### F17 — P2：可容納足夠片段，隨機抽樣仍可能永久卡住【已重現】

位置：[lilia/segment_sampling.py:8](/home/bps-yichin/lilia_analysis/lilia/segment_sampling.py:8)、[quality_check.py:104](/home/bps-yichin/lilia_analysis/quality_check.py:104)。

目前隨機選第一段後不回溯。`n_total=200, seg_len=100, n_segs=2, seed=42` 只回傳 `[9]`；實際 `[0,100]` 可容納兩段。再嘗試一萬次也無法在固定第一段的情況找到第二段。quality_check 還會因少於 N_SEGS 而略過整份圖。

修復：先檢查容量，使用保證可配置的方法或有限回溯；若僅承諾 best effort，應讓 caller 接受不足，不把這種情況誤認資料不足。這是 index 非重疊，仍須另外處理 F04 的時間連續性。

### F18 — P2：驗證腳本可能把失敗回報成功【shell 程式碼確認】

位置：[validate_signal_processing.sh:46](/home/bps-yichin/lilia_analysis/validate_signal_processing.sh:46)。此檔不是 Python，但直接影響 Python 驗證可信度。

`python ... | tee test_output.txt` 沒有 pipefail，`$?` 是 tee 的狀態；unit tests 失敗仍可能 exit 0。compile/help 的 failure counters 也未納入最後 exit code。

修復：保留 Python 的 exit code 或設定 pipefail，整合所有階段失敗狀態。本次直接執行 unittest 取得 28/28，而不是採信此 shell summary。

### F19 — P3／介面正確性：quality 保留了不影響分數的參數【靜態確認＋局部重現】

位置：[lilia/quality.py:19](/home/bps-yichin/lilia_analysis/lilia/quality.py:19)、[lilia/quality.py:110](/home/bps-yichin/lilia_analysis/lilia/quality.py:110)、[lilia/quality.py:353](/home/bps-yichin/lilia_analysis/lilia/quality.py:353)，兩份 quality 複本亦同。

`target_score` 只被寫入 preset，scorer 不使用；同一輸入以 .1 與 .99 設定得到完全一樣的分數。以下 10 個 preset key 也沒有任何計分讀取：`alpha_artifact_floor`、`alpha_dpr_good/bad`、`nonalpha_artifact_floor`、`nonalpha_dpr_good/bad`、`low_freq_penalty_floor/ceiling`、`line_noise_penalty_floor/ceiling`。

目前 spectrum 分數主要只看 PSD slope，參數名稱不能當作具有 line-noise 或 DPR detector 的證據。應標記不支援／棄用並清理文件；不要為了讓舊參數生效，未經驗證直接補回一套不同計分方法。

## 4. 需要確認分析定義的風險

以下是結果解讀與方法一致性的問題，不全部等同程式 bug，也尚未驗證會把某份研究結論改成相反。

| 項目 | 證據與影響 | 重構時的處理 |
| --- | --- | --- |
| 隨機基線先拼接再推論 | `plot_tflite_summary.py:288` 把選到的不連續 1 秒 epochs concatenate，再 resample、以 2 秒模型窗推論及 Welch。人工接縫進入濾波／模型／頻譜。`spectral_entropy.compute_state_entropy_from_epochs` 已有逐 epoch 算 PSD 的不同策略。 | 定義基線統計單位。優先在原連續區段完成推論，再按真實視窗挑選並聚合；須用數值案例比較，不應直接當等價搬移。 |
| 名稱相同的 qEEG 不一定數值一致 | `lilia/qeeg.py:50` 的 Welch 4 秒；`spectral_entropy.py:323` 預設 1 秒。beta 邊界前者 `<30`、後者 `<=30`。合成 7.8/10/20 Hz 混合信號，theta 比例分別约 .158 與 .056。 | 共用 PSDConfig／band 定義；保留舊 profile 以重現，不只共用 Focus 公式就宣称結果相同。 |
| 基線策略差異 | event markers 的 pre-event-rest、summary 的 session／pre-event、index_vs_raw 的向前借用安靜區、compare_subjects 的第一活動前、TYY 的固定事件前不同。event markers 連續事件無基線時 bar 又 fallback 到所有先前資料，缺資料可顯示 0；heatmap 則可能 NaN。 | 以具名 baseline policy、實際 baseline 起訖、有效視窗數、fallback reason 使方法可追蹤；缺資料不得默認成「變化量 0」。 |
| 品質前處理與匯總不同 | event markers 對 raw 評分；summary 對 200 Hz 前／後訊號評分；Goertzel 对 BP 評分再加 raw artifact gate；TYY heatmap 沒有 quality mask。summary before 中位數是 4 ch，after 是 2 ch，不能說唯一差別只有 denoising。 | 明確記錄 signal stage、channel set、preset、aggregation 與 hard gate。判斷研究需要哪種比較，不強制套單一值。 |
| NaN／缺資料可能視為合格 | 部分流程用 `q < threshold` 當 bad；NaN 比較為 False。未對齊尾段默認 good；`lilia.quality` spectrum 例外回 0.5，正好等於多處 accepted 邊界。 | 建立 invalid 狀態與 reason；finite、shape、timestamp 不合法時不應當正常數值。 |
| 60 Hz 的 Goertzel 指標意義 | `plot_goertzel_vs_raw.py:99` 是對 0.5–45 Hz BP 後訊號算 60 Hz，不是 raw 60 Hz；core Goertzel 是未按窗長／Hann energy 正規化的 DFT magnitude squared。 | 可以保留作為既有校準指標，但必須標記 BP、fs、window、normalization；改窗長、fs 或濾波後不可直接沿用 67.15 dB 門檻。 |
| MI 的顯著性及命名 | band-event 特徵預設 1 秒／0.5 秒，有重疊，surrogates 卻逐列 permutation；baseline/event 統計可使用重疊視窗。joint_mi 把 sum MI−joint MI 稱為 redundancy percent；實作為 continuous/discrete estimator，但欄名叫 KSG。 | 在正式統計解讀前確認自相關、交換單位、跨事件 pooling、multiple comparisons 與估計器名稱；保留「差值」本身，勿自動把它解釋為純 redundancy。 |
| 模擬資料與實驗圖混淆 | `plot_timevarying_mi.py` 明確使用 simulate() 與預設尖峰，HTML 圖的標題卻沒有 simulated/demo。 | 放到 examples/，圖面加模擬標示。不能拿來支持某活動 MI 比較高。 |

## 5. 可共用與架構問題

`lilia/` 的既有抽取值得保留。最大依賴中心卻仍是 **`plot_event_markers.py`，被 13 個根目錄 production 腳本直接匯入**；`spectral_entropy` 的 quality path 還依賴 `plot_tflite_summary` 私有函式。修改畫圖或 session 常數會連動科學計算。

| 應共用的責任 | 目前散布位置 | 建議邊界 |
| --- | --- | --- |
| CSV metadata／訊號／時間契約 | io、merge、data_analysis、extract、analyze_jenqwei、eye_open_close、index_vs_raw | `lilia.io` 回傳 Recording：samples、fs、time_us/time origin、channel mapping、units、source。原始／merged／分析表是不同 schema。 |
| 濾波與 resampling | io.bandpass_filter、signal.bandpass、各 session 的 gcd/resample/interp | `lilia.signal` 單一核心。原 API 的 dtype 差異要保留或版本化：io 轉 float32，signal 現在不強制。 |
| 連續區段及 window 座標 | 各 compute_*_windowed、sampling、event mapping | 新 `lilia.windowing` 統一 start/end/center、segment ID、缺口／尾段策略、fs 驗證。 |
| session 與 events | plot_event_markers、plot_tyy、joint_mi、peri_event_delta | 新 `lilia.sessions/events`。活動、日期、時區、參與者由設定輸入；繪圖只讀資料。 |
| qEEG／quality 多通道批次 | plot_event_markers、quality_check、plot_tyy、plot_tflite_summary、spectral_entropy | `lilia.qeeg`、`lilia.quality` 的純計算 API；不同 PSD／quality profile 明示，不再藉匯入畫圖檔取得設定。 |
| Goertzel selection policy | distribution、global/per-subject sampling、hard-filter、render | 統一 finite、quality_final、hard clip、threshold 規則與輸入 schema。 |
| 模型載入／推論／訊號鏈 | data_analysis、extract、analyze_jenqwei、plot_event_markers、summary、TYY、compare | 模型 backend 與 PipelineConfig 分離，允許重用 interpreter/model，產出統一中間结果。 |
| baseline／delta／smoothing | event、summary、meditation_zoom、index_vs_raw、compare_subjects | 純函式計算結果與 metadata；渲染器只畫。NaN smoothing 規則要明確。 |
| 輸出／快取／圖表樣式 | 幾乎所有 root plotting 腳本 | 共用保存 PNG/SVG、建立輸出目錄、設定指紋與 metadata；各圖保留專有 layout。 |

兩條 neural pipeline **目前不等價**：data_analysis 有 notch＋33.25 Hz bandstop＋MAD 補點，PyTorch 用 50% overlap/Hann，保留原始長度；TFLite 主路徑用 BP、非重疊 400 samples、捨棄尾段。重構可共用元件，但不能無意改變其中一條的科學與邊界語義。

效能改善機會：analyze_jenqwei 每個 channel 都整份重跑 pipeline；Goertzel 每個 channel 重新讀檔／濾波且每窗重新算全通道 quality；global sampling 每張小圖重新載完整 recording；spectral state entropy 每窗重算 Welch 兩次；`np.correlate(..., full)` 的長序列 lag 搜尋沒有區間上限。先共用已算出的中間結果，再評估向量化／FFT，不宜先增加 multiprocessing。

## 6. 過時、重複與移除建議

### 可低風險清理

- Pyflakes 確認的未用 import：`data_analysis.py:1 sys`、`convert_to_tflite.py:13 sys`、`extract_tflite_signal_pipeline.py:25 gcd`、`:34 resample_polyphase`。後者名稱恰好在 summary 缺少，但仍須按各檔責任修正。
- 過時 docstring／help：index_vs_raw 開頭仍描述 first-part 兩基線，但實作已是逐事件 pre/post；eye_open_close 的 fmax help 寫 100、default 是 50；peri_event_delta 標題固定 30 秒卻可讀其他 pre/post 設定產物。
- quality 的未使用 key（F19）與未用設定，如模擬圖的 `SPIKE_TASKS`：可清理，但對外參數先明示棄用。

### 舊入口：已過時的實作位置，不等於可立即刪檔

`eeg_utils.py`、`eeg_quality_v2.py`、`qeeg_indices.py` 是清楚標示 Deprecated shim 的相容性入口；後者仍支援 CLI，README 及外部使用者可能依賴。`plot_index_vs_raw_session.py`、`resample_clean_goertzel_samples.py` 也刻意保留 CLI。

建議先把內部匯入及文件改為 canonical API，發出 deprecation 訊息、列出替代指令，再依版本政策移除。Goertzel resample 舊入口預設 n=10，新入口 n=6、outdir 也不同，不能只換指令名字就宣称完全相容。`power_spectral_app.py` 是獨立實用 CLI，不因為 wrapper 很短就刪除。

### 實體複本：應自動產生，不應維護多份來源

- SHA-256 比對：bundle 的 14 個 `lilia/*.py`、`plot_event_markers.py`、`merge_subject_csvs.py`，共 **16 檔完全相同**於主來源。
- `signal_quality_package/quality.py` 與 `lilia/quality.py` 完全相同，共有 root／bundle／standalone 三份 quality 實作。
- bundle 的 `plot_index_vs_raw.py` **不相同**：987 行對主版 750 行，含 custom markers、日期解析、baseline fallback 與 summary 匯出等獨有功能。

做法：把 bundle custom 功能帶回 canonical implementation，修 F12；獨立包若必須無 repo 依賴，就使用建置流程產生副本並做匯入／功能驗證。若 standalone 不需獨立散布，可改相容轉接。**本次未直接刪除任何複本或舊入口。**

### 適合移位而非刪除

`plot_timevarying_mi.py` 移到 examples/；`twse_index_lstm_rnn.py` 與 EEG 主題及依賴差異大，可獨立子專案／examples/market，仍需修其邏輯。Jenqwei、TYY、眼開閉等 session 腳本是可用研究流程，宜改設定＋薄入口，不因為日期較早就判定過時。

`tf.lite.Interpreter` 在本地 TensorFlow 2.21.0 執行時產生 deprecation warning，但本次推論確實成功；應規劃 backend adapter／遷移驗證，不能說它現在已不可用而直接移除。

## 7. 漸進重構順序與驗收

| 階段 | 變更 | 驗收條件 |
| --- | --- | --- |
| 1：保護數據正確性 | 修 F01/F02/F05/F06/F07/F08/F18；補最小失敗回歸案例。 | 秒／微秒精度、額外欄位讀取、merge 自訂 outname 重跑、完整 target 區間隔離、輸出 header 與 data 一致；故意讓測試失敗時驗證器非零退出。 |
| 2：時間與分段 | Recording、WindowTable、連續區段、timestamp event lookup；重算受影響 fixture。 | 合成中斷與 Hardy fixture 都不能跨 gap 做分析；事件時間與抽樣 index 一致。 |
| 3：抽離計算責任 | session/events、PSD/quality/windowing、baseline policy；保留 root CLI 轉接。 | 純計算模組不 import 根目錄 plot 腳本；未刻意變更的方法有數值容許誤差比對，變更的方法另立新 profile／版本。 |
| 4：推論與輸出統一 | PyTorch/TFLite backend、pipeline config、cache metadata；共享中間結果。 | 每 backend 的 shape、RMS、channel、尾段、短輸入有明確測試；cache 不相容不能重用；CLI 主要參數實際執行有驗证。 |
| 5：整理交付與舊碼 | bundle 生成、standalone quality 打包、demo／market 移位、棄用入口政策。 | bundle 與 canonical 功能一致，獨立環境可用；舊入口仍轉接到相同結果；文件與 dependency profiles 可重現。 |

建議的核心資料流：`CLI → session/config → IO/Recording → continuous segments → signal/model → window metrics/quality → baseline/summary → output/plot`。不需要一開始就加入龐大的 class framework；資料類型、純函式與明確設定已足夠。

依賴管理：目前根目錄 `requirement.txt` 註明只供 extraction，不能拿它安裝整個 repo；全專案還需 matplotlib、torch、plotly、scikit-learn、動畫所需 Pillow 等，market 路徑另有 yfinance。建議用 pyproject 或多份明確命名的 requirements profiles，並記錄外部 TinyUNetV4 程式碼版本。`lilia/__init__.py` 現在 eager import qEEG→matplotlib，連只需要 Goertzel／時間工具也會載入繪圖依賴，可降低此耦合。

目前不建議估算「刪掉多少行就是完成」；真正驗收是數值契約、歷史結果可重現、缺資料不冒充有效結果，以及 bundle 不再分叉。

## 8. 逐檔盤點

以下由檔案盤點附加；相同 SHA 的副本承接 canonical 檔案的問題，並另外列出打包處置。標示「未見額外問題」只代表此次審查未發現，並非完全正確的保證。

### 根目錄：33 檔

| 檔案 | 行數 | 責任 | 評估與處置 |
| --- | ---: | --- | --- |
| [analyze_jenqwei_pipeline.py](/home/bps-yichin/lilia_analysis/analyze_jenqwei_pipeline.py) | 449 | Jenqwei 原始 EEG→TFLite→時域/PSD/STFT 比較 | 保留薄入口；共用 pipeline、時間與 CSV 契約（F03–F05）。每個 channel 重跑全流程可避免；同名來源輸出可能碰撞；失敗 batch 目前仍正常返回。 |
| [build_jenqwei_tflite_dataset.py](/home/bps-yichin/lilia_analysis/build_jenqwei_tflite_dataset.py) | 318 | 濾波、降採樣、分割、整模型窗匯出 | 保留；接共用 pipeline、schema 與分段（F04/F16）。fs 非整數有驗證值得保留；metadata 名稱寫死 200hz，而 CLI fs-out 可改。 |
| [compare_subjects.py](/home/bps-yichin/lilia_analysis/compare_subjects.py) | 546 | 跨受試者/群組 BP 與 TFLite qEEG delta | 保留；抽 baseline/quality/推論。F04 與方法比較風險；mean/std 是跨 channel，不能誤讀為跨受試者推論誤差。 |
| [convert_to_tflite.py](/home/bps-yichin/lilia_analysis/convert_to_tflite.py) | 248 | TinyUNetV4 PyTorch→TF 權重映射及 TFLite 轉換 | 保留工具；去未用 sys。PT/TF 誤差超標只 return、exit 仍為 0；TFLite 誤差只印不判定，模型檔也在最終驗證前寫入。應驗證通過才正式輸出。 |
| [data_analysis.py](/home/bps-yichin/lilia_analysis/data_analysis.py) | 620 | APP/NUC 前處理、PyTorch 去噪、lag/PSD/STFT/qEEG | 優先 F01/F05；抽模型、pipeline、比較計算。MAD 全部被標為 bad 時 np.interp 無可用點，短輸入需明確處理；長序列 correlate 可另測效能。 |
| [eeg_quality_v2.py](/home/bps-yichin/lilia_analysis/eeg_quality_v2.py) | 14 | 舊 quality API 轉接 | 已棄用入口；先維持相容，再依遷移政策移除；不再放計算邏輯。 |
| [eeg_utils.py](/home/bps-yichin/lilia_analysis/eeg_utils.py) | 4 | 舊 CSV/bandpass API 轉接 | 已棄用入口；canonical 是 lilia.io，確認外部使用者後再移除。 |
| [extract_tflite_signal_pipeline.py](/home/bps-yichin/lilia_analysis/extract_tflite_signal_pipeline.py) | 304 | 匯出真實 TFLite pipeline 各階段及 JSON/NPZ | 保留作 canonical CLI 候選；F04/F05。已跑 10 秒真實推論；輸入不足 4ch 現在 min 截取而非提前拒絕；階段 CSV 是一般表格格式，需與 raw loader 明確區分。 |
| [filter_and_plot_hard_artifacts.py](/home/bps-yichin/lilia_analysis/filter_and_plot_hard_artifacts.py) | 284 | 剔除 hard artifact、CSV 報表與 gallery | 保留；F03/F10。timeline 用原 quality、main 優先 quality_final，圖表與匯出規則應一致；共享原始 index 定位。 |
| [joint_mi.py](/home/bps-yichin/lilia_analysis/joint_mi.py) | 365 | 逐活動 MI 及跨受試者 sum−joint 差值摘要 | 保留；F03/F11；抽 metrics/events。固定受試者/路徑改 config；gap-summary 依賴 surrogate_p_value，來源未算 surrogate 時缺欄會失敗。 |
| [merge_subject_csvs.py](/home/bps-yichin/lilia_analysis/merge_subject_csvs.py) | 180 | raw 相對 timestamp 加 offset、合併、排序 | 優先 F06；保留目前額外欄位解析修復及整列去重語義。補空資料、schema/取樣率衝突與 timestamp collision 報告。 |
| [plot_event_markers.py](/home/bps-yichin/lilia_analysis/plot_event_markers.py) | 1246 | session registry、quality/qEEG/TFLite、完整事件圖 | 需拆分的依賴中心，13 個 root 腳本依賴；F04、基線缺資料與 fallback、quality NaN、窗長空陣列；保留繪圖入口。 |
| [plot_focus_relax_anim.py](/home/bps-yichin/lilia_analysis/plot_focus_relax_anim.py) | 160 | 從 entropy CSV 重算 Focus/Relax 並產 GIF | 保留 renderer；共用 analysis CSV loader 與有效性策略。依賴上游時間 F03；補 fps/max_frames/window 正值驗證；finite 篩列後動畫會跳過缺資料時段。 |
| [plot_goertzel_histograms.py](/home/bps-yichin/lilia_analysis/plot_goertzel_histograms.py) | 143 | Goertzel dB 分布圖 | 已使用共用 collector；保留。finite 資料與 bins 驗證；若跨組比圖需要共用 bin edges，目前每張自動選邊界。 |
| [plot_goertzel_vs_raw.py](/home/bps-yichin/lilia_analysis/plot_goertzel_vs_raw.py) | 575 | Goertzel、quality、raw artifact metrics 及圖表/CSV | 抽出 WindowResult 計算與 renderer；F04/F10；按 channel 重算全通道 quality 浪費。60Hz 是 BP 後指標，應記錄設定。 |
| [plot_index_vs_raw.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw.py) | 750 | entropy CSV→指標、baseline/逐活動/session 圖 | 保留；優先 F03/F13/F14；統一 loader/window metadata，移入 bundle custom 功能；split 模式重讀/重算可共享。缺 quality 欄時全部 mask，與可選欄介面不一致。 |
| [plot_index_vs_raw_session.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_session.py) | 55 | session baseline 舊 CLI 轉接 | 已合併實作，可棄用但先維持入口；F03 的映射風險承接主版。 |
| [plot_meditation_zoom.py](/home/bps-yichin/lilia_analysis/plot_meditation_zoom.py) | 190 | 單活動 EEG 與 qEEG heatmap 放大 | 保留 renderer；重用預先算好的 window/baseline 結果。事件範圍無 bin 時 fallback 全 bins、零 bin 時取 index 會失敗；不應默認全段等於最近 bins。 |
| [plot_peri_event_delta.py](/home/bps-yichin/lilia_analysis/plot_peri_event_delta.py) | 112 | 跨受試者活動前後 delta 比較圖 | 保留；共用 registry/result schema。標題硬編 30 秒；CSV 的實際 pre/post 應主導標示。讀同一 CSV 兩次可減少。 |
| [plot_raw_eeg.py](/home/bps-yichin/lilia_analysis/plot_raw_eeg.py) | 107 | 隨機原始訊號片段 | 保留；F04/F17，片段應保證 timestamp 連續。固定 ±100 顯示只適合既定尺度；不要把原始圖的採樣壓縮當實際時間。 |
| [plot_tflite_summary.py](/home/bps-yichin/lilia_analysis/plot_tflite_summary.py) | 822 | 前後品質、TFLite qEEG、基線/delta summary | 優先 F02；抽 baseline/quality/pipeline；檢查不連續 epochs 拼接。API 標註 list[str]，缺檔回空字串；batch 捕捉 ValueError 後仍 Done/exit0，需結構化狀態。 |
| [plot_timevarying_mi.py](/home/bps-yichin/lilia_analysis/plot_timevarying_mi.py) | 152 | 合成事件 MI 動畫示範 | 移 examples，圖面明示 simulated。是有意模擬，不列為真實研究流程或自動判定要刪除。 |
| [plot_tyy_meditation.py](/home/bps-yichin/lilia_analysis/plot_tyy_meditation.py) | 435 | TYY 特定 session 冥想圖 | 改 session config＋renderer；共用 multi-channel qEEG/heatmap。F04、缺 quality mask／基線不足當 0；不應因同名指標就當與其他腳本可直接比較。 |
| [power_spectral_app.py](/home/bps-yichin/lilia_analysis/power_spectral_app.py) | 45 | 單頻 Goertzel CLI 與相容函式 | 保留 CLI；core 已共用。補 fs>0、frequency 範圍與有限值契約；不要因短 wrapper 而視為無用。 |
| [process_lilia_eye_open_close.py](/home/bps-yichin/lilia_analysis/process_lilia_eye_open_close.py) | 320 | 8ch 原始→兩次 4→2 PyTorch 推論及匯出 | 優先 F01/F08；抽 loader/pipeline/export metadata。fmax help 與 default 不一致。 |
| [qeeg_indices.py](/home/bps-yichin/lilia_analysis/qeeg_indices.py) | 16 | 舊 qEEG imports/CLI 轉接 | 維持相容直到文件/外部呼叫遷移；承接 F09 與 qEEG PSD 定義。 |
| [quality_check.py](/home/bps-yichin/lilia_analysis/quality_check.py) | 556 | 隨機片段品質與時序 anomaly 診斷 | 保留；window scorer 下沉，diagnostics/render 分離。F17；目前知道如何找 gap，但未形成全 pipeline 的 gate。 |
| [resample_clean_goertzel_samples.py](/home/bps-yichin/lilia_analysis/resample_clean_goertzel_samples.py) | 41 | 逐受試者重新抽樣舊 CLI | 可棄用；與新 CLI 預設 n/outdir 不同，轉接需保留舊預設；共用 argparse builder。 |
| [run_goertzel_single_csv.py](/home/bps-yichin/lilia_analysis/run_goertzel_single_csv.py) | 120 | 單 CSV Goertzel pipeline/舊 CSV 重畫 | 保留薄入口；F14/F16，改依賴公開 service，而非 pgv._plot_subject 與 pgv.pem。 |
| [sample_segments_by_goertzel_db.py](/home/bps-yichin/lilia_analysis/sample_segments_by_goertzel_db.py) | 311 | 全域或逐受試者選窗及波形圖 | 保留；F03/F10；全域 collector 應保留 window indexes；載入/濾波結果按 recording 重用。per-subject 是 nearest pool 抽樣，tol-db 不適用應寫明。 |
| [spectral_entropy.py](/home/bps-yichin/lilia_analysis/spectral_entropy.py) | 3565 | entropy、BASD、MI、顯著性、事件、圖與 CLI | 3565 行，最需要按責任拆分；F03/F04/F11、PSD/quality/統計策略。保留 BASD 公開 API，即使當前 CLI 未接通也不能只按內部引用數刪除。 |
| [summarize_goertzel_distribution.py](/home/bps-yichin/lilia_analysis/summarize_goertzel_distribution.py) | 167 | Goertzel 分布統計及共同邊界 histogram CSV | 保留，已共用 collector；F16 與 finite 值驗證；ALL_SUBJECTS 是樣本加權，不是每個受試者等權。 |
| [twse_index_lstm_rnn.py](/home/bps-yichin/lilia_analysis/twse_index_lstm_rnn.py) | 754 | 市場時序 LSTM/RNN、多步/grid search/baseline | 移獨立子專案或 examples/market；優先 F07/F15。best trial metadata、標準化 artifact 與資料完整性比拆 class 更重要。 |

### 核心 lilia：14 檔

| 檔案 | 行數 | 責任 | 評估與處置 |
| --- | ---: | --- | --- |
| [lilia/__init__.py](/home/bps-yichin/lilia_analysis/lilia/__init__.py) | 27 | 公開 re-export | 保留；減少 eager import qEEG/plotting，讓低階 core 不必依賴 matplotlib。 |
| [lilia/constants.py](/home/bps-yichin/lilia_analysis/lilia/constants.py) | 21 | fs、模型 shape、濾波範圍、品質門檻/顏色 | 保留；逐步讓重複預設值引用它；實驗差異放 profile，不宜全部當全域常數。 |
| [lilia/goertzel.py](/home/bps-yichin/lilia_analysis/lilia/goertzel.py) | 46 | DC/Hann/Goertzel recurrence | 保留 canonical；與 C 演算法對照及 DFT 合成訊號測試值得補；明示未正規化 power，並驗證 fs/frequency。 |
| [lilia/goertzel_distribution.py](/home/bps-yichin/lilia_analysis/lilia/goertzel_distribution.py) | 134 | 依 subject/channel 收集、聚合、hist edges | 保留；F10 的品質契約；finite power/db、重複 channels 及缺根目錄需驗證。 |
| [lilia/goertzel_sampling.py](/home/bps-yichin/lilia_analysis/lilia/goertzel_sampling.py) | 117 | 全域與逐受試者 Goertzel 選窗 | 優先 F10；兩個 collector 不再悄悄用不同 quality。廣泛 except Exception 會把格式錯誤当缺檔忽略；fallback 固定 5 秒應由 metadata 取代。 |
| [lilia/io.py](/home/bps-yichin/lilia_analysis/lilia/io.py) | 86 | 四列 metadata CSV、offset、bandpass | 優先 F05；raw/merged 時間與欄位契約。bandpass core 移 signal，保留 dtype 相容 wrapper；offset 按固定 col3 解析需 schema 錯誤訊息。 |
| [lilia/pathing.py](/home/bps-yichin/lilia_analysis/lilia/pathing.py) | 55 | repo root 與外部模型 import path | 保留；候選依序 insert(0) 使实际 precedence 與文件順序相反。只捕捉真正缺模組，別把任意外部 import 錯誤都包成路徑缺失。 |
| [lilia/qeeg.py](/home/bps-yichin/lilia_analysis/lilia/qeeg.py) | 282 | 相對頻帶功率、wellness indices、window/plot/CLI | 優先 F09；拆 pure metric 與 plot/CLI；PSD profile、指標範圍、NaN/短輸入契約。 |
| [lilia/quality.py](/home/bps-yichin/lilia_analysis/lilia/quality.py) | 472 | 裝置參數 preset 與品質評分 | 保留唯一來源；F19；finite/空輸入與錯誤狀態需明確。flat 子窗 range 未包含最後合法起點，恰 0.5 秒輸入沒有任何子窗；不應無意改校準。 |
| [lilia/segment_sampling.py](/home/bps-yichin/lilia_analysis/lilia/segment_sampling.py) | 35 | 隨機非重疊片段 | 優先 F17；目前為 best effort greedy，不能保證可行數量；增加 segment continuity 輸入邊界。 |
| [lilia/signal.py](/home/bps-yichin/lilia_analysis/lilia/signal.py) | 171 | filter chain、resample 與 time interpolation | 優先 F01/F04/F14；建立單位/fs/dtype/shape 契約；io 的等效 bandpass 可共享但保留輸出 dtype。 |
| [lilia/subject_paths.py](/home/bps-yichin/lilia_analysis/lilia/subject_paths.py) | 33 | subject/merged/Goertzel 路徑探索 | 保留；能判定 merged 存在值得沿用。root 缺失策略一致化；不要與 merge 的 raw-source 探索混成同一條件。 |
| [lilia/tflite.py](/home/bps-yichin/lilia_analysis/lilia/tflite.py) | 54 | 每窗 RMS normalize、TFLite、rescale | 保留 backend；已真實推論成功。驗證 input tensor shape/dtype/channel/win；明示尾段丟棄/空輸出，重用 interpreter；分段責任在呼叫層。 |
| [lilia/time_utils.py](/home/bps-yichin/lilia_analysis/lilia/time_utils.py) | 34 | local datetime 與 UTC us | 保留；已有回歸測試，未見本次高優先單位換算錯誤。統一有時區/naive datetime 邊界；F01 是錯誤呼叫契約而非此檔 UTC 工具。 |

### 獨立 quality：2 檔

| 檔案 | 行數 | 責任 | 評估與處置 |
| --- | ---: | --- | --- |
| [signal_quality_package/eeg_quality_v2.py](/home/bps-yichin/lilia_analysis/signal_quality_package/eeg_quality_v2.py) | 14 | standalone 相容轉接 | 以相對 import 支援獨立散布；保留與否依交付用途，不應單靠無內部引用判死碼。 |
| [signal_quality_package/quality.py](/home/bps-yichin/lilia_analysis/signal_quality_package/quality.py) | 472 | 獨立包 quality 完整複本 | SHA 與 lilia/quality.py 相同；承接 F19 與核心問題。建置生成或依赖 canonical，避免手動同步。 |

### bundle：17 檔

| 檔案 | 行數 | 責任 | 評估與處置 |
| --- | ---: | --- | --- |
| [plot_index_vs_raw_bundle/lilia/__init__.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/__init__.py) | 27 | 獨立 bundle：公開 re-export | 與 lilia/__init__.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/constants.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/constants.py) | 21 | 獨立 bundle：fs、模型 shape、濾波範圍、品質門檻/顏色 | 與 lilia/constants.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/goertzel.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/goertzel.py) | 46 | 獨立 bundle：DC/Hann/Goertzel recurrence | 與 lilia/goertzel.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/goertzel_distribution.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/goertzel_distribution.py) | 134 | 獨立 bundle：依 subject/channel 收集、聚合、hist edges | 與 lilia/goertzel_distribution.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/goertzel_sampling.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/goertzel_sampling.py) | 117 | 獨立 bundle：全域與逐受試者 Goertzel 選窗 | 與 lilia/goertzel_sampling.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/io.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/io.py) | 86 | 獨立 bundle：四列 metadata CSV、offset、bandpass | 與 lilia/io.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/pathing.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/pathing.py) | 55 | 獨立 bundle：repo root 與外部模型 import path | 與 lilia/pathing.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/qeeg.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/qeeg.py) | 282 | 獨立 bundle：相對頻帶功率、wellness indices、window/plot/CLI | 與 lilia/qeeg.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/quality.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/quality.py) | 472 | 獨立 bundle：裝置參數 preset 與品質評分 | 與 lilia/quality.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/segment_sampling.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/segment_sampling.py) | 35 | 獨立 bundle：隨機非重疊片段 | 與 lilia/segment_sampling.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/signal.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/signal.py) | 171 | 獨立 bundle：filter chain、resample 與 time interpolation | 與 lilia/signal.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/subject_paths.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/subject_paths.py) | 33 | 獨立 bundle：subject/merged/Goertzel 路徑探索 | 與 lilia/subject_paths.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/tflite.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/tflite.py) | 54 | 獨立 bundle：每窗 RMS normalize、TFLite、rescale | 與 lilia/tflite.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/lilia/time_utils.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/lilia/time_utils.py) | 34 | 獨立 bundle：local datetime 與 UTC us | 與 lilia/time_utils.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/merge_subject_csvs.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/merge_subject_csvs.py) | 180 | 獨立 bundle：raw 相對 timestamp 加 offset、合併、排序 | 與 merge_subject_csvs.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/plot_event_markers.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/plot_event_markers.py) | 1246 | 獨立 bundle：session registry、quality/qEEG/TFLite、完整事件圖 | 與 plot_event_markers.py 完全相同；承接同檔分析，改由 canonical source 建置產生，保持獨立交付測試。 |
| [plot_index_vs_raw_bundle/plot_index_vs_raw.py](/home/bps-yichin/lilia_analysis/plot_index_vs_raw_bundle/plot_index_vs_raw.py) | 987 | bundle 指標圖＋custom marker 功能 | 987 行，非可直接刪除複本。獨有 marker/date/fallback/summary 要先回收主版；F12，並承接 F03/F13/F14。 |

### 測試：6 檔

| 檔案 | 行數 | 責任 | 評估與處置 |
| --- | ---: | --- | --- |
| [tests/test_goertzel_distribution_regression.py](/home/bps-yichin/lilia_analysis/tests/test_goertzel_distribution_regression.py) | 96 | 3 個 collector/aggregate/bin-edge 測試 | 保留；補 nonfinite values 與跨頻率 metadata。temporary CSV fixture 可共用，但現在主要用來存在性探索，不是 loader 格式驗證。 |
| [tests/test_goertzel_sampling_regression.py](/home/bps-yichin/lilia_analysis/tests/test_goertzel_sampling_regression.py) | 96 | 3 個選窗與種子重現測試 | 保留並修規格；目前 global 測試期待包含 hard clip，正好固定了 F10 差異；新增統一 policy 與真實 index 追蹤案例。 |
| [tests/test_phase_smoke.py](/home/bps-yichin/lilia_analysis/tests/test_phase_smoke.py) | 132 | 14 個 import/CLI smoke tests | 保留為 smoke 層；不能代替真正 numerical branch。TensorFlow/torch 匯入成本高，適合與純核心測試分組。 |
| [tests/test_segment_sampling_regression.py](/home/bps-yichin/lilia_analysis/tests/test_segment_sampling_regression.py) | 50 | 3 個非重疊/短檔/種子測試 | 保留；缺 F17 緊密可配置情境與 gap 檢查，現測試只檢查 <= n。 |
| [tests/test_subject_discovery_regression.py](/home/bps-yichin/lilia_analysis/tests/test_subject_discovery_regression.py) | 58 | 2 個目錄探索測試 | 保留；補 missing root 與排除 generated-output 情境；raw merge 需另一套 schema 測試。 |
| [tests/test_time_utils_regression.py](/home/bps-yichin/lilia_analysis/tests/test_time_utils_regression.py) | 35 | 3 個 local/UTC/roundtrip 測試 | 保留；補 aware datetime、微秒精度與 F01 的秒/微秒跨 API 整合測試。 |

## 9. 重要重現案例摘要

在目前環境 NumPy 2.2.6、SciPy 1.15.3、pandas 2.3.3、TensorFlow 2.21.0、torch 2.11.0 下執行；模型推論使用本地檔案，無網路下載。

```python
# F01：呼叫端給秒；400 個 200Hz timestamps 只剩 0、1。
from data_analysis import downsample_data
import numpy as np
out_t, _ = downsample_data(np.arange(1000) / 500, np.ones((1000, 4)))
print(out_t[:10], np.unique(out_t))
# [0 0 0 0 0 0 0 0 0 0] [0 1]

# F02：並非 import 或 --help 可以發現的執行分支。
from plot_tflite_summary import _resample_500_to_200
_resample_500_to_200(np.ones((1000, 4)))
# NameError: name 'resample_polyphase' is not defined
```

```python
# F09：允許的兩秒 qEEG 視窗。
from lilia.qeeg import compute_qeeg_indices
compute_qeeg_indices(np.sin(2*np.pi*10*np.arange(1000)/500), fs=500)
# ValueError: noverlap must be less than nperseg.

# F17：容量足夠，greedy 的第一個隨機位置使第二段不可配置。
from lilia.segment_sampling import pick_non_overlapping_segments
print(pick_non_overlapping_segments(200, 100, 2, np.random.default_rng(42)))
# [9]，但 [0,100] 本來可容纳兩段。
```

Pyflakes 完整回報為四個未使用 import 及一個 undefined name；本報告的其他結果來自呼叫鏈、資料契約、真實 timestamp、合成資料重現，不能只靠 lint 推得。

本次工作起始已存在 README.md、merge_subject_csvs.py 的修改，以及 bundle、requirement.txt、TWSE script/output 等未追蹤內容；報告均以此工作樹為基礎，保留這些既有改動。
