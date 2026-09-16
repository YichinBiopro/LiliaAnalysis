# 方法校準：舊 profile 與比較契約

日期：2026-09-16。凍結版本：`aa0df5f4ebaefddae1fee8c9e50e95323e777aed`（Stage18 驗收後）。
本文件中的 ID 是校準盤點名稱，尚未新增產品 CLI/profile API。記錄現行程式行為，不代表方法已獲科學驗證。

## Baseline policies

| 舊 profile ID | 入口／實作 | 保留的選取與聚合語義 | 本輪新基準涵蓋 |
| --- | --- | --- | --- |
| B-event-session | `plot_event_markers` → `lilia.event_qeeg.summarize_branch` | 第一個參與事件前的完整窗口；bar 為逐通道事件均值減 baseline 均值，再通道 mean/SD；heatmap 先每 6 窗口取通道／窗口 median，再對 baseline bins 取 median；無參與事件時取前 `max(1,n_bins//5)` bins。 | M0 真實與合成 BP session／無事件；M1 合成多事件 BP／實際模型分支。 |
| B-event-rest | 同上及 event zoom | 每事件前、前一事件結束後的完整窗口；非參與事件也更新前一事件終點；不向較早區間 fallback。 | M1 完整 TYY protocol BP、多事件／非參與者合成 BP／實際模型；bar 有效但 heatmap baseline bins 不足的差異明示。 |
| B-subject | `compare_subjects` → `lilia.subject_comparison` | 有事件：第一參與事件前完整窗口；無事件：前 `max(1,N//5)` 候選列；先定候選再套品質；不是 elapsed time 的前 20%；無 fallback。 | 兩政策、候選／接受／排除列與通道 delta。 |
| B-meditation | `lilia.meditation.summarize_meditation` | 裁切後各來源段每 6 窗口成 bin；完整截止於 baseline_end 的 bins；median reference；quality disabled。 | M1 完整 TYY 預設入口（實際模型）與合成缺口／無 baseline；逐段 bin、reference、delta 與來源 reader。 |
| B-tflite | `plot_tflite_summary` → `lilia.tflite_baseline` | 500 Hz filtered 四通道按 1 秒子窗品質 median ≥ .5；raw 任通道 ≥2047 的樣本比例 > .02 排除；全部子窗合格才接受模型窗；200 Hz／400 點模型窗，所需 10 秒取 ceil 個完整窗；seed 42 無放回取樣排序；逐模型窗兩通道 qEEG 後取均值。事件搜尋預設 onset−15 到 onset−3 秒，session 搜尋全段。 | M0 真實模型 Before／After、時間／索引、catalog 與選取；M1 合成缺口完整 summary 的 session／pre-event 兩政策、實際模型與 reader；不足保留明確錯誤。 |
| B-legacy-sampler | `plot_tflite_summary.build_baseline_epochs/build_session_baseline` | 舊 1 秒 filtered/raw 品質篩選及抽樣 API；packed baseline 不可交給 `_tflite_qeeg_reference` 重跑模型。 | M1 talk 真實來源、品質 =.5 接受／.49 不足、缺口 guard；兩個抽樣 API 的 seed／接受與輸出。 |
| B-marker | `plot_index_vs_raw` | `_baseline_mean` 用窗口中心落在 `[lo,hi)` 的 nanmean；custom marker 預設前 30 秒；僅 marker 早於錄製且前窗無有效分數時，fallback 到錄製開始後 30 秒。 | M1 talk／move-head 鍵盤絕對微秒；合成錄製前 fallback、中心半開邊界、缺口及錄製外 marker；繪圖缺口修正另驗數值不變。 |
| B-entropy-state | `spectral_entropy.compare_baseline_event` | 使用者指定半開 baseline/event 實際時間；各區間與來源段交集重新排完整窗；pooled entropy 與逐窗分布的 Mann–Whitney U 各自保留。 | M1 talk 鍵盤區間 ordinary／clean；合成缺口、品質 =.5 與無完整窗口；pooled PSD／entropy、逐窗、delta、U／p 全精度比較。 |

raw／BP／model 索引與品質 stage 不共用。一般 event baseline 接受 `>= .5`；診斷 invalid 不自動取代既有 legacy policy。缺 baseline 的 NaN／status／例外也是比較契約。

## PSD profiles

| 舊 profile ID | 實作／參數 | 差異與界線 |
| --- | --- | --- |
| P-qeeg | `lilia.qeeg.compute_relative_powers`：float64，Hann；`nperseg=min(N,max(1,round(4fs)))`、overlap `min(round(2fs),nperseg//2)`；其餘沿用安裝的 SciPy defaults。 | θ[4,8)、α[8,13)、β[13,30)；trapezoid；三帶總和 +1e−9 正規化；單一頻點的 trapezoid 為 0。 |
| P-entropy | `spectral_entropy._compute_welch_psd`：float64，Hann、constant detrend、density；預設 `min(N,max(8,round(fs)))`，50% overlap。 | 1 秒 PSD；band helper 可控制 upper bound，單一頻點用 `PSD×df`；不可以共用 helper 抹平與 P-qeeg 的差異。 |
| P-hardy | `lilia.hardy2.window_metrics`：4 秒 Welch，`int(fs*4)`／`int(fs*2)`。 | delta 1–4、theta 4–8、alpha 8–13、beta 13–30、gamma 30–45，各帶**兩端包含**；五帶總和 +1e−12；qEEG 四指標另外呼叫 P-qeeg。 |
| P-comparison | `lilia.comparison.segmented_psd`：固定 `nfft=round(4fs)`；短段縮 Welch 窗再補零至同一格點。 | 逐來源段、不跨缺口；線性 PSD 以 Welch 子窗數加權，再 `10log10(power+1e−12)`；本輪保存完整頻率／dB／權重 audit。 |
| P-jenqwei | `lilia.jenqwei_plot.compute_panels`：200 Hz、`nperseg=min(800,Before段長,當前branch長)`、density，其餘 SciPy defaults。 | Before 保留完整重採樣尾端，After 段內裁 400 點；每段各畫 PSD，不串接／平均；本輪只凍結程式，Stage16 保存真實模型與 PSD/STFT 比較。 |
| P-quality | `lilia.quality` spectrum component：`welch(ch_data,fs,nperseg=fs*2)`；fit `(1,spectrum_fit_hi)`。 | 2 秒 profile 與 component fallback／preset 相連，不能用分析圖 PSD 替換；新基準透過實際 scorer 間接涵蓋。 |
| P-quality-plot | `quality_check` sample PSD：float64、`nperseg=4fs`、overlap `2fs`，raw/filtered 分畫。 | 圖形與 quality component 是不同 profile；本輪只凍結程式，Stage18 R6 已有來源表與圖形證據。 |

新增基準對 P-qeeg 保存 relative powers，P-hardy 保存各 band／index，P-entropy／P-comparison 保存頻譜。這不是所有 PSD 入口的全量驗收。SciPy／NumPy 版本與凍結原碼 hash 均記在 `config.json`，未把依賴 defaults 偷換成新參數。

## Goertzel profile

- G-legacy：`lilia.goertzel.goertzel_power` 使用原始目標頻率（不取最近 FFT bin）、移除 mean、對稱 Hann `N−1`、Goertzel recurrence；沒有除以 N、N² 或 Hann gain；空輸入回 0。
- CLI／single CSV 預設 500 Hz、60 Hz、5 秒窗／5 秒步、smooth 5。輸入為來源段內 float32 **0.5–45 Hz BP**；因此 60 Hz 值與前端衰減綁定，現有 dB 門檻不能直接移植至 raw 或新 normalization。
- `10log10(max(power,1e−12))`；raw ch 的 abs≥1950 比例≥.12、PTP≥1000，或 filtered 首尾 1 秒 median 差≥80 任一成立為 hard artifact；max diff 僅記錄，不是門檻。
- 共用品質選取 `valid_goertzel_rows` 使用 **quality > .5**（嚴格大於）；預設排 hard artifact，須同時有限時間／dB。filtered 全通道評分後取指定通道 legacy overall。
- 顯示先 mask，再 pandas 中置 rolling median 5／min_periods=1，再 mask；這個 rolling 本身沒有 segment 分組，仍可能受缺口另一側窗口影響；圖上斷線不代表平滑已分段。本輪凍結此行為，分段平滑須另開明示差異。
- 新基準涵蓋 5 秒與 .5 秒窗、raw artifacts／品質、接受 mask／平滑；下游 distribution/sampling 的共用選取規則不改。

## MI profiles

| 舊 profile ID | 現行設定 | 不可混用的意義 |
| --- | --- | --- |
| M-histogram | `compute_joint_probability`：每信號去 mean、population std、clip ±5；16 bins；helper default uniform；joint CLI default quantile；ties 用 unique 邊界。 | bits，plug-in／Miller–Madow（負值截 0）／MI÷min(Hx,Hy) 都保留；population 與 window 各自 normalization，不可互換。 |
| M-lagged | `compute_lagged_interhemispheric_sync_windowed`：2 秒／步長預設同窗；雙向非零 lag histogram MI 平均；tau 預設 5／10／15／20 ms。 | 不等於零 lag joint；顯式 grid 時必須先按 timestamp 濾波，再 `apply_bandpass=False`。本輪保存完整返回陣列。 |
| M-circular-null | `compute_joint_mi_significance`：200、seed 0；每來源段獨立 circular shift，offset `[max(1,n//100),n)`；p 加一校正、null std ddof=0；短於16樣本的 run 停用 null。 | joint CLI 的全人口通常不套逐窗品質 mask；不能把 population p-value 當每個窗口顯著性；與 label permutation 不同。 |
| M-event-KSG | `compute_band_event_joint_mi`：θ/α/β bandpass＋Hilbert amplitude envelope；1秒子窗／.5秒步；k=min(3,n_pre−1,n_post−1)；scale＋jitter seed0；multivariate continuous/discrete estimate 與 sklearn 各 band 加總都存，nats→bits。 | feature 是 amplitude envelope，不能因文字「band-power」改成平方功率；label-shuffle null std ddof=1，與 circular-null 不同。helper 預設0、spectral CLI預設200、joint_mi per-event固定500 surrogates；新探針基準明示200。 |

`plot_timevarying_mi.py` 是模擬展示，不是實際 estimator，排除校準數值基準。`joint_mi.py gap-summary` 是既有表彙整；需隨未來 estimator 變更追蹤表來源。

## 固定基準與涵蓋界線

- 執行：`python tools/freeze_method_profiles.py --out /path/to/new-directory`；輸出目錄必須不存在。
- 凍結版從固定 commit 擷取 root／lilia Python；凍結版、工作區版由**不同 Python process**執行，使用同一模型／來源 hash／參數與環境。每個陣列比較 keys、dtype、shape、值及 NaN 位置，rtol=atol=0；JSON selections／status／audit 精確相同。
- 五份 Jenqwei 完整真實錄製（前四個宣告通道）＋60 秒合成連續／兩個30秒且間隔10秒／半秒短資料，共8案例；沒有截短真實來源、修改正式來源或覆寫模型。
- baseline 事件區間 [30,60) 秒與 KSG 每段中點 ±5 秒皆是**驗證探針**，不是受試者行為標註。KSG baseline 不代表真實事件效應；未宣稱顯著性或估計器優劣。
- `local/` 留真實 NPZ／圖、逐例 audit、凍結原碼；Git 保留合成 NPZ／圖、config（含來源／模型 hash）、比較摘要／log。repository manifest 不要求本機正式來源；local manifest 額外綁定來源／模型與真實結果。
- Stage18／Stage16 的原證據按各報告範圍保留；本輪不重算舊 hash、不把歷史報告當本輪重跑。尚未新捕獲的 profiles 已在上表明列，不可標成「所有方法已驗收」。

## M1 完整入口補充

- 執行：`python tools/freeze_baseline_entries.py --out /path/to/new-directory`；`--cases` 可明示子集；完整 19 案例、585 陣列與 metadata 精確相同，20 來源表 reader。實際 TFLite 輸入／輸出也保存比對。
- Jenqwei talk／move-head `time_marker.csv` 是鍵盤 trigger；原始相對時間只在獨立四通道副本加 header offset。talk 的 `Abs Time − (offset + Rel Time)` 為 −1 µs，move-head 為 0；保留絕對欄六位小數，不自動修正。TYY 為完整 2,009,328 列正式來源，其事件／meditation 時段來自既有 protocol 設定，不是觀測標註。
- 合成注入品質 .5／.49 的案例保留 diagnostics unavailable；不把注入分數當實際品質診斷。legacy sampler continuity guard 保留。
- 目視發現 custom marker 在兩個完整窗口間仍跨缺口連線；僅繪圖依來源段插 NaN，保留有效端點／孤立點，raw 同樣不丟端點。修正後真實兩例＋合成缺口 72 陣列、3 reader 與原版一致；CSV、baseline、品質政策未改。
- 原 19 案例產物及預期 hash 不重算；舊繪圖程式 hash 綁定同 bytes 快照，當前版本由 marker 重驗與最新全套檢查支持。詳見 [驗收範圍](validation/method_calibration/m1_acceptance_2026-09-16/scope.json)；其餘 PSD capture 仍待 M2。

## 後續校準順序與驗收門檻

1. **Baseline 完整入口基準：M1 完成**。各入口與缺口／不足／品質邊界的候選、排除、selected indices、reference、delta 已核對；範圍與限制見 [M1 報告](METHOD_CALIBRATION_M1_REPORT_2026-09-16.md)。政策仍各自保留。
2. **PSD 校準**：補 P-jenqwei／P-quality-plot 新 capture；用同輸入評估1秒／4秒、頻帶邊界及單頻點規則。校準前先定比較量與容許差異，保留 legacy profile；不能把 PSD 差異混入搬移。
3. **Goertzel 校準**：先單獨比較分段平滑，再評估 raw/BP、窗長與 normalization；須列線性功率、dB、接受／排除窗口差異，既有門檻不直接套新單位。
4. **MI 校準**：分開 histogram、event KSG、各 null 與品質 population；新增有已知關係的合成控制及真实事件標註；評估 bins／樣本數／k／seed／surrogate數的敏感度。方法判讀前另查一手方法文獻，這一輪不下科學結論。

每次方法變更需具名新 profile、明示預設／相容策略、獨立原基準、數值差異與 NaN／索引／品質差異、来源讀回與目視；受影響測試及階段完整檢查通過後才標完成。架構／F19、批次原子發佈仍排在方法校準之後。
