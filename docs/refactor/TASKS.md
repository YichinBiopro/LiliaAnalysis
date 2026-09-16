# 階段任務單

只讀當前卡片。14–18 已完成；方法校準已完成 M0 核心基準與 M1 baseline 完整入口，當前為 M2：補 PSD 基準並定義比較契約。
所有任務共用 [WORKFLOW](WORKFLOW.md) 的驗證與收尾規則。方法校準另立任務，勿混入等價搬移。

<a id="stage-14"></a>
## 14：qEEG CLI — 已完成（2026-09-08）

- 成果：221 tests／0 skipped、完整 Hardy 845 窗口、三入口表相同；表重讀／hash／數值／目視已完成，CLI guard 已解除。[增量報告](STAGE14_REPORT_2026-09-08.md)。

- 目標：`lilia/qeeg.py`、`qeeg_indices.py`、必要的新 IO／測試與生成 bundle。
- 先讀：上述兩檔、`lilia/windowing.py`；用 `rg` 盤點 `compute_qeeg_indices_windowed` callers。
- 保留：單一 raw channel 的 theta／alpha／beta 相對功率、四指標、連續窗口中心、summary／PNG；無 BP／TFLite／品質／baseline。無 timestamp 的共用 helper 保留既有語義。
- 實作：保留整數微秒；WindowGrid 原始索引／segment／真實時間；不跨缺口窗口；短尾、非有限與無合法窗口有 audit。CSV/meta 核對來源／channel／設定／全部窗口。兩面板圖斷線且孤立窗口可見。
- 完成條件：舊連續數值基準通過；缺口、短段、非有限、錯誤 channel、邊界案例通過；全短／全排除明確失敗；完整有缺口 Hardy 真實分析；主版、相容 CLI 與 bundle 核對；表重讀、產物 hash、目視完成後才解除 guard。
- 起始測試：`python tools/refactor_check.py check --tests tests/test_signal_contract_regression.py tests/test_time_utils_regression.py tests/test_entropy_windows_regression.py tests/test_qeeg_cli_regression.py`
- 已新增入口專屬測試並加入命令；階段完成依據另含真實資料、產物與目視證據。
- 收尾：完整檢查＋實跑 evidence manifest＋增量報告；更新短進度至任務 15。

<a id="stage-15"></a>
## 15：眼開閉 — 已完成（2026-09-09）

- 成果：[完整驗收報告](STAGE15_ACCEPTANCE_REPORT_2026-09-09.md)。分段 TD／STFT、elapsed 軸、ch5/6 來源映射與標題完成；CLI guard 的解除已通過驗收。
- 驗證：246 tests／0 skipped、151 Python 靜態／bundle／diff；88 項數值／產物 evidence、9 項持久目視產物 hash、六張正式圖與兩張細節圖目視通過。
- 真實連續 18,109 列、真實訊號分段副本 1,403 列、合成連續 602 列及小缺口 800 列；模型與 STFT dB 最大誤差 0，整數時間精確一致。
- 保留：八通道映射、float32 bandpass、兩組四通道 PyTorch 輸入與 ch1/2/5/6 輸出、鏡射 OLA、完整重採樣尾樣本；不引入 notch／bandstop 或品質 scorer。
- 專屬測試：`tests/test_eye_open_close_regression.py`、`tests/test_eye_plot_regression.py`；實跑：`tools/validate_eye_stage15.py --out /path/to/new-directory`。
- 歷史：[起步基準](STAGE15_REPORT_2026-09-09.md)／[CLI／IO 增量](STAGE15_IO_REPORT_2026-09-09.md)。其中 guard 保留、繪圖未完成均為當時狀態。
- Git：基準與 CLI／IO 已提交 `b84029e`；繪圖／測試／完整驗收證據已提交 `141bde2`，未 push。

<a id="stage-16"></a>
## 16：Jenqwei 分析 — 已完成（2026-09-10）

- 成果：[完整驗收報告](STAGE16_ACCEPTANCE_REPORT_2026-09-10.md)。main 每來源只處理一次；Before／After 專屬 CSV／sidecar／reader／audit、分段 TD／PSD／STFT、elapsed 與裁尾完成。
- 驗證：269 tests／0 skipped、160 Python 靜態／bundle／diff；217 項 evidence（18 表重讀／25 組 NPZ）、13 項目視 hash；五份真實與合成／分段案例數值最大誤差 0，十二張圖目視通過。
- 保留：四通道 float32 bandpass、500→200 Hz、400 點不重疊 TFLite；Before 全尾端／After 段內裁尾、max_samples 先濾完整段再截斷、raw／Before／After 各自索引。PSD 逐段顯示，無跨缺口串接或新增平均。
- 通道：顯示索引 0/1 比較真正來源 ch1/2；2/3 僅顯示來源 ch3/4 Before，After 明示不存在。STFT 顯示 ≤128 點時明示省略，未改方法。
- Guard：分段 main 已驗收可接受缺口，舊 `run_pipeline` 的 continuity guard 保留；不影響其他入口 guard。
- 專屬測試：`tests/test_jenqwei_pipeline_regression.py`、`tests/test_jenqwei_output_regression.py`；完整實跑 `tools/validate_jenqwei_stage16.py --out /path/to/new-directory`。
- 歷史：[起步報告](STAGE16_REPORT_2026-09-09.md) 已提交 `e8ebe18`；CLI／IO／繪圖／驗收後續已提交 `ef9eada`，代表圖補提交 `23aa2df`；未 push。

<a id="stage-17"></a>
## 17：Jenqwei 資料集 — 已完成（2026-09-10）

- 成果：[驗收報告](STAGE17_REPORT_2026-09-10.md)／[資料集契約](STAGE17_DATASET_CONTRACT.md)。先等分再裁窗口，逐來源段獨立處理；來源／原始與重採樣索引／窗口／裁尾與排除原因均可回讀。
- 盤點：訊號準備，不執行模型或 train/test split；專案內未找到其他下游讀取者。保留連續 CSV、檔名與舊欄位，另明示真實窗口數及實際採樣率。
- 驗證：291 tests／0 skipped、165 Python 靜態／bundle／diff；545 evidence checks（327 hash、202 表重讀、16 組 NPZ），另 17 項舊基準及 4 項持久目視 hash。
- 數值：五份真實資料兩模式共 170 CSV；連同合成、199.5 Hz 與真實訊號分段共 202 CSV，float32 值與 int64 微秒最大誤差 0。
- 邊界：22 項專屬測試，含窗口錨點、短段／padlen、污染、同名來源、上採樣、manifest 缺漏／錯序及來源／產物篡改；全排除仍失敗。三張診斷圖目視通過。
- Guard：入口已完成分段遷移與驗收，可接受缺口；不改其他入口 guard。`--allow-short-drop` 才允許排除；既有輸出不覆寫。
- 實跑：`python tools/validate_jenqwei_dataset_stage17.py --out /path/to/new-directory`；專屬測試 `tests/test_jenqwei_dataset_regression.py`。
- Git：第十六階段後半及第十七階段已提交 `ef9eada`；代表圖／manifest 補提交 `23aa2df`，未 push。下一工作單元為第十八階段品質 scorer。

<a id="stage-18"></a>
## 18：品質 scorer — 完成

- 目標：`lilia/quality.py` 與品質呼叫端／audit／圖形；[盤點與契約](STAGE18_QUALITY_INVENTORY.md)。
- 已實作：保留 legacy overall／detail，新增 valid／invalid reasons、component diagnostics、usable_overall、preset／stage／config context；主版與兩份 bundle 同步。
- 舊基準：`23aa2df` 的四 presets、五真實來源 raw／BP 118 窗口，加合成邊界共 130 輸入／520 評分／1,950 陣列；數值與 NaN 位置精確一致。
- 核心：[報告](STAGE18_CORE_REPORT_2026-09-10.md)，已提交 `a0a7ba6`，未 push。
- 前增量：[raw 呼叫端報告](STAGE18_RAW_CALLERS_REPORT_2026-09-11.md)。event qEEG／event markers／zoom／subject comparison 的逐窗診斷、CSV／audit／來源 reader 完成；event 品質圖明示無效／不可用及實際 preset。程式已提交 `40d2a23`，驗收 CSV／圖有漏提交。
- TFLite 增量：[報告](STAGE18_TFLITE_CALLERS_REPORT_2026-09-11.md)。baseline filtered／Before filtered_resampled／After model_output 三種 stage、子窗口 audit／來源 reader／品質圖完成；legacy sampler 亦保存診斷，短路及 API 相容語義保留。程式已提交 `40d2a23`，產物漏提交詳見 R3。
- 前增量驗證：當時 317 tests／0 skipped、176 Python 靜態／bundle；九案例共 107 qEEG 窗口／271 模型窗口／510 子窗口，9 NPZ 組共 399 陣列、18 表重讀，最大誤差與選取差異均 0，三圖目視通過。
- entropy：[entropy 診斷增量](STAGE18_ENTROPY_DIAGNOSTICS_REPORT_2026-09-15.md)：R2／R4 的 raw／filtered stage、component 診斷、來源 reader 與品質圖完成；327 tests／0 skipped、180 Python 靜態／bundle，35 案例、714 陣列精確相同、77 reader 檢查、五圖目視。R3 的 73 檔已由 `0dd9ff3` 補提交。
- R5：[Goertzel 增量](STAGE18_GOERTZEL_QUALITY_REPORT_2026-09-15.md)：filtered 品質診斷、raw／filtered artifact 來源、CSV／reader／品質標記與四面板缺口斷線完成；336 tests／0 skipped、354 陣列精確相同、36 reader 檢查、七圖目視。舊 rolling median 保留，分段平滑另待方法校準。
- 最新：[R6 quality_check 增量](STAGE18_QUALITY_CHECK_REPORT_2026-09-15.md)：NaN qmed 明確列異常；samples raw／filtered 與 anomalies raw 保存診斷、CSV／sidecar／來源 reader、圖形。345 tests／0 skipped、188 Python 靜態／bundle；20 案例及污染 helper、974 陣列精確相同、970 表格列／17 來源 reader、八圖目視。三個 NaN 異常原因／severity 有意修正，有限分數與接受政策不變。
- 相容政策：仍用 `legacy_overall`，診斷無效不暗改既有接受／排除結果；注入舊 scorer 明示 unavailable，無診斷的舊表仍可讀。原始 stage 與 BP／模型分析 branch 不混用。
- 收斂：[完整驗收報告](STAGE18_ACCEPTANCE_REPORT_2026-09-15.md)。舊 direct helper 的實際 caller `plot_custom_markers` 已保存 raw 診斷、CSV／sidecar／來源 reader 與品質圖；16 helper 案例及 5 plot 案例共 102 陣列精確相同，5 來源 reader／五圖目視。當前 348 tests／0 skipped、192 Python 靜態／bundle；R1–R6 舊產物及新 helper 本機來源綁定 1,364 項、可提交範圍 1,351 項證據核對通過。
- 相容與限制：舊雙值 API、外部 scorer 簽名、`legacy_overall` 門檻和缺口 qEEG mask 保留；無法重跑的外部 scorer 診斷明示 unavailable。正式來源僅留來源 hash 與本機核對，不納入 Git 副本。
- 完成條件：callers／readers／圖形與 preset／stage 可追蹤；真實舊新數值、短窗／污染／fallback／例外、接受排除、目視、整階段證據及完整檢查均已通過。Git 發佈狀態以分支遠端為準。

<a id="method-calibration"></a>
## 方法校準：進行中

- 範圍：[舊 profiles／比較契約](METHOD_CALIBRATION_PROFILES.md)；與等價重構分開，不改目前 CLI 預設、方法數值或品質接受政策。
- M0 已完成（2026-09-16）：baseline／PSD／Goertzel／MI 盤點，固定 Stage18 後 `aa0df5f4` 原碼與來源／模型／環境；五份真實＋三份合成案例共 1,255 陣列與選取 audit 精確一致。詳見 [首輪增量報告](METHOD_CALIBRATION_BASELINE_REPORT_2026-09-16.md)。這是核心 helper／模型比較基準，不代表所有入口或科學方法已驗收。
- M1 已完成（2026-09-16）：meditation／marker／legacy sampler／entropy-state 完整入口與多事件、非參與者、鍵盤標記／protocol 案例；19 案例、585 陣列與 metadata 精確一致、20 來源 reader。目視另修正 marker 窗口間缺口連線，3 案例／72 陣列重驗；357 tests／0 skipped。詳見 [M1 增量報告](METHOD_CALIBRATION_M1_REPORT_2026-09-16.md)。來源標註不等於觀測行為，缺 baseline／品質邊界／guard 政策保留。
- **M2 當前任務**：先補 P-jenqwei／P-quality-plot 的獨立舊新 PSD capture；沿用固定 `aa0df5f4` 原碼與來源／模型 hash，真實模型 Before／After、逐段與短段都保存頻率／線性 PSD／顯示值。
- M2 完成條件：先定窗長、頻帶上下界、單頻點積分、denominator 的比較量與容許差異，再做同輸入敏感度比較；1 秒／4 秒及各 legacy profile 不暗改。數值、NaN／索引／品質差異、reader 與必要圖形可追蹤；若引入新方法，需具名 profile 與明示相容策略。
- M2 起始測試：依 `lilia.jenqwei_plot`、`quality_check` 與共享 PSD 實作查相應回歸；先讀 profiles 的 PSD 表，按需讀 Stage16 與 Stage18 R6 證據。涉及科學判讀時另查一手方法文獻；收尾 `--full`。
- M3：Goertzel 分段平滑獨立比較，再處理 normalization／前端濾波與門檻單位；舊 > 門檻、不正規化 power 及 rolling 行為保留可重現版本。
- M4：MI histogram／KSG／null 分開校準；含合成控制、真實標註、bins／k／樣本數／seed／surrogates 敏感度；方法結論另查一手文獻。
- 重跑核心基準：`python tools/freeze_method_profiles.py --out /path/to/new-directory`；只寫新目錄。真實衍生產物在 ignored `local/`，合成與摘要可提交；完整來源版與 repository manifest 分開。
- 校準完成條件：具名 legacy／新 profile、明示預設與相容政策、預先定義的數值差異驗收、真實與合成／圖形／完整檢查；不得以 M0 精確相同替代方法有效性驗證。

## 入口遷移後的其餘待辦（尚未編號）

| 工作單元 | 接手範圍／完成依據 |
| --- | --- |
| 方法校準 | 進行中；M0／M1 完成，下一步 M2 PSD 基準與比較契約，見上方卡片。 |
| 架構與 F19 | 大入口拆分、移除計算對繪圖私有函式的依賴、明示 deprecated 參數的相容策略。 |
| 批次與產物 | 批次失敗報告、來源／設定追蹤、CSV／sidecar／圖形成套原子發佈。 |
| 整體驗收 | 所有入口、真實基準、必要方法差異、bundle、文件與發佈狀態；提交／推送依當時授權。 |
