# 第十六階段：Jenqwei CLI／輸出／分段圖形驗收完成

日期：2026-09-10。狀態：完成。範圍：Jenqwei main、專屬 IO／繪圖、測試與驗證工具。

## 本階段變更

- [main](../../analyze_jenqwei_pipeline.py) 已接 `process_segments`，同一來源只處理一次，各顯示通道共用結果；新增 `--csv` 指定來源，預設仍掃描 `jenqwei/`、`--channels 0 1`、`--max-sec 60`。有任一來源失敗時 CLI exit 1，保留每來源成功／失敗 audit。
- 保留前四通道 float32 bandpass、500→200 Hz polyphase、400 點不重疊 TFLite RMS 推論；Before 保留全部重採樣尾樣本，After 才逐段裁尾。`max_samples` 的完整來源段濾波後截斷語義及三種索引映射不變；未引入品質 scorer、baseline 或新模型。
- [專屬 IO](../../lilia/jenqwei_io.py) 新增兩張平面 CSV 與各自 sidecar：`sample_idx`／整數 `Time[us]`／`segment_id`／`raw_fractional_idx`，After 另有 `before_idx`。Before 為 ch1–4，After 為來源 ch1/2；這是新增輸出格式，不冒充裝置 raw CSV。
- `jenqwei_signal` reader 要求模型路徑，核對來源／模型／方法設定、完整 inference／filter context／裁尾、逐列原始與 Before／After 映射、列數／欄名／有限值及 table hash。reader 不重跑模型，數值等價另由凍結舊流程證據確認。
- audit 記錄 planned／completed inference、來源／模型／程式 hash、污染來源段、處理步驟、表重讀、繪圖範圍及成功／失敗產物。輸出碰撞明確失敗；已有 audit 時另存唯一檔名，不覆寫正式產物。
- [分段圖形](../../lilia/jenqwei_plot.py)：TD 分段連線且 Before／After 各用自己的時間；STFT 逐段、保留 Hann 256／overlap 128／zero padding 方法並裁於各自來源與輸出範圍。elapsed 從來源 epoch 起算；來源缺口留白、排除短段斜線、After 裁尾點紋明示。
- Welch PSD 對每來源段、每分支分別計算並畫出 S0/S1 等曲線，不串接缺口或新增平均；仍用 density、`nperseg=min(800, before段長, 分支段長)`、預設 Hann／半重疊／constant detrend。PSD 單位標籤修為 ADC²/Hz，未添加校正倍率。
- 明示相容差異：顯示 ch=2/3（0-based）保留來源 ch3/4 的 Before，After 面板說明沒有對應模型輸出，移除舊 clamp 到模型 ch2 的錯誤比較。`--max-sec` 用於 elapsed TD／STFT，PSD 仍用完整段；STFT 顯示樣本 ≤128 時明示省略，不改 overlap 或製造頻譜。
- 分段 main 經驗收可接受缺口；**舊 `run_pipeline` 的 continuity guard 仍保留**，不將 packed 分段結果交給舊計算。`analyze_csv` 保留單通道返回路徑／失敗返回 None 的包裝介面，重複寫同一來源輸出會明確碰撞失敗。

## 驗證

- [完整檢查](validation/stage16/final/checks.json)：**269 tests、0 skipped**；**160 Python** 編譯、Pyflakes、bundle、diff 通過。`python build_bundles.py` 新增兩個共享模組副本，後續生成為 0 updates；bundle 不提供 Jenqwei CLI 入口。
- [實跑腳本](../../tools/validate_jenqwei_stage16.py) 先核對既有七組處理基準／真實分段數值，再完成五份真實、合成連續、短顯示、真實訊號分段副本及小缺口 **9 個成功案例**；每案重讀 Before／After，成功 CLI 共產生 21 張 PNG。
- [數值摘要](validation/stage16/final/analysis.json)：五份真實 Before **11,994／12,260／12,391／12,925／12,260** 列，After **11,600／12,000／12,000／12,800／12,000** 列；模型／Before／PSD／STFT dB 最大誤差 **0**，所有整數微秒精確相同。
- 真實訊號分段 Before **1,403 × 4**／After **1,200 × 2**；兩保留段 After 裁尾 202／1 點，兩短段排除。大 epoch／6 ms 小缺口案例兩分支各 **800** 列；與凍結舊入口逐段推論／頻譜精確一致。
- STFT elapsed 座標最大差 **7.1055e-15 秒**，為浮點表示；NPZ 浮點 tolerance `rtol=atol=1e-6`、整數精確比較，專屬時間／裁切測試使用 `rtol=0, atol=1e-12`。 ch3/4 僅核對真正來源 Before，不把舊誤映射 After 視為正確基準。
- [Evidence manifest](validation/stage16/final/evidence.json)／[結果](validation/stage16/final/evidence_checks.json)：**217 checks = 174 hashes＋18 表重讀＋25 組 NPZ 比較**；[持久目視產物驗證](validation/stage16/final/visual_evidence_checks.json) 另 **13 hashes**，預期 hash 取自生成時紀錄。
- 全短段、污染短段、缺來源、錯誤顯示通道、非有限顯示時間、輸出碰撞 **6 個預期 CLI 失敗**均符合 exit／audit；單元測試另含模型第二段失敗、缺模型、寫圖失敗與關閉 figure、metadata／table 竄改、max_samples、timestamp jitter、每來源只處理一次及批次退出狀態。
- [目視](validation/stage16/final/visual_review.json)：直接檢視 **10 張正式 PNG＋2 張補充細節圖**，涵蓋所有五份真實來源、ch3 Before-only、After 裁尾、半秒顯示、省略 STFT、分段 Welch、6 ms 缺口與 20 ms 短前段。補充 renderer 只改 viewport 並交集來源 clip；沒有新增 CLI zoom 功能。
- 初次測試的 CSV 還原將 CRLF 轉成 LF，造成 hash 不符；改為還原原始 bytes。後續 timestamp jitter 測試以浮點精確相等遇到 **2.22e-16 秒**差異，改用上述既定時間容差；首次 `--full` 同項失敗，更正後完整通過。未變更訊號／頻譜數值或放寬整數時間檢查。[成功與失敗完整 log](validation/stage16/final/logs/) 均留檔，無未解驗收失敗。
- 大型產物在 `/tmp/lilia-stage16-final-v1/`；摘要、manifest、audit、完整 log 與十二張目視圖在 [final](validation/stage16/final/)，舊基準仍在 `tests/fixtures/jenqwei_*_reference.*`。暫存不保證永久存在。
- 可重現：`MPLCONFIGDIR=/tmp/lilia-mpl python tools/validate_jenqwei_stage16.py --out /tmp/lilia-jenqwei-final-new-run`，再 `python tools/refactor_check.py evidence /tmp/lilia-jenqwei-final-new-run/evidence.json`；階段檢查 `python tools/refactor_check.py check --full`。

## 未解問題與下一步

- 第十六階段無新增阻塞。成套原子發佈仍留待工程收尾：寫檔／繪圖失敗可能留下部分 CSV／sidecar／PNG，須依 failed audit 判讀，不能當完整成功組。過短 STFT 顯示明示省略；PSD 不因縮短顯示區間重算。
- 下一步：[任務 17：Jenqwei 資料集](TASKS.md#stage-17)，先盤點切片單位、模型窗與下游讀取者；本輪未開始第十七階段。
- Git：起步基準／分段處理已提交 `e8ebe18`；本次 CLI／IO／繪圖／完整驗收修改尚未提交，未 push，後續推送由使用者負責。起步歷史見 [2026-09-09 報告](STAGE16_REPORT_2026-09-09.md)。
