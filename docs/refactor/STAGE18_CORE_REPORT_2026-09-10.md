# 第十八階段：品質 scorer 核心診斷增量

日期：2026-09-10。狀態：核心增量完成；整階段進行中。範圍：`lilia/quality.py`、生成 copies、舊基準／測試／驗證工具。

## 本階段變更

- 先完成前階段提交：`ef9eada`（第十六階段後半＋第十七階段）、`23aa2df`（被 ignore 排除的代表圖與 manifest）；未 push。
- [盤點](STAGE18_QUALITY_INVENTORY.md) 確認短窗／fit bins 不足／Welch、kurtosis、correlation 例外可能回傳有限 fallback；舊分數本身無法表示是否成功計算。
- 保留 `overall`／`detail`、舊 flat exclusive-stop loop、PSD／權重／門檻與 fallback 數值；新增逐通道／component 有效性、明確原因、`usable_overall` 及完整 preset／stage／config context。
- `valid` 表示可計算性，不代表通過品質門檻；有效低分保留。非有限污染及 correlation 對其他通道的影響明示；停用的 component 不計算／不額外使結果無效。
- `stage` 未提供時為 `unspecified`，不猜 raw／BP；不合法 fs、零通道、空 stage 拒絕，設定 context 使用有限數值。參數校準與 deprecated 參數語義不變。
- 既有 callers 仍讀原 `overall`；本增量不自動採用 NaN masking，也未宣稱完整 audit／圖形遷移完成。

## 驗證

- [完整檢查](validation/stage18/core/checks.json)：301 tests／0 skipped、168 Python 編譯／Pyflakes／bundle／diff 通過；新增 10 項專屬測試，完整 log 同目錄。
- 舊基準取自 `23aa2df:lilia/quality.py`；保存舊碼與 hash、真實來源 hash、input NPZ／reference NPZ／四 presets 及每窗口 stage。
- [實跑摘要](validation/stage18/core/analysis.json)：五真實錄製 raw／完整 BP 後的五秒窗口共 118 筆，加空窗／短窗／常數／非有限／單通道共 130 輸入；四 presets 共 520 次評分。
- [持久 evidence](validation/stage18/core/evidence_checks.json)：20 checks，1,950 陣列 `rtol=atol=0` 比較；最大誤差 0，NaN 位置精確一致。診斷記錄 1,973 個有效及 95 個無效通道評分，不將無效誤算成品質合格。
- 專屬測試另強制注入 spectrum／kurtosis／corr 例外，核對保留 fallback 值與失敗原因；驗證 99／100／101 點、污染 peer effects、停用 components 及 preset／stage config identity。
- [目視](validation/stage18/core/visual_review.json)：200 Hz、flat-only 的 100／101 點邊界、低分與無效標記清楚；診斷圖數值沿用舊 loop，沒有聲稱修正其窗口公式。
- 已解檢查失敗：兩次全套在舊測試直接 `pandas.read_csv` 時 SIGSEGV；單行 `0e555…` SHA-256 可重現 parser numeric-inference 崩潰。正式 reader 原已指定 ID 字串；測試改用正式 `load_tflite_table` 並核對來源，9 項專屬及完整 301 項通過。[原因紀錄](validation/stage18/core/native_failure_resolution.json)／[原 trace](validation/stage18/core/native_failure_trace.log)。

## 未解問題與下一步

- 無未解核心驗收失敗；**第十八階段尚未完成**。下一增量須將 stage／diagnostics 傳至各 caller 的 audit／reader／圖形，明示是否採用 usable_overall 並核對接受／排除窗口差異。
- 真實／合成舊基準、當前比較 NPZ、完整 JSON／log／代表圖持久留存；驗證腳本使用新目錄，不覆寫正式錄製或模型。
- 本輪核心診斷與測試修正尚未 commit；未 push。[當前任務](TASKS.md#stage-18)。
