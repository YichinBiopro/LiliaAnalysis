# 第十八階段增量：entropy 品質診斷與來源回讀

日期：2026-09-15。狀態：本增量完成，第十八階段仍進行中。範圍：R2 剩餘診斷一致性、R4 entropy／joint-MI／state 與圖形；另完成 R3 補提交。

## 本次變更

- [spectral_entropy.py](../../spectral_entropy.py) 保存逐窗 scorer context／component valid／reasons；一般 entropy 評全部來源通道、joint-MI 評指定兩通道，clean state 評全部來源通道。`--no-bandpass` 明示 raw，其餘 filtered；denoised MI 仍關閉品質評分並標 model_output。
- 維持 `legacy_overall`、channel median、原門檻與 precheck 順序；有限 fallback 的診斷無效不改原接受／排除。舊注入 scorer 保留原簽名並明示 unavailable，普通窗口 scorer 例外仍向外傳遞，clean state 仍保存 quality_error。
- [entropy_quality.py](../../lilia/entropy_quality.py) 與各 IO 新表保存 `quality_diagnostics_version=1`、四個窗口 CSV 診斷欄位及 sidecar audit；state summary 保存診斷計數、stage、policy，完整逐窗診斷在 states audit。
- 來源 reader 重建原始分段與 float32 濾波 context，再核對 stage、通道、窗口、分數、component 診斷；clean 另重核 raw 削波／非有限／filter precheck。重新計算 hash 後的偽造欄位仍會被拒絕；沒有整組診斷的舊表仍可讀。
- 新增配套 `*_quality.png`，分別標示低分、診斷無效與 unavailable；state baseline／event 各有品質圖。MI 全排除保留原時間範圍並顯示空結果原因；主 population 圖 footer 使用實際 fs／win／step。
- bundle 由 `python build_bundles.py` 生成；未修改濾波、PSD、MI、surrogate、原始／模型索引或既有數值方法。
- R3 的 73 份原產物已補提交 `0dd9ff3`，詳見 [恢復報告](STAGE18_ARTIFACT_RESTORE_REPORT_2026-09-15.md)；[逐 Git blob 核對](validation/stage18/entropy_diagnostics/r3_commit_verification.json) 確認提交內容仍符合原 hash。

## 驗證

- [完整檢查](validation/stage18/entropy_diagnostics_checks/summary.json)：327 tests／0 skipped；180 Python 編譯／Pyflakes、bundle、diff 全通過。完整 log 同目錄；先前 57 項 caller／audit 與新增 6 項反例測試紀錄保存在實跑目錄。
- 新測試涵蓋有限 spectrum fallback、raw／filtered／disabled／external scorer、重新計算 hash 後的 stage／component／score／欄位矛盾、state 削波及 scorer 例外、半秒污染與全排除時間軸。
- [數值摘要](validation/stage18/entropy_diagnostics/analysis.json)：固定 `40d2a233473c3aced382f6eea7f1473842f38cf6` 入口及五項未變更數值依賴。五份完整 Jenqwei 八通道來源各跑 joint-MI／entropy／普通 state／clean state 新舊對照。
- 加上缺口、非有限拒絕、全排除、disabled 及 joint／entropy／clean 的 raw、半秒、fallback 共 35 案例；714 組陣列精確相同，最大及容許誤差均 0，NaN、窗口選取與原 state audit 相同。只允許新增 `quality_diagnostics`，未忽略既有 audit 欄位差異。
- [產物驗證](validation/stage18/entropy_diagnostics_evidence/summary.json)：356 checks＝289 hash＋66 來源表重讀＋1 NPZ 組（714 陣列）；實跑另讀 11 新 summary，共 77 次 reader 檢查。
- [目視紀錄](validation/stage18/entropy_diagnostics/visual_review.json)／[6 項目視 hash](validation/stage18/entropy_diagnostics_visual_checks/summary.json)：MI、entropy、clean fallback 品質圖、全排除 MI 圖及真實 population 圖共五張；分數與診斷分離、時間範圍、空結果提示及 footer 均符合預期。
- 實跑：`python tools/validate_entropy_quality_state_stage18.py --diagnostics --out /path/to/new-directory`；[manifest](validation/stage18/entropy_diagnostics/manifest.json)、凍結舊入口、NPZ、CSV／sidecar、圖與完整 log 均留檔，新增產物已設局部 ignore 例外。

## 限制與下一步

- reader 的歷史例外只能驗證 context、fallback 與其餘可重算 component，不能證明例外必然再次發生；外部無診斷 scorer 不冒充可重算結果。模型計算未變，未重跑真實神經模型；完整測試保留 neural timeline／disabled 診斷路徑。
- manifest 的真實錄製仍依賴本機正式來源，合成缺口／污染輸入在證據目錄；非有限 CSV 保留入口拒絕，污染窗口另以 helper 回歸覆蓋。
- R1–R4 已完成；下一步為 R5 Goertzel、R6 quality_check 與剩餘 direct helper，見 [TASKS](TASKS.md#stage-18)。第十八階段尚未完成。
- R3 補提交完成，未 push；先前 R1／R2 與本輪 R4 的程式、測試及證據仍未 commit。保留其可驗收工作區，不把產物提交誤稱為程式已提交。
