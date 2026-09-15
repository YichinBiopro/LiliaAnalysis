# 第十八階段增量：quality_check 診斷與 NaN 異常

日期：2026-09-15。狀態：R6 增量完成，第十八階段仍進行中。範圍：`quality_check.py` 的 samples／anomalies、CSV／sidecar／來源 reader、圖形。

## 本次變更

- [quality_check.py](../../quality_check.py) 的 samples 每個選取片段分別保存 raw／filtered 全通道舊 `overall`、scorer context、component 診斷；anomalies 保留 raw 10 秒窗口的 qmed／clip／flat／gap，另存 raw 診斷。兩路圖示明示實際 stage／preset，區分低分、診斷無效與 unavailable。
- 舊 `qmed=NaN` 因比較均不成立，原 `reasons=[]` 而漏掉 flagged。現在增加 `invalid-Q non-finite` 原因、有限 severity，並在圖上顯示；僅此類窗口的分類有意改變。有限 qmed 的門檻、jump、hard-artifact、severity、抽樣及 qEEG 數值政策保留；沒有改用 `usable_overall`。
- 兩路新增 CSV／`.meta.json`，宣告 `quality_diagnostics_version=1`、來源／表格 hash、設定與逐窗 audit。[quality_check_io.py](../../lilia/quality_check_io.py) 來源 reader 重建 float32 raw／filtered 片段或 raw 異常窗口，重算分數、診斷、特徵、時間及原因；重算 hash 後篡改亦拒絕。[refactor_check.py](../../tools/refactor_check.py) 白名單加入兩種 reader。
- 異常 raw 波形於來源時間缺口斷線，只修顯示連線；窗口與 gap 判定未改。NaN 圖圖例移開三個 baseline 異常標記。bundle 由 `python build_bundles.py` 生成並在完整檢查中核對。

## 驗證

- [完整檢查](validation/stage18/quality_check_r6_release_checks/summary.json)：345 tests／0 skipped、188 Python 編譯／Pyflakes、bundle／diff 通過。新增 9 項 [專屬測試](../../tests/test_quality_check_diagnostics_regression.py)，涵蓋 NaN 分類、raw／filtered 來源回讀、缺口／削波、fallback／外部 scorer、空／短／非有限、例外與重算 hash 後篡改。
- [數值摘要](validation/stage18/quality_check_r6_release/analysis.json)：凍結 `40d2a233473c3aced382f6eea7f1473842f38cf6` 舊入口並核對八項未改數值依賴 hash；五份完整 Jenqwei、James samples、Hardy anomalies（850 窗口），加合成邊界共 20 案例及污染 helper。
- 新舊 974 非空陣列精確相同，容許／最大誤差均 0，NaN 位置一致；兩份 NPZ 保存對照。970 個新表列、17 個來源 reader 回讀通過。只有強制 NaN scorer 的三個異常原因由空變 `invalid-Q non-finite`、severity 由 0.0002 變 1.0002；有限分數、原選取起點與其他原因／severity 相同。
- [產物驗證](validation/stage18/quality_check_r6_release_evidence/summary.json)：156 checks＝138 原始 hash＋17 新表來源回讀＋1 NPZ 組。[原 manifest](validation/stage18/quality_check_r6_release/manifest.json) 保存生成時 hash，凍結舊碼、CSV／sidecar 與 [完整實跑 log](validation/stage18/quality_check_r6_release/validation.log) 留檔。開發期 skipped 真實來源與 Welch 注入失敗的目錄是修正前證據，以此 release run 為準。
- Git 收斂另保存 [來源清單](validation/stage18/quality_check_r6_release/source_manifest.json)／[156 項核對](validation/stage18/quality_check_r6_source_evidence/summary.json)：五份 Jenqwei 驗證副本逐一確認與正式來源 byte-identical，改以 repo 相對正式來源路徑回讀，原預期 hash 未改。[可提交清單](validation/stage18/quality_check_r6_release/repository_manifest.json)／[142 項核對](validation/stage18/quality_check_r6_repository_evidence/summary.json) 保留 131 hash、10 個合成來源 reader、1 NPZ 組；正式錄製來源不進 Git。
- [目視紀錄](validation/stage18/quality_check_r6_release/visual_review.json)／[9 項 hash](validation/stage18/quality_check_r6_release_visual_checks/summary.json)：真實 Jenqwei／Hardy／James、缺口、有限 fallback、外部 scorer、NaN、空窗口八圖檢視通過；raw 缺口不連線，NaN 三點可見，stage／preset 與無效／不可用標記可辨。
- 實跑：`python tools/validate_quality_check_stage18.py --out /path/to/new-directory`。數值驗證與來源 reader 分開執行，目視不由自動檢查替代。

## 限制與下一步

- samples 保留舊兩段 30 秒選取，短來源不生成表；anomalies 保留舊窗口／缺口政策。歷史 component 例外只核對保存 fallback 與可重算部分，不能證明例外再次發生；注入舊 scorer 的診斷明示 unavailable。七個真實來源 reader 需本機正式錄製，乾淨 checkout 只能重核可提交清單的 142 項；真實來源副本只留本機驗證目錄。
- R1–R6 已完成；下一步為 [剩餘 direct helper 與整階段驗收](TASKS.md#stage-18)，尤其 event 舊 `compute_quality_windowed`。本輪 Git 包保存實作、最終證據及合成來源，忽略重複開發期 run 與正式錄製副本；正式來源、模型及歷史產物未覆寫。遠端狀態以分支紀錄為準。
