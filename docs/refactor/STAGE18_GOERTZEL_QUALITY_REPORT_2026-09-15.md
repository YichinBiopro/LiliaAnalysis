# 第十八階段增量：Goertzel 品質診斷與來源回讀

日期：2026-09-15。狀態：R5 增量完成，第十八階段仍進行中。範圍：Goertzel caller／CSV／sidecar／reader／圖形。

## 本次變更

- [plot_goertzel_vs_raw.py](../../plot_goertzel_vs_raw.py) 保存全部 filtered 通道的 scorer context／component valid／reasons，仍取指定通道的 `legacy_overall`；不採用 `usable_overall`，保留 `fs=int(fs)`、原門檻、Goertzel／hard-artifact／平滑數值及 scorer 例外向外傳遞。
- CSV 新增四個診斷欄位；sidecar 宣告 `kind=goertzel`、`quality_diagnostics_version=1`、逐窗 audit 與 feature sources。sat／PTP／max diff 為 raw，edge shift 及 Goertzel power 為 filtered；`quality_final` 仍在 hard artifact 時歸零。
- [goertzel_io.py](../../lilia/goertzel_io.py) 重建 float32 分段濾波、原始窗口與整數微秒，重算 Goertzel、raw／filtered artifact 指標與品質診斷；驗證分數、指定通道、hard 判定及新欄位。重算 hash 後的偽造仍拒絕；缺少整組診斷的完整舊表仍可讀。
- cache 重繪先保留既有來源／設定／程式指紋檢查，再使用來源 reader，並載入原 audit 畫圖。此路徑會重算來源特徵及診斷供驗證，不再宣稱跳過全部昂貴計算；歷史 fallback 分數仍取保存值。
- 品質圖分別標示低分、診斷無效、unavailable 與 hard artifact；標題明示 preset、filtered 與全通道診斷範圍。四面板於來源缺口斷線，孤立保留窗口可見；空窗口有提示。分段只改顯示連線，未改 rolling median 方法。
- [provenance.py](../../lilia/provenance.py) 保留舊 writer 呼叫簽名相容；[驗證工具](../../tools/refactor_check.py) 新增 `goertzel` reader。bundle 已由 `python build_bundles.py` 生成。

## 驗證

- [完整檢查](validation/stage18/goertzel_quality_checks/summary.json)：336 tests／0 skipped、184 Python 編譯／Pyflakes、bundle／diff 全通過。新增 9 項專屬測試，含 11 類重算 hash 後的篡改、舊注入 scorer、fallback、半秒、污染、空窗口、例外與四面板斷線。
- [數值摘要](validation/stage18/goertzel_quality_final/analysis.json)：凍結 `40d2a233473c3aced382f6eea7f1473842f38cf6` 舊入口，核對六項未改數值依賴 hash；五份完整真實來源各跑 ch1／ch2，加缺口、raw step、半秒、fallback、external、全排除、空窗口、499.5 Hz、非有限拒絕共 19 案例，另有污染 helper。
- 新舊共 288 表格窗口，354 陣列精確一致、最大／容許誤差均 0，NaN 位置及兩種 hard 政策／兩個門檻的選取一致。335 非空陣列存入 NPZ；19 空陣列在實跑核對 shape／欄位，不放寬 evidence 工具的非空要求。
- [產物驗證](validation/stage18/goertzel_quality_evidence/summary.json)：203 checks＝184 hash＋18 新表來源回讀＋1 NPZ 組；實跑另回讀 18 舊表，共 36 reader checks。[manifest](validation/stage18/goertzel_quality_final/manifest.json) 保存原 hash，凍結舊碼、CSV／sidecar、來源副本及 logs 均留檔。
- [目視紀錄](validation/stage18/goertzel_quality_final/visual_review.json)／[8 項 hash](validation/stage18/goertzel_quality_visual_checks/summary.json)：真實 talk、缺口、半秒、有限 fallback、external、空窗口與 raw step 七圖通過；hard 圖由保存 CSV 回讀生成，參數與原 hash 見 [生成紀錄](validation/stage18/goertzel_quality_final/hard_plot_generation.json)。
- 開發期空表 dtype 失敗已修，原 [測試 logs](validation/stage18/goertzel_quality_development/initial_tests/summary.json) 留存；[修正前實跑](validation/stage18/goertzel_quality/README.md) 數值通過但缺口圖仍連線，不當作最終目視證據。
- 實跑：`python tools/validate_goertzel_quality_stage18.py --out /path/to/new-directory`；專屬測試：[test_goertzel_quality_regression.py](../../tests/test_goertzel_quality_regression.py)。

## 限制與下一步

- 歷史 component 例外只驗證 fallback 與可重算部分，不證明當時例外必然重現；外部舊 scorer 明示 unavailable。來源 reader 的 source hash／重算依賴本機正式來源，缺口／污染副本已持久保存。
- 舊 rolling median 仍可能取到缺口另一側的窗口；本次只修圖形連線。平滑分段、Goertzel 門檻／normalization 留待方法校準，未暗改研究結果。抽樣／distribution helper 與其他入口的 guard 不變。
- R1–R5 已完成，下一步為 [R6 quality_check 與剩餘 direct helper](TASKS.md#stage-18)：NaN anomaly、raw／BP 診斷及可回讀輸出。第十八階段尚未完成。
- 本輪與既有工作區修改均未 commit／push；保留先前程式、資料、模型及歷史研究產物。
