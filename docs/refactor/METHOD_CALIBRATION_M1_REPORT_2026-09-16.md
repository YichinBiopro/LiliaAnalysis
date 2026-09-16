# 方法校準 M1：baseline 完整入口與事件基準

日期：2026-09-16。狀態：完成本輪入口基準；尚未改變或完成科學方法校準。

## 本階段變更

- 新增 [獨立原碼／工作區實跑工具](../../tools/freeze_baseline_entries.py)：B-meditation、B-marker、B-legacy-sampler、B-entropy-state，以及多事件／非參與者與完整 TFLite summary；兩個獨立 process 保存窗口／索引、選取、reference、delta、模型輸入輸出與完整精度 entropy／PSD／U／p。
- 真實來源涵蓋完整 TYY（2,009,328 列）與 Jenqwei talk／move-head；後兩者只在獨立四通道副本加 header offset。鍵盤 trigger 保留絕對微秒；talk 絕對欄與 offset＋相對欄差 −1 µs，move-head 為 0，差異明記 config。
- TYY 事件／meditation 時段採既有 protocol 設定；鍵盤 trigger 與 protocol 都不能當作已確認行為標註。合成事件亦只作驗證探針。
- 目視發現 custom marker 在兩個完整窗口間仍跨缺口連線；[繪圖修正](../../plot_index_vs_raw.py) 改按來源段插 NaN，保留 raw 端點與孤立有效點。CSV、baseline、品質分數／接受政策不變；bundle 已由 `python build_bundles.py` 生成。
- 新增 harness 失敗偵測與政策 witness 測試；繪圖回歸含連續、8 ms／10 秒缺口，確認不丟有效點且不改 CSV 接受列。

## 驗證

- [完整檢查](validation/method_calibration/m1_final_checks_2026-09-16/summary.json)：357 tests、0 skipped；197 Python 編譯、Pyflakes、bundle、diff 全通過。修正前 356 tests 的紀錄保留為歷史。
- [完整基準](validation/method_calibration/m1_final_2026-09-16/analysis.json)：固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed`；19 案例（7 真實入口＋12 合成）、585 陣列，dtype／shape／值／NaN 位置與 metadata 精確一致，rtol=atol=0，最大絕對誤差 0；20 個來源 reader 通過。
- [繪圖修正重驗](validation/method_calibration/m1_marker_gap_2026-09-16/analysis.json)：上述 marker 子集 3 案例、72 陣列、3 reader 再通過，不將重跑子集加算成新案例。
- 缺 baseline meditation、低品質 sampler、sampler continuity guard、無完整 state 窗口共 4 個預期失敗案例，均符合固定版狀態／錯誤；非未處理失敗。另驗品質 =.5、中心半開邊界、錄製前 fallback、錄製外 marker、非參與事件更新 rest 起點，以及 bar 有效但 heatmap baseline bins 不足。
- 歷史基準 evidence：[本機 576 項](validation/method_calibration/m1_preserved_local_2026-09-16/summary.json)／[repository 314 項](validation/method_calibration/m1_preserved_repository_2026-09-16/summary.json)；原產物、NPZ 及預期 hash 原封保留，只將舊繪圖程式路徑綁定相同 bytes 快照。原 manifest 不覆寫，不冒充當前程式驗證；詳見 [scope](validation/method_calibration/m1_acceptance_2026-09-16/scope.json)。
- 修正後 evidence：[本機 281 項](validation/method_calibration/m1_marker_evidence_local_2026-09-16/summary.json)／[repository 114 項](validation/method_calibration/m1_marker_evidence_repository_2026-09-16/summary.json)；含當前程式與真實兩例／合成缺口產物。原始資料、模型均未覆寫。
- [目視 15 圖](validation/method_calibration/m1_acceptance_2026-09-16/visual_review.json)：12 張原基準圖＋3 張修正圖；原 marker 缺口缺陷明記為已被修正結果取代。檢查 baseline／事件位置、缺口、無 baseline、品質 stage／低品質及孤立點；目視 hash [本機 17](validation/method_calibration/m1_visual_local_2026-09-16/summary.json)／[repository 11](validation/method_calibration/m1_visual_repository_2026-09-16/summary.json) 項通過。

## 未解問題與下一步

- 無新增阻塞。品質 .5／.49 注入案例明示 diagnostics unavailable；legacy sampler 缺口 guard 保留。全精度舊新相同證明相容性，不代表 baseline 政策或事件效應已獲科學驗證。
- 真實 NPZ／圖／凍結原碼留 ignored `local/`；合成與摘要保留可提交證據。完整 log 在各 run 目錄，歷史證據範圍與當前來源分開記錄。
- 下一步：[M2 任務卡](TASKS.md#method-calibration)，先補 P-jenqwei／P-quality-plot PSD capture，再定窗長、頻帶邊界、積分與 denominator 的比較契約；保留 legacy profile。
- Git：工作區保留本輪及前輪修改，尚未提交／推送。
