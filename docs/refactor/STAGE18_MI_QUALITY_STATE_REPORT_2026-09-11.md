# 第十八階段增量：MI population／series 與品質狀態回讀

日期：2026-09-11。狀態：本增量完成，第十八階段仍未完成。範圍：驗收 R1 與 R2 的狀態／遮罩／audit 一致性；scorer 診斷仍待 R4。

## 本次變更

- R1：population 維持未遮罩方法，`population_quality_state=disabled`、`population_quality_masked=false`；另存 series 評分狀態、總窗口與品質通過窗口數。全部 series 被排除也不改 population／surrogate。
- 新增 summary sidecar 與 `load_joint_mi_summary`，綁定 series CSV／sidecar hash、來源及 population／series scope；支援 denoised series 的來源／模型驗證。既有 summary CSV 欄位保留，新 reader 需要新 sidecar。
- R2：entropy／joint-MI reader 核對 enabled、state、分數與門檻遮罩，排除列不可保留有限指標；MI 同時核對 signal_valid 與兩個 MI 指標。denoised 品質維持 disabled。
- state reader 核對 clean 設定、逐窗狀態／分數／門檻／削波前置檢查、audit 計數與 summary 狀態。檢查一致性，不重算 scorer／PSD 或模型。
- 新輸出宣告 `quality_state_version=1`；已有狀態欄位的舊表同樣檢查，無整組狀態且未宣告版本的舊 entropy 表保留相容。CSV 採 round-trip 浮點解析，避免門檻相鄰值回讀後改變遮罩判定。
- [入口](../../spectral_entropy.py)、[共用 IO](../../lilia/entropy_io.py)、[state IO](../../lilia/state_entropy_io.py)、[neural IO](../../lilia/neural_io.py) 與生成 bundle 同步；未改濾波、PSD、MI、surrogate、品質門檻或索引方法。

## 驗證

- [完整檢查](validation/stage18/mi_quality_state_checks/summary.json)：321 tests／0 skipped，177 Python 編譯／Pyflakes、bundle、diff 通過；完整 log 同目錄保存。
- 反例涵蓋重新計算 hash 後的狀態／遮罩／有限指標矛盾、summary scope／父檔案變動、state audit／削波矛盾、門檻相鄰浮點值、全排除 population 及 denoised summary 回讀。
- [數值摘要](validation/stage18/mi_quality_state/analysis.json)：固定 `40d2a233473c3aced382f6eea7f1473842f38cf6` 入口並核對五個數值依賴 hash；五份完整 Jenqwei 原始八通道來源，joint pair 1/2、entropy ch1＋同步指標、普通／clean state 各跑新舊版本。
- 加入有限缺口、非有限輸入拒絕、全排除、disabled 共 26 案例；513 陣列、NaN 位置、窗口選取與 state audit 精確相同，容許及最大誤差均 0。非有限 CSV 仍由既有 loader 拒絕，不冒充可分析污染資料；窗口內污染另由回歸測試覆蓋。
- [產物驗證](validation/stage18/mi_quality_state/evidence_checks.json)：250 checks＝201 hash＋48 來源表重讀＋1 NPZ 組（513 陣列）。實跑另驗證 8 份新 summary，共 56 次 reader 檢查；summary reader 未加入共用 evidence 白名單。
- [實跑工具](../../tools/validate_entropy_quality_state_stage18.py)：`python tools/validate_entropy_quality_state_stage18.py --out /path/to/new-directory`；[manifest](validation/stage18/mi_quality_state/manifest.json)、固定舊入口、NPZ、CSV／sidecar、產物及完整 log 留檔。正式來源保持唯讀，manifest 仍依賴本機正式來源路徑。
- [目視紀錄](validation/stage18/mi_quality_state/visual_review.json)：檢視真實 talk 的 population／series 與強制全排除 series 三圖；population unmasked 標題及排除後無曲線符合 R1。既有全排除時間軸與提示、distribution footer step 顯示問題留待 R4，未宣稱完整圖形驗收通過。
- 開發期實跑腳本最初誤把非有限 CSV 當可分析輸入，觸發既有拒絕；已修驗證案例並保留 [development.log](validation/stage18/mi_quality_state/development.log)，最終 26 案例通過。未重跑模型推論；模型計算未改，本輪完整測試含 neural timeline 回歸。

## 未解問題與下一步

- R1 可關閉；R2 狀態／遮罩一致性完成，stage／component diagnostics 仍與 R4 一起待補，不能視為第十八階段契約全數完成。
- 接著補 R3 漏提交的 73 個產物，再做 R4 entropy 診斷／圖形、R5 Goertzel、R6 quality_check 與剩餘 helper；任務見 [TASKS](TASKS.md#stage-18)。
- 本輪產物目錄加入局部 ignore 例外，避免新 CSV／PNG 再漏列；本輪程式、測試、文件與證據尚未 commit／push，HEAD 仍為 `40d2a23`。
