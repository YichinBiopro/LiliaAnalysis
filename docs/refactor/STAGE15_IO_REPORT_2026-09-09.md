# 第十五階段：main 接入與可重讀輸出

日期：2026-09-09。狀態：本次 CLI／IO 工作完成；第十五階段仍未完成，繪圖遷移及完整驗收待續。範圍：眼開閉 main、專屬 IO、回歸及驗證工具。

## 本階段變更

- [main](../../process_lilia_eye_open_close.py) 改用既有 `process_segments`；CSV 直接保存 timeline 整數微秒，保留四行裝置 header 與 ch1/2/5/6 欄位。原 float32 bandpass／兩組模型／鏡射 OLA／全部真實重採樣尾樣本不變。
- [eye_io](../../lilia/eye_io.py) 新增 `eye_model_signal` CSV sidecar 與 reader：核對 raw／checkpoint／目前架構／固定方法設定、完整 inference windows／來源段、逐列 output／segment／raw fractional index、CSV header／timestamps／有限輸出及 table hash。quality 明示 disabled；不引入 notch／bandstop。
- reader 需要 `model_path`，目前仍限定眼開閉既有固定 checkpoint 與架構；重讀只驗證來源契約及產物完整性，不重新推論，數值等價另由舊基準與獨立舊流程驗證。
- main 保存成功／失敗 audit：處理階段、planned／completed inference、來源與模型／程式指紋、逐段非有限通道計數、例外原因、已完成產物 hash。模型錯誤保留來源段與輸入通道組。缺口仍在 CLI guard 明確拒絕；loader 本身只讀取資料。
- 輸出碰撞明確失敗並要求新輸出目錄；已有 audit 時另存唯一失敗 audit，不覆寫舊 CSV／sidecar／研究圖。此增量未實作成套原子發布；寫檔或繪圖中途失敗仍可能留下部分產物，以 failed audit／table_verified 判讀。
- `refactor_check evidence` 新增專屬 reader；[實跑腳本](../../tools/validate_eye_io_stage15.py) 使用新目錄保存真實 CLI、分段 IO 對照與失敗案例。bundle 由 build 工具生成，新增 1 個 eye_io 副本；既有 bundle 未包含眼開閉 CLI。

## 驗證

- [測試](validation/stage15/cli_io/checks.json)：**53 tests、0 skipped**，含 16 項眼開閉測試（本次新增 8 項），以及 neural／signal／CSV／pipeline／驗證工具測試。[靜態](validation/stage15/cli_io/static.json)：149 Python 編譯、Pyflakes、bundle、diff 通過；本次未跑階段 `--full`。
- [數值與 CLI 結果](validation/stage15/cli_io/analysis.json)：完整真實連續輸出 **18,109 × 4**，最後 **90,540,000 μs**；CSV 重讀後與凍結舊真實模型 output／time 最大誤差 **0**。模型浮點 tolerance 為 `rtol=atol=1e-6`，整數時間精確比較。
- 真實錄製前 4,511 筆另建測試副本，分為 10／1,503／2,001／997 raw 樣本四段；兩短段排除，兩保留段共 **1,403 × 4** 輸出。逐段執行凍結舊程式及真實模型，before／模型值／時間與新 adapter／CSV 最大誤差 **0**；這是分段 IO 證據，未解除 gapped CLI guard。
- [Evidence manifest](validation/stage15/cli_io/evidence.json)／[結果](validation/stage15/cli_io/evidence_checks.json)：**38 checks = 34 hashes＋2 表重讀＋2 組 NPZ 比較**。真實 CLI exit 0；缺口／全短／非有限／缺來源四案例皆 exit 1，保存 failed audit。單元測試另涵蓋大整數 epoch、metadata／時間／header／模型竄改、模型第二組失敗、輸出碰撞。
- [目視紀錄](validation/stage15/cli_io/visual_review.json)：檢視本次連續 ch1/2 TD／STFT 及 ch5/6 比較圖。後者仍誤標 ch3/4 並使用錯誤 before 通道，明示 known failure，不能當作有效通道比較。尚未完成分段圖形或完整階段目視驗收。
- 初次命令誤填不存在的測試檔而中止；更正後發現舊 header 測試替身未接受 `window`／`hop` 且未建立圖形 hash 所需檔案。已修正替身，保留原 CSV 斷言，最後 53 項全通過。[完整成功與失敗 log](validation/stage15/cli_io/logs) 均留檔。
- 可重現：`MPLCONFIGDIR=/tmp/lilia-mpl python tools/validate_eye_io_stage15.py --out /tmp/lilia-eye-io-new-run`，再執行 `python tools/refactor_check.py evidence /tmp/lilia-eye-io-new-run/evidence.json`。大型產物目前在 `/tmp/lilia-stage15-cli-io-v1/`；摘要、manifest、log 與兩張目視圖持久保存，tmp 路徑不保證永久存在。

## 未解問題與下一步

- 接續[任務 15](TASKS.md#stage-15)：分段 TD／STFT、elapsed 軸、ch5/6 真正來源映射及標題；補分段圖形／CLI 與失敗驗收，完成真實／合成目視及 `--full` 後才解除 guard。
- 本次實跑成功只代表 guarded CLI／IO 完成，不代表眼開閉整階段或舊圖正確。基準起步歷史另見[原報告](STAGE15_REPORT_2026-09-09.md)。保留所有既有修改；本次未 commit／push，後續 push 由使用者負責。
