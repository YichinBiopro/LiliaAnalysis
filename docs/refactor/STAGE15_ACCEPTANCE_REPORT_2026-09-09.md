# 第十五階段：眼開閉分段繪圖驗收完成

日期：2026-09-09。狀態：完成。範圍：既有未提交的眼開閉繪圖／CLI 修改、專屬測試與本次新增驗收腳本。

## 本階段變更

- 驗收 [眼開閉入口](../../process_lilia_eye_open_close.py) 的分段 TD／STFT：逐來源段計算，TD 分開連線，STFT 保留舊 padding 數值、顯示裁至來源段範圍；elapsed 從來源 epoch 起算，短段斜線標記、缺口留白。
- ch5/6 比較圖 before 明確取八通道輸入的第 5/6 通道，after 取 packed output 的第 3/4 欄，標題使用真正來源 ch5/6；TD／STFT 面板時間軸等寬。
- 保留 float32 bandpass、500→200 Hz polyphase、兩組 PyTorch 推論與鏡射 Hann OLA、真實重採樣尾樣本；未引入 notch／bandstop、品質 scorer 或方法校準。
- CLI 已使用分段繪圖並記錄 plot timeline，現有移除 continuity guard 的修改經本次完整驗收確認；未改動正式資料、checkpoint 或歷史研究圖。本輪沒有再修改入口或使用者既有測試。
- 新增 [完整實跑腳本](../../tools/validate_eye_stage15.py)，保留舊 CLI／IO 增量腳本供歷史查閱。現有新增 [繪圖測試](../../tests/test_eye_plot_regression.py) 納入完整測試收集。

## 驗證

- [完整檢查](validation/stage15/final/checks.json)：**246 tests、0 skipped**；151 Python 編譯、Pyflakes、bundle、diff 全通過。先前 [專屬檢查](validation/stage15/final/focused_checks.json) 62 tests／0 skipped；最終完整檢查涵蓋新增驗收腳本後的程式指紋。
- `python build_bundles.py`：0 copies updated；完整檢查中的 bundle check 通過。既有 bundle 未提供眼開閉 CLI，本次未新增 bundle 入口。
- [模型與 CLI 數值](validation/stage15/final/analysis.json)：真實連續 **18,109 × 4**、真實訊號分段副本 **1,403 × 4**、合成連續 **602 × 4**、大 epoch／小缺口合成 **800 × 4**；模型／before 最大誤差 **0**，整數微秒精確一致。真實連續最後時間 90,540,000 μs。
- 舊基準來自 `96fee9d` 凍結入口、既有 `eye_*continuous_reference.npz` 與真實 checkpoint；腳本核對基準／來源／模型 hash，分段期望值由舊入口逐段獨立推論，未以新 adapter 自建預期值。大 epoch 案例以已知均勻整數格點核對時間，避免舊浮點 epoch 精度損失。
- 四組 before／after、來源 ch1/2/5/6 的 **STFT dB 最大誤差 0**；elapsed 座標最大差 **1.4211e-14 秒**，來自浮點表示。NPZ 浮點 tolerance `rtol=atol=1e-6`，專屬繪圖測試另以 `atol=1e-12` 驗證均勻時間中心及裁切邊界。
- [Evidence manifest](validation/stage15/final/evidence.json)／[結果](validation/stage15/final/evidence_checks.json)：**88 checks = 74 hashes＋5 表重讀＋9 組 NPZ 比較**；另有 [持久目視產物驗證](validation/stage15/final/visual_evidence_checks.json) **9 checks**，hash 取自原生成紀錄。
- 四個完整 CLI 成功案例皆產出 CSV／sidecar／四張 PNG、成功 audit；全短段、非有限及缺來源三個 CLI 案例預期 exit 1。專屬測試另涵蓋模型失敗、污染短段、缺模型、寫圖失敗、輸出碰撞、metadata 竄改與大整數 epoch。
- [目視紀錄](validation/stage15/final/visual_review.json)：直接檢視六張正式 CLI 圖及兩張補充細節圖；確認來源 ch5/6、elapsed、分段留白、短尾、6 ms 缺口與 20 ms 短前段。小於總覽像素寬度的區間由細節圖、來源 metadata 與渲染物件測試共同確認。
- 初次補充放大圖因來源 clip 未與縮小 viewport 取交集導致版面異常；只修正外部 detail renderer，重畫及目視通過。此失敗不涉及正式 CLI 圖，原失敗與修正腳本均 [留檔](validation/stage15/final/detail_initial_failure.json)，不計入通過產物。無未解驗收失敗。
- 完整 log、audit、摘要、manifest 與八張目視圖在 [final](validation/stage15/final/)；大型 CSV／NPZ 及全部 CLI 圖在 `/tmp/lilia-stage15-acceptance-v1/`，暫存不保證永久存在，舊連續基準仍持久保存在 fixtures。
- 可重現：`MPLCONFIGDIR=/tmp/lilia-mpl python tools/validate_eye_stage15.py --out /tmp/lilia-eye-final-new-run`，再 `python tools/refactor_check.py evidence /tmp/lilia-eye-final-new-run/evidence.json`。階段完整檢查：`python tools/refactor_check.py check --full --input 2026-07-03-lilia-eye-open-close.csv`。

## 未解問題與下一步

- 第十五階段無新增阻塞；成套原子發佈仍屬後續工程工作，寫圖失敗可能留下已驗證 CSV／sidecar，須依 failed audit 判讀。補充縮放是驗收視圖，不是新增 CLI 功能。
- 下一步：[任務 16：Jenqwei 分析](TASKS.md#stage-16)，先盤點 callers、真實模型及連續基準；本次未開始第十六階段實作。
- Git：前次基準／CLI／IO 已提交 `b84029e`；本次繪圖、測試、驗收工具與文件仍未提交，未 push，後續推送由使用者負責。
