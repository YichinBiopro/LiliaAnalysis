# 第十五階段：眼開閉基準與分段處理起步

日期：2026-09-09。狀態：未完成，已完成基準與處理函式；CLI 尚未遷移。範圍：眼開閉入口、專屬測試與基準工具。

## 本階段變更

- 先完成第十四階段提交 `96fee9d`，包含報告引用的三張圖；後續 push 由使用者負責。本輪第十五階段修改未提交。
- 由提交 `96fee9d` 凍結舊入口，保存[原程式快照](validation/stage15/baseline_legacy_entry.py.txt)、共享程式／checkpoint／架構 hash、環境、原始輸入、bandpass、重採樣、兩組模型輸出與各通道 STFT。
- [完整真實基準](../../tests/fixtures/eye_real_continuous_reference.json)：`2026-07-03-lilia-eye-open-close.csv`，45,272 × 8、單一連續段，500→200 Hz 後 18,109 × 4，最後時間 90,540,000 μs。
- [合成基準](../../tests/fixtures/eye_synthetic_continuous_reference.json)：seed 15015、1,503 × 8，602 × 4 輸出，非零 epoch、非整除尾端；同名 NPZ 保存完整數值。
- [process_segments](../../process_lilia_eye_open_close.py) 新增獨立可測處理函式：逐來源段 bandpass／polyphase／兩組四通道推論，回傳 InferenceTimeline、八通道模型輸入及 ch1/2/5/6 輸出；未接入 main，CLI continuity guard 保留。
- 保留 float32 bandpass、RMS／Hann overlap-add、鏡射邊界與全部真實重採樣尾樣本；不能套用 denoise_with_time 的 notch／bandstop。短段記錄排除，非有限（含短段污染）整次失敗；模型錯誤標示段及輸入通道組，不回傳部分成功結果。
- 盤點發現既有 ch5/6 圖標題誤標 ch3/4；比較圖 before 也誤取原始 ch3/4。尚未修正。基準 STFT 明確依真正來源 ch1/2/5/6 保存，不能將舊錯誤圖當映射契約。

## 驗證

- [專屬及共享測試](validation/stage15/checks.json)：**36 tests、0 skipped**，含 8 項新入口測試；真實／合成模型、八通道映射、跨段污染隔離、短段、998 raw→400 model 邊界、小缺口、整數 epoch、裁尾與失敗案例通過。
- [靜態檢查](validation/stage15/static.json)：146 Python 編譯、Pyflakes、bundle、diff 通過。`python build_bundles.py` 為 0 copies updated；本輪沒有共享模組或 bundle 變更。階段未完成，未跑 `--full`。
- [數值與時間對照](validation/stage15/adapter_analysis.json)：兩組 before／模型輸出／四通道前後 STFT 最大絕對誤差皆 **0**；整數微秒精確一致。模型與 STFT 比較容許 `rtol=atol=1e-6`，before 另在測試精確比較。
- [基準 evidence](validation/stage15/baseline_evidence_checks.json) 5 hashes；[adapter evidence](validation/stage15/adapter_evidence_checks.json) 8 hashes＋2 組 NPZ 比較。預期值來自修改前凍結基準，未以新函式重新建立預期數值。
- [完整 log](validation/stage15/logs) 已保存；真實來源 hash 已重新核對。數值比較產物在 `/tmp/lilia-stage15-adapter-evidence/`，不是正式研究輸出。fixtures／基準 hashes／數值摘要持久保存。
- 基準可重現：`MPLCONFIGDIR=/tmp/lilia-mpl python tools/capture_eye_baseline_stage15.py --out /tmp/lilia-eye-new-baseline`（須新目錄，讀固定舊提交並核對共享依賴）；不會覆寫 fixtures 或正式產物。
- 本輪未生成新 CLI CSV／圖，**未做輸出重讀或目視驗證**，不能視為階段十五完成或解除 guard 的依據。

## 未解問題與下一步

- [任務 15](TASKS.md#stage-15) 接續：main 改接分段函式、輸出來源／模型／設定／索引 metadata 與失敗 audit，保持已修好的 CSV header。
- TD 依來源 segment 斷線，STFT 逐段計算並放回 elapsed 軸；修正 ch5/6 圖形映射，保留既有 STFT 方法，來源短段／缺口及尾端必須可辨識。
- 補專屬輸出重讀／竄改／CLI 失敗測試，真實模型分段對照、表與 hash、圖形目視及 `--full`，完成後才解除 CLI guard。品質與方法校準不在本次等價遷移內。
