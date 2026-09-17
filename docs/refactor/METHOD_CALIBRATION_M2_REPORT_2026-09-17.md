# 方法校準 M2：PSD 基準與同輸入敏感度比較

日期：2026-09-17。狀態：M2 完成；捕獲／檢查於 2026-09-16 留檔，收尾核對指紋仍匹配。範圍：驗證工具、比較契約與證據；產品 CLI、方法預設、品質／索引政策不變。

## 本階段變更

- [PSD 捕獲工具](../../tools/freeze_psd_profiles.py) 明確綁定同一模型，修復凍結版本預設路徑找不到模型的失敗；兩個 process 檢查實際匯入原碼目錄，原碼／來源／模型 hash 保留。沒有修改模型或正式資料。
- P-jenqwei 四個來源通道逐段 Before／After，P-quality-plot 原始／filtered 的實際 Welch 輸入、頻率、密度、Line2D 顯示值、品質、表格及索引均保存。缺圖線／缺通道／重複 panel／意外 skip 會失敗，不能以兩邊相同漏算通過。
- [敏感度工具](../../tools/compare_psd_sensitivity.py) 依預先固定的 [比較契約](PSD_COMPARISON_CONTRACT.md)，比較 1／4 秒、半開／閉區間、單點積分與分母；對合成控制及全部成功產品頻譜的相同輸入分別計算，不串接來源段。
- [7 項專屬測試](../../tests/test_psd_profile_freeze_regression.py) 涵蓋模型預設覆寫、缺漏／虛構 panel、短資料、奇偶單邊能量、密度篡改、單點／邊界／分母 witness 與 metadata 保留。

## 驗證

- [完整檢查](validation/method_calibration/m2_final_checks_2026-09-16/summary.json)：364 tests／0 skipped；200 Python 編譯、Pyflakes、bundle、diff 通過。收尾以 `status` 確認沿用範圍匹配，沒有重跑無變更測試。
- [舊新數值](validation/method_calibration/m2_final_2026-09-16/analysis.json)：固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed`，五份完整真實 Jenqwei＋合成連續／缺口／半秒共 8 案例、938 陣列、19 來源 reader；keys／dtype／shape／值／NaN 位置及 metadata 一致，rtol=atol=0、最大絕對誤差 0。
- [本機證據 394 項](validation/method_calibration/m2_evidence_local_2026-09-16/summary.json)／[repository 證據 151 項](validation/method_calibration/m2_evidence_repository_2026-09-16/summary.json) 通過；完整／可提交 manifest 在 capture 目錄。後者不含真實衍生產物、模型與需模型的 Jenqwei reader；前者驗證全部 19 reader，未降低驗收範圍。
- [合成控制](validation/method_calibration/m2_final_2026-09-16/sensitivity.json)：160 個窗長控制＋1 個單頻點探針；已知能量與常數控制通過。NumPy FFT 對 SciPy 最大誤差 1.78e−15；[128 個捕獲輸入／256 次窗長計算](validation/method_calibration/m2_final_2026-09-16/sensitivity_capture_summary.json) 最大誤差 2.27e−13，均符合 rtol=1e−10、atol=1e−12。
- [補充核對](validation/method_calibration/m2_acceptance_2026-09-16/captured_psd_check.json)：128 份產品原 PSD 與敏感度 4 秒 PSD 精確一致；先對保存的 manifest hash 驗證輸入，再比對頻率／密度。[可重跑腳本](validation/method_calibration/m2_acceptance_2026-09-16/check_captured_psd.py.txt) 以 Python 執行，參數為 capture 目錄及新的 result JSON。
- [10 圖目視](validation/method_calibration/m2_acceptance_2026-09-16/visual_review.json)：真實 talk、合成連續／缺口的 ch1／ch3；真實／合成品質圖；合成及真實敏感度圖。PSD 段別、來源／模型分支、缺口／裁尾／排除及無 After ch3 標示正確。目視與補充證據 hash [本機 15](validation/method_calibration/m2_visual_local_2026-09-16/summary.json)／[repository 11](validation/method_calibration/m2_visual_repository_2026-09-16/summary.json) 通過。

## 方法差異與限制

- 合成窗長控制三帶相對功率最大差 0.07055；閉／半開積分最大差 0.83333（輸入單位平方），分母最大相對值差 0.91667。8 點單頻點探針：P-qeeg β 積分 0，P-entropy 為 1.33333。這些不是等價誤差或方法優劣評分。
- 真實輸入 1／4 秒三帶相對功率最大差 0.06666，分母差最大 0.48939。合成缺口 S1 Before ch1 的 602 點案例，nperseg 200→602、df 1→0.33223 Hz，theta 相對值 0.14347→0.81280、alpha 0.85599→0.18655，最大差 0.66944；窄頻訊號在 8 Hz 邊界附近的格點／積分敏感度不能被默認等價。
- 峰值搜尋使用完整單邊頻譜，非圖中 0–50 Hz；真實 vibration-level3 的 raw sample 1 ch2 從 121→25.25 Hz（差 95.75 Hz），不可解讀為特定腦波頻帶移動。頻譜與完整逐列比較保存在各案例 `sensitivity.npz/json`。
- 缺口 S0／S3 短段排除、S1／S2 分開計算；全短模型案例保留原錯誤。move-head 29,984 點及合成 gap／short 無兩個完整 30 秒品質窗口，依舊政策 skip，沒有縮短窗口補出結果。全部品質、選取、NaN 與索引政策維持原狀。
- 方法說明依 [SciPy 1.15.3 Welch 文件](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.welch.html) 的窗口／重疊／density 定義；數值容差只驗證實作，不證明 EEG 行為效應，不新增 profile 或替使用者選窗長。

## 未解問題與下一步

- 無新增阻塞。[原失敗 log](validation/method_calibration/m2_capture_2026-09-16/expected.log) 原封保留；模型路徑開發 run 留本機並由最終 capture 取代，不冒充當前驗收。[範圍明細](validation/method_calibration/m2_acceptance_2026-09-16/scope.json)。
- 下一步：[M3 Goertzel](TASKS.md#method-calibration)，先獨立比較分段平滑，保留 legacy rolling 與嚴格品質門檻，再評估 normalization／濾波／單位。
- Git：收尾時前輪 `8dfbf00` 與本機 `origin/spectral-entropy-flow-rework` 一致，遠端追蹤 reflog 記為 push；本輪 M2 修改與驗收證據留工作區，未提交／推送。
