# 第十八階段：品質 scorer 與呼叫端完整驗收

日期：2026-09-15。狀態：完成。範圍：品質 scorer、raw／filtered／TFLite／entropy／Goertzel／quality_check 呼叫端與 direct helper。

## 本階段變更

- 唯一剩餘實際呼叫端為 [plot_index_vs_raw.py](../../plot_index_vs_raw.py) 的 `plot_custom_markers`，它使用 [plot_event_markers.py](../../plot_event_markers.py) 的 `compute_quality_windowed`。現在保留原 raw 單通道 `legacy_overall` 評分及門檻，另輸出逐窗診斷、來源索引／整數時間、qEEG 缺口 mask、CSV／sidecar 和品質圖。
- [custom_marker_quality_io.py](../../lilia/custom_marker_quality_io.py) 的來源 reader 重建原 raw 切片、窗口、qEEG 對齊、診斷及門檻 mask；`refactor_check` 可核對 `custom_marker_quality`。品質圖顯示實際 raw stage／preset，將低分、無效、不可用與缺口窗口分開標記。缺口窗口保留舊分數，但品質線在缺口處斷開。
- `compute_quality_windowed` 的預設雙值回傳、注入舊 scorer 簽名、例外傳遞、窗口與分數均保留；`return_audit=True` 才回傳第三項。`usable_overall` 不代替舊分數門檻；外部舊 scorer 無法重算診斷時明示 unavailable。
- Stage18 其他入口的增量結果見 [盤點與相容契約](STAGE18_QUALITY_INVENTORY.md)。唯一有意改變的接受語義是 R6 的三個 NaN qmed 異常列明確 flagged；有限分數、選取、時間與既有品質政策維持原結果。

## 驗證

- [最後完整檢查](validation/stage18/direct_quality_helper_final_checks/summary.json)：348 tests／0 skipped，192 Python 編譯，Pyflakes／bundle／diff 全通過；[direct helper 專屬測試](../../tests/test_custom_marker_quality_regression.py) 覆蓋舊 API、注入 scorer、來源 reader 與重算 hash 的表格篡改。
- [凍結舊碼數值對照](validation/stage18/direct_quality_helper_visual_final/analysis.json) 使用提交 `996fb110` 的兩支原程式及六個未變數值依賴；五份真實 Jenqwei 來源各兩通道、缺口、有限 fallback、外部 scorer、NaN、短窗與例外共 16 helper 案例，再比較真實 talk 與四種合成 plot。本機舊新 NPZ 的 102 個陣列／5 組 marker summary 精確相同，NaN 位置一致，最大絕對誤差 0，容忍值 0；舊接受／排除窗口沒有差異。來源綁定 NPZ 與 talk 圖表保留本機；[可提交 manifest](validation/stage18/direct_quality_helper_visual_final/manifest.json) 僅列合成資料產物。
- 本機來源版核對 78 項，含 5 張來源表重讀；[可提交版](validation/stage18/direct_quality_helper_visual_final/repository_evidence_release/summary.json) 核對 59 項，保留四張合成來源表，不發佈錄製衍生的 NPZ／圖表。[目視紀錄](validation/stage18/direct_quality_helper_visual_final/visual_review.json)／[5 項可提交圖形 hash](validation/stage18/direct_quality_helper_visual_final/visual_evidence_publication/summary.json)：真實 talk、缺口、有限 fallback、外部 scorer、NaN 共五圖已在本機檢查；四張合成圖可在 Git 核對。缺口線已斷開，fallback 的有限低分與無效診斷同時可見。
- [整階段證據清單](validation/stage18/final_acceptance/scope.json) 及 [核對摘要](validation/stage18/final_acceptance/evidence_rollup.json)：本機來源綁定 8 組模組共 1,135 個原檔案 hash、204 張來源表、25 組原 NPZ 精確比較，合計 1,364 項通過；可提交範圍保留 1,123 個 hash、204 張表及 24 組原 NPZ，共 1,351 項通過。較早增量的 6 個程式 hash 因後續修正而失效，已逐一列名；不重算舊產物 hash 或數值基準，現版程式由上述最後完整檢查覆蓋。

## 未解問題與下一步

- 無新增方法阻塞。正式 Jenqwei／James／Hardy 來源和舊 TFLite 模型的來源重讀仍依賴本機原檔；direct helper 的真實錄製衍生 NPZ／圖表只留本機，可提交 manifest 保留合成來源與讀表證據。歷史注入例外／外部 scorer 只能驗證已存 fallback 與可重算部分，不能證明當時例外再次發生。
- 下一工作單元為 [方法校準](TASKS.md)：baseline、PSD、Goertzel 門檻／平滑與 MI estimator／surrogates；保留舊 profile 及明確數值比較。Stage18 的方法門檻不因 metadata 重構暗改。Git 提交／推送狀態以目前分支為準。
