# 方法校準首輪：舊 profiles 與核心比較基準

日期：2026-09-16。狀態：M0 首輪增量完成；方法校準仍進行中。範圍：方法盤點、驗證工具與基準；分析程式／bundle／正式資料／模型未修改。

## 本階段變更

- 原先只有「方法校準」待辦，現在 [profiles／比較契約](METHOD_CALIBRATION_PROFILES.md) 明列 baseline、PSD、Goertzel、MI 的來源、參數與數值涵蓋界線；[任務卡](TASKS.md#method-calibration) 分 M0–M4，當前下一步 M1。
- [capture 工具](../../tools/freeze_method_profiles.py) 從 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed` 取凍結 Python 原碼，以獨立 processes 跑凍結版與工作區版；新輸出目錄拒絕覆寫，來源／模型 hash、程式 hash、套件與設定留 [config](validation/method_calibration/freeze_final_2026-09-16/config.json)。
- 保留各入口 baseline 聚合與門檻差異、1秒／4秒 PSD、Goertzel 未正規化功率與舊跨列平滑、histogram／KSG 與不同 surrogate null。沒有切換預設或作科學方法優劣結論。
- 真實 NPZ／圖與逐例 audit、凍結原碼保留 ignored `local/`；合成基準及來源 hash／摘要可提交。新增 [四項失敗偵測測試](../../tests/test_method_profile_freeze_regression.py) 覆蓋數值／NaN、遺漏 keys／dtype、空 shape、來源篡改／缺檔。

## 驗證

- [最終完整檢查](validation/method_calibration/final_checks_2026-09-16/summary.json)：352 tests／0 skipped、195 Python compile、Pyflakes／bundle／diff 通過；只新增驗證工具與測試，不需重建未變的 bundle。
- [數值摘要](validation/method_calibration/freeze_final_2026-09-16/analysis.json)：五份完整 Jenqwei 真實來源（各取前四宣告通道）＋60秒合成連續／兩段30秒且間隔10秒／半秒短資料，共8案例、1,255陣列；keys／dtype／shape／NaN位置／整數時間／數值精確相同，rtol=atol=0，最大絕對誤差0；baseline選取與audit JSON亦完全一致。
- 實際 TFLite 推論、Before／After、模型時間／索引、完整候選與 session／pre-event 選取均保存；不足基線及沒有完整模型窗為預期 status，不列成成功選取。空陣列另核對 shape／dtype，不放進要求非空的通用 NPZ evidence。
- [本機來源 manifest](validation/method_calibration/freeze_final_2026-09-16/manifest_local.json)／[274項核對](validation/method_calibration/evidence_local_2026-09-16/summary.json)；[repository manifest](validation/method_calibration/freeze_final_2026-09-16/manifest_repository.json)／[100項核對](validation/method_calibration/evidence_repository_2026-09-16/summary.json)。repository 核對不需正式錄製／模型或真實衍生 NPZ。這一輪保存 helper arrays/audit，沒有新增產品 CSV；不宣稱新增來源表 reader 驗收。
- [目視紀錄](validation/method_calibration/visual_2026-09-16/visual_review.json)：真實 talk、合成缺口、短資料三張四面板總覽及三張 Goertzel 補圖。PSD舊新曲線重合；baseline選取／不足／無模型窗可辨識，MI null可見。初版Goertzel半秒全遮罩空白，已由[補圖工具](../../tools/review_method_profiles.py)顯示未遮罩功率、排除標記與說明，沒有重算科學數值。
- Goertzel補圖：talk 5秒接受12/12、半秒0/122；缺口5秒8/12、半秒0/120，30–40秒間斷線且孤立有效窗口可見；短來源5秒0窗、半秒1窗被排除。[補圖本機11項](validation/method_calibration/visual_evidence_local_2026-09-16/summary.json)／[repository 4項](validation/method_calibration/visual_evidence_repository_2026-09-16/summary.json) hash通過。
- 開發期失敗：第一個capture在lagged MI使用顯式grid仍要求helper再次濾波，被既有guard拒絕；工具已改成按來源段先濾波再 `apply_bandpass=False`。原log留 `validation/method_calibration/freeze_2026-09-16/expected.log`（本機忽略）；最終兩個worker log與上述數值／完整檢查通過，無未解失敗。

## 未解問題與下一步

- **不是全入口基準完成**：B-meditation本輪只測summary helper、B-marker／legacy sampler／entropy-state與P-jenqwei／quality-plot只凍結程式並列既有歷史證據；新完整入口／多事件與真實行為標註待M1/M2。歷史 hash／產物不重算，歷史檢查不冒充本輪重跑。
- [30,60)秒baseline事件及KSG每段中點±5秒是驗證探針，不是真實行為標註。MI的統計顯著性／估計器有效性及參數校準尚未驗收。
- Goertzel 60Hz仍從0.5–45Hz BP輸入計算、`quality > .5`仍嚴格大於、rolling仍按整串列；這些是後續校準項目，本輪保留並固定原結果。
- 下一步 [M1：補 baseline 完整入口與事件案例](TASKS.md#method-calibration)。本輪未提交／推送。
