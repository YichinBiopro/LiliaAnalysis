# 方法校準 M3：Goertzel 分段平滑與功率單位比較

日期：2026-09-17。狀態：M3 比較驗收完成。範圍：校準工具／測試／證據；未修改產品預設、Goertzel recurrence、品質／hard／索引或 downstream 選取政策。

## 本階段變更

- [獨立捕獲工具](../../tools/freeze_goertzel_profiles.py) 凍結 Stage18 後原碼，分別在舊／新 process 實跑完整入口，核對實際匯入目錄、來源／原碼／環境 hash。Goertzel 不使用模型，無模型推論或模型變更。
- [比較工具](../../tools/compare_goertzel_methods.py) 依實跑前固定的 [契約](GOERTZEL_COMPARISON_CONTRACT.md)，保留 G-legacy-rolling，另量測 G-segment-rolling：只限制 rolling 到連續來源段，品質 mask／中心窗／不足窗與被拒列的位置不改；linear／dB 分開平滑。
- 同窗口 raw／完整段 float32 BP、.5／1／5 秒窗、未正規化／N²／Hann coherent mean-square／單頻 density 分開保存。直接複數 DTFT 作獨立數值核對，門檻換單位有明示 offset，沒有套用到產品 CSV 或 sampling／distribution。
- [7 項新增測試](../../tests/test_goertzel_method_comparison_regression.py)：跨段污染、連續相容、NaN／拒絕列、重現 segment labels／單窗／空陣列、奇偶 rolling、任意頻率 DTFT、錯誤功率、單位／floor 與意外空結果。

## 驗證

- [完整檢查](validation/method_calibration/m3_release_checks_2026-09-17/summary.json)：371 tests／0 skipped；203 Python 編譯、Pyflakes、bundle、diff 通過。初次完整檢查只有驗證工具未使用 import 失敗，已移除；[原失敗 log](validation/method_calibration/m3_final_checks_2026-09-17/pyflakes.log) 保留，release 重新綁定最終工具 hash。
- [舊新數值](validation/method_calibration/m3_release_2026-09-17/analysis.json)：固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed`；20 案例（13 真實入口／7 合成）、350 表格窗口、932 陣列與 metadata 精確一致，rtol=atol=0、最大誤差 0；包括 actual 圖線、原品質政策兩門檻／兩 hard 選项與 legacy 平滑。
- 其中 884 非空陣列供一般 NPZ evidence 比對，48 空陣列在實跑逐 key／dtype／shape 核對並記於 metadata；全短無候選與非有限來源拒絕是預期結果，不算未處理失敗。舊／新各 19 張表來源重讀共 38 次。
- [本機 evidence 531 項](validation/method_calibration/m3_evidence_local_2026-09-17/summary.json)／[repository 215 項](validation/method_calibration/m3_evidence_repository_2026-09-17/summary.json) 通過；原始錄製與真實衍生產物不放 repository manifest。完整 hash／tables／NPZ manifest 在 release 目錄。
- [30 組功率控制](validation/method_calibration/m3_release_2026-09-17/controls.json)：.5／1／5 秒 × 5 訊號 × raw／filtered；已知振幅、常數、off-grid、noise 與 mixture。350 窗 × raw／filtered 的 DTFT 核對亦通過 rtol=atol=1e−8；最大絕對誤差 3.55e−6 出現在大功率，逐值誤差占允許誤差的最大比例僅 4.81e−5，未放寬門檻。
- [16 圖目視](validation/method_calibration/m3_acceptance_2026-09-17/visual_review.json)：真實 talk／缺口副本、合成 gap／品質注入／空窗各 dB／linear 原圖與平滑比較圖，另 normalization／前端控制圖。缺口、低品質、diagnostics unavailable、空候選與單位標示核對完成；目視與 scope hash [本機 18](validation/method_calibration/m3_visual_local_2026-09-17/summary.json)／[repository 12](validation/method_calibration/m3_visual_repository_2026-09-17/summary.json) 通過。

## 觀測差異與相容策略

- 真實完整 talk 的缺口副本：G-segment-rolling 改變窗口 `[4,6,7]`，最大 2.07965 dB／48,327.8395 legacy linear power；合成 gap 改變 4 窗，最大 11.72248 dB。已知四點 witness 最大 50 dB。這些是分段計算差異；產品原圖斷線不能證明原 rolling 沒有跨段影響。
- 連續來源的候選與 legacy 精確相同；全部比較的接受／排除和 NaN 位置差異均 0。`.5 s` 真實／合成案例的原品質政策全部拒絕，1 s talk 為 61 窗接受；不能把短窗更多列解讀為更多合格資料。
- 振幅 2 的 60 Hz 原訊號：.5→5 秒 legacy 為 41.90338→61.93473 dB，coherent mean-square 仍約 2；BP 後同窗 legacy 約低 22.53 dB。此結果包含現有濾波及邊界行為，不替新前端挑門檻。
- 5 秒、legacy 67.15 dB 的單位 witness：P/N² 為 −0.80880 dB、coherent 項 8.22557 dB、density 項 13.45262 dB。比較使用 ±.5 探針，正式 sampler 預設 ±.6 不變；正值未觸 floor 才可用明示 offset 換算。不同單位的相同數值 floor 不代表同一物理 floor。
- G-segment-rolling 與 normalization 只是比較 profile，沒有新增產品 CLI 選項、變更 CSV、重設品質或發布新科學方法。Hann 與單位定義參照 [SciPy symmetric Hann](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.windows.hann.html)／[spectrum scaling](https://docs.scipy.org/doc/scipy-1.15.3/tutorial/signal.html#spectral-analysis)；不從這些比較推論 EEG 行為效應。

## 未解問題與下一步

- 無新增驗收阻塞。[scope](validation/method_calibration/m3_acceptance_2026-09-17/scope.json) 明記真實缺口是完整來源副本的人造 10 秒缺口，.7 品質為 diagnostics unavailable 的探針；不是新行為標註。未來若採用分段平滑，須另具名產品選項並保留 legacy 相容策略。
- 下一步：[M4 MI 校準](TASKS.md#method-calibration)，histogram／KSG／null 分別比較，先定已知控制與標註界線，再比較 bins／k／樣本數／seed／surrogates。
- Git：M2 已提交並確認遠端為 `6eef079`；本輪 M3 修改與證據留工作區，未提交／推送。
