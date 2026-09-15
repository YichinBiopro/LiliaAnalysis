# 第十八階段待驗收清單

日期：2026-09-11。用途：交給下一位 agent 做收斂驗收，不要把這份清單當成階段完成證據。

驗收結果（更新 2026-09-15）：**整階段通過**，詳見 [完整驗收](STAGE18_ACCEPTANCE_REPORT_2026-09-15.md)。本檔保留修正前的問題清單與核對門檻。

## 已先修正

- [x] R1：[spectral_entropy.py](../../spectral_entropy.py) 的 population 維持 `disabled`／未遮罩，series 另存 scored／disabled 與窗口計數；新 summary sidecar／reader 核對 scope。
- [x] R2：entropy／joint-MI／state／denoised reader 核對設定、分數、遮罩與 audit；R4 已補 stage／component diagnostics 與可重算來源檢查。
- [x] 最新完整檢查 348 tests／0 skipped、192 Python／Pyflakes／bundle／diff；新 helper 102 陣列精確相同、5 來源 reader／五圖目視；整階段本機 1,364 項、可提交範圍 1,351 項證據核對通過。

## 仍待驗收／可能還要修

- [x] [spectral_entropy.py](../../spectral_entropy.py)：entropy clean／state／MI 的 quality mask、CSV／sidecar／audit、stage 與配套品質圖已驗收；全排除時間軸／提示及主 population footer 已修。
- [x] [plot_goertzel_vs_raw.py](../../plot_goertzel_vs_raw.py)：Goertzel filtered quality／raw hard-artifact 的 stage、診斷、reader 與圖形已驗收。
- [x] [quality_check.py](../../quality_check.py)：samples raw／filtered、anomalies raw 的 stage／診斷／CSV／sidecar／來源 reader 與圖形已驗收；NaN qmed 明確 flagged。
- [x] 舊 `compute_quality_windowed` 的實際 caller `plot_custom_markers`：raw 診斷、CSV／sidecar／來源 reader、品質圖及新舊精確數值已驗收；雙值 API 與舊 scorer 注入相容。
- [x] R1／R2 修改後已由 `python build_bundles.py` 重新生成 bundle，完整檢查確認無漂移。
- [x] `0dd9ff3` 補齊 `40d2a23` 漏掉的 73 個驗收 CSV／PNG／SVG，含六張代表圖；Git blob hash 與原證據一致。

## 建議驗收順序

1. 已完成 direct helper 的專屬數值與圖形驗收。
2. 已以原 hash 與原 NPZ 對照核對全部增量產物；較早的程式 hash 由後續變更與最後完整檢查取代。

## 驗收結論門檻

- 所有新增或修正的 CSV / audit / sidecar 欄位都要能從 reader 回讀。
- `quality_state` / `population_quality_state` 不得再和實際 masking 狀態不一致。
- 若只是文件與待辦整理，不可把第十八階段標成完成。
