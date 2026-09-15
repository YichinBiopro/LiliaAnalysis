# 第十八階段待驗收清單

日期：2026-09-11。用途：交給下一位 agent 做收斂驗收，不要把這份清單當成階段完成證據。

驗收結果（更新 2026-09-15）：**整階段仍未通過**。R1–R6 已完成，詳見 [entropy 增量](STAGE18_ENTROPY_DIAGNOSTICS_REPORT_2026-09-15.md)、[R3 補提交](STAGE18_ARTIFACT_RESTORE_REPORT_2026-09-15.md)、[R5 Goertzel](STAGE18_GOERTZEL_QUALITY_REPORT_2026-09-15.md) 與 [R6 quality_check](STAGE18_QUALITY_CHECK_REPORT_2026-09-15.md)；剩餘 direct helper 與整階段驗收待完成。

## 已先修正

- [x] R1：[spectral_entropy.py](../../spectral_entropy.py) 的 population 維持 `disabled`／未遮罩，series 另存 scored／disabled 與窗口計數；新 summary sidecar／reader 核對 scope。
- [x] R2：entropy／joint-MI／state／denoised reader 核對設定、分數、遮罩與 audit；R4 已補 stage／component diagnostics 與可重算來源檢查。
- [x] 最新完整檢查 345 tests／0 skipped、188 Python／Pyflakes／bundle／diff；R6 20 案例及污染 helper、974 陣列精確相同，17 來源 reader 與八圖目視。

## 仍待驗收／可能還要修

- [x] [spectral_entropy.py](../../spectral_entropy.py)：entropy clean／state／MI 的 quality mask、CSV／sidecar／audit、stage 與配套品質圖已驗收；全排除時間軸／提示及主 population footer 已修。
- [x] [plot_goertzel_vs_raw.py](../../plot_goertzel_vs_raw.py)：Goertzel filtered quality／raw hard-artifact 的 stage、診斷、reader 與圖形已驗收。
- [x] [quality_check.py](../../quality_check.py)：samples raw／filtered、anomalies raw 的 stage／診斷／CSV／sidecar／來源 reader 與圖形已驗收；NaN qmed 明確 flagged。
- [ ] 剩餘 direct helper：event 舊 `compute_quality_windowed` 不等同已驗收的 main 路徑；依實際使用情況完成遷移、測試與整階段驗收。
- [x] R1／R2 修改後已由 `python build_bundles.py` 重新生成 bundle，完整檢查確認無漂移。
- [x] `0dd9ff3` 補齊 `40d2a23` 漏掉的 73 個驗收 CSV／PNG／SVG，含六張代表圖；Git blob hash 與原證據一致。

## 建議驗收順序

1. 完成剩餘 direct helper；依變更選測試，無新變更不重跑已通過的 entropy／Goertzel／R6 數值對照。
2. 全部完成後才進行第十八階段整體收斂驗收。

## 驗收結論門檻

- 所有新增或修正的 CSV / audit / sidecar 欄位都要能從 reader 回讀。
- `quality_state` / `population_quality_state` 不得再和實際 masking 狀態不一致。
- 若只是文件與待辦整理，不可把第十八階段標成完成。
