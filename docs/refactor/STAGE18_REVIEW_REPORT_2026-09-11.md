# 第十八階段清單驗收：未通過

日期：2026-09-11。範圍：[待驗收清單](STAGE18_REVIEW_CHECKLIST_2026-09-11.md)、HEAD `40d2a23` 與工作區 joint-MI summary／測試兩處修改。第十八階段維持進行中。

後續更新：[R1／R2 品質狀態增量](STAGE18_MI_QUALITY_STATE_REPORT_2026-09-11.md) 已修 R1 與 R2 狀態／遮罩／audit 一致性；321 tests／0 skipped、26 案例／513 陣列精確相同。stage／component 診斷及 R3–R6 仍待辦；下列保留修正前發現與當時證據。

2026-09-15 更新：[entropy 增量](STAGE18_ENTROPY_DIAGNOSTICS_REPORT_2026-09-15.md) 已完成 R2 剩餘診斷與 R4，R3 由 `0dd9ff3` 補提交；327 tests／35 案例／714 陣列對照及五圖目視通過。R1–R4 可關閉，R5／R6 與剩餘 helper 仍待完成。

## 阻擋驗收的發現

- **R1／P1：population 狀態誤用 series 狀態。** `spectral_entropy.py:3322` 改成 `quality_state`，但 3253–3278 的 population 在遮罩前建構，3297 只遮罩 window series。重現六窗口全排除、series MI 全 NaN，summary 仍用全部 1,600 個樣本，MI 與未遮罩計算相同（CSV 回讀差 6.94e-18），卻標 `population_quality_state=scored`。應保留 population 未遮罩的獨立語義，必要時另加 series 欄位；不可為了對齊標籤暗改 population 方法。現有新斷言只確認字串，不能作修正成功證據。
- **R2／P1：reader 未核對 quality 狀態。** `lilia/entropy_io.py:128` 的 joint-MI reader 只加上 kind／channel 檢查，共用 loader 驗來源、hash 與窗口。將 CSV `quality_state` 改成 disabled 並重算 table hash，sidecar 仍 `quality_enabled=true`，來源 reader 仍接受。entropy／state readers 也未完成 scorer stage／component 診斷驗證；需加入狀態、mask、設定與 audit 的一致性檢查。
- **R3／P1：上次提交缺少驗收表與圖。** `40d2a23` 未納入 raw_callers／tflite_callers 的 51 CSV、19 PNG、3 SVG（共 73 檔）；六張 visual manifests 引用的 PNG 全部不在 Git tracked files。檔案目前只存在本機，乾淨 checkout 無法重核代表圖與表格。上次回覆「持久化證據已提交」不完整；需明確加入被 ignore 的驗收產物，保留其既有 hash。
- **R4／P2：entropy fallback 原因被丟棄。** `spectral_entropy.py:814` 與 `:882` 只取 overall，未保存 stage／component diagnostics。強制 spectrum 例外回傳 0.5，兩個 clean 子窗口仍按舊門檻接受並記 scored，但四通道 scorer valid 都是 false；接受政策可保留，無效原因及圖形標記必須另存，不能把有限 fallback 當成功評分。
- **R5／P2：Goertzel stage 尚未串接。** `plot_goertzel_vs_raw.py:106` 只取 BP 的 overall；CSV／metadata（250–267）沒有 scorer 診斷。sat／ptp／max diff 取 raw，edge shift 取 BP；現有 hard-artifact 欄位應保留並分別記錄來源，品質圖目前只有分數／門檻，無診斷失敗標記。
- **R6／P2：quality_check 未標出 NaN 評分異常。** `quality_check.py:390` 的 NaN 比較不成立，重現 `qmed=NaN` 卻 `reasons=[]`，後續 414 行的 flagged 篩選將其漏掉。samples 的 raw／filtered 分數與 anomalies 的 raw 分數亦缺少 stage／diagnostics 及可回讀表格／sidecar；須明示無法評分並補輸出，保留原閾值與數值方法。

## 清單核對結果

| 項目 | 結果 |
| --- | --- |
| joint-MI 已先修正項 | 11 tests 通過；R1 顯示語義仍不符合驗收，不能關閉。 |
| entropy clean／state／MI | 窗口／原遮罩回歸通過；R1／R2／R4，未通過第十八階段診斷契約。 |
| Goertzel | raw／BP 與 hard-artifact 路徑已盤點；診斷／stage／reader／圖形仍未完成。 |
| quality_check | samples／anomalies 均已盤點；R6 待修，兩路輸出尚未滿足契約。 |
| 其他 direct caller | 剩餘包含 `plot_event_markers.compute_quality_windowed`；已完成的 event main、TFLite helpers／baseline 與 reader 重算呼叫不可重複算成缺口。 |
| bundle／測試 | 靜態檢查通過，無 bundle drift；本輪沒有產品程式修正或需要重新生成的 bundle。 |

## 驗證證據與限制

- [joint-MI](validation/stage18/review/joint_mi/summary.json) 11 tests；[其餘回歸](validation/stage18/review/related/summary.json) entropy 13、state 10、Goertzel sampling 3／distribution 3，共 40 tests／0 skipped。
- [靜態檢查](validation/stage18/review/static/summary.json)：176 Python 編譯／Pyflakes、bundle、diff 通過。未把本輪 focused tests 說成重跑全部 317 tests。
- [重現結果](validation/stage18/review/findings.json)／[重現腳本](validation/stage18/review/reproduce_review.py)：population 數值對照、重算 hash 的 reader 反例、有限 fallback、NaN anomaly 與 Git 產物盤點。執行 `python docs/refactor/validation/stage18/review/reproduce_review.py --out /path/to/new-directory`；腳本目前以重現缺陷為成功條件，非修復後應保持通過的測試。
- 本輪沒有重跑真實模型或新圖目視；檢查圖形程式與既有產物追蹤範圍，不以此宣稱圖形驗收完成。重現 CSV 在 `/tmp/lilia-stage18-review-repro-v3/`，程式、findings 及檢查完整 log 已留 repo；暫存 CSV 可用腳本重建。
- 下一步先修 R1／R2，補 R3 提交產物，再完成 R4–R6 與剩餘 helper，依變更補反例測試、真實數值、來源 reader、目視及完整檢查。本輪保留原工作區程式／測試修改，僅新增驗收證據與更新交接文件，未 commit／push。
