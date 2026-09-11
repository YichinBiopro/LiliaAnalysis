# 第十八階段：共用 raw 品質呼叫端診斷

日期：2026-09-11。狀態：本增量完成；第十八階段仍進行中。範圍：event qEEG／event markers／zoom／subject comparison。

## 本增量變更

- 原本呼叫端只留下分數，無法分辨有效低分與例外 fallback；新增 [quality_audit](../../lilia/quality_audit.py)，逐窗保存診斷、實際 raw stage、完整 scorer context 及來源索引。
- [event_qeeg](../../lilia/event_qeeg.py) 保留舊注入 scorer 的 `(data, fs, params)` API；未提供診斷者明示 unavailable，短窗／非有限來源／評分例外分別記錄原因。
- [event IO](../../lilia/event_qeeg_io.py) 與 [comparison IO](../../lilia/subject_comparison_io.py) 新增四個診斷 CSV 欄位及 sidecar 版本；zoom 共用來源 reader，三入口 audit 均保留逐窗診斷與摘要。
- reader 重建 raw 切片與設定、重算一般診斷，拒絕 preset／stage／欄位／覆蓋範圍及重算 hash 後的篡改。歷史注入例外只核對 fallback 值、原因、有效性與可重算部分，不宣稱能重現當時例外。
- 舊表缺少整組診斷欄位及版本時仍可讀；部分缺漏或互相矛盾的新診斷會失敗，不將舊資料猜成有效。
- [event markers](../../plot_event_markers.py) 品質圖新增無效／不可用標記，標題取實際 components／preset。zoom 與 subject comparison 原有圖形保留，新增診斷由相鄰 CSV／audit 提供。
- 品質政策明示 `legacy_overall`，未採用 `usable_overall`；保留原分數、門檻、接受／排除窗口、濾波／模型／qEEG／baseline 與 raw／模型索引語義。bundle 由 `build_bundles.py` 生成。

## 驗證

- [完整檢查](validation/stage18/raw_callers/checks.json)：310 tests／0 skipped；172 Python 編譯／Pyflakes、bundle／diff 通過，執行期間來源指紋一致。
- 新增 9 項 [專屬測試](../../tests/test_quality_audit_regression.py)：缺口映射、舊 scorer／舊表、短窗、污染與例外、有限 fallback、重算 hash 後篡改及圖形標記／實際標題。
- [真實實跑](../../tools/validate_quality_callers_stage18.py) 對照 `a0a7ba6` 的凍結 event qEEG，核對共享數值模組未變；五份 Jenqwei 原始八通道做 BP，另明示取前四通道副本做 BP＋真實 TFLite。
- [來源盤點](validation/stage18/raw_callers/source_inventory.json) 保存原始／副本 hash 與通道選取方式；共 10 案例、15 branches、177 窗口。首次直接把八通道交給四通道模型的驗證失敗 log 已保存，改正驗證輸入後完整通過，未變更產品通道政策。
- [數值摘要](validation/stage18/raw_callers/analysis.json)：10 組新舊 NPZ 比較，rtol／atol 均 0；分數、窗口遮罩、raw 索引、四項 qEEG、heatmap 與平滑趨勢最大誤差 0，NaN 位置一致，選取差異 0 窗口。
- [實跑 evidence](validation/stage18/raw_callers/evidence_checks.json)：168 checks＝125 hash＋33 表重讀＋10 數值比較；表格為 17 event、15 comparison、1 zoom。
- [目視紀錄](validation/stage18/raw_callers/visual_review.json)：真實 event、強制 spectrum fallback、30–60 秒 zoom 三圖通過；fallback 保留 0.5 及舊接受結果，同時以橘色叉號明示無效，標題正確顯示 spectrum [custom]。
- 持久副本另通過 [數值 30 checks](validation/stage18/raw_callers/numeric_evidence_checks.json) 與 [目視產物 4 checks](validation/stage18/raw_callers/visual_evidence_checks.json)；NPZ、表格、sidecar、圖、完整 log 與摘要均保存在本報告相鄰的 validation 目錄。

## 未解問題與下一步

- 本增量無新增阻塞；[任務 18](TASKS.md#stage-18) 尚餘 TFLite baseline／summary、entropy clean、Goertzel、quality_check 與舊直接 helper。需依實際 raw／filtered 來源逐入口驗收，不能把本次通過當整階段完成。
- 完整來源重讀的原始 manifest 保留實跑絕對路徑 `/tmp/lilia-stage18-callers-v3/`，四通道衍生來源仍在該目錄；清除暫存後需用腳本在新目錄重建。repo 內 numeric／visual manifests 使用相對路徑，可獨立重核持久 NPZ／圖，未把暫存來源視為永久證據。
- Git：第十八階段核心已提交 `a0a7ba6`；本次 raw 呼叫端增量尚未提交，未 push。CSV／PNG 受全域 ignore 規則影響，提交驗收產物時須明確加入相應檔案。
