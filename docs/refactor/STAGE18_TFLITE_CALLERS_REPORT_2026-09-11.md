# 第十八階段：TFLite baseline／summary 品質診斷

日期：2026-09-11。狀態：本增量完成；第十八階段仍進行中。範圍：TFLite baseline／summary、相關 reader、相容 helper 與生成 bundle。

## 本增量變更

- [baseline](../../lilia/tflite_baseline.py) 逐子窗口保存原始索引、物理時間、削波比例、篩選結果及診斷；遇到第一個不合格子窗口仍立即停止，未新增評分呼叫或改變 `quality_min` 意義。
- [summary](../../plot_tflite_summary.py) 明示三種 stage：baseline 為 filtered／500 Hz 四通道，Before 為 filtered_resampled／200 Hz 來源 ch1/2，After 為 model_output／200 Hz 兩通道。
- [tflite_quality](../../lilia/tflite_quality.py) 與 [IO](../../lilia/tflite_io.py) 保存前後各四個 CSV 診斷欄位、逐窗 sidecar／audit、baseline catalog 與選取紀錄；品質圖分別標記前後無效診斷並顯示實際 components／preset。
- [共用品質 audit](../../lilia/quality_audit.py) 抽出可按 stage 驗證的 record helper；既有 raw caller 回歸通過。舊注入 scorer 仍標 unavailable，舊二元 helper 回傳與無診斷舊表可讀性保留。
- reader 有來源時重做完整來源段的濾波／保留重採樣、raw 削波與一般品質診斷；依保存 scorer 回應重建 baseline 短路順序、接受結果與 seed 選取，拒絕重算 hash 後的 stage／preset／通道映射／聚合方式／索引與覆蓋範圍篡改。
- legacy baseline sampler 另保存診斷與選取 audit；其 continuity guard、串接回傳相容語義及不可直接作模型 baseline 的限制保留。
- 全程使用 `legacy_overall`；不採用 `usable_overall`，未改門檻、分數、filter／model／qEEG、raw／模型索引、baseline 候選／選取與圖中數值。scorer 拋錯仍依舊向外傳遞。

## 驗證

- [完整檢查](validation/stage18/tflite_callers/checks.json)：317 tests／0 skipped；176 Python 編譯／Pyflakes、bundle／diff 通過，執行期間來源指紋一致。
- 新增 7 項 [專屬測試](../../tests/test_tflite_quality_regression.py)：三種 stage、舊 API／舊表、污染／短窗／削波短路、有限 fallback、scorer 例外、重算 hash 篡改及圖形標記。
- [實跑腳本](../../tools/validate_tflite_quality_stage18.py) 載入 `a0a7ba6` 凍結 summary／baseline，確認六個共用數值模組未改；[來源盤點](validation/stage18/tflite_callers/source_inventory.json) 明示五份原始八通道取前四通道的副本及 hash，原始資料未覆寫。
- 九案例：五份真實來源、talk 加 100 秒缺口副本、明示 threshold=0 對照、0.5 秒 baseline、強制 spectrum fallback；其餘案例保留預設 0.5 門檻，未調整產品預設。
- [數值摘要](validation/stage18/tflite_callers/analysis.json)：107 個 qEEG 窗口、271 個模型窗口、510 個實際評估子窗口；九組共 399 陣列精確比較，含模型 Before／After、品質／mask、baseline 選取與參考、qEEG、趨勢及 heatmap。
- rtol／atol 均 0，最大誤差 0、NaN 位置一致、選取差異 0。半秒案例新舊皆明確失敗為無可用 baseline，新增診斷追到 30 個無效子窗口；有限 fallback 則保留 0.5 分及舊接受結果。
- [完整 evidence](validation/stage18/tflite_callers/evidence_checks.json)：145 checks＝118 hash＋18 新舊表來源重讀＋9 NPZ 比較；持久 NPZ 另通過 [27 checks](validation/stage18/tflite_callers/numeric_evidence_checks.json)。
- [目視紀錄](validation/stage18/tflite_callers/visual_review.json)：真實 talk、缺口、fallback 三圖通過，最終 PNG hash 與已檢視版本一致；[4 項產物 hash](validation/stage18/tflite_callers/visual_evidence_checks.json) 通過。開發期欄位對接及測試設定問題已修正，失敗 log 留存。

## 未解問題與下一步

- reader 不重跑模型或 baseline qEEG；After 只核對模型身分、context 與診斷內部一致性，歷史例外不保證重現。獨立新舊模型／baseline 實跑數值證據另如上，不把 reader 通過當模型數值證明。
- NPZ、CSV／sidecar／audit、PNG、凍結舊碼、摘要及完整 log 保存在 `validation/stage18/tflite_callers/`；原始 evidence 使用 `/tmp/lilia-stage18-tflite-real-v2/` 實跑路徑，衍生來源仍在 inputs 暫存。清除後可由腳本重建；numeric／visual manifests 用 repo 相對路徑獨立重核。
- 本增量無未解失敗；[任務 18](TASKS.md#stage-18) 下一步是 entropy clean／state／MI，再接 Goertzel、quality_check 與剩餘直接 helper；整階段尚未完成。
- Git：核心 `a0a7ba6` 已提交；raw 與本次 TFLite 呼叫端增量均尚未提交，未 push。提交驗收 CSV／PNG 時須明確處理 ignore 規則。
