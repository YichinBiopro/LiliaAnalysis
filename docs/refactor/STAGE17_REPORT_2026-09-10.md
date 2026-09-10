# 第十七階段：Jenqwei 資料集分段與完整驗收

日期：2026-09-10。狀態：完成。範圍：`build_jenqwei_tflite_dataset.py`、`lilia/jenqwei_dataset.py`、生成 bundle、測試及驗證工具。

## 本階段變更

- 盤點確認本入口只準備訊號，不執行模型或 train/test split；專案內未找到其他讀取者。舊版 `e8ebe18` 入口及依賴已凍結，詳見[契約](STAGE17_DATASET_CONTRACT.md)。
- 保留 float32 四通道 bandpass、polyphase、先等分再裁 400 點窗口；連續資料的 CSV、檔名和既有 metadata 值不變。缺口資料逐來源段套用相同流程。
- 來源段、完整濾波上下文、packed／段內重採樣座標、原始 fractional sample centres、真實模型窗口數、每等分裁尾及排除原因均可回讀；舊 `*_200hz` 別名另以 `fs_out` 明確解釋。
- 預設短段／短等分失敗；只有 `--allow-short-drop` 可排除，全排除仍失敗。所有宣告訊號欄位污染均拒絕，包含未選通道／被排除點；不加入品質 fallback。
- CLI 使用新／空目錄、排他建立檔案；manifest 發佈前驗證完整來源與所有 CSV。失敗留下 run audit，既有正式資料及第十六階段修改均保留。
- 入口 continuity guard 已隨分段遷移完成驗收而解除；其他入口 guard 不變。

## 驗證

- [完整檢查](validation/stage17/final/checks.json)：291 tests、0 skipped；165 Python 編譯、Pyflakes、bundle、diff 通過。新增 22 項專屬測試，完整 log 留在同目錄。
- [舊基準](validation/stage17/final/baseline_evidence_checks.json)：17 項 frozen entry／依賴／七份 NPZ／五份真實來源 hash 通過；基準為實際舊版 CSV 逐檔回讀，未用新實作生成舊預期。
- [實跑摘要](validation/stage17/final/analysis.json)：五份真實來源整片模式 25 CSV、單窗口模式 145 CSV；每模式保留 145 個模型窗口、58,000 列。
- 合成 200／199.5 Hz 兩模式另 20 CSV；真實訊號分段副本另 12 CSV。所有 202 CSV、八份 manifest 與來源 audit 均逐檔回讀及核對對齊。
- [Evidence](validation/stage17/final/evidence_checks.json)：545 checks＝327 次 hash（313 個不同路徑）、202 表重讀、16 組 NPZ。`rtol=atol=0`，float32 CSV 值及 int64 微秒最大誤差均為 0。
- 分段數值另按完整來源段獨立濾波／重採樣對照；不跨缺口、不合併窗口。測試另含 padlen 27／28、非整數採樣率、上採樣最後 fractional centre 與同名不同來源。
- [目視紀錄](validation/stage17/final/visual_review.json)：真實 move-head、真實訊號分段、199.5 Hz 合成三張診斷圖均確認窗口／裁尾／缺口與波形同軸對齊，無跨段連線；[四項持久 hash](validation/stage17/final/visual_evidence_checks.json) 通過。
- 預期失敗：嚴格短段、全排除及被排除尾端污染均非零退出、有失敗 audit、無成功 manifest。
- 已解驗收問題：共用 evidence 原預設逐 CSV sidecar，初次讀不到來源級 `dataset.json`；已補專屬路徑與成功／kind／signal 篡改回歸測試，完整檢查重跑通過。[原錯誤 log](validation/stage17/final/resolved_sidecar_failure.log) 留存。

## 未解問題與下一步

- 無新增未解驗收失敗。自訂形狀未載模型驗證；資料集準備沒有宣稱任何自訂參數適合特定模型。跨來源失敗可留下先前成功檔案，成套原子發佈仍屬後續任務。
- 持久保存基準、摘要、完整 log、manifest／audit 及三張圖；完整 CSV／比較 NPZ 位於 `/tmp/lilia-stage17-final-v2/`，不保證永久存在。可用 `tools/validate_jenqwei_dataset_stage17.py` 在新目錄重建，不能將重建 hash 當舊證據。
- 下一步：[品質 scorer 待辦](TASKS.md)，先盤點短窗、非有限與例外 fallback；尚未開始。第十六階段後半及本階段尚未 commit，未 push。
