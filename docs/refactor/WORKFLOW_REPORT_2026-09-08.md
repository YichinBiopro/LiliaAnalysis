# 重構工作流程改善

日期：2026-09-08。狀態：五項流程改善完成；分析入口仍完成至第十三階段。

## 本輪變更

- 新增 [短進度](../../REFACTOR_STATUS.md) 與根目錄 `AGENTS.md` 接手規則；舊交接加入口連結，保留歷史內容。
- [任務卡](TASKS.md) 固定 14–17 的範圍、保留行為、起始測試與完成條件；其後方法／架構工作留待拆分。
- [共用驗證工具](../../tools/refactor_check.py) 提供選測試、全套／靜態檢查、指紋適用性檢查，以及 manifest 驅動的表重讀、hash 與 NPZ 比較。完整 log 留檔，stdout 只顯示摘要。
- [WORKFLOW](WORKFLOW.md) 定義按變更選測試與沿用既有成功結果的條件；不使用隱式跳過測試的 cache。
- [報告模板](REPORT_TEMPLATE.md) 只記增量與證據連結；不再向長交接複製歷史摘要。

## 驗證

- `python tools/refactor_check.py check --full --out /tmp/lilia-workflow-full`：**206 tests、0 skipped、138 Python 編譯、Pyflakes／bundle／diff 通過**。[持久摘要](validation/workflow_checks.json)
- 新增 9 項工具測試：數值／NaN、整數時間精度、空值／shape／keys／tolerance、hash 篡改／schema、未知 table kind、零測試、失敗退出、fixture／模型／來源變動及缺檔狀態查詢。
- 舊工具測試摘要在程式修改後 `status` 返回不適用；最終全套摘要指紋相同，返回適用且明示沒有重跑。
- `python tools/refactor_check.py evidence /tmp/lilia-workflow-evidence-inputs/manifest.json --out /tmp/lilia-workflow-evidence`：12 項通過，含 8 個既存 hash、2 份各 803 列的真實 TYY 表重讀、2 組 BP／TFLite 數值對照。[摘要](validation/workflow_evidence.json)／[manifest](validation/workflow_evidence_manifest.json)
- NPZ actual 取自第十三階段真實 metrics CSV；expected 取自當時保存的 `/tmp/lilia-stage13/real_before.json` 舊主函式結果；比較四指標及整數窗口中心，最大浮點誤差 `1.1102230246251565e-16`，時間差 0。
- 本輪未改分析／繪圖程式；圖形沿用第十三階段目視及原 audit hash，沒有重新宣稱做過目視驗證。

## 限制與下一步

- 指紋不是完整依賴追蹤，額外輸入要用 `--input`；`status` 不能代替新疑慮下的驗證。工具尚未支援的新表 kind 必須加入正式 loader，不降級成普通 CSV 讀取。
- 持久 JSON 留存結果，完整 log、真實產物及本次 NPZ 仍位於 `/tmp`，可能消失；範例 manifest 是本輪紀錄，重跑需保留／重建所指檔案並使用新輸出目錄。
- 下一步：[任務 14 qEEG CLI](TASKS.md#stage-14)。本輪未 commit／push，保留第九至十三階段既有修改。
