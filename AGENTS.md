# 重構工作入口

- 接手重構先讀 `REFACTOR_STATUS.md`，再讀 `docs/refactor/TASKS.md` 的當前任務；歷史交接與報告按需查閱，勿每輪全文重讀。
- 工作流程與驗證工具見 `docs/refactor/WORKFLOW.md`。依變更選測試；階段完成跑必要完整檢查。相同來源與環境已通過的檢查，沒有新疑慮不重跑。
- 完成階段只更新短進度、任務狀態與增量報告；使用 `docs/refactor/REPORT_TEMPLATE.md`，完整 log 留檔，對話回報摘要及失敗項目。
- 保留現有工作區修改；bundle 由 `python build_bundles.py` 生成。不要覆寫正式資料、模型或歷史研究產物。
- 方法、數值、原始／模型索引與品質語義不能因重構而暗改；驗證指令通過不取代必要的真實數值對照及目視檢查。
