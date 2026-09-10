# 精簡重構流程

1. 接手只讀 `REFACTOR_STATUS.md` 與當前任務卡，執行 `git status --short`；只在遇到方法疑問時查歷史報告。
2. 先保存需要的舊行為基準，按任務範圍改程式；確定性的檢查交給腳本，不反覆手寫同一段分析。
3. 開發中跑受影響測試；階段收尾跑必要完整檢查與真實／合成證據。不要為節省用量省略模型／方法數值對照。
4. 同一程式／fixtures／模型／环境及輸入已有通過結果，無新疑慮不重跑；用 `status` 檢查指紋，記明沿用的檢查範圍。
5. 用短報告模板記本次增量，更新短進度與任務狀態；完整 log 不貼進對話，也不再向長交接累加每階段摘要。

## 自動檢查

```bash
# 指定測試檔；拼錯或零測試會失敗。任務卡列了各入口的起始集合。
python tools/refactor_check.py check --tests tests/test_signal_contract_regression.py
# 只跑編譯、Pyflakes、bundle 與 Git tracked diff whitespace。
python tools/refactor_check.py check --static
# 階段收尾：上述檢查＋全部 unittest（包含連續數值 fixtures）。
python tools/refactor_check.py check --full
# 檢查某次成功結果的程式／環境與輸入指紋是否仍匹配，不執行測試。
python tools/refactor_check.py status /tmp/lilia-refactor-XXXX/summary.json
```

各次執行預設建立唯一 `/tmp/lilia-refactor-*` 目錄，完整 log、失敗 traceback 及 `summary.json` 留檔，stdout 只列每項結果。可用 `--out /path/to/new-run` 指定**空目錄**，避免覆寫舊證據。工具不重建 bundle；先自行執行 `python build_bundles.py`。

`check` 記錄根目錄及已知程式目錄（含 bundle／工具）的 Python、fixtures、根目錄模型、requirements／toml／shell scripts、Python／套件版本與主要計算環境變數。額外測試資料使用可重複的 `--input /path/to/file` 納入指紋；未宣告的外部資料、硬體、套件原地修改不在此保證內。檢查期間上述指紋改變也會失敗。Markdown 不影響此指紋，因此純文件變更不要求重跑科學分析。

`status` 只確認**原報告範圍**仍有相同指紋的成功結果，並非本次重跑、整階段完成或完整依賴證明；缺檔、失敗、指紋不同均不可沿用。它不沿用 `evidence` 結果。任何新增失敗、相關環境變化或新疑慮都要重跑受影響檢查。此工具沒有自動跳過測試的隱式 cache。

## 依變更選測試

| 變更 | 開發中 | 階段完成 |
| --- | --- | --- |
| 純文件／任務狀態 | 檔案連結與 diff | 不重跑 197 項既有科學測試 |
| 單一入口 | 專屬回歸＋其呼叫的共享模組測試 | `--full`＋該入口真實／合成證據 |
| windowing／signal／qeeg／共用 IO | 新增邊界測試、所有受影響 callers；不確定就跑全套 | `--full`＋受影響真實方法／模型對照 |
| 驗證工具 | 工具本身的失敗偵測測試＋端到端命令 | 靜態檢查；首次導入執行一次 `--full` 確認收集能力 |

測試通過後若只改報告，不再重跑。`git diff --check HEAD` 不涵蓋 untracked 檔案，新增檔案仍需檢視。全套的測試數會隨新增案例增加，不把 197 寫成永久門檻；skipped 必須在報告解釋。

## 產物重讀、hash 與數值對照

建立 JSON manifest，相對路徑都相對於 manifest 所在目錄，絕對路徑也可用。只列本任務需要的項目，至少一項；未知 schema、未知 table kind、缺檔、hash／數值／來源不符均非零退出。

```json
{
  "schema_version": 1,
  "files": [
    {"path": "result.png", "sha256": "替換成產生時保存的 SHA-256"}
  ],
  "tables": [
    {"path": "result_bp_metrics.csv", "kind": "meditation_bp", "raw_csv": "raw.csv"},
    {"path": "result_tflite_metrics.csv", "kind": "meditation_tflite", "raw_csv": "raw.csv", "model_path": "model.tflite"}
  ],
  "comparisons": [
    {"actual": "new.npz", "expected": "baseline.npz", "rtol": 1e-12, "atol": 1e-12, "equal_nan": false}
  ]
}
```

```bash
python tools/refactor_check.py evidence /path/to/manifest.json
```

`files` 的預期 hash 必須取自先前產生時的 audit／固定基準；不要驗證時重新算一份相同 hash 冒充舊證據。可列 PNG、SVG、CSV、sidecar、audit、來源、模型及已記錄的程式檔。表格使用 `tools/refactor_check.py` 的 `READERS` 白名單呼叫既有來源驗證 loader；zoom 自動依 scope 使用專屬 reader，模型表必須提供模型。第十四階段新增 `raw_qeeg` reader，重建原始格點並核對 channel、完整 audit 與重算數值。APP／NUC 專屬配對等未列支援的 kind 仍由原專屬測試驗證，不降級成僅 pandas 讀取。

NPZ 對照要求相同 keys、非空相同 shape；整數 indices／timestamps 精確比對，浮點依明示 tolerance，NaN 需明確允許且位置一致。缺值相符不代表品質合格；全非有限陣列的最大誤差記 null。工具不生成新基準、不決定方法容許差異、不自動目視圖形，也不以表重讀取代重新計算的數值對照。

第十五階段新增 `eye_model_signal` reader，必須提供 `model_path`；核對保留四行 header 的 CSV、來源／模型／固定設定、完整 inference 與逐列來源映射。reader 不重跑模型，數值對照仍需獨立舊基準；第十五階段已完成分段繪圖／CLI／目視及完整驗收，眼開閉 CLI guard 已解除，詳見 [驗收報告](STAGE15_ACCEPTANCE_REPORT_2026-09-09.md)；表重讀仍不取代數值與目視驗證。

第十六階段新增 `jenqwei_signal` reader，需提供 `model_path`；分開核對 Before／After 表的來源／模型、完整段與 filter context、原始 fractional index、After-to-Before 映射、裁尾、整數微秒與 table hash。表重讀不重跑模型；[完整驗收](STAGE16_ACCEPTANCE_REPORT_2026-09-10.md) 另含舊模型／PSD／STFT 數值與目視。分段 main 已可接受缺口，舊 `run_pipeline` guard 保留。

第十七階段新增 `jenqwei_dataset` reader，不需要模型；sidecar 使用片段所在目錄的 `dataset.json`。核對完整來源切片計畫、檔案涵蓋範圍、hash 與逐列微秒；`lilia.jenqwei_dataset.load_manifest` 另驗證整份 manifest 並回讀所有片段。此 reader 不重新濾波，獨立舊基準數值、短段／污染及目視見 [第十七階段報告](STAGE17_REPORT_2026-09-10.md)。

真實資料的生成命令、專屬分析參數及必要的特殊比較仍由任務記錄；共用工具處理機械性核對。完整報告只貼摘要與失敗項目；持久保存重要 JSON／manifest，原始大型產物留指定資料目錄。不要為補證據覆寫正式來源。

第十八階段核心診斷以 `tools/validate_quality_stage18.py --out /path/to/new-directory` 對照四 presets 的舊分數；`valid` 與 `usable_overall` 是新增介面，不自動取代呼叫端原有品質政策。呼叫端遷移仍在任務 18，詳見 [核心增量報告](STAGE18_CORE_REPORT_2026-09-10.md)。
