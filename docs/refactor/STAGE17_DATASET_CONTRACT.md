# 第十七階段資料集契約

盤點日期：2026-09-10。入口：`build_jenqwei_tflite_dataset.py`；實作：`lilia/jenqwei_dataset.py`。

## 切分單位與舊基準

- 這是訊號準備入口，不執行模型、不建立標籤，也不分 train／validation／test。專案內搜尋入口與 manifest 欄位，未發現其他 Python 下游讀取者。
- 舊版 `e8ebe18`：前四通道 float32 → 整段四階 Butterworth 0.5–45 Hz 零相位濾波 → polyphase 500→200 Hz → 五等分 → 每等分從起點裁成 400 的倍數。
- 等分邊界為 `floor(i*N/n_splits)`；不能先裁整段尾端，也不能把模型窗口改錨在整段第 0 點。各等分起點可以不是 400 的倍數。
- 凍結入口、依賴 hash 與五份真實／合成／199.5 Hz 基準；`tools/capture_jenqwei_dataset_stage17.py` 可在新目錄重現。持久 fixtures 使用 `dataset_jenqwei_*`，避免前階段的 `jenqwei_*` 搜尋誤收。
- 預設四通道、400 點契約保留。既有自訂採樣率／通道數／窗口 CLI 仍可用；本工具不載入模型，因此不保證自訂形狀適用任一特定模型。

## 分段與排除

- 缺口沿用共用政策：相鄰時間差 **大於**三個輸入採樣週期才分段；相同或逆序時間拒絕。
- 各來源段獨立濾波、重採樣，然後各自做 `n_splits` 等分；不跨缺口補值、不合併短段，也不調整濾波方法。
- 每段所有等分均在 `dataset.json` 留存原範圍、保留終點、排除樣本數及 `window_tail`／`short_split`／`filter_too_short`／`none` 原因。
- 四階 bandpass 的預設 sosfiltfilt padlen 由係數計算，目前為 27。來源段不超過 padlen 時不能濾波；不採用未經驗證的 fallback。
- 預設遇到無窗口的等分即失敗，該來源尚未寫出檔案；`--allow-short-drop` 才允許排除。全資料無窗口仍非零退出。
- 所有宣告訊號欄位（包含未選通道及將被排除的短段／尾端）仍須有限；保留舊 loader 的污染拒絕政策。品質 scorer 未啟用。

## 檔案與座標

- 訊號 CSV 保留 `time_us,ch1,...`、int64 微秒與 float32 十進位輸出；連續資料的檔名與舊欄位值保留。分段資料在檔名加入 `_seg0000` 等來源段 ID。
- `source_csv`／`source_id` 為絕對來源路徑／SHA-256；來源目錄名仍由來源、內容與數值設定決定。`allow_short_drop` 舊識別公式不包含，但完整參數保存在 audit／manifest sidecar。
- 所有 start/end 區間為左閉右開；`raw_start_idx/raw_end_idx` 是**完整來源段濾波上下文**，不是這個片段的原始樣本包圍範圍。
- `resampled_start_idx/resampled_end_idx` 是依來源段順序串列的重採樣座標，計入所有排除點；`segment_local_*` 則從各來源段的重採樣起點算起。
- `raw_fractional_first_idx/raw_fractional_last_idx` 是片段首末樣本中心，公式 `raw_start + local_idx * down/up`；最後中心可能超過最後原始中心，仍在原始半開區間內。它們不是濾波支撐範圍。
- 原有 `*_200hz` 欄位作為重採樣座標／樣本數的相容別名保留，實際頻率以 `fs_out` 為準；舊檔名整數 Hz 標籤也保留，自訂小數採樣率必須讀 metadata。
- 舊 `window_index_in_split/window_count_in_split` 表示等分內的**檔案**位置／數量；整片模式仍為 1／1。新增 `model_window_start_in_split`（0 起）、`model_window_count`（本檔）及 `split_model_window_count`（整個等分）表達真正模型窗口。
- 第 j 個模型窗口從 `segment_local_start_idx + j*tflite_win` 起，長度固定；每一窗口都在同一等分、同一來源段內。

## 回讀與失敗狀態

- `load_fragment(path, raw_csv)` 重建來源分段與完整切片計畫，核對來源／設定、檔案涵蓋範圍、hash、欄位、每列整數時間及有限數值；不重新濾波，數值對照另由舊基準驗收。
- `load_manifest(path)` 檢查來源 audit 清單、設定、manifest 每列及**所有**片段；缺檔、多檔、錯序、重複或修改座標均拒絕。產生 manifest 前亦檢查全來源與全部輸出。
- 輸出根目錄必須新建或為空；來源子目錄、CSV／JSON 使用排他建立。既有正式資料、舊輸出不被覆寫。
- CLI 對處理失敗非零退出，留下 `run_audit.json` 與錯誤／可用切片計畫；先前成功來源可能仍留在輸出目錄。此階段沒有宣稱跨來源的成套原子發佈，該工作仍在後續任務。
- 產物驗證：`python tools/validate_jenqwei_dataset_stage17.py --out /tmp/new-stage17-run`，再對其中 `evidence.json` 執行共用 evidence 工具。診斷 PNG 僅供驗收，資料集 CLI 不另產生分析圖。
