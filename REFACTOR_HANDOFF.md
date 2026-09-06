# Python 修正接手狀態

更新日期：2026-09-07。**狀態：第二階段普通 entropy 分段分析、窗口 metadata 與下游繪圖已完成驗證；獨立 MI／baseline 模式及大型架構重構待續。**

## 先讀文件

1. [PYTHON_FIX_REPORT_2026-09-07.md](PYTHON_FIX_REPORT_2026-09-07.md)：本輪原因、方法、成果與限制。
2. [PYTHON_FIX_VALIDATION_2026-09-07.json](PYTHON_FIX_VALIDATION_2026-09-07.json)：目前版本驗證摘要及 Python 檔案指紋。
3. [PYTHON_FIX_REPORT_2026-09-06.md](PYTHON_FIX_REPORT_2026-09-06.md)：第一階段修正歷史。
4. [PYTHON_REVIEW_2026-09-06.md](PYTHON_REVIEW_2026-09-06.md)：原始 72 個 Python 檔案完整盤點、F01–F19 及研究方法風險；行號是歷史版本。

2026-09-06 的驗證 JSON 是舊版本基準；不可拿其 SHA-256 宣稱目前所有 Python 檔案仍相同。

## 授權與 Git 狀態

使用者已授權全面分析與直接修正 Python scripts，要求記錄原因、方法、成果，並留下後續接手狀態。可繼續既有範圍內的程式修正與測試，不需再次確認是否開始。

第一階段已建立 commit `8c07401`；開始第二階段時工作區乾淨，分支為 `spectral-entropy-flow-rework`，比 origin 同名分支領先 1 個 commit。前一版文件「尚未 commit」的描述已過時。本輪第二階段修改尚未 commit。

先前 push 被自動審核拒絕：要求明確授權把包含程式、報告、TWSE 模型的 payload 推送至 `https://github.com/YichinBiopro/LiliaAnalysis.git` 的 `spectral-entropy-flow-rework`。使用者這輪要求的是繼續修正，沒有確認那個 push 問題，因此本輪沒有重試；不要繞過審核。

沒有自動續跑排程，也不能保證五小時後自動啟動下一個工作時段。此文件供手動接手。

## 已完成，避免重做

第一階段：時間單位與分段 core、CSV schema／merge、短 Welch、眼開閉 header、Goertzel 品質與抽樣政策、TWSE label 切分與 scaler 匯出、已知 CLI／圖表錯誤、快取／產物識別、bundle canonical 建置，54 個測試通過。

第二階段：

- `lilia.windowing.WindowGrid`／`build_window_grid`：共同原始樣本格點、排除跨缺口窗口、保留真實中心 timestamps。
- 普通 spectral_entropy、`--sync-pair` 與 quality 使用同一個 grid；先逐段濾波，沒有合法窗口的短段不參與運算。
- `lilia.entropy_io`：輸出及驗證 CSV + `.csv.meta.json`，逐列記錄原始索引／時間／segment/source/config；讀取時核對 raw 內容與重建窗口。
- NaN 品質明確遮罩數值但不刪窗口列；未評分狀態明示，metadata 指標圖要明確 `--quality-threshold -1` 才跳過品質遮罩。
- entropy、sync、composition、focus/relax 以及 session/event/absolute/split 指標圖不跨缺口連線、平滑；保留缺口後首個有效窗口。raw 顯示降採樣也保留各段首尾。
- 新版 entropy CSV 支援有缺口的 raw；沒有 metadata 的舊 CSV 仍保留 continuity guard，不猜測對齊。
- 66 個測試通過（本輪新增 12 個），87 個 Python 編譯、Pyflakes、diff whitespace、bundle 一致性通過。
- 完整 Hardy：2,126,712 樣本、6 段，2,121 個合法窗口，1,944 個品質合格；最後中心真實時間 4791.920551 秒，樣本計數時間只有 4251 秒，相差 540.920551 秒。
- 完整 Hardy 的四種 entropy 圖與指定活動的下游圖／summary 已執行；合成缺口案例完成 entropy+sync、主版／bundle session 繪圖。

## 下一個具體任務

**遷移尚未支援分段的獨立 MI／事件與 baseline 流程；不要直接移除 guard。**

優先閱讀：

- `spectral_entropy.py`：`_run_joint_mi_mode`、`compute_joint_mi_windowed`、`compute_event_pre_onset_joint_mi`、`_resolve_event_onsets`、`_run_band_event_mi_mode`、`_run_baseline_event_mode`。
- `joint_mi.py`：`resolve_events`、`analyze_subject`。
- `plot_tflite_summary.py`：隨機 baseline epochs 的拼接／推論。

建議順序與驗收：

1. 先遷移不需神經網路的 event onset 定位：以真實 timestamps 定位半開事件區間，不能用 elapsed_seconds × fs。每個候選 pre/post 區間要確認完整落在連續段內；不足或跨缺口則明示排除原因。
2. 頻帶包絡與 Hilbert/filter 運算也要逐段；不能先跨缺口濾波，再只修事件索引。
3. 獨立 joint-MI 時序可延用 WindowGrid，所有 channels、quality、metadata 同一組窗口；推論型路徑要使用實際保留的 TFLite 時間軸。
4. baseline 不可把不相鄰的 1 秒 epochs 拼成 2 秒模型窗口。優先考慮先在原連續段推論，再選完整輸出窗口；這會改變統計單位，須記錄與比較。
5. 加入連續資料基準、缺口位於 onset/pre/post 的案例、區間不足、未參與活動與品質無效案例；通過後只解除該已遷移模式的 guard。

普通 band-entropy 已完成上述分段視窗遷移，不要從頭重寫。主版／bundle renderer 使用 `lilia.entropy_io.load_entropy_table`；新 CSV 的來源與窗口核對不可為方便而移除。

其後仍需：品質 scorer 短窗末端／失敗 fallback 的語義、拆分 spectral_entropy/plot_event_markers、大型批次錯誤退出與 artifact 原子發佈、PSD/Goertzel/MI 方法一致性。F19 公開參數僅標記棄用，尚未移除。

## 工作區與相容性保護

先看 git status/diff，保留使用者及前輪修改。第一階段 commit 已含既有 README、TWSE 腳本／輸出與 bundle；不要把它們當作本輪生成的可刪除檔案。

bundle 是生成副本：改根目錄主程式／lilia 後跑 `python build_bundles.py`，不要分別手改兩份 Python。README 與資料不會被建置工具覆蓋。不要為測試覆寫正式錄製、舊研究圖表或模型。

新版 CSV 與 `.csv.meta.json` 必須一起保存。模型／分析方法的改變要另留數值對照，不能只用 help／編譯通過宣稱正確。降低繪圖 threshold 也不能恢復上游已設為 NaN 的指標；需要重新分析才能重新取得那些數值。

## 接手檢查

```bash
cd /home/bps-yichin/lilia_analysis
git status --short --branch
python build_bundles.py --check
MPLCONFIGDIR=/tmp/lilia-audit-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
git diff --check
```

依下一階段實際修改選測試；沒有新修改或失敗，不需要反覆重跑全部驗證。套件 qEEG CLI 用 `python -m lilia.qeeg` 或根目錄相容入口 `qeeg_indices.py`。

## 暫存與持久資源

- `/tmp/lilia-stage2/`：完整 Hardy、合成缺口、主版與 bundle 繪圖結果。
- `/tmp/lilia-stage2-all-tests.log`、`/tmp/lilia-stage2-tests.log`：全套與新增測試記錄。
- `/tmp/lilia-stage2-hardy.log`、`/tmp/lilia-stage2-hardy-event.log`：真實資料 CLI 記錄。
- `tests/fixtures/entropy_continuous_reference.json`：持久保存的第一階段連續資料數值基準，包含 seed、shape 與 reference commit。
- 第一階段 `/tmp/lilia-fix-backup/`、`/tmp/lilia-fix-integration/`、`/tmp/lilia_python_audit_pipeline/` 若尚存在，可用於額外追查；不保證跨環境保留。

重要驗證數字已記錄在本輪報告與 JSON；即使暫存消失，仍可依測試與持久文件接手。

## 下一時段接手指令

> 請讀 REFACTOR_HANDOFF.md 與第二階段修正報告，保留現有工作區。普通 entropy 的 WindowGrid／metadata 遷移已完成；接著處理獨立 MI/event 的真實時間區間、逐段頻帶處理及排除原因。以連續資料基準與缺口案例驗證後才移除對應 guard，更新報告並同步 bundle。尚未取得先前 push 審核要求的明確確認，不要自行重試。
