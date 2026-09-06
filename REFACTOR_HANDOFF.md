# Python 修正接手狀態

更新日期：2026-09-06。**狀態：第一階段已完成修改、驗證與報告；完整分段分析及大型架構重構待續。**

## 先讀這三份文件

1. [PYTHON_FIX_REPORT_2026-09-06.md](PYTHON_FIX_REPORT_2026-09-06.md)：本輪修改原因、方法、成果、相容性變更與剩餘工作。
2. [PYTHON_FIX_VALIDATION_2026-09-06.json](PYTHON_FIX_VALIDATION_2026-09-06.json)：測試摘要、整合驗證數字與 Python 檔案指紋。
3. [PYTHON_REVIEW_2026-09-06.md](PYTHON_REVIEW_2026-09-06.md)：修正前全部 72 個 Python 檔案盤點、F01–F19 證據與研究方法風險；行號是歷史版本。

## 使用者意圖與授權

使用者要求全面檢查共用程式、過時程式、邏輯問題與重構可能性，已授權直接進行修正並撰寫詳細報告；也要求若本輪未完成，先留下狀態供下一個工作時段接手。可在現有授權範圍繼續程式修正與測試，不需要再次詢問是否開始。

沒有建立自動續跑排程；目前不能保證五小時後自動啟動下一個工作時段。此文件是持久的手動接手紀錄，不代表背景仍有 agent 運行。

## 已完成、不要重做的工作

- 共用秒／微秒契約與分段濾波、重取樣、TFLite extraction、Goertzel window metadata。
- CSV 宣告欄位解析、合併輸入篩選與原子輸出、短 Welch、眼開閉輸出 header。
- Goertzel 品質政策、抽樣容量、MI 參與者篩選、負 delta 圖軸與 CLI 選項。
- TWSE 完整 target horizon 切分、零 Volume 特徵、scaler／模型匯出與 selected-trial counts。
- dataset／summary 檔名識別、Goertzel 快取內容／設定／程式指紋檢查。
- bundle custom marker 功能回收至根目錄 canonical source；build_bundles.py 產生副本，--check 檢查差異。
- 54 個測試通過；84 個 Python 編譯、35 個 help 入口、Pyflakes、diff whitespace、bundle 檢查通過。
- 真實 Hardy 前 10 秒模型結果與修正前完全一致；合成缺口推論時間正確；本機兩組 RNN 小型訓練及模型/scaler reload 通過。

## 下一階段第一個具體任務

**先完成 F03/F04 的真正分段支援，不要直接移除 require_continuous。**

目前 spectral_entropy、特殊 baseline/event 流程及舊 entropy CSV 繪圖遇到缺口會明確報錯。下一步從 spectral_entropy 的普通 band-entropy 路徑開始：

1. 使用 lilia.windowing.window_starts 產生合法窗口，所有指標與 quality 共用同一組起點。
2. 每筆結果保存 start/end index、start/end/center timestamp、source/config 身分；time_s 一律是真實 elapsed time。
3. renderer 從 metadata 對齊資料，不再猜測 CSV 第幾列對應第幾個原始窗口，也不能把缺口窗口刪除後壓縮時間。
4. 以連續資料數值基準，加上缺口、jitter、無效品質、短 segment 案例驗證，通過後才解除該路徑的 guard。
5. 再遷移 MI/event 和 baseline；其中 baseline 不能把不相鄰 1 秒 epochs 拼成 2 秒模型窗。

接續工作依修正報告第 7 節：品質無效狀態與短窗邊界、拆分大型模組、批次錯誤退出／artifact schema、PSD 與 MI 方法一致性。F19 只明示棄用，公開參數尚未刪除。

## 工作區保護

本輪開始前已存在 README.md、merge_subject_csvs.py 的修改，以及未追蹤的 plot_index_vs_raw_bundle/、requirement.txt、twse_index_lstm_rnn.py、twse_tracker_output/。README.md 保持原狀；merge 的整列去重與衝突樣本保留語義必須保留。不要用整體 reset/checkout 清除差異，也不要把所有未追蹤內容都當作本輪新增。

尚未 commit；先看 git status/diff，再讀新增檔案。bundle 是生成副本，請改根目錄／lilia 的主版本後執行建置；不要分別手改兩套程式。不要為了測試覆寫原始錄製或正式研究產物。

## 接手檢查命令

```bash
cd /home/bps-yichin/lilia_analysis
git status --short
python build_bundles.py --check
MPLCONFIGDIR=/tmp/lilia-audit-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
git diff --check
```

依實際下一階段修改選擇測試；沒有新修改或失敗，不需要反覆重跑全部驗證。套件 CLI 使用 `python -m lilia.qeeg`，舊 root 入口 `python qeeg_indices.py` 也保留。

## 暫存資源（不保證跨環境存在）

- `/tmp/lilia-fix-backup/`：開始修正時的來源快照；windowing.py 在快照建立前已新增，因此它並非原始 72 檔的一部分。
- `/tmp/lilia-fix-integration/`：Hardy／缺口 TFLite 輸出、小型市場訓練、模型重新載入、Goertzel 重算／重畫、驗證 log。
- `/tmp/lilia-fix-tests.log`：單獨執行 unittest 的記錄。
- `/tmp/lilia_python_audit_pipeline/`：審查階段的 Hardy 前 10 秒模型輸出，可與修正後比較。

若上述暫存已消失，持久報告與測試程式仍足以繼續，不需要依賴上一輪的聊天內容。

## 下一時段可以使用的接手指令

> 請讀取 REFACTOR_HANDOFF.md、PYTHON_FIX_REPORT_2026-09-06.md 與目前 git diff，延續已授權的 Python 修正工作。優先完成 spectral_entropy 的分段時間與窗口 metadata，驗證後才解除該路徑的 continuity guard。保留既有工作區修改、同步 bundle，並更新修正報告與接手狀態。
