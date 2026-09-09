# 第十四階段：qEEG CLI 分段時間與可核對輸出

日期：2026-09-08。狀態：完成。範圍：`lilia.qeeg`、相容 CLI、raw qEEG adapter／IO、專屬測試、驗證工具與生成 bundle。

## 本階段變更

- CLI 原本拒絕缺口、將微秒轉絕對 float 秒；現在保留整數微秒，由 [qeeg_raw.py](../../lilia/qeeg_raw.py) 建立原始樣本格點，保留 segment／真實中心時間，排除跨缺口窗口。已解除此 CLI guard。
- 保留直接單一 raw channel、三頻帶半開積分／Welch 與四指標公式；沒有 BP、模型、baseline 或品質評分，`quality_state=disabled`。無 timestamp 的共用 helper 完全保留；已盤點 `data_analysis.py`、根相容入口與套件匯出。
- CLI 的窗口樣本數沿用 `int(win_sec*fs)`；傳入 WindowGrid 的是有效樣本長度。延續原始全域格點，不在段首重啟；未覆蓋段首、短尾、短段及跨缺口候選均有 audit。
- 非有限只排除所選 channel 的污染窗口，候選列保留 NaN／原因，summary 使用有限窗口。全短、全排除、錯誤 channel／設定明確 exit 1 並保存 audit；有合法但全排除窗口時仍保存 CSV／meta／缺值 PNG。
- [qeeg_io.py](../../lilia/qeeg_io.py) 新增 `raw_qeeg` 表與 sidecar，回讀核對來源 hash、channel／政策、每個整數窗口映射、段／排除／summary，並重算全部數值；已接入 evidence reader。
- 兩面板圖使用明示 elapsed 時間及完整來源範圍，缺口／污染列斷線，孤立窗口有點；全排除圖明示無有限窗口。舊絕對 float 軸改為 elapsed，原始時間仍在 metadata。
- 本轮接續已有實作、基準與驗證，目視發現右上圖例遮住末段孤立點、缺值文字與零線重疊；將分段图圖例置於軸外，缺值文字加白底。未傳入 segment IDs 的舊 caller 布局保留；重新建置 bundle 及完成最終實跑。

## 驗證

- [最終完整檢查](validation/stage14/checks.json)：**221 tests、0 skipped、144 Python 編譯**，Pyflakes／bundle／diff 通過。新增 15 項專屬測試；[完整 log 已持久保存](validation/stage14/logs/checks)，原執行目錄 `/tmp/lilia-refactor-3xw1as5v/`。
- 修改前 fixture：[合成連續基準](../../tests/fixtures/qeeg_cli_continuous_reference.json)／同名 NPZ，保存舊程式 hash、seed／raw、5 秒／1.0019 秒／9 樣本窗口、中心時間、summary 與 PNG hash；[真實 Hardy 基準](../../tests/fixtures/qeeg_hardy_reference.json)／同名 NPZ 保存舊 helper 數值及來源 hash。
- 同步 summary 格式與無 timestamp helper 數值通過；三種合成連續 CLI 的 CSV 回讀最大誤差 **4.45e-16 以下**，真實連續 **1.12e-16 以下**，容許誤差 `rtol=atol=1e-12`；整數索引／微秒精確核對。
- 完整 Hardy：**2,126,712 樣本、6 段、845 個有限窗口**，排除 5 個跨缺口原格點窗口，共 14,212 個未覆蓋段首／尾樣本。最後中心 elapsed **4788.420551 秒**，舊樣本計數為 4247.5 秒，差 **540.920551 秒**。
- 三個 Hardy 入口的表與 analysis 完全相同；和修改前 helper 相同 raw 窗口的 CSV 回讀最大誤差 **2.23e-16 以下**。舊 CLI 對缺口會失敗；850 個串接 raw helper 窗口僅作診斷，不能當合法舊 CLI 結果。移除 5 個跨缺口窗口造成的 summary 差異另存 [數值摘要](validation/stage14_validation.json)。
- [最終產物 evidence](validation/stage14/evidence_checks.json)：**94 checks = 12 張表重讀＋75 個 hash＋7 組 NPZ 比較**；[manifest](validation/stage14/evidence.json) 固定生成時 hashes。含三入口 Hardy／合成缺口、channel 2、連續／短窗及四個預期失敗案例，無未解失敗；[最終數值與逐窗口 audit](validation/stage14/analysis.json)。
- 目視：此前檢查連續舊／新曲線、完整 Hardy、合成污染、channel 2 與全排除圖，並修正空圖時間範圍。最終圖例／文字修正後再次直接檢視 Hardy、合成孤立點與全排除三張圖；末段點完整可見、缺口不連線、缺值明示。[目視紀錄與三張原尺寸圖](validation/stage14/visual_review.json) 已持久保存，複本 hash 與生成 audit 一致。
- 重現實跑：`python tools/validate_qeeg_stage14.py --out /tmp/lilia-stage14-new`（須新目錄），再 `python tools/refactor_check.py evidence /tmp/lilia-stage14-new/evidence.json`。最終產物在 `/tmp/lilia-stage14-review/`；[完整 CLI log](validation/stage14/logs/cli) 已持久保存。暫存可能消失，可由 fixtures／腳本重現。
- `validation/stage14_*.json` 與 `/tmp/lilia-stage14/final/` 保留為圖例修正前的歷史驗證；最終證據使用 `validation/stage14/`。舊基準程式 SHA-256 `c98f2937f86b567cb86b9a7796a70ac9b203d5a3ea14b8519d4f21f8e4efb0e6` 已核對為 `28e0086:lilia/qeeg.py`。

## 未解問題與下一步

- 無新增阻塞。沿用 `>3` nominal periods 缺口門檻，原格點政策會排除部分段首；有限數值不等於品質合格。全套產物的原子發佈、既有套件 eager import 所致 `python -m` runpy warning 留待工程收尾。
- 下一步：[任務 15：眼開閉](TASKS.md#stage-15)，先盤點入口與真實 PyTorch 模型、保存舊連續模型基準。第十四階段未變更模型、原始錄製或歷史研究產物。
- Git：HEAD `28e0086`；本輪接手時已有第十四階段工作區修改，已保留並接續。本階段尚未 commit／push。
