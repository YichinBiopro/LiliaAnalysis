# Python 修正接手狀態

更新日期：2026-09-07。**狀態：第八階段 plot_event_markers 一般主流程的分段 BP／TFLite、raw 品質映射、事件 baseline 與圖表已完成；下一步為 data_analysis.py CLI 分段遷移。**

## 先讀文件

- [第八階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE8.md)：event markers 一般主流程、baseline 政策差異與限制。
- [第八階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE8.json)：目前 Python 指紋、129 tests、真實模型連續基準與完整 Hardy 兩模式。

- [第七階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE7.md)：普通 entropy state 的窗口政策、連續數值、排除紀錄與限制。
- [第七階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE7.json)：第七階段歷史指紋、116 tests、連續基準與完整 Hardy 兩模式。

- [第六階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE6.md)：TFLite 逐段裁尾、baseline 統計單位變更與限制。
- [第六階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE6.json)：第六階段歷史指紋、106 tests、連續模型／baseline 對照與完整 Hardy。

- [第五階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE5.md)：denoise 重採樣錯誤、模型時間軸、短段政策與限制。
- [第五階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE5.json)：第五階段歷史指紋、97 tests 與完整 Hardy 真實模型數值。

- [第四階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE4.md)：本輪 joint-MI 遷移、surrogate 語義與限制。
- [第四階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE4.json)：第四階段歷史指紋、87 個測試與完整 Hardy 數值。

- [第三階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE3.md)：第三階段歷史修改、方法變更與限制。
- [第三階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE3.json)：第三階段歷史指紋、76 個測試與完整 Hardy 數值。

下列文件保留為歷史；第一至第七階段指紋不代表目前版本。

1. [PYTHON_FIX_REPORT_2026-09-07.md](PYTHON_FIX_REPORT_2026-09-07.md)：本輪原因、方法、成果與限制。
2. [PYTHON_FIX_VALIDATION_2026-09-07.json](PYTHON_FIX_VALIDATION_2026-09-07.json)：第二階段歷史驗證摘要及 Python 檔案指紋。
3. [PYTHON_FIX_REPORT_2026-09-06.md](PYTHON_FIX_REPORT_2026-09-06.md)：第一階段修正歷史。
4. [PYTHON_REVIEW_2026-09-06.md](PYTHON_REVIEW_2026-09-06.md)：原始 72 個 Python 檔案完整盤點、F01–F19 及研究方法風險；行號是歷史版本。

2026-09-06 的驗證 JSON 是舊版本基準；不可拿其 SHA-256 宣稱目前所有 Python 檔案仍相同。

## 授權與 Git 狀態

使用者已授權全面分析與直接修正 Python scripts，要求記錄原因、方法、成果，並留下後續接手狀態。可繼續既有範圍內的程式修正與測試，不需再次確認是否開始。

第一階段 commit 為 `8c07401`，第二階段已提交為 `c76bc43`。第三階段開始時工作區乾淨，分支為 `spectral-entropy-flow-rework`，與本機 origin tracking ref 相同（本輪未 fetch）。第三至第八階段的程式、測試、fixtures、報告、驗證摘要與 bundle 變更收錄於本次整合提交。各階段報告中的「未 commit／push」是撰寫當時的歷史狀態；目前提交與遠端同步狀態以 Git 為準。

使用者於本次對話已明確要求將這幾輪修改 commit and push，授權將目前累積變更推送至既有遠端 `https://github.com/YichinBiopro/LiliaAnalysis.git` 的 `spectral-entropy-flow-rework`。先前 push 審核拒絕保留為歷史；這次依新授權正常提交及推送，不使用 force push。

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

第三階段：

- `lilia.event_windows.select_event_windows`：真實 UTC timestamps 半開 pre/post 區間，完整位於同一連續段才接受，明示排除原因。
- `spectral_entropy --band-event-mi`／`joint_mi.py per-event`：依每個區間長度選事件、逐段 front-end／bandpass／Hilbert，已解除這兩條路徑的 continuity guard。
- 輸出 `*_band_event_mi_events.csv`／`*_per_event_mi_events.csv`，保留區間索引、時間、segment、channel 與排除原因；全數排除仍保存審核紀錄。
- `compute_event_pre_onset_joint_mi` 支援實際 timestamps，要求完整區間；非 denoise caller 傳入 raw 時间並輸出 audit。joint-MI 整體 guard 仍在。
- 品質 scorer 尚未用於 band-event MI，明示 `Quality_State=disabled`；NaN/Inf 不是合格品質資料，受污染 channel/segment 會排除。
- **76 tests、90 Python 編譯、Pyflakes、bundle 與 diff check 通過**。修改前連續 MI／surrogate 基準在 `1e-12` 內一致。
- 完整 Hardy pooled 得 8 列、逐事件得 64 列，均完成圖表及 64 列事件審核；5 次 surrogate 僅供流程驗證。缺口後舊 onset 索引最多偏移 49,664 樣本。

第四階段：

- 非 denoise 的 `--joint-mi` 共用 WindowGrid，兩 channels、quality 與真實窗口 metadata 對齊；只解除這條已驗證模式的 guard。
- `lilia.entropy_io.write_joint_mi_table`／`load_joint_mi_table`：`kind=joint_mi`，核對來源、ordered channel pair、設定及全部原始窗口，與 entropy 共用驗證內部實作但不能混用 kind。
- MI 品質 NaN/低分明確遮罩、保留窗口；缺 scorer 或列數不符失敗，`--no-quality-mask` 明示 disabled。
- 整體 histogram 使用可提供 grid 窗口的段內成對有限樣本，每筆一次，保留段末餘樣本；summary／pre-onset histogram 明示品質未遮罩。
- surrogate 在各連續 run 內獨立 circular shift，再估計相同 pooled MI；不跨段交換樣本。短於 16 樣本的 run 使 null 明確停用，單段 RNG/數值保留。
- MI series 按有效 run 平滑，static/interactive peri-event 不跨缺口或品質 NaN 插值；保留 run 邊界與繪圖斷點。
- **87 tests、91 Python 編譯、Pyflakes、bundle 與 diff check 通過**。joint histogram、windowed MI、quality、surrogate 與改前連續基準在 `1e-12` 內一致；另保存新舊兩段 null 的數值對照。
- 完整 Hardy 得 2,121 窗口、1,975 品質通過／146 遮罩，最後真實中心 4791.920551 秒；完成 8 活動、五類 PNG/SVG、peri-event HTML、CSV/metadata 及來源／窗口重讀核對。5 次 surrogate 僅供流程驗證。

第五階段：

- 修正舊 `denoise_channels` 把 sample index 當秒，致 500→200 Hz 未執行却誤報 200 Hz 的錯誤。保存舊 1503×2 與正確 602×2 參考；新版和正確重採樣＋舊 OLA 真實模型輸出逐值相同。
- `lilia.neural`：每原始 segment 分別濾波／polyphase／PyTorch 400/200 overlap-add，至少400個真實重採樣樣本才接受，保存段與模型窗口映射。
- `data_analysis.apply_filters` 接受實際 fs；`run_model` 拒絕不完整模型窗口、錯誤 shape／非有限 output／無權重位置，修正短輸入可能被反向裁切的問題。
- WindowGrid 支援原始 segment ID 與原 epoch。Denoised MI 每段重啟輸出格點，timestamp 由原始 fractional sample position 對應，不能把模型輸出索引當 raw 索引。保留降採樣後的小缺口，區間終點不超出來源段。
- `lilia.neural_io`：`kind=denoised_joint_mi`／`index_space=resampled_model_output`，核對 raw、inference mapping、MI grid 與可選 checkpoint。成功 CLI 保存 inference segments CSV。
- 只解除 `--joint-mi --denoise` guard；品質明示 disabled，`--no-bandpass` 與 denoise 不相容。`data_analysis.py` 本身 CLI、TFLite／baseline guard 未移除。
- **97 tests、96 Python 編譯、Pyflakes、bundle／diff check 通過**。完整 Hardy：6 段、4,257 模型窗口、850,686 輸出樣本、2,125 MI 窗口，最後中心 4793.344551 秒，完成五類圖、HTML、CSV/metadata 與來源／checkpoint 重讀核對。

第六階段：

- `lilia.tflite` 保存 raw segment／重採樣／逐段裁尾／完整 400 點模型窗口映射。前後陣列完全對齊，只在原始段內推論；不相鄰完整窗口打包不會形成跨段模型窗口。
- 上游重採樣可回傳原 segment ID，extract pipeline 已傳入 TFLite adapter，小缺口不因降採樣遺失。
- TFLite summary baseline 從已推論的完整 2 秒窗口抽樣，原始每 1 秒子段都需通過 QC／ADC；預設 10 秒選 5 窗口，逐窗口計算 qEEG 再平均。不再串接不相鄰 1 秒 raw epochs 交給模型／Welch。
- 品質前後與 qEEG 共用每段 5 秒 grid、兩通道 median；NaN 明示遮罩。趨勢／平滑／30 秒熱圖保留缺口，熱圖按真實矩形中點判定事件並保存邊界及 metric rows。
- `lilia.tflite_io`：`kind=tflite_qeeg`／`index_space=retained_tflite_output`，驗證 raw／模型 hash、完整 inference mapping 與 qEEG grid。另存 baseline catalog／抽樣／排除 audit。CLI 部分失敗以非零退出。
- **106 tests、101 Python 編譯、Pyflakes、bundle／diff check 通過**。真實連續 TFLite 模型輸出和舊版逐值相同；baseline 抽樣單位／PSD 聚合有意改變，數值差異已記錄，不宣稱等價。
- 完整 Hardy：6 段、2,125 模型窗口、850,000 保留輸出樣本、686 裁尾樣本；848 qEEG 窗口，557 QC 合格，最後中心 4791.844551 秒。session 與 5 個 pre-event baseline 成功；另 3 個 pre-event baseline 各只有 4 個合格窗口而明示排除。兩類 PNG/SVG、CSV/meta/audit 與重讀來源核對完成。

第七階段：

- `lilia.state_windows` 按原始 epoch + 秒數建立半開區間，每個區間／原始 segment 交集從首個樣本重新起窗；只納入完整窗口，保留尾端不足與缺資料時間 audit。
- `spectral_entropy --baseline/--event` 只對有完整候選窗口的來源段濾波。普通模式分析指定 channel，clean 的有限值／ADC／品質檢查仍用全部 channels；污染段、短段不能濾波、品質 NaN／錯誤／低分均有明確排除紀錄。
- `collect_clean_epochs` 共用分段 selector；保留逐 epoch PSD、等權平均與 separate per-window entropy。clean 若指定不同於 win 的 step，明確拒絕，避免默默忽略。
- `compare_baseline_event(..., time_us=...)` 使用真實時間。新半開邊界取代 nearest-sample rounding；連續整齊邊界與真實品質基準最大誤差 0，非整數邊界的起點／納入數量／數值差異另留對照。
- `lilia.state_entropy_io` 輸出 `kind=state_band_entropy`／`index_space=raw_samples` 的 summary CSV + metadata，保存所有候選與排除／缺口／QC／per-window entropy。重讀核對來源、設定、候選窗口與缺資料範圍；兩個 state 都審核後才判定成功，全數排除仍保存 audit 並非零退出。
- **116 tests、106 Python 編譯、Pyflakes、bundle／diff check 通過**。完整 Hardy 6 段，普通 2,125 窗口；clean 1,664 合格、108 低品質、353 飽和，另 4 個尾端不足。最後接受窗口中心 4793.344551 秒；兩模式 summary／meta 與來源重讀完成。

第八階段：

- `lilia.event_qeeg`：一般 `plot_subject` 分段 BP、BP 品質／qEEG 同樣本窗口、TFLite 沿用完整模型窗口 timeline，品質由每個模型 metric 實際時間回找 raw 區間。保留小缺口原始 segment ID。
- 品質 NaN／Inf／shape／例外明示排除，保留所有窗口。trend 依有效 run 平滑；raw 顯示保留段首尾並斷線；30 秒 heatmap 保存／繪製真實矩形邊界，不跨段著色。
- session-start 的 event bars 與 heatmap 統一使用第一參與活動前的 baseline；pre-event-rest 缺休息時不 fallback／不填 0。baseline／event 需完整窗口或 bin 包含，channel SD 與不同聚合方法明示。連續無事件基準保留；事件政策改變另存舊程式實跑對照。
- `lilia.event_qeeg_io` 保存 `event_marker_bp`／`event_marker_tflite` CSV + sidecar，核對 raw／model／原始及模型 grid、quality raw indices、heatmap／baseline rows。全排除保留 audit；指定模型失敗保存 BP partial outputs 後報錯。一般 CLI 逐受試者續跑並彙總非零失敗。
- **129 tests、111 Python 編譯、Pyflakes、bundle／diff check 通過。** 完整 Hardy 兩種 baseline 模式均實際 BP＋TFLite，各 848 窗口、683 合格／165 低品質、139 heatmap bins。最後中心 4791.844551 秒。pre-event-rest 的 Color Agility Ladder 無 baseline，明示排除；PNG/SVG、CSV/meta/audit 與來源重讀完成。
- 未遷移 `--hardy2-analysis` 特殊分支及其他 legacy helper callers；不要據一般主流程完成而宣稱所有入口已支援缺口。

## 下一個具體任務

**遷移 `data_analysis.py` CLI 的完整分段流程，驗證後才移除其 guard。**

1. 保存目前連續資料主入口的模型、品質、時間軸與輸出數值基準；優先檢查載入秒／微秒、濾波、重採樣及模型推論呼叫順序。
2. 重用已完成的 `lilia.neural` timeline／來源映射，逐原始 segment 濾波、重採樣、推論；不要重新拼接不相鄰樣本，也不要混用 PyTorch OLA 與 TFLite 裁尾政策。
3. 下游圖表與匯出使用正確模型輸出索引／timestamps，品質失敗、短段及全數排除明示；保留來源與模型 hash 驗證。
4. 完成連續基準、合成缺口／邊界、真實錄製與輸出重讀後，才移除 CLI guard；更新報告及 bundle。

已完成的普通 entropy 時序／baseline-event／clean、band-event MI、raw joint-MI、denoise MI、TFLite summary、event markers 一般主流程避免重写。

其後仍有其他入口（含 `--hardy2-analysis`、compare_subjects、meditation 圖、qEEG CLI、眼開閉與 Jenqwei）的盤點／遷移、品質 scorer 短窗／fallback、baseline 方法整理、大型入口拆分、批次與 artifact 原子發佈、PSD／Goertzel／MI 方法校準、F19 相容介面及整體驗收。不同模式的 baseline／品質與索引空間不能直接混用。

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

- `/tmp/lilia-stage8/hardy/`：session-start／pre-event-rest 的 BP＋TFLite PNG/SVG、metrics CSV/meta、analysis audit。
- `/tmp/lilia-stage8-all-tests.log`、`/tmp/lilia-stage8-hardy.log`、`/tmp/lilia-stage8/hardy_validation.json`。
- `tests/fixtures/event_qeeg_continuous_reference.json`：修改前真實連續 BP／TFLite／quality／summary 數值與程式／模型 hash。
- `tests/fixtures/event_qeeg_policy_reference.json`：舊主入口實跑與新 session-start／無事前休息政策差異。

- `/tmp/lilia-stage7/hardy/`：普通／clean baseline-event summary CSV + sidecar，全候選／排除 audit。
- `/tmp/lilia-stage7-all-tests.log`、`/tmp/lilia-stage7-hardy.log`、`/tmp/lilia-stage7/hardy_validation.json`。
- `tests/fixtures/state_entropy_continuous_reference.json`：修改前連續普通／clean PSD、entropy、品質篩選數量與程式 hash。

- `/tmp/lilia-stage6/hardy/`：session／pre-event PNG/SVG、metrics CSV/meta、完整推論／baseline／熱圖 audit。
- `/tmp/lilia-stage6-all-tests.log`、`/tmp/lilia-stage6-hardy.log`、`/tmp/lilia-stage6/continuous_comparison.json`。
- `tests/fixtures/tflite_continuous_reference.json/.npz`：修改前連續真實 TFLite 輸出、舊 baseline 數值／抽樣、模型 hash。

- `/tmp/lilia-stage5/hardy/` 與 `/tmp/lilia-stage5-all-tests.log`、`/tmp/lilia-stage5-hardy.log`：完整真實模型推論與測試。
- `tests/fixtures/denoise_continuous_reference.json`、`denoise_continuous_reference.npz`（錯誤舊路徑）、`denoise_resampled_reference.npz`（正確重採樣＋舊OLA）。

- `/tmp/lilia-stage4/hardy/`：joint-MI 五類圖、互動 HTML、summary/events/audit/timeseries/metadata。
- `/tmp/lilia-stage4-all-tests.log`、`/tmp/lilia-stage4-hardy.log`。
- `tests/fixtures/joint_mi_continuous_reference.json`：第四階段修改前的連續 joint-MI／quality／surrogate 基準，保存來源 Python SHA-256。

- `/tmp/lilia-stage3/`：完整 Hardy pooled／逐事件 CSV、audit、PNG/SVG。
- `/tmp/lilia-stage3-all-tests.log`、`/tmp/lilia-stage3-hardy.log`、`/tmp/lilia-stage3-per-event.log`。
- `tests/fixtures/band_event_continuous_reference.json`：第三階段修改前 `c76bc43` 的連續 MI 基準。

- `/tmp/lilia-stage2/`：完整 Hardy、合成缺口、主版與 bundle 繪圖結果。
- `/tmp/lilia-stage2-all-tests.log`、`/tmp/lilia-stage2-tests.log`：全套與新增測試記錄。
- `/tmp/lilia-stage2-hardy.log`、`/tmp/lilia-stage2-hardy-event.log`：真實資料 CLI 記錄。
- `tests/fixtures/entropy_continuous_reference.json`：持久保存的第一階段連續資料數值基準，包含 seed、shape 與 reference commit。
- 第一階段 `/tmp/lilia-fix-backup/`、`/tmp/lilia-fix-integration/`、`/tmp/lilia_python_audit_pipeline/` 若尚存在，可用於額外追查；不保證跨環境保留。

重要驗證數字已記錄在本輪報告與 JSON；即使暫存消失，仍可依測試與持久文件接手。

## 下一時段接手指令

> 請讀 REFACTOR_HANDOFF.md 與第八階段報告，從已整合的第三至第八階段版本接續，並保留任何新的工作區修改。plot_event_markers 一般主流程的分段 BP／TFLite、raw 品質映射、事件 baseline、缺口圖形及完整 Hardy 已驗證；接著處理 data_analysis.py CLI，沿用 lilia.neural 的逐段重採樣／OLA timeline，保存連續基準、驗證後才移除 guard。特殊入口／品質 scorer／方法校準與整庫工程收尾仍未完成。更新報告並同步 bundle；後續發布依當次使用者授權範圍執行。
