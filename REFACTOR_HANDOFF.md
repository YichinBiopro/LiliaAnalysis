# Python 修正接手狀態

> **日常接手入口已改為 [REFACTOR_STATUS.md](REFACTOR_STATUS.md)。** 本檔保留第十三階段以前的完整交接與歷史證據，按需查閱，不再逐階段累加。下文「下一步 qEEG」為歷史狀態；第十四階段現已完成，當前任務見 [任務單](docs/refactor/TASKS.md)，執行與驗證見 [精簡流程](docs/refactor/WORKFLOW.md)。

更新日期：2026-09-08。**狀態：第十三階段 plot_tyy_meditation.py 已完成；下一步為 lilia.qeeg／qeeg_indices.py CLI。**

## 先讀文件

- [第十三階段修正報告](PYTHON_FIX_REPORT_2026-09-08_STAGE13.md)：TYY 固定 baseline、全 raw 裁切／模型映射、四通道 BP 行為更正與限制。
- [第十三階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE13.json)：目前 Python 指紋、197 tests、舊主函式真實模型基準、完整 TYY 與四個合成案例。

- [第十二階段修正報告](PYTHON_FIX_REPORT_2026-09-08_STAGE12.md)：單事件 zoom、完整 pre-event-rest bins、顯示對齊、hash parser 崩潰修復與限制。
- [第十二階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE12.json)：第十二階段歷史指紋、184 tests、舊連續基準、Hsin 與一般入口逐值核對及缺資料案例。

- [第十一階段修正報告](PYTHON_FIX_REPORT_2026-09-08_STAGE11.md)：跨受試者分段 BP／TFLite、兩組 baseline、缺值圖形與限制。
- [第十一階段驗證](PYTHON_FIX_VALIDATION_2026-09-08_STAGE11.json)：第十一階段歷史指紋、171 tests、連續基準、全部 8 位真實受試者與合成不同模型裁尾窗口。

- [第十階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE10.md)：Hardy_2 五頻帶／時段政策、缺口圖形與限制。
- [第十階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE10.json)：第十階段歷史指紋、155 tests、連續基準、完整 Hardy_2 主版／bundle 與合成污染段。

- [第九階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE9.md)：APP／NUC 分段 MAD／OLA、elapsed 配對、圖表與限制。
- [第九階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE9.json)：第九階段歷史指紋、142 tests、連續基準與真實 APP／NUC 兩案例。

- [第八階段修正報告](PYTHON_FIX_REPORT_2026-09-07_STAGE8.md)：event markers 一般主流程、baseline 政策差異與限制。
- [第八階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE8.json)：第八階段歷史指紋、129 tests、真實模型連續基準與完整 Hardy 兩模式。

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

下列文件保留為歷史；第一至第十二階段指紋不代表目前版本。

1. [PYTHON_FIX_REPORT_2026-09-07.md](PYTHON_FIX_REPORT_2026-09-07.md)：本輪原因、方法、成果與限制。
2. [PYTHON_FIX_VALIDATION_2026-09-07.json](PYTHON_FIX_VALIDATION_2026-09-07.json)：第二階段歷史驗證摘要及 Python 檔案指紋。
3. [PYTHON_FIX_REPORT_2026-09-06.md](PYTHON_FIX_REPORT_2026-09-06.md)：第一階段修正歷史。
4. [PYTHON_REVIEW_2026-09-06.md](PYTHON_REVIEW_2026-09-06.md)：原始 72 個 Python 檔案完整盤點、F01–F19 及研究方法風險；行號是歷史版本。

2026-09-06 的驗證 JSON 是舊版本基準；不可拿其 SHA-256 宣稱目前所有 Python 檔案仍相同。

## 授權與 Git 狀態

使用者已授權全面分析與直接修正 Python scripts，要求記錄原因、方法、成果，並留下後續接手狀態。可繼續既有範圍內的程式修正與測試，不需再次確認是否開始。

第一階段 commit 為 `8c07401`，第二階段已提交為 `c76bc43`。第三階段開始時工作區乾淨，分支為 `spectral-entropy-flow-rework`，與本機 origin tracking ref 相同（本輪未 fetch）。第三至第八階段的程式、測試、fixtures、報告、驗證摘要與 bundle 變更收錄於本次整合提交。各階段報告中的「未 commit／push」是撰寫當時的歷史狀態；目前提交與遠端同步狀態以 Git 為準。

第三至第八階段已整合為 `edd119b`。第九階段開始與結束時，HEAD 與本機 `origin/spectral-entropy-flow-rework` tracking ref 均指向該提交；本階段沒有 fetch，不能據此宣稱已即時查驗 GitHub。先前 push 曾被自動審核拒絕，不能將本機 tracking ref 狀態當作該次 push 成功的證據。

使用者此前的 commit and push 要求針對前幾輪累積修改。第九至十三階段依「繼續下一階段／確認交接並繼續」執行程式修正與測試，五階段修改保留在工作區，未 commit／push。第十一階段接手時程式與 13 項測試已存在但未記入交接，本輪補齊例外處理、圖形邊界、真實驗證與文件。後續發布依當次明確授權及審核結果執行；不要自行重試已被拒絕的操作。

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

第九階段：

- `lilia.comparison` 逐 raw segment 濾波、MAD 插值、500→200 Hz 重採樣及 PyTorch OLA；短段排除保留 epoch／原 segment ID，保留段資料或模型失敗明示報錯。
- APP／NUC 使用各自 elapsed clock，最長共同連續段估單一 ch1 lag，一對一配對保留兩端索引及 residual。這是波形對齊，不是 absolute clock 同步；週期訊號可能有多個相關峰。
- Welch 逐段計算、依子窗口數平均線性 power；各自完整錄製 PSD 範圍沿用舊版。TD、STFT 與 5 秒 qEEG 保留缺口及真實 timestamps，孤立 qEEG 窗口有可見點。
- `lilia.comparison_io` 的 signal／qEEG／alignment CSV + sidecar 核對 raw、模型、設定、原始及模型索引映射。保存修復區間、短段／窗口及失敗 audit。quality 明示 disabled，不與 MAD 修復混淆。
- **142 tests、116 Python 編譯、Pyflakes、bundle／diff check 通過。** 等間距連續 fixture 的真實模型、MAD、PSD、STFT、qEEG 與時間全部最大誤差 0。
- 真實 10 Hz APP／NUC：11725／12000 模型輸出、11／12 qEEG 窗口、lag −32、11725 配對。從兩份真實來源刪列構造不同大／小缺口：10724／10999 輸出、9／10 qEEG、10394 配對分為 4 段；短前綴排除。每案例 5 表重讀與 30 產物 hash 核對通過；STFT／qEEG 目視完成。
- 固定 500 Hz／前四通道、無 quality scorer、單一 lag 與非原子整批發布等限制見報告；尚未完成所有其他入口。

第十階段：

- `lilia.hardy2` 保留每通道五頻帶正規化／median 及獨立四指標公式，使用來源分段 grid、逐段 BP、有效 run 的 5 窗口平滑。每段短尾、非有限段／窗口、filter／metric 失敗均明示。
- 四 period 只納入完整包含的窗口；跨事件邊界 rows 留在時序圖與表，但不計入時段 mean。底色依有效相鄰窗口的實際範圍，不跨缺口或污染窗口。quality 明示 disabled。
- `lilia.hardy2_io` 保存 metrics CSV/meta，核對 raw、完整窗口／段／短尾、通道 median、平滑及 period rows／spans／means。成功／失敗均有 analysis audit，全短／全排除非零退出。
- 新增 `--hardy2-csv`，預設路徑相容，bundle 可指定外部錄製；x 軸限來源範圍，只標示範圍內事件。
- **155 tests、121 Python 編譯、Pyflakes、bundle／diff check 通過。** 連續 raw／BP 五頻帶、指標、平滑及同政策 period mean 最大誤差 0。
- 完整 Hardy_2：958384 樣本、2 段、313.290063 秒缺資料、383 有限窗口、884 未滿窗口尾樣本。最後中心 2225.790063 秒；3 個跨事件窗口明示不納入 period，四 period 分別 149／35／179／17 窗口。分段 BP 與 period 政策造成的真實數值差異另存對照。
- 主版／bundle 真實 metrics 表完全相同；合成短前綴／小大缺口／污染段 9 候選、7 有限。三案例各 14 產物 hash 與 table 重讀完成；真實及合成圖目視通過。

第十一階段：

- `compare_subjects.py` 改用 `lilia.event_qeeg`／`lilia.tflite` 分段 BP、模型裁尾與各自指標 grid；品質按真正 raw 範圍評分，不再截齊列數或讓 unmatched tail 默認合格。
- `lilia.subject_comparison` 保留 iBrain 第一參與事件前完整窗口 baseline、YoGa 前 `max(1,N//5)` 候選指標列 baseline，先選候選再遮罩品質。每通道先算窗口 mean 差，再算通道 mean／population SD。
- `lilia.subject_comparison_io` 保存並核對 raw／model／config、原始與模型窗口、品質索引及所有 baseline／事件候選／接受／排除與 delta。缺 baseline、缺事件、未參與、模型失敗均明示；BP 部分成果保留，CLI 續跑其他受試者並彙總非零失敗。
- 共享 `score_branch` 的單窗口 metric 例外清除部分通道結果並保留失敗列；非有限指標不能接受。圖表區分 NA／NP／OFF／真實零值，全缺值子圖固定受試者軸範圍。
- **171 tests、126 Python 編譯、Pyflakes、bundle／diff check 通過。** 連續 BP／TFLite／raw 品質及兩種 delta 最大絕對誤差均為 0。新完整事件選窗與舊中心選窗差異另留數值，不宣稱政策等價。
- 真實 iBrain 5 位、YoGa 3 位全成功，共 16 表重讀、32 表／sidecar、8 subject audit、12 圖形 hashes 核對。Hardy 6 段、848 窗口／683 合格，最後中心 4791.844551 秒；Jammie 3 段、1128／954。其餘數字見報告。
- 合成短前綴／5 秒段／小大缺口使用真實模型與 scorer，BP 9 窗口、TFLite 8 窗口；後續品質無位移。Session／event 兩案例 4 表重讀與圖形目視完成。
- bundle 同步新增模組與共享修正；既有 manifest 未包含 compare_subjects CLI，沒有自行新增發行入口。

第十二階段：

- `plot_meditation_zoom.py` 共用一般 event markers 的逐來源段 BP／raw 品質與 `pre-event-rest` summary。`lilia.event_zoom` 保存 raw 顯示 runs、缺資料區間、baseline／event 候選與排除、完整／邊界／有效 bins，無空選區 fallback。
- 每個 30 秒 bin 按真正起訖畫矩形；跨事件邊界灰色、沒有完整 bin 留白。raw 逐 run 降採樣保留首尾、逐段斷線／底色；獨立 colorbar 欄讓 raw 與 heatmap 實際時間位置對齊。
- `lilia.event_zoom_io` 的 event_marker_bp／scope=event_zoom 表可重讀，重建來源 grid／品質映射、有效性、整份 summary、顯示索引／缺資料／選區／段短尾。未參與、缺 raw、全短、缺 baseline、全排除保存圖與 audit 並非零退出。新增 `--csv`，沿用 subject schedule。
- 實際來源 hash `68e6869767633…` 使 pandas 2.3.3 自動型別推斷 segfault；單欄子程序重現。共享 `_load_window_table` 強制 source_id／config_id 為文字後正常，另測前導零 hash，不再觸發數值推斷。
- **184 tests、131 Python 編譯、Pyflakes、bundle／diff check 通過。** 舊連續 24 窗口的 BP／raw 品質／absolute／delta heatmap 最大差異 0；bin 中心有意改正 2.5 秒。Hsin 新版與一般完整 pre-event-rest 表／summary 逐值相同。
- 真實 Hsin 1978896 樣本、791 窗口／748 合格、131 bins；meditation 20 完整有效 event bins、5 baseline bins、1 邊界灰色 bin。舊 baseline 及 delta 政策差異另留數值。真實 Color Agility Ladder 明示缺 baseline，保存缺值圖並 exit 1。
- 四個合成案例包含短前綴、8 ms／20.008 秒缺口、短事件與缺 baseline；兩個真實與四個合成 zoom 案例均表重讀、每例 4 產物 hash 核對及圖形檢查。共享模組同步 bundle，既有 manifest 未包含 zoom CLI。

第十三階段：

- 舊 TYY 註解稱 BP 只用 Ch1／Ch2，但舊主函式實跑證實 BP 使用四通道，只有 raw 圖顯示前兩通道。保留實際四通道 BP、四通道模型輸入／兩通道輸出；第十二階段下一步的誤記已更正。
- `lilia.meditation` 按原始微秒半開裁切，逐段 BP、重採樣與 TFLite 完整模型窗／裁尾。BP 保持完整 raw 索引，模型輸出索引另外明示；source crop／原 segment ID／epoch／model-to-raw 映射均保留。
- baseline 保留固定 14:24 前完整 30 秒 bins，載入 14:10–15:19、顯示 15:00–15:19；不套 pre-event-rest，也不加入品質 scorer。quality 明示 disabled，無 baseline 不再 fallback 零。
- `lilia.meditation_io` 重讀核對 source／crop／model／原始與模型 grid、顯示 indices、有限性、baseline／heatmap 與 audit。來源短尾、污染段、模型缺失／推論失敗明示；BP 部分成果保留後非零退出。新增 --csv／--model。
- 真正 30 秒矩形不跨段補色，raw 首尾保留／斷線，colorbar 不改變面板軸寬；原 ±150 µV raw 顯示限保留。
- **197 tests、136 Python 編譯、Pyflakes、bundle／diff check 通過。** 連續 BP／TFLite 指標與 delta 最大誤差 0，真實模型 output 逐值相同。舊 bin 中心有意改正 2.5 秒。
- 完整 TYY：2009328 樣本、1 段；兩分支各 803 有限窗口、133 bins、27 baseline bins、34 顯示 bins。TFLite 803600 輸出／2009 模型窗。新舊真實 delta 差異 0，CSV round-trip 不超過 1.12e-16。
- 真實與 4 合成案例共 9 表重讀、28 產物 hash 核對；合成 crop 從原索引 6100／原 segment 2開始，保留segment 3 與 8 ms／20.008 秒缺口。缺 baseline／缺模型案例明確失敗並保留圖與 audit。共享模組同步 bundle；manifest 未包含 TYY CLI。

## 下一個具體任務

**遷移 `python -m lilia.qeeg`／根目錄 `qeeg_indices.py` 相容 CLI，完成驗證後解除 guard。**

1. 保存目前直接分析單一 raw channel 的連續三頻帶／四指標、窗口中心、CLI summary／PNG 基準。此入口沒有 BP、TFLite、品質 scorer 或 baseline，不要直接套用其他入口的方法。
2. CLI 保留原始整數微秒，不再先轉絕對 float 秒再 round-trip。用 WindowGrid 保留來源 segment ID／原始樣本窗口／真實 elapsed 或明示的絕對時間；確保不跨缺口窗口、保留各段短尾與非有限值排除。
3. `compute_qeeg_indices_windowed` 是共用／相容 helper：盤點 callers，保留無 timestamp 呼叫的既有語義，若增加 timestamps／grid 參數需驗證形狀及設定；不要為 CLI 改變所有既有 callers。
4. 兩面板圖不跨段連線，孤立窗口有可見點。輸出 CSV/meta＋analysis audit，核對來源、channel、設定與全部窗口；全短／全排除／錯誤 channel 明確失敗。品質若未評分明示 disabled，不把有限值稱為品質合格。
5. 完成連續基準、合成缺口／短段／非有限／邊界、完整有缺口 Hardy 真實資料，主版／相容入口／bundle 來源重讀與圖形目視，再移除 guard、更新報告與 bundle。

已完成的普通 entropy、band-event MI、raw／denoise joint-MI、TFLite summary、event markers 一般主流程、APP／NUC CLI、Hardy_2、compare_subjects、meditation zoom 與 TYY meditation 避免重寫。

其後仍有眼開閉與 Jenqwei 的盤點／遷移、品質 scorer 短窗／fallback、baseline 方法整理、大型入口拆分、批次與 artifact 原子發佈、PSD／Goertzel／MI 方法校準、F19 相容介面及整體驗收。不同模式的 baseline／品質與索引空間不能直接混用。

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

- `/tmp/lilia-stage13/real/`：完整 TYY BP／TFLite PNG/SVG、metrics CSV/meta、analysis audit。
- `/tmp/lilia-stage13/gapped/`、`large_gap/`、`no_baseline/`、`missing_model/`：四合成案例，來源為 `synthetic_source.csv`／`large_source.csv`，裁切原始索引／模型對齊與失敗輸出。
- `tests/fixtures/tyy_meditation_continuous_reference.json/.npz`：舊主函式 BP 四通道／真實模型兩通道基準與原始模型輸出。`/tmp/lilia-stage13/capture_before.py`、`plot_tyy_meditation_before.py`、`real_before.json` 保存實跑程序與真實舊數值。
- `/tmp/lilia-stage13/validate_stage13.py`、`evidence.json`：完整來源／模型／產物重讀與新舊比對；重要摘要持久保存於第十三階段驗證 JSON。
- `/tmp/lilia-stage13-all-tests.log`、`/tmp/lilia-stage13-tests.log`、`/tmp/lilia-stage13-real.log`、`/tmp/lilia-stage13-validation.log`。

- `/tmp/lilia-stage12/real/`、`general/`：Hsin meditation zoom 與獨立一般 pre-event-rest 完整圖／表／summary 對照；`real_no_baseline/` 為真實 Color Agility Ladder 缺 baseline 案例。
- `/tmp/lilia-stage12/gapped/`、`large_gap/`、`short_event/`、`no_baseline/`：四個合成案例，來源在 `synthetic_source.csv`／`large_gap_source.csv`。
- `/tmp/lilia-stage12/capture_before.py`、`plot_meditation_zoom_before.py`、`hsin_before.json`：舊主入口實跑基準；連續 fixture 已持久保存於 `tests/fixtures/meditation_zoom_continuous_reference.json`。
- `/tmp/lilia-stage12/validate_stage12.py`、`evidence.json`：完整來源／產物重讀與數值對照；重要摘要保存於第十二階段驗證 JSON。
- `/tmp/lilia-stage12-all-tests.log`、`/tmp/lilia-stage12-tests.log`、`/tmp/lilia-stage12-parser-tests.log`、`/tmp/lilia-stage12-validation-fixed.log`。`/tmp/lilia-stage12-validation.log` 保留修復前 native parser crash traceback，不能當作成功驗證 log。

- `/tmp/lilia-stage11/real/iBrainCenter/`、`real/YoGa/`：全部 8 位受試者的 BP／TFLite CSV/meta、subject/group audit 與 12 張 PNG/SVG。
- `/tmp/lilia-stage11/synthetic/session/`、`synthetic/events/`、`synthetic_source/`：短前綴、不同模型裁尾、小大缺口及未參與／無資料事件案例。
- `/tmp/lilia-stage11/validate_stage11.py`、`evidence.json`：本輪實跑重讀與連續／選窗政策比對腳本及完整證據；重要摘要已持久保存於驗證 JSON。
- `/tmp/lilia-stage11-all-tests.log`、`/tmp/lilia-stage11-tests-current.log`、`/tmp/lilia-stage11-real.log`、`/tmp/lilia-stage11-validation.log`。
- `tests/fixtures/subject_comparison_continuous_reference.json`：第十一階段改前 BP／TFLite／raw 品質、兩種 delta、seed、shape 與程式／模型 hash。

- `/tmp/lilia-stage10/real/`、`/tmp/lilia-stage10/bundle_real/`：Hardy_2 真實資料主版／bundle 的 6 類 PNG/SVG、metrics CSV/meta、analysis audit。
- `/tmp/lilia-stage10/synthetic/` 與 `synthetic_source/`：短前綴／小大缺口／污染段的合成輸出與來源。
- `/tmp/lilia-stage10-all-tests.log`、`/tmp/lilia-stage10-real.log`、`/tmp/lilia-stage10-bundle.log`、`/tmp/lilia-stage10-synthetic.log`。
- `tests/fixtures/hardy2_continuous_reference.json`：第十階段修改前的 raw／BP 五頻帶、四指標、平滑、時段 mean 及程式 hash。
- `/tmp/lilia-stage10/real_before.json`：修改前真實資料數值；重要新舊差異與 period means 已保存於第十階段驗證 JSON。

- `/tmp/lilia-stage9/real_10hz/`、`/tmp/lilia-stage9/gapped_10hz/`：真實 APP／NUC 原始與人工缺口案例的 PNG、CSV/meta、analysis audit。
- `/tmp/lilia-stage9/gapped_source/`：從真實來源刪列產生的測試副本；刪除範圍與 source hashes 見第九階段報告／JSON。
- `/tmp/lilia-stage9-all-tests.log`、`/tmp/lilia-stage9-real.log`、`/tmp/lilia-stage9-gapped.log`。
- `tests/fixtures/app_nuc_continuous_reference.json/.npz`：第九階段修改前的連續濾波／MAD／真實 PyTorch／PSD／STFT／qEEG 數值與模型／程式 hash。

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

> 請讀 REFACTOR_HANDOFF.md 與第十三階段報告，保留工作區第九至十三階段修改，從 lilia.qeeg／qeeg_indices.py CLI 接續。TYY 已驗證原本四通道 BP／兩通道模型、未啟用品質遮罩與固定 14:24 前 baseline，不需重做。qEEG CLI 本身直接分析單一 raw channel，沒有 BP／品質／baseline；先保存連續三頻帶／四指標及 summary，再接上原始微秒窗口與缺口／短尾／非有限排除、不跨段圖形及可重讀輸出。保護 compute_qeeg_indices_windowed 的既有 callers。完成有缺口 Hardy 真實資料與主版／相容／bundle 驗證後再解除 guard。更新報告並同步 bundle；其餘入口／方法校準／工程收尾仍待處理，發布依當次授權執行。
