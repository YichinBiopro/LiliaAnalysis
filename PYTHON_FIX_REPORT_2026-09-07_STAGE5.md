# Python 第五階段修正報告 — 2026-09-07

本輪完成 **PyTorch `--joint-mi --denoise` 的逐段濾波、重採樣、推論與實際輸出時間軸**，使用本機真實 TinyUNetV4 checkpoint 驗證後，解除此模式的 continuity guard。TFLite 與 baseline 模式未在本輪解除 guard，`data_analysis.py` 自身的 APP/NUC CLI 也仍要求連續錄製。

開始時 HEAD 為 `c76bc43`，保留未提交的第三、四階段修改；本輪未 commit、fetch 或 push。最新數值與 Python 指紋見 `PYTHON_FIX_VALIDATION_2026-09-07_STAGE5.json`，先前驗證檔只代表歷史版本。

## 1. 接手時發現的實際錯誤

### Denoise 的 500→200 Hz 根本沒有執行

舊 `spectral_entropy.denoise_channels` 建立 `np.arange(N)` 作為 `time_s`，送入 `data_analysis.downsample_data`。秒數 adapter 因而把每個樣本看成相隔一秒，按缺口逐個樣本重採樣，每段一個樣本仍輸出一筆。結果保留原本 N 筆，却宣稱模型輸入／輸出是 200 Hz，連 MI 時間軸也跟著錯誤拉長。

固定 seed=531、1503×4、500 Hz 的真實模型舊基準得到 **1503×2**；正確 polyphase 2/5 應得到 **602×2**。本輪保留舊結果及其來源程式／checkpoint／外部 architecture SHA-256，不能用「連續結果應完全不变」保留這個錯誤。

另建立獨立參考：正確的 `scipy.signal.resample_poly(filtered, 2, 5)`，接上 `HEAD` 的舊 `run_model`。新版與這個參考在本機 CPU **逐值一致，max absolute error=0**。回歸測試容差是 `rtol=atol=1e-6`，供同一 checkpoint 在 CPU 上重現。

驗證 JSON 另保存舊／新模型結果在共同物理時間上的 RMSE：舊輸出按其實際 500 Hz 原始樣本位置線性插值到新的 200 Hz 格點，只比較不需外推的 601 點。這是修正前後差異，不是相對乾淨 EEG ground truth 的準確度評估。

### 濾波固定 fs 與短輸入裁切

`data_analysis.apply_filters` 原本固定使用全域 FS=500，即使 denoise caller 指定其他 fs。現在接受 fs 參數，預設值不變，逐段推論傳入實際 fs。

原 `run_model` 在 N 小於 hop 時只有 N 筆可供鏡像 padding，卻仍從固定 hop 裁切。以 100 筆 ramp 和 identity 模型驗證，輸入 `0…99` 會回傳 `99…0`。新版要求至少一個完整模型窗口（預設 400 筆），明確拒絕不足輸入，並驗證 window/hop、模型回傳 shape、有限值與輸出位置確實有 overlap-add 權重。這會改變其他直接使用 `run_model` 的短資料呼叫：它們現在報錯，不再接受不完整或錯位推論。

## 2. 來源段、模型窗口與輸出時間

新增 `lilia.neural`：

- `build_inference_timeline`：按原始 timestamps 分段，記錄 raw 半開索引、原始時間界限、source segment ID、polyphase 整數比例、resampled sample count 與保留／排除原因。
- 每段至少有 **400 個真實重採樣樣本**才推論；這是本輪明訂的納入政策。短段不濾波、不推論，不把它補成全長模型輸入或與另一段拼接。
- `denoise_with_time`：每個保留段分別執行 bandpass→notch→bandstop、polyphase、400/200 的 Hann overlap-add。checkpoint 載入一次，各段分別推論；無有效段或模型失敗會明確報錯。
- 400/200 模型窗口沿用原本的邊界鏡像與 RMS normalization；鏡像會複製邊界樣本，是 symmetric padding。模型窗口紀錄含相對該段的輸入區間（可因鏡像 padding 為負或超出段長），以及對應的 packed output 索引。
- `InferenceTimeline.grid`：MI 在每個來源段的**重採樣輸出格點重新起算**，完整窗口才保留。這與 raw MI 沿原始錄製格點的政策不同，因此兩模式窗口數不必相同。

時間映射共用 `lilia.signal.resample_segment_time_us`：由 polyphase 的 fractional source position 插值原始 timestamps，最後不足一個原始樣本的部分依名目 period 外推，不丟失 jitter 或 UTC 精度。`resample_with_time` 也改用同一函式，避免兩份時間公式分歧。

輸出 segment ID 明確保留原始分段，即使原始 8 ms 缺口在 200 Hz 上小於「3 個 sample periods」，也不會被重新合併。`continuous_slices`／`build_window_grid` 新增可選 segment IDs；原始 entropy／非 denoise MI 的預設政策不變。

若前面短段排除，MI 的 `time_s` 仍以原始 raw epoch 為零點。模型輸出 sample indexes 不冒充 raw indexes。窗口 exclusive end 會截於原始段的物理終點，避免降採樣名目 sample period 讓區間越過來源涵蓋範圍；pre/onset 選取也使用原 segment ID 與同樣的 end cap。

## 3. Denoised metadata 與品質語義

新增 `lilia.neural_io.write_denoised_joint_mi_table`／`load_denoised_joint_mi_table`。

CSV/sidecar 使用獨立的 **`kind=denoised_joint_mi`、`index_space=resampled_model_output`**。窗口 `window_start_idx/end_idx` 指向 packed 模型輸出，sidecar 的 inference mapping 才把它對應回 raw source。普通 joint-MI loader 會拒絕此 kind。

sidecar 保存：來源 hash／raw samples／epoch、input/output fs、input/output channels、模型 window/hop、完整 segment 與 model-window mapping、filter/RMS/padding/OLA 規則、checkpoint hash、外部 architecture hash，以及本庫相關 Python 指紋。

`load_denoised_joint_mi_table(path, raw_csv, model_path)` 不执行模型：先核對來源與表格指紋，再從 raw timestamps 重建整條 inference timeline、所有模型窗口紀錄與 MI grid，逐項核對。可選 model_path 另外驗證 checkpoint hash；architecture hash 保存為追溯依據，此 reader 不載入外部模型程式。若未傳 raw_csv，只能核對表格及 metadata 自身，不能宣稱 raw 對應已確認。

每次成功 CLI 額外輸出 `*_inference_segments.csv`，包含短段排除原因。全部段都不足模型窗口時會明確失敗；此情況沒有成功的 MI 表格或 segment CSV。

品質維持 **disabled**：denoised channel 不套用 raw 裝置校準 scorer。summary、事件 histogram 與時序均明示狀態。`--no-bandpass` 與 `--denoise` 現在明確不相容，避免使用者以為已停用模型需要的固定濾波。

Denoised summary 的樣本數與 excluded_samples 使用模型輸出空間；原始短段丟棄量見 inference mapping。source-coordinate mapping 描述樣本位置與推論窗口，不宣稱濾波後每個樣本只依賴附近一個 raw 點；零相位濾波在該完整來源段內運算。

## 4. 驗證

- **97/97 tests 通過**，新增 10 個 `tests/test_neural_timeline_regression.py` 測試。
- 真實模型可載入：PyTorch 2.11.0+cu130，CPU single thread、TinyUNetV4 19,608 parameters；本輪沒有修改模型或外部 architecture。
- 完整連續模型數值對照：新版 602×2 與正確重採樣＋舊 OLA 完全一致；原錯誤 1503×2 另存歷史基準。
- identity 模型驗證 400、401、602、799 長度的真實首尾樣本不丟失，short／invalid hop／錯 shape／非有限 output 都會拒絕。
- 250 Hz 測試確認濾波使用 caller fs，改變缺口前的訊號不影響缺口後輸出。
- 512.5 Hz 與 ±20 µs jitter 驗證 fractional source positions、output count 及 physical end cap。
- 8 ms 原始缺口在降採樣後仍保留两个 segment，MI 起點分別為 0、401，不跨來源段。
- 排除首段後仍保留 raw epoch 與原 segment ID；全部太短／重複 timestamps／模型輸出長度不符明確失敗。
- CLI 輸出核對 inference index space、source mapping、checkpoint、quality disabled；竄改 raw segment mapping、窗口索引後重新計算 CSV hash，仍被 reader 拒絕。
- 原 denoise guard 測試改成保留 baseline guard。TFLite、ordinary entropy、band-event、非 denoise MI 原有測試也全部通過。
- Pyflakes、**96 個 Python 編譯**、bundle 同步與 diff whitespace 通過。

## 5. 完整 Hardy 真實模型驗證

來源 `iBrainCenter/Hardy(SN036)/merged.csv`；500 Hz、2,126,712 原始樣本、6 個連續段。400/200 模型窗口，輸出 200 Hz；MI win=step=2 秒。

| 項目 | 結果 |
| --- | ---: |
| 保留來源段 | 6 |
| 真實模型呼叫窗口 | 4,257 |
| 模型輸出樣本 | 850,686 |
| MI 窗口 | 2,125 |
| 最後中心真實時間 | 4,793.344551 秒 |
| pre/onset 活動比較 | 8 |
| 品質狀態 | disabled |

2,125 個 denoised MI 窗口與第四階段 raw MI 的 2,121 不同，原因是輸出 fs、各段重採樣 rounding，以及本輪在每段重新起算輸出 MI grid，不能只比列號或列數。

成功產生 distribution、excess、timeseries、events、peri-event 五類 PNG/SVG、peri-event HTML，以及 summary、events/audit、inference segments、timeseries CSV/sidecar。以 raw source 與 checkpoint 重新讀取，所有 inference mapping 與窗口核對通過；目視時序圖保留大型錄製缺口。

真實資料只用 **5 次 surrogate 作流程驗證**，不據此作 p/z 的研究判讀，預設 surrogate 次數未改。這輪驗證處理與時間對應的正確性，不代表模型 denoising 的生理訊號保真度已獲驗證。

## 重現與後續

```bash
MPLCONFIGDIR=/tmp/lilia-stage5-mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLBACKEND=Agg python -m unittest discover -s tests -v
python build_bundles.py --check
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
MPLCONFIGDIR=/tmp/lilia-stage5-mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLBACKEND=Agg python spectral_entropy.py \
  --csv 'iBrainCenter/Hardy(SN036)/merged.csv' --joint-mi --denoise --ibrain-events \
  --peri-event-html --mi-surrogates 5 --out /tmp/lilia-stage5/hardy
```

持久基準：`tests/fixtures/denoise_continuous_reference.json`、`denoise_continuous_reference.npz`（錯誤舊路徑）、`denoise_resampled_reference.npz`（正確重採樣＋舊 OLA）。最新驗證：`PYTHON_FIX_VALIDATION_2026-09-07_STAGE5.json`。暫存 log：`/tmp/lilia-stage5-all-tests.log`、`/tmp/lilia-stage5-hardy.log`；圖表／CSV：`/tmp/lilia-stage5/hardy/`。未覆寫正式录製、舊研究產物或 checkpoint。

下一步為 **TFLite 的來源 segment 傳遞及 baseline epochs 拼接修正**。`lilia.tflite.apply_tflite_with_time` 雖已按 timestamps 切完整窗口，仍需確認上游降採樣後的小缺口不會遺失原 segment ID，並保存被裁尾的實際 model-window/source mapping。`plot_tflite_summary` 的隨機 baseline 不可把不相鄰 1 秒 epochs 拼成 2 秒模型輸入；需比較「先原段推論，再選完整輸出窗口」帶來的納入與統計單位變更。

`data_analysis.py` APP/NUC CLI、baseline/clean 模式的連續 guard、quality scorer 內部短窗/fallback、band-event 品質政策及 artifact 原子發佈仍待續。CSV 與 sidecar 延用非交易式兩檔發佈；中断需重算，沒有在本輪完成原子發佈。
