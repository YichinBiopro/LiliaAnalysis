# Python 修正第六階段：TFLite 分段時間軸與 baseline 抽樣

日期：2026-09-07。承接第五階段 PyTorch denoise；本階段完成 `plot_tflite_summary.py` 的分段推論、完整模型窗口 baseline、品質／qEEG 共用窗口，以及可重讀核對的來源映射。第三至第六階段修改仍在工作區，未 commit 或 push。

## 問題與修正

舊 summary 要求整份錄製連續；推論前後的陣列裁尾長度不一致。baseline 從不同位置抽取 1 秒 raw epochs，串接後重採樣並交給 2 秒 TFLite 模型，模型與後續 Welch 可能跨過人造接縫。各自計算的品質／qEEG 窗口、平滑與熱圖也沒有共同原始 segment 資訊。

- `lilia.tflite.TFLiteTimeline` 記錄 raw segment、重採樣比例、裁尾數量、完整模型窗口、原始索引與微秒時間。每原始段分別濾波、500→200 Hz 重採樣，只保留完整 400 點窗口；不套用 PyTorch 的鏡像 padding／overlap-add。
- 各段裁尾後才打包完整窗口供單一 interpreter 執行。前後訊號使用相同保留位置，模型窗口不跨原始缺口。模型輸出 shape、dtype 與有限值明確檢查。
- `resample_with_time(..., return_segment_ids=True)` 與 `apply_tflite_with_time(..., segment_ids=...)` 傳遞原始段 ID；`extract_tflite_signal_pipeline.py` 已接上，避免原始小缺口在降採樣後被誤認為連續。
- `lilia.tflite_baseline` 在完整錄製推論後抽取已存在的模型窗口。每個 2 秒窗口內的兩個原始 1 秒子段，都必須通過四通道 median 品質、有限值與 ADC 飽和檢查。預設品質門檻 0.5、飽和比例上限 0.02、seed 42；預設需求 10 秒即抽 5 個完整窗口。其他需求以完整模型窗口向上取整，audit 記錄 nominal duration。
- 每個被選模型窗口分別計算兩通道 qEEG，再取各指標算術平均。baseline 不再額外推論，也不將非相鄰窗口串接交给 Welch。舊 raw 抽樣 helper 保留 continuity guard；舊 `_tflite_qeeg_reference` 明確拒絕 packed raw baseline，主流程已改用新 selector。
- 事前搜尋仍為事件前 `[-15, -3]` 秒，要求整個模型窗口位於搜尋區間。參與者與 Color Agility Ladder 借用 Agility Ladder 錨點的規則保留。`skip` 記錄排除原因；沒有可用 baseline 時失敗，不產生零值參考或無 baseline 的空圖。
- 品質前後與 qEEG 共用每段重新起算的 5 秒 WindowGrid，前後比較都使用兩通道 median。NaN／低品質保留 CSV 列但遮罩指標；不再靠截短陣列掩蓋窗口數不一致。
- 趨勢線、平滑保留段界及失效窗口；30 秒熱圖只聚合單段內完整的六個窗口，按真實起訖畫矩形，缺口不著色。事件歸屬使用矩形真正的中點，修正原本第 4 個 5 秒窗口中心造成的 2.5 秒偏移。audit 保存熱圖邊界、中點與來源 metric rows。沒有資料的事件長條值改為 NaN。
- `lilia.tflite_io` 輸出／讀取 `kind=tflite_qeeg`、`index_space=retained_tflite_output` 的 CSV + sidecar。提供 raw 時核對檔案 hash，重新建立完整 inference mapping 與每列窗口；可另核對模型 hash。不能把 output index 當 raw index。
- Summary 額外保存 `*_tflite_analysis.json`：raw／model hash、完整推論映射、baseline eligibility、抽樣明細、排除原因與熱圖窗口。批次入口遇到 ValueError／OSError 會記錄失敗並以非零狀態退出。

## 數值與統計單位

修改前保存 `tests/fixtures/tflite_continuous_reference.json/.npz`。seed 631、16000×4 raw 樣本、相同 bandpass 與真實 TFLite 模型，新版 6400×2 輸出和修改前 **最大絕對差 0**，timestamps 完全相同；測試容差為 `atol=rtol=1e-6`。

Baseline 的納入單位與聚合方法有意改變，因此不宣稱新舊 baseline 等價。固定 seed 42、相同連續資料、所有 raw 品質分數為 1 的對照：

| 指標 | 舊 ch1 | 新 ch1 | 新−舊 ch1 | 新−舊 ch2 |
|---|---:|---:|---:|---:|
| relaxation | 0.2538132183 | 0.2837587901 | +0.0299455718 | −0.0104379511 |
| calm | −0.5482080314 | −0.5197910820 | +0.0284169494 | +0.0155750022 |
| flow | −0.5877928435 | −0.5449594710 | +0.0428333725 | +0.0154042980 |
| focus | 0.4582169505 | 0.3927125473 | −0.0655044032 | −0.0177402563 |

舊法抽取 10 個 1 秒 epochs 串接，再模型推論與整段 qEEG；新法選取起點 2、12、18、20、30 秒的 5 個完整 2 秒模型輸出窗口，再平均逐窗指標。差異包含抽樣單位、模型上下文與 PSD／指標聚合三部分，不能歸因為单一因素。完整兩通道結果與選段記錄在第六階段 validation JSON。

## 驗證結果

- **106 tests 全數通過，無 skip；101 個 Python 編譯、Pyflakes、bundle 一致性與 diff whitespace 檢查通過。** 本階段新增 9 tests，涵蓋真實模型連續基準、逐段裁尾、小缺口、原始段 ID 傳遞、baseline 不重跑模型／不拼接 PSD、NaN／飽和／不足、搜尋邊界、錯誤模型輸出，以及 summary 圖形和 metadata 篡改拒絕。
- 完整 Hardy：2,126,712 raw 樣本、6 個來源段，保留 850,000 TFLite 輸出樣本，共 2,125 個完整模型窗口；每段裁尾 `[170, 0, 394, 122, 0, 0]`，共 686 個重採樣樣本。
- qEEG 共 848 個 5 秒窗口，557 個通過目前 after-model 品質門檻，291 個遮罩；最後中心真實 elapsed time 為 4791.844551 秒。
- baseline catalog：1603 個 eligible、171 個 low_quality、351 個 raw_saturation。session 及 5 個事前 baseline 各成功選 5 個模型窗口。Agility Ladder、Color Agility Ladder、Mindfulness Meditation 各僅 4／5 候選合格，需要 5 個，故以 `on_insufficient='skip'` 明確排除事前 baseline；session 模式仍可呈現這些事件。
- 已實際輸出 session／pre-event 各 PNG 與 SVG、metrics CSV／sidecar、analysis audit。重讀核對 raw、模型、inference mapping、qEEG grid，並檢查全部 baseline 與熱圖窗口的來源段及搜尋邊界。已目視檢查 session 圖的缺口呈現。

重跑主要驗證：

```bash
MPLCONFIGDIR=/tmp/lilia-stage6-mpl MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
python build_bundles.py --check
git diff --check
```

完整 Hardy 以 Python 呼叫 `plot_subject_tflite_summary('Hardy', SUBJECTS['Hardy'], '/tmp/lilia-stage6/hardy', heatmap_baseline_mode='both', on_insufficient='skip')`。產物 hash 與目前全部 Python 指紋在 [第六階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE6.json)。第一至第五階段指紋為歷史版本。

## 尚存限制與下一步

1. 新 baseline 是 2 秒模型窗口指標平均，trend 仍是 5 秒 qEEG；兩者的 PSD 估計時間長度不同，這次修正不等於完成研究方法或生理效度校準。前後品質都改用兩通道，不能直接比較舊四通道 before median；after-model 的品質評分仍沿用既有參數，未宣稱適合模型輸出分布。
2. 品質 scorer 在原始 1 秒子段上仍會出現 SciPy 將 `nperseg=1000` 縮為 500 的警告。此階段未更動 scorer 短窗／fallback 語義。完整段的離線濾波仍含雙向濾波上下文，事前 baseline 不代表因果、即時估計。
3. 30 秒熱圖捨棄各段不足 30 秒的尾部；事件值依窗口中心歸屬，尚未改成要求完整位於事件範圍。`on_insufficient='raise'` 仍立即失敗，不保證保存失敗當次 audit；本次實際驗證使用 `skip`。多檔輸出尚非原子發佈。
4. 只解除已遷移的 TFLite summary continuity guard。`plot_event_markers.py` 主入口、`data_analysis.py` CLI 與 `spectral_entropy --baseline/--event` 的 guard 保留。其他 TFLite 呼叫者不能僅因共享 backend 更新就宣稱整條流程已遷移。
5. 下一階段優先處理普通 entropy 的 baseline/event 真實時間區間、合法窗口與 audit；`--clean` 已逐 epoch 計算 PSD，應保留這個聚合語義，補足原始 segment／選段映射後才解除 guard。
