# Python 修正第八階段：event markers 主流程分段遷移

日期：2026-09-07。完成 `plot_event_markers.plot_subject` 與一般 CLI 的 BP／TFLite 分段分析、raw 品質窗口映射、事件 baseline 選段、缺口圖形與輸出驗證。第三至第八階段修改仍在工作區，未 commit 或 push。

## 問題與修正

舊 `plot_subject` 要求整份錄製連續，TFLite 重採樣與推論使用整段陣列；raw 品質與 TFLite 指標依列數直接對齊，超出品質列數的尾端默認 good。rolling mean 可填回低品質空窗；熱圖依相鄰中心推算邊界，可能跨缺口著色。baseline 缺資料時，bar 可改用所有事前資料或顯示 0，heatmap 又採不同 fallback。

- 新增 `lilia.event_qeeg`：各 raw segment 分別 BP，BP qEEG／quality 共用每段重啟的 5 秒窗口。品質直接使用該窗口的同一份 raw 樣本索引，不因 timestamp jitter 多算下一個樣本。TFLite 重用已驗證的 `run_tflite_recording`，逐段重採樣、400 樣本模型窗口裁尾並保存來源映射。
- TFLite qEEG 使用 retained output 的獨立 5 秒 grid，保留原始 segment ID；每列品質依模型窗口實際時間找到 raw 區間，再用全部 raw channels 評分。原始時間軸的小缺口不因降採樣被合併，不能依兩個表格的列號猜測對齐。
- 品質 scorer shape、有限值、[0,1] 範圍與例外明確記錄。NaN／Inf／錯誤品質、低品質或非有限指標都使該列無效；CSV 保留窗口及原因，遮罩指標。污染 raw segment 不送入 BP；有其他合法段時 BP 可繼續，TFLite 需要的段若污染則整個模型分支明確失敗。
- 保存每段完整 BP 窗口數、tail samples、short_segment／nonfinite_segment／filter_error；模型 mapping 另記錄其完整窗口與裁尾。沒有合法 BP 窗口或全部品質排除仍保存 analysis audit，再非零退出。
- 品質保持既有 **raw 全通道評分** 的主流程政策，並未改成第六階段 summary 的 after-model 兩通道評分。兩者品質合格數不可直接比較。
- `summarize_branch` 集中 BP／TFLite 共用的事件、baseline、trend 與 heatmap 計算。rolling 只在相同 segment 的有效 run 內執行，不填回品質 NaN。heatmap 每段只取完整六個 5 秒窗口，保存窗口列、有效列、實際起訖與真正中點；每個矩形獨立繪製，缺口不著色。
- raw 顯示降採樣保留各段首尾；raw、quality、BP／TFLite trend 都在段界斷線。修正固定 trend Y 軸可能裁掉 engagement 的問題，調整多面板間距；缺少 bar baseline/event 時標示原因而不畫 0。
- 新增 `lilia.event_qeeg_io`：`kind=event_marker_bp`／`event_marker_tflite`，分別使用 `raw_samples`／`retained_tflite_output` 索引空間。CSV + sidecar 保存來源／設定／程式／table／analysis hash、quality raw 索引、每窗品質與指標狀態、完整推論及 baseline／heatmap audit。
- `load_event_qeeg_table(path, raw_csv, model_path=None)` 重建原始或模型 grid，逐列核對 metric 與 quality raw 索引，再核對 heatmap／baseline 選段。即使重算 table／analysis hash，錯誤原始索引、inference 或 baseline mapping 仍拒絕。此驗證不重跑品質 scorer 或模型。
- 一般 CLI 逐受試者記錄錯誤並繼續其他受試者；失敗清單寫入 `event_markers_failures.json`，任一失敗非零退出。指定 TFLite 但模型缺失／推論失敗，會保存可用 BP 圖表與 audit，再回報失敗，不默默以 BP-only 宣稱成功。

## Baseline 與數值相容性

`session-start` 現在對所有事件都使用「第一個參與活動之前」的同一區間，與 CLI 說明及 heatmap 政策一致。舊 bar 則對每個事件使用所有較早資料，可能把先前活動納入 baseline。`pre-event-rest` 使用緊接事件前、前一排定活動結束後的區間，缺資料不再借用更早區間。未參與活動仍占用排程時間，不被當成休息。沒有參與事件時，保留前 `max(1, n_bins // 5)` 個完整 heatmap bins 作參考的既有策略，明示於 audit。

Baseline 與事件皆要求完整窗口／完整 bin 位於半開時間範圍內，取代舊中心歸屬。bar 仍為「先逐 channel 計算事件 mean − baseline mean，再取 channel mean／SD」；heatmap 仍使用「逐窗口 channel median，再取 bin median」，baseline reference 亦為 bin median。這兩種統計量保留區別。

修改前保存 `tests/fixtures/event_qeeg_continuous_reference.json`：seed 831、75000×4 float32 raw、500 Hz、150 秒、真實 TFLite 模型與真實品質 scorer，包含舊程式及模型 hash。新版 BP／TFLite 的逐窗 qEEG、raw 品質、連續有效趨勢、無事件 heatmap absolute／delta 均符合 `atol=rtol=1e-12` 基準。詳細最大誤差在 validation JSON；真正的 heatmap 中點從舊 17.5 秒修正為 15 秒，起點不再偏移 2.5 秒。

另用修改前的 `plot_subject` 執行固定 24 個連續 5 秒窗口、四通道分數由 −0.5 線性增加到 +0.5、品質全為 1 的政策對照，保存在 `tests/fixtures/event_qeeg_policy_reference.json`：

| 政策／第二個事件 | 舊 Focus delta | 新 Focus delta | 原因 |
|---|---:|---:|---|
| session-start，事件 `[90,120)` | 0.5217391304 | 0.7826086957 | 改用第一活動 `[30,60)` 之前的共同 baseline，避免混入先前活動 |
| pre-event-rest，事件 `[60,90)` | 0.3913043478 | NaN／missing baseline | 緊接前活動，沒有休息區間；取消所有較早資料的 fallback |

這些是有意的納入政策修正，不能宣稱所有舊事件 delta 都會保持相同。

## 驗證結果

- **129 tests 通過、無 skip；111 個 Python 編譯、Pyflakes、bundle 一致性與 diff whitespace 通過。** 本階段新增 13 tests。
- 覆蓋真實連續 BP／模型／品質基準、各段獨立濾波、非整齊段界、小缺口原 segment ID、timestamp jitter、NaN／scorer exception、baseline 無休息／全部缺失、完整窗口邊界、輸出重讀與篡改拒絕、短段／全排除 audit、模型缺失／批次部分失敗，以及 raw 斷線與熱圖真實矩形邊界。
- 完整 Hardy：2,126,712 raw 樣本、6 段。`session-start` 與 `pre-event-rest` 各自實際執行 BP＋TFLite，兩分支各 **848 個 5 秒窗口、683 個品質合格、165 個低品質**。最後中心 elapsed time 4791.844551 秒；各有 **139 個完整 30 秒 heatmap bins**。
- session-start：全部 8 個參與活動都使用相同 21 個合格事前窗口。pre-event-rest：baseline 合格窗口依序為 `[21, 11, 42, 54, 47, 0, 31, 23]`；Color Agility Ladder 沒有事前休息區間，BP／TFLite 的 bar 與 heatmap 明示缺 baseline，其餘 7 個活動有可用參考。
- 兩種模式各輸出 PNG、SVG、BP／TFLite metrics CSV + sidecar、analysis JSON；使用公開 loader 核對來源、模型、raw quality mapping、heatmap 與 baseline rows。已目視檢查完整 pre-event-rest 圖的缺口、缺 baseline 呈現及面板排版。

目前完整數值、artifact hashes 與全部 Python 指紋見 [第八階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE8.json)。第七階段及更早指紋為歷史版本。

```bash
MPLCONFIGDIR=/tmp/lilia-stage8-mpl MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
python build_bundles.py --check
git diff --check
```

完整 Hardy 以 Python 呼叫 `plot_subject('Hardy', SUBJECTS['Hardy'], outdir, 500, use_tflite=True, baseline_mode=mode)`，mode 分別為 `session-start` 與 `pre-event-rest`。產物位於 `/tmp/lilia-stage8/hardy/`。

## 限制與下一步

1. 本階段遷移一般 `plot_subject` 路徑；`--hardy2-analysis` 的專用分析，以及其他入口直接呼叫舊 helper 的流程，未因此取得整條分段正確性的保證。
2. 品質 scorer 核心的短子窗末端、內部 fallback 與裝置校準尚未修改。此層能攔截顯式 exception／NaN，不能辨識 scorer 內部已轉成有限值的 fallback。
3. 延續既有三倍 nominal sample period 的缺口判定。完整段 zero-phase filter 會使用事件邊界以外的上下文，不能解讀為因果即時估計。
4. heatmap 每段不足 30 秒尾部不呈現。完整區間納入會捨棄跨活動邊界的 bin；bar 5 秒與 heatmap 30 秒的有效樣本數可以不同。bar SD 是通道差異，不是受試者推論誤差。
5. CSV／metadata／PNG／SVG／audit 多檔發布尚非原子操作。部分失敗可留下有效 BP 產物或前次檔案，應以本次 analysis 的來源、branches 與 errors 判定，不能只看資料夾內是否有圖。讀取失敗或無效設定等前置錯誤不保證有個別 subject audit，一般 CLI 會記錄 batch failure。
6. 下一步優先遷移 `data_analysis.py` CLI，保留已驗證的 PyTorch／TFLite 差異；再盤點其他入口、品質 scorer、大型模組拆分與方法校準。前述 10 個工作包中的第 1 項一般事件圖流程已完成；特殊入口與整庫收尾仍在清單內。
