# Python 修正第七階段：普通 entropy baseline/event 分段遷移

日期：2026-09-07。完成 `spectral_entropy --baseline/--event` 與 `--clean` 的真實時間選段、逐段濾波、共同窗口與排除 audit，解除這條模式的 continuity guard。第三至第七階段修改保留在工作區，未 commit 或 push。

## 問題與修正

普通模式原本以秒數乘 fs、四捨五入為 sample index；缺口後會選錯資料。舊 `collect_clean_epochs` 雖用 timestamps 找區間首尾，卻在中間直接按樣本數切窗，可能跨缺口。CLI 的全錄製 continuity guard 原本阻止這些錯誤進入分析，但也阻止有缺口的錄製使用 baseline/event。

- 新增 `lilia.state_windows.select_state_windows`：區間為原始 epoch 加相對秒數形成的 `[lo_us, hi_us)`。在每個來源 segment 與區間的交集內，從首個 in-range sample 重新起算窗口，原始索引不壓縮；只有完整窗口、且最後樣本加 nominal sample period 不超出區間終點才納入。每段尾端不足窗口與缺資料的實際時間範圍都保留 audit。
- 普通模式可使用 `--step` 重疊窗口；`--clean` 保留 non-overlapping epochs，若明確指定不同於 `--win` 的 `--step`，現在報錯，避免過去默默忽略 step。
- CLI 只對能提供完整候選窗口的原始段濾波。普通模式分析指定 channel；clean 的有限值、飽和與品質仍檢查所有來源 channels。預設濾波時，相關 channel/segment 受 NaN/Inf 污染則排除整段；`--no-bandpass` 僅排除受污染窗口。太短而不能執行濾波的完整候選窗口記錄 `filter_error`。
- `_collect_state_epochs` 共用候選 catalog 作為 PSD、QC 與 audit 的來源。分別記錄 accepted、incomplete_window、nonfinite_signal、raw_saturation、low_quality、invalid_quality、quality_error、raw_unavailable、filter_error；品質 NaN/Inf、錯誤 shape 或超出 [0,1] 不當成合格資料。Scorer 拋錯時保存原因、排除該窗，不使用 fallback 分數。
- `collect_clean_epochs` 使用同一 selector；`compute_state_entropy_from_epochs` 保留逐 epoch Welch、等權平均 PSD 與每窗 entropy，另拒絕不等長／非有限 epochs。不串接缺口兩側或不相鄰合格 epochs 計算 PSD。
- `compare_baseline_event(..., time_us=...)` 支援原始時間戳。未傳 timestamps 的 caller 明確假設均勻時鐘，不能據此對有缺口資料宣稱正確對齊。
- 新增 `lilia.state_entropy_io`，輸出原摘要檔名的 CSV + `.csv.meta.json`，`kind=state_band_entropy`、`index_space=raw_samples`。CSV 保留 baseline/event 兩列，加入 status、quality_state 與 pooled band energies；sidecar 保存設定、程式／來源／table／audit 指紋、全部候選原始窗口／時間／segment、QC／排除理由、缺資料範圍與 accepted 順序的 per-window entropy。
- `load_state_entropy_table(path, raw_csv, channel)` 核對 hash、設定、state/accepted 數量，依 raw timestamps 重建每個候選窗口與缺資料區間。篡改窗口即使重算 audit hash，仍會在 raw mapping 核對時失敗。這是來源與選段驗證，不是重新計算 QC/PSD。
- 兩個 state 都完成審核後才判定成功；其中一個或兩個全數排除，仍保存兩列 summary 與 sidecar，再以非零狀態失敗。失敗列保留 metric 欄位並留空，不回傳零值 entropy。
- 只解除 baseline/event guard。MI/sync/denoise 與 baseline/event 同時指定現在明確拒絕。其他已完成模式的運算保持既有流程；bundle 已由 canonical builder 同步。

## 數值與納入政策

修改前保存 `tests/fixtures/state_entropy_continuous_reference.json`，含修改前程式 hash、seed 731、16000×4 float32 raw、500 Hz、0.5–45 Hz 濾波與兩區間 `[1,15)`／`[18,31)`。普通模式 2 秒窗／1 秒 step 與 clean 2 秒 epochs、真實品質 scorer 的 mean PSD、pooled energies/proportions/entropy、per-window entropy 及普通模式差值／U/p，修改後最大絕對差均為 **0**；回歸容差 `atol=rtol=1e-12`。另驗證 CLI float32 輸出與來源重讀。

邊界政策有意改變，不宣稱所有連續資料區間都與舊法等價。相同連續資料、區間 `[1.0005,15.0005)`，舊法從 index 500 起算；新法從 index 501 起算，且最後窗口必須完整結束於區間內。舊法納入 13 窗、新法 12 窗，pooled entropy 由 1.1709843605818215 變成 1.1698941626410626，差 −0.0010901979407589302。差異同時包含窗口起點及納入數量，詳細數值保存在 validation JSON。

Pooled band energy 的 entropy 和 per-window entropy 的平均仍為不同統計量；這次沒有更動兩者的聚合定義。

## 驗證结果

- **116 tests 通過，無 skip；106 個 Python 編譯、Pyflakes、bundle 一致性、diff whitespace 全部通過。** 新增 10 tests，另更新 2 個原先依賴 baseline guard 的既有測試。
- 測試涵蓋修改前連續數值與真實品質、缺口兩側獨立濾波／PSD、跨缺口 state 的逐窗 pooling、半開與非整數邊界、短段／太短濾波窗口、所有排除原因中的主要路徑、全排除／完全無資料區間、來源／channel／窗口篡改拒絕，以及 clean step 明示。
- 完整 Hardy：2,126,712 樣本、6 段、實際 recording end 4794.344551 秒。使用 baseline `[0,2400)`、event `[2400,4794.344551)`，2 秒 non-overlapping windows，ch1；clean 使用預設 threshold 0.5、全通道品質／飽和檢查。

| 模式／state | 接受窗口 | 低品質 | 飽和 | 尾端不足 | pooled entropy |
|---|---:|---:|---:|---:|---:|
| 普通 baseline | 1159 | 未評分 | 未評分 | 2 | 1.5540299386607173 |
| 普通 event | 966 | 未評分 | 未評分 | 2 | 1.3448724955917282 |
| clean baseline | 928 | 64 | 167 | 2 | 1.5782597217813683 |
| clean event | 736 | 44 | 186 | 2 | 1.1464058863579556 |

普通共 2,125 合法窗口，clean 共 1,664 合格窗口；最後接受窗口中心為 4793.344551 秒。兩模式各輸出 summary CSV／metadata，全部 accepted 窗口核對原始 segment 與區間邊界，並使用公開 loader 重讀核對來源與所有候選窗口。完整數值、artifact hashes 及目前全部 Python 指紋在 [第七階段驗證](PYTHON_FIX_VALIDATION_2026-09-07_STAGE7.json)。

重跑檢查：

```bash
MPLCONFIGDIR=/tmp/lilia-stage7-mpl MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
python build_bundles.py --check
git diff --check
```

完整 Hardy 可使用：

```bash
python spectral_entropy.py --csv 'iBrainCenter/Hardy(SN036)/merged.csv' --baseline 0 2400 --event 2400 4794.344551 --out /tmp/lilia-stage7/hardy/ordinary
python spectral_entropy.py --csv 'iBrainCenter/Hardy(SN036)/merged.csv' --baseline 0 2400 --event 2400 4794.344551 --clean --out /tmp/lilia-stage7/hardy/clean
```

## 限制與後續

1. 分段採既有三倍 nominal sample period 的缺口門檻；沒有改成每個微小遺失樣本都切段。每段先用完整上下文作 zero-phase filter，因此事前區間仍非因果即時估計。
2. 普通模式明示品質 disabled；clean 的 QC 時長跟隨 entropy win，和第六階段 TFLite baseline 的 1 秒 raw QC／2 秒模型指標不是相同統計設計，不能直接混用。Scorer 本身的短窗／fallback 語義尚待後續處理。
3. 重疊窗口與時間序列相關性仍存在；既有 Mann–Whitney U/p 保留，但不能把窗口當成獨立受試者、也不能把通過 regression 等同研究方法或生理效度驗證。本次全錄製前後半區間用於工程驗證，不是活動效果分析。
4. CSV 與 sidecar 多檔發佈尚非原子操作；中斷／不完整配對須重新產生。無效 CLI 設定、不可讀來源等在選窗前發生的錯誤，不保證產生 audit；合法選段但全部排除的失敗已驗證保存 audit。
5. 下一步可遷移 `plot_event_markers.py` 主入口，其後為 `data_analysis.py` CLI；兩者仍保留 continuity guard。大型入口拆分、批次錯誤退出與 artifact 原子發佈、PSD/Goertzel/MI 方法一致性仍未完成。
