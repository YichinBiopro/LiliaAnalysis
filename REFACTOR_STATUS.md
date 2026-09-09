# 重構短進度

更新：2026-09-09。接手先讀本檔，再只讀當前任務；歷史細節按需查閱。

- 已完成：分析入口遷移至第十五階段（眼開閉）。
- 當前：[任務 16](docs/refactor/TASKS.md#stage-16) Jenqwei 分析進行中：舊基準／分段處理完成，main／IO／繪圖待續，guard 保留。
- 本輪：第十五階段已提交 `141bde2`；第十六階段五份真實＋兩個合成基準凍結，新增 `process_segments`，41 tests／0 skipped、154 Python 靜態、28＋55 項 evidence 通過。
- 數值：Jenqwei 七個連續模型／PSD／STFT 案例精確一致；真實訊號分段 Before 1,403／After 1,200 列，舊模型／時間／來源映射最大誤差 0。
- 下一步：main 每來源只推論一次，接入獨立 Before／After IO／audit；分段 TD／PSD／STFT、elapsed 及 ch3/4 誤映射處理，再做 CLI／目視／完整驗收。
- 順序：16 Jenqwei 分析 → 17 Jenqwei 資料集；細節於接手盤點。
- 其後：品質短窗／fallback、baseline／PSD／Goertzel／MI 方法校準、架構／F19、批次／原子發佈、整體驗收。

## 必須保留

- qEEG CLI 直接分析單一 raw channel；無 BP、模型、品質 scorer 或 baseline。無 timestamp 共用 helper 保留舊語義。
- qEEG 保留 `int(win_sec*fs)` 與原始全域格點；段首／短尾明示排除，圖為 elapsed，CSV 保留原始微秒。
- 不同入口的 baseline、品質與 raw／模型索引空間不可混用。
- 只有完成分段遷移與驗證的路徑才能解除 continuity guard；legacy／抽樣 guard 個別判定。
- 普通 entropy、MI、TFLite summary、event markers、APP／NUC、Hardy_2、compare_subjects、zoom、TYY、qEEG CLI、眼開閉已完成，勿重寫。
- 眼開閉只做 float32 bandpass，不混入 notch／bandstop；模型輸出依序為來源 ch1/2/5/6，before 比較圖取來源 ch1/2/5/6；STFT 數值保留原法、逐段繪於 elapsed 軸。
- 不覆寫原始錄製、模型或舊圖。bundle 改主版後執行 `python build_bundles.py`，再 `--check`。
- Jenqwei Before 保留全重採樣尾端、After 才按 400 點裁尾；`max_samples` 在完整來源段濾波後截斷，raw／Before／After 使用各自索引與明確映射。
- Git：第十五階段基準／CLI／IO `b84029e`、完整繪圖驗收 `141bde2`；第十六階段基準／分段處理與驗證證據隨本次提交收錄，整階段尚未完成。未 push，後續推送由使用者負責。

## 驗證入口

- 本輪：[第十六階段專屬／共享檢查](docs/refactor/validation/stage16/checks.json)：41 tests／0 skipped；[靜態](docs/refactor/validation/stage16/static.json) 154 Python 編譯／Pyflakes／bundle／diff 通過，尚未做第十六階段 `--full`。
- [數值摘要](docs/refactor/validation/stage16/adapter_analysis.json)／[adapter evidence](docs/refactor/validation/stage16/adapter_evidence_checks.json)：47 hashes＋8 組 NPZ 比較；[持久基準](docs/refactor/validation/stage16/baseline_fixture_evidence_checks.json) 28 hashes。
- 本輪尚未接入 main／輸出表／分段圖形，未做表重讀與目視；不得把處理測試通過當作 guard 解除依據。
- 前階段：[第十五階段完整檢查](docs/refactor/validation/stage15/final/checks.json) 246 tests／0 skipped；[目視](docs/refactor/validation/stage15/final/visual_review.json) 已完成，為當時版本證據。
- 開發中依任務卡選專屬與共享模組測試，不重跑無變更且已有同來源／環境成功證據的檢查。
- 階段完成：`python tools/refactor_check.py check --full`。
- 舊結果是否適用：`python tools/refactor_check.py status /path/to/run/summary.json`。
- 產物／數值：`python tools/refactor_check.py evidence /path/to/manifest.json`。
- 完整操作、限制與選測試規則：[WORKFLOW](docs/refactor/WORKFLOW.md)。

## 按需參考

- [當前與後續任務單](docs/refactor/TASKS.md)／[報告模板](docs/refactor/REPORT_TEMPLATE.md)。
- [第十六階段起步報告](docs/refactor/STAGE16_REPORT_2026-09-09.md)／[數值驗證腳本](tools/validate_jenqwei_adapter_stage16.py)。
- Jenqwei 凍結基準在 `tests/fixtures/jenqwei_*_reference.*`，舊入口／盤點／摘要／log 在 `docs/refactor/validation/stage16/`；大型產物 `/tmp/lilia-stage16-baseline-v1/`、`/tmp/lilia-stage16-adapter-v1/` 不保證永久存在。
- [第十五階段驗收報告](docs/refactor/STAGE15_ACCEPTANCE_REPORT_2026-09-09.md) 已完成；`tests/fixtures/eye_*reference.*` 與 `docs/refactor/validation/stage15/final/` 保留其持久基準及目視證據。
- [完整歷史交接](REFACTOR_HANDOFF.md) 保留至第十三階段；其中歷史下一步、Git 描述與指紋不代表當前版本。

維護方式：階段完成時替換目前狀態、下一步與最新驗證連結，維持約 50 行；不累加歷史摘要。
