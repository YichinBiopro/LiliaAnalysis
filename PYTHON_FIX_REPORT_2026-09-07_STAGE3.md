# Python 第三階段修正報告 — 2026-09-07

本輪依 `REFACTOR_HANDOFF.md` 接續完成 **band-event MI 的真實時間事件區間與逐段頻帶處理**，涵蓋 `spectral_entropy.py --band-event-mi` 及 `joint_mi.py per-event`。另遷移 `compute_event_pre_onset_joint_mi` 的區間選取；獨立 `--joint-mi` 時序、神經網路與 baseline 路徑尚未完成，因此其 continuity guard 仍保留。

## 接手檢查

開始時工作區乾淨，HEAD 為 `c76bc43`（第二階段已提交），分支為 `spectral-entropy-flow-rework`，與本機 origin tracking ref 相同。交接文件所稱「第二階段尚未 commit」已過時。本輪未 fetch、commit 或 push；先前 push 審核紀錄僅保留為歷史，不據此推斷遠端現況。

## 原因與方法

1. **事件位置錯誤：** 原程式將經過秒數乘以 fs 當成樣本位置，缺口後會取錯資料。新增 `lilia/event_windows.py`，依整數微秒 timestamps 用 `searchsorted` 選取 `[onset-pre, onset)` 與 `[onset, onset+post)`。兩區間的物理邊界必須完整落在同一連續段；末端以最後樣本加一個名目採樣週期定義。缺口規則沿用超過三個名目採樣週期。
2. **跨缺口濾波：** `extract_band_envelopes` 可接收 timestamps，先按段執行 0.5–45 Hz front-end，再按頻帶濾波與 Hilbert。只計算能提供候選事件區間的段，保留原始樣本位置，其餘填 NaN；沒有候選事件的極短段不會拖垮分析。
3. **每個長度獨立審核：** `joint_mi.py` 以前要求每個事件能容納最大的 30 秒區間，才計算所有長度。現在 5、10、15、30 秒各自驗證，保留合法短區間。這是有意的納入政策變更，結果可能因新增事件而改變。
4. **事件審核輸出：** pooled 模式另輸出 `*_band_event_mi_events.csv`；逐事件模式輸出 `*_per_event_mi_events.csv`。記錄事件 UTC onset、區间時間／半開原始索引、segment、channel、長度、accepted/excluded 與原因。全數排除時仍保存審核 CSV，再以非零狀態結束。
5. **排除語義：** onset 在缺口或錄製範圍外、pre/post 跨缺口或不足、沒有足夠 sub-epochs、未參與活動、非有限值 signal/envelope 都有明示原因。不再裁切不完整的 pre/onset 區間。若某 channel 的連續段含 NaN/Inf，該段濾波會受污染，因此保守排除整個 channel/segment，其他 channel 可繼續。
6. **避免重複演算法：** `joint_mi.analyze_subject` 使用相同 pipeline 的 `pool_events=False`，維持逐活動估計，並保存 `Event_Abbr`、`Gap_Bits` 等既有繪圖欄位。
7. **pre/onset histogram API：** 支援明確 `time_us` 與 `audit`；未給 timestamps 的相容介面只描述均勻連續陣列。非 denoise 的 joint-MI caller 已傳入原 timestamps 並保存 `*_events_audit.csv`，但 joint-MI 模式整體尚未解除 guard。

## 品質與方法限制

band-event MI 原本沒有套用品質 scorer，本輪維持該分析方法，結果及區間紀錄新增 `Quality_State=disabled`，CLI 也會明示。`--quality-threshold` 不會在此模式產生品質遮罩；非有限值檢查不等於 EEG 品質評分。後續若加入評分，必須另定 sub-epoch 納入政策與數值對照。

Hilbert/filter 仍有段內邊界效應；本輪處理的是跨錄製缺口運算，不宣稱已解決估計方法有效性。MI surrogate 與重疊 sub-epochs 的統計假設未改動。API 若只傳 signal/onset index 而不提供 timestamps，仍無法自行辨識缺口。

audit CSV 是事件區間紀錄，不是普通 entropy CSV 的 source/config 指紋 sidecar 格式；不要交給 `load_entropy_table` 當成 entropy 表格。本輪產物不是多檔案交易式發佈。

## 驗證結果

- **76/76 tests 通過**，新增 `tests/test_event_windows_regression.py` 的 10 個測試。
- 修改前從 `c76bc43` 產生連續資料基準：`tests/fixtures/band_event_continuous_reference.json`，seed=314、6000×2、200 Hz、onset=15 秒、5/10 秒區間、5 次 surrogate。
- 新路徑的 MI、樣本數、鄰居數與 surrogate 統計皆在 `rtol=atol=1e-12` 內一致。
- 缺口後非整樣本 onset 驗證半開物理切片；onset/pre/post 缺口、錄製兩端不足、末端恰好完整、非有限值、非參與者、極短未使用段均有案例。
- 濾波與 Hilbert 結果逐段和獨立分析精確相等；改變缺口前信號不會影響缺口後的包絡。
- 合成有缺口 CSV 經實際 `main()` 成功輸出；onset 落在缺口時保存原因並失敗。
- 全庫 Pyflakes、90 個 Python 檔案編譯、bundle 同步與 whitespace 檢查通過。

### 完整 Hardy

輸入 `iBrainCenter/Hardy(SN036)/merged.csv`：2,126,712 樣本、6 個連續段、500 Hz。

- pooled CLI：2 channels × 4 種區間 = **8 列 MI**，64 列事件區間全部合法，涵蓋 8 個活動；PNG/SVG 成功。
- `joint_mi.analyze_subject` 與逐事件繪圖：8 活動 × 2 channels × 4 種區間 = **64 列 MI**，另有 64 列審核紀錄；PNG/SVG 成功。
- 本次真實資料驗證使用 **5 次 surrogate** 作流程驗證，不用這些 p 值作研究推論；程式預設次數未更改。

| 活動 | 真實 onset index | 舊 elapsed × fs index | 舊索引偏移 |
| --- | ---: | ---: | ---: |
| Color Agility Ladder | 1,135,655 | 1,175,751 | 40,096 |
| Cone Rotation | 1,366,087 | 1,415,751 | 49,664 |
| Mindfulness Meditation | 1,606,087 | 1,655,751 | 49,664 |

最大索引偏移以 500 Hz 換算為 99.328 秒。這是舊索引算法的對照，沒有解除舊版 guard 後重跑錯誤分析來建立研究結果。

## 產物與重現

```bash
MPLCONFIGDIR=/tmp/lilia-stage3-mpl MPLBACKEND=Agg python -m unittest discover -s tests -v
python build_bundles.py --check
python -m pyflakes lilia *.py tests plot_index_vs_raw_bundle signal_quality_package
MPLCONFIGDIR=/tmp/lilia-stage3-mpl MPLBACKEND=Agg python spectral_entropy.py \
  --csv 'iBrainCenter/Hardy(SN036)/merged.csv' --band-event-mi --ibrain-events \
  --mi-surrogates 5 --out /tmp/lilia-stage3/hardy
```

持久數值與 Python 指紋保存在 `PYTHON_FIX_VALIDATION_2026-09-07_STAGE3.json`。測試與真實 CLI log：`/tmp/lilia-stage3-all-tests.log`、`/tmp/lilia-stage3-hardy.log`、`/tmp/lilia-stage3-per-event.log`。逐事件圖與 CSV 在 `/tmp/lilia-stage3/hardy-per-event/`。暫存不保證跨環境保存，正式錄製與舊研究產物未覆寫。

## 下一步

接續 `compute_joint_mi_windowed`／`_run_joint_mi_mode`：共同 WindowGrid、兩 channels 與品質／metadata 對齊、逐段 front-end、series/peri-event 不跨缺口平滑或連線；whole-recording circular-shift null 也不能跨段任意 roll。先處理非 denoise 路徑並保存連續基準，完成驗證後才解除該 guard。

神經網路路徑需保留實際推論輸出時間軸；baseline 不可把不相鄰 1 秒 epochs 拼進 2 秒模型窗口。這些與品質評分內部語義、架構拆分及 artifact 發佈仍待續。本輪修改尚未 commit。
