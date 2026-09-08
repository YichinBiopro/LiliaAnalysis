# Python 修正報告：第九階段

日期：2026-09-07。範圍：`data_analysis.py` 的 APP／NUC CLI 分段遷移。

本階段已完成；整體重構仍有其他入口與工程收尾。起始版本為 `edd119b`，本階段未 commit／push。

## 問題與處理

原 CLI 遇到時間缺口會拒絕執行。若直接移除 guard，整段濾波、MAD 插值、重採樣、模型與分析窗口就會跨越不相鄰的樣本。第九階段改用既有 `lilia.neural.InferenceTimeline`，完成下游遷移與驗證後解除此入口的 guard。

- 新增 `lilia.comparison`：逐原始連續段執行 BP 0.5–45 Hz、60 Hz notch、33.25 Hz bandstop、MAD 插值、500→200 Hz 重採樣及 PyTorch OLA。保留 400 點模型窗口、200 點 hop、RMS 正規化與鏡射／Hann 政策。MAD 是此 CLI 的既有步驟，沒有加到 denoised MI 或 TFLite 流程。
- 重採樣後不足 400 點的段明示排除，保留原始 epoch、segment ID 與 raw 範圍；不為短段補造有效訊號。保留段含非有限值、無 MAD 插值錨點、模型輸出錯誤時，保存失敗 audit 並拋出例外，CLI 非零退出。
- `load_file_us` 使用整數微秒，保留 header gain／absolute offset，沿用前四通道與固定 500 Hz 假設。秒數相容 adapter 仍保留。
- APP／NUC 先按各自錄製起點配對 elapsed timestamps；從最長共同連續區間的 ch1 估單一 lag，然後以半個輸出週期容差、一對一重新配對。任一來源跳段或索引不相鄰即開始新配對段。沒有足夠共同區間或參考訊號為常數時報錯。
- TD 與 qEEG 圖按 segment 斷線；qEEG 每段重新建立完整 5 秒窗口，無完整窗口時 audit 明示。單一窗口也有可見資料點。既有流程沒有 quality scorer，CSV／圖表明示 `quality_state=disabled`，MAD 修復不被當作品質合格。
- Welch 在段內計算，以實際 Welch 子窗口數加權平均線性 power，再轉 dB；短段縮短窗口，zero-pad 到 4 秒的共同頻率格點。APP／NUC PSD 比較沿用舊版的各自完整錄製範圍，不裁成 lag 配對範圍。
- STFT 逐段計算，中心映射回真實 elapsed 時間，只繪製來源範圍內的中心。原 STFT padding 算法保留；不把補齊尾端畫到來源之外，也不跨缺口著色。

## 輸出與驗證契約

每次成功分析可產生原有類型的 20 張 PNG，以及以下 5 個 CSV 和各自 `.csv.meta.json`，另有 `app_nuc_analysis.json`：

- `app_signal.csv`、`nuc_signal.csv`：模型輸出索引、整數 timestamps、elapsed 秒、原始 segment ID、重採樣對應 fractional raw index、四通道模型輸入與兩通道模型輸出。`filtered_ch*` 欄包含濾波、MAD 修復與重採樣後的數值。
- `app_qeeg.csv`、`nuc_qeeg.csv`：完整 5 秒窗口的模型輸出索引、來源時間邊界及每通道指標；沒有完整窗口時不產生此表／對應圖。
- `app_nuc_alignment.csv`：配對兩端各自的索引與原始 timestamp、配對 segment 及 lag 後的時間殘差。

`lilia.comparison_io` 的 loader 核對 table／config／raw 指紋，從 raw timestamps 重建 inference timeline 與所有窗口／配對映射；signal／qEEG loader 可另驗模型 checkpoint。audit 保存模型及外部 architecture hash、程式指紋、MAD 修復範圍、短段排除、Welch 窗口設定、lag 參考範圍及產物 hashes。

這是來源與映射完整性檢查，不是重新執行模型來認證每個分析值；alignment loader 也不重新估計波形 lag。CSV 與 sidecar 需一起保存。

## 數值與測試

- **142 項測試通過，0 skipped**，含新增 13 項 APP／NUC 回歸測試。
- **116 個 Python 檔案編譯、Pyflakes、bundle 一致性及 diff whitespace 全通過。**
- 修改前保存 `tests/fixtures/app_nuc_continuous_reference.json/.npz`：seed 931、6003×4 連續輸入、真實模型與來源程式 hashes。
- 連續基準的濾波、MAD 修復、重採樣、模型輸出、時間、PSD、STFT、qEEG 指標，本環境最大絕對誤差全部為 **0**。MAD 修復數為 `[1536, 1770, 1376, 1721]`。
- 測試覆蓋小缺口在降採樣後仍保留、段間資料變動不污染另一段、短前綴 epoch、正負 planted lag 與舊連續裁切對照、逐段 PSD/STFT、圖形斷線及孤立點、來源／模型／重算 hash 後的錯誤映射拒絕、CLI 失敗 audit。

## 實際錄製驗證

原始資料為 `20260424_compare_10Hz.csv`（APP）及 `20260424_10Hz.csv`（NUC），均使用真實 PyTorch checkpoint 執行完整 CLI。

| 案例 | APP 輸出樣本 | NUC 輸出樣本 | APP／NUC qEEG 窗口 | 配對樣本／段 |
| --- | ---: | ---: | ---: | ---: |
| 原始連續錄製 | 11725 | 12000 | 11／12 | 11725／1 |
| 從實際錄製刪列構造不同缺口 | 10724 | 10999 | 9／10 | 10394／4 |

缺口案例的兩份檔案都保留 raw `[0,100)` 作為不足模型窗口的前綴，刪除 `[100,1000)`；APP 再刪 `[5000,6500)`，NUC 再刪 `[6000,7500)`，兩者都刪 `[12000,12004)`，形成大缺口及 10 ms 小缺口。正式原始 CSV 沒有修改。各保留三段，模型輸出從原始 elapsed APP 2.000896 秒／NUC 2 秒開始，未歸零成第一個保留樣本。

原始資料 lag 為 −32 samples／−160 ms，與舊 direct correlation 相同。缺口案例最長共同區間得到 −72 samples／−360 ms。原始／缺口案例最大配對時間殘差分別為 960／1744 µs，均在半個輸出週期 2500 µs 內；每案例的 5 表重讀及 30 個產物 hash 核對通過，缺口 STFT／qEEG 圖已目視檢查。

輸出位於 `/tmp/lilia-stage9/real_10hz/` 與 `/tmp/lilia-stage9/gapped_10hz/`。持久驗證摘要 [PYTHON_FIX_VALIDATION_2026-09-07_STAGE9.json](PYTHON_FIX_VALIDATION_2026-09-07_STAGE9.json) 保存數值、source/model/code hashes、來源段、qEEG 中心、配對與產物指紋。

## 限制與下一步

APP／NUC header absolute offsets 相差約 38 分鐘，因此此比較沿用各自 elapsed 的波形對齊，**不是兩台裝置的絕對同步證據**。10 Hz 週期訊號的相關峰可能有多個；不同參考區間的 lag 不應解讀成已校準的裝置時間延遲。單一 lag、時鐘漂移及研究方法校準仍屬後續工作。

APP 真實 timestamps 帶有 jitter，qEEG 中心保留原始時間映射，例如第一個連續窗口為 2.499616 秒，不再強制寫成樣本數推算的 2.5 秒。上述逐值相同基準是等間距的連續 fixture。

本 CLI 固定假設 500 Hz、使用前四通道、未加入品質評分；缺口前後的濾波邊界與 PSD 聚合是明示方法政策，不宣稱與舊串接方法等價。舊 `align_for_comparison` 相容 helper 仍只接受連續陣列，CLI 已不使用它。

路徑解析失敗發生在 audit 建立之前。分析期失敗可留下 partial outputs；尚未實作整批產物原子發佈或自動移除上次輸出，重跑建議用新 outdir，並以本次 audit 的狀態／產物清單判讀結果。

下一階段處理 `plot_event_markers.py --hardy2-analysis` 特殊入口：其整段濾波、按樣本起窗、跨段平滑與 period summary 尚未遷移。其後仍有 compare_subjects、meditation、qEEG CLI、眼開閉與 Jenqwei、品質 scorer／baseline 方法、入口拆分與整體驗收。
