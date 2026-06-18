# Lilia EEG Analysis

本專案包含多個 Python scripts / modules，涵蓋 EEG 訊號分析、qEEG wellness indices、band entropy、EEG 品質評分、受試者資料合併、活動標記驗證圖、以及模型轉檔。

## Python Scripts 總覽

| Script | 功能 |
| --- | --- |
| `data_analysis.py` | 比較 APP 與 NUC EEG 資料，進行前處理、500→200 Hz downsampling、TinyUNetV4 模型推論，並輸出分析圖（含 qEEG indices 圖） |
| `qeeg_indices.py` | 實作 Appendix J §3 的 qEEG wellness indices（Focus / Flow / Calm / Relaxation），可獨立執行或被其他 script 匯入 |
| `spectral_entropy.py` | 使用 Welch PSD 計算 theta / alpha / beta 三個頻段能量、正規化成比例後計算 BandEn；另含 baseline-vs-event 比較、左右腦非零時滯互資訊同步，以及 joint probability distribution → mutual information 模式（可選 TinyUNetV4 denoise 後的 ch1/ch2） |
| `eeg_quality_v2.py` | 從 SleepStage 抽出的獨立 EEG quality v2 scorer，核心為 `get_eeg_quality_index_v2_parametric()`，搭配 `DEFAULT` / `BEST_MEAN_ABS_CORR` 參數集與 `get_best_eeg_quality_v2_flat_spectrum_only_params()`；另含針對 iBrainCenter 4-ch 消費級裝置實測校準的 `get_ibrain_device_eeg_quality_v2_params()`（主流程預設採用） |
| `eeg_utils.py` | 共用低階工具：`load_merged_csv()`（4-row header CSV 載入）、`bandpass_filter()`（零相位 Butterworth bandpass） |
| `merge_subject_csvs.py` | 將 `iBrainCenter/` 與 `YoGa/` 底下各 subject 的多個 CSV 合併為單一 `merged.csv` |
| `plot_event_markers.py` | 將 evt_time.docx 的活動時間點疊加至 iBrainCenter 各 subject 的 merged.csv，輸出驗證圖（含 EEG quality、qEEG heatmap、30s smooth summary、event-level delta，以及 TFLite 模型對照組） |
| `plot_tflite_summary.py` | 以 `plot_event_markers.py` 的 Summary 風格產生 TFLite 專屬分析圖：原始 Relax / Calm / Flow / Focus 趨勢、TFLite 前後品質比較、qEEG Δ heatmap，並採「緩衝區 + 1s 微分段 + 盲抽樣（固定種子）」建構事件前基線 |
| `check_quality_anomalies.py` | 診斷 EEG Quality 曲線上的異常線段：逐視窗偵測資料斷點 / ADC 削波 / 平線 / 低品質 / 驟跳，並與原始時域波形對照，分辨「連線假影」與「真實壞訊號」 |
| `plot_tyy_meditation.py` | 針對 TYY (SN041) 的 Mindfulness Meditation 區段，輸出 Ch1 / Ch2 波形與 BP / TFLite qEEG Δ heatmap 的聚焦圖（PNG + SVG） |
| `compare_subjects.py` | 跨受試者比較 BP 與 TFLite qEEG delta index（iBrainCenter 以 event block、YoGa 以整段 session 計算） |
| `sample_quality_check.py` | 從每份 merged.csv 隨機取樣 2 個 30 秒片段，計算 EEG quality 與 qEEG indices，輸出品質檢查圖 |
| `plot_raw_eeg.py` | 從每份 iBrainCenter merged.csv 隨機取樣 N 個 30 秒非重疊片段，繪製未濾波、未下採樣的 4-ch 原始 EEG 波形 |
| `convert_to_tflite.py` | 將 PyTorch checkpoint (`tiny_v4_optimized.pth`) 轉換成 float32 TFLite 模型 |

## 專案結構

```text
lilia_analysis/
├── data_analysis.py           # APP vs NUC 分析主程式
├── qeeg_indices.py            # qEEG wellness indices（Appendix J §3）
├── spectral_entropy.py        # Welch PSD + three-band entropy + joint-MI
├── eeg_quality_v2.py          # 獨立 EEG quality v2 scorer
├── eeg_utils.py               # 共用工具：CSV 載入、bandpass filter
├── merge_subject_csvs.py      # 合併各 subject 的多個 CSV
├── plot_event_markers.py      # 活動時間標記驗證圖（含 quality / qEEG / TFLite 對照面板）
├── plot_tflite_summary.py     # TFLite Summary 分析圖（raw 趨勢 + 品質前後比較 + Δ heatmap + 微分段基線）
├── check_quality_anomalies.py # EEG Quality 異常線段診斷（斷點/削波/平線 vs 原始時域對照）
├── plot_tyy_meditation.py     # TYY 冥想區段聚焦圖（Ch1/Ch2 + BP/TFLite qEEG Δ heatmap）
├── compare_subjects.py        # 跨受試者 BP / TFLite qEEG delta 比較
├── sample_quality_check.py    # 隨機取樣品質檢查圖
├── plot_raw_eeg.py            # 隨機取樣原始 EEG 波形圖（未濾波、未下採樣）
├── convert_to_tflite.py       # PyTorch → TFLite 轉換
├── Appendix_J_qEEG_Description.pdf
├── tiny_v4_optimized.pth
├── tiny_v4_optimized.tflite
├── 20260424_compare_10Hz.csv
├── 20260424_10Hz.csv
├── 20260424_compare_1Hz.csv
├── 20260424_1Hz.csv
├── 20260424_compare_ECEO.csv
├── 20260424_ECEO.csv
├── 10Hz/                      # data_analysis.py 輸出（--group 10Hz）
├── 1Hz/                       # data_analysis.py 輸出（--group 1Hz）
├── ECEO/                      # data_analysis.py 輸出（--group ECEO）
├── iBrainCenter/
│   ├── Ann(SN027)/
│   │   ├── 20260512_141051.csv
│   │   └── merged.csv
│   ├── Hardy(SN036)/
│   │   ├── 20260512_141048.csv  ┐
│   │   ├── 20260512_144743.csv  │
│   │   ├── 20260512_144907.csv  ├ 6 files → merged.csv
│   │   ├── 20260512_145126.csv  │
│   │   ├── 20260512_152411.csv  │
│   │   ├── 20260512_152642.csv  ┘
│   │   └── merged.csv
│   ├── Hsin(SN032)/  ...
│   ├── James(SN035)/ ...
│   ├── TYY(SN041)/   ...
│   ├── event_verification/    # plot_event_markers.py 輸出的驗證圖
│   ├── tflite_summary/        # plot_tflite_summary.py 輸出的 TFLite Summary 圖
│   ├── quality_anomalies/     # check_quality_anomalies.py 輸出的品質異常診斷圖
│   ├── comparison/            # compare_subjects.py 輸出（iBrainCenter group）
│   └── TYY_meditation/        # plot_tyy_meditation.py 輸出（PNG + SVG）
├── sample_quality/            # plot_raw_eeg.py 輸出的原始波形圖
└── YoGa/
    ├── James(SN035)/ ...
    ├── comparison/            # compare_subjects.py 輸出（YoGa group）
    ├── Jammie(SN036)/
    │   ├── 20260513_135528.csv  ┐
    │   ├── 20260513_151700.csv  ├ 3 files → merged.csv
    │   ├── 20260513_152819.csv  ┘
    │   └── merged.csv
    └── TYY(SN041)/ ...
```

## 執行環境

主要使用套件：`numpy`、`pandas`、`matplotlib`、`scipy`、`torch`、`tensorflow`

`data_analysis.py` 與 `convert_to_tflite.py` 從下列路徑匯入 `TinyUNetV4`：

```python
sys.path.insert(0, '/home/bps-yichin/tommy')
from eeg_denoise.tiny_model_v4 import TinyUNetV4
```

若移至其他機器，需修改上述路徑。

---

## data_analysis.py

比較 APP 與 NUC 的 EEG CSV 檔案，並產生 time-domain、PSD、STFT 及 qEEG indices 等分析圖。
`qeeg_indices` 的計算邏輯已拆分至 `qeeg_indices.py`，透過 import 使用。

目前流程會先以 500 Hz 進行濾波與 artifact removal，之後將訊號以 `scipy.signal.resample_poly` 降採樣到 200 Hz，再送入模型與後續分析。

### 分析流程

1. 讀取 APP 與 NUC CSV。
2. 訊號前處理：0.5–45 Hz bandpass → 60 Hz notch → 33.25 Hz bandstop。
3. MAD threshold artifact 偵測與 interpolation 修補。
4. 將清理後訊號從 500 Hz downsample 到 200 Hz。
5. TinyUNetV4 模型推論（4-ch in → 2-ch out）。
6. 以 200 Hz 訊號做 cross-correlation 估計 APP/NUC lag 並對齊。
7. 輸出 time-domain、PSD、STFT 比較圖。
8. 對模型輸出 channel 計算 qEEG wellness indices 並輸出圖。

### 重要參數

```python
BASE_DIR     = '/home/bps-yichin/lilia_analysis'
FS           = 500       # Hz
DOWNSAMPLED_FS = 200     # Hz, artifact removal 後供模型與後續分析使用
N_CH         = 4         # 模型輸入 channels
N_CH_OUT     = 2         # 模型輸出 channels
MODEL_WINDOW = 400       # samples
MODEL_PATH   = os.path.join(BASE_DIR, 'tiny_v4_optimized.pth')
```

補充：artifact removal 前後圖仍基於原始 500 Hz 訊號；模型前後比較、APP/NUC lag、PSD、STFT、qEEG indices 則使用 downsample 後的 200 Hz 訊號。

### 執行方式

```bash
# 使用 group label（自動尋找對應 CSV）
python data_analysis.py --group 10Hz
python data_analysis.py --group 1Hz
python data_analysis.py --group ECEO

# 指定檔案
python data_analysis.py \
  --app /path/to/compare.csv \
  --nuc /path/to/nuc.csv \
  --outdir /path/to/output_dir
```

### CSV 命名規則

| 類型 | 搜尋規則 | 範例 |
| --- | --- | --- |
| APP | `*compare_<LABEL>.csv` | `20260424_compare_10Hz.csv` |
| NUC | `*_<LABEL>.csv`（排除含 `compare`） | `20260424_10Hz.csv` |

### 輸出檔案

| 檔名 | 說明 |
| --- | --- |
| `app_artifact_removal.png` | APP artifact removal 前後波形 |
| `app_psd_artifact_removal.png` | APP artifact removal 前後 PSD |
| `nuc_artifact_removal.png` | NUC artifact removal 前後波形 |
| `nuc_psd_artifact_removal.png` | NUC artifact removal 前後 PSD |
| `app_model_before_after_td.png` | APP 模型前後 time-domain |
| `app_model_before_after_psd.png` | APP 模型前後 PSD |
| `nuc_model_before_after_td.png` | NUC 模型前後 time-domain |
| `nuc_model_before_after_psd.png` | NUC 模型前後 PSD |
| `time_domain_comparison.png` | APP vs NUC filtered 4-ch 波形比較 |
| `psd_comparison.png` | APP vs NUC filtered 4-ch PSD 比較 |
| `model_output_time_domain.png` | APP vs NUC 模型輸出波形比較 |
| `model_output_psd.png` | APP vs NUC 模型輸出 PSD 比較 |
| `app_model_stft.png` | APP 模型前後 STFT |
| `nuc_model_stft.png` | NUC 模型前後 STFT |
| `stft_filtered_comparison.png` | APP vs NUC filtered STFT 比較 |
| `stft_model_output_comparison.png` | APP vs NUC 模型輸出 STFT 比較 |
| `qeeg_indices_app_ch<N>.png` | APP 模型輸出 ch N 的 qEEG indices |
| `qeeg_indices_nuc_ch<N>.png` | NUC 模型輸出 ch N 的 qEEG indices |

---

## qeeg_indices.py

實作 Appendix J Chapter 3 的 qEEG wellness indices。可作為 module 被 `data_analysis.py` import，也可獨立執行。

### 實作內容（對應 Appendix J）

| 函式 | §  | 說明 |
| --- | --- | --- |
| `compute_relative_powers` | 3.1 | 計算 θ / α / β 相對功率（排除 Delta 與 Gamma），三者相加 ≈ 1 |
| `bounded_ratio(E, I)` | 3.2 | `clamp((E−I)/(E+I+ε), −1, 1)` |
| `focus_index` | 3.3 | 持續注意力：高 β（去 EMG）vs 抑制 α/θ |
| `flow_index` | 3.3 | 心流：α-θ 同步，加入 β flexibility 與 α-θ imbalance 懲罰項 |
| `calm_index` | 3.3 | 平靜清醒：θ+α vs β+excess-theta（避免將嗜睡誤判為平靜） |
| `relaxation_index` | 3.3 | 深度放鬆：高 α 主導，高 β 與過量 θ 均有懲罰 |

頻段定義為半開區間 `[fmin, fmax)`，相鄰頻段不共用邊界 bin：Theta 4–8 Hz、Alpha 8–13 Hz、Beta 13–30 Hz（與 `spectral_entropy.py` 一致）。

### 獨立執行

```bash
python qeeg_indices.py --csv <path.csv> [--fs 500] [--ch 1] [--win 5] [--out <dir>]
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--csv` | （必填） | 輸入 CSV（lilia 格式） |
| `--fs` | 500 | 取樣率 (Hz) |
| `--ch` | 1 | 1-based channel index |
| `--win` | 5 | 視窗長度（秒） |
| `--out` | CSV 所在目錄 | 輸出目錄 |

---

## spectral_entropy.py

以 2 秒 EEG window 為單位，先用 Welch's method 計算 PSD，再整合三個任務相關 EEG 頻段能量：

- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz

> **為何丟棄 Delta / Gamma：** 與 `qeeg_indices.compute_relative_powers` 一致，僅以 θ/α/β 正規化、排除 delta（0.5–4 Hz，乾電極在動作時主要為移動 / 汗液 / 漂移假影）與 gamma（30–45 Hz，主要為 EMG 並逼近電源雜訊區）。這讓 BandEn 與 Flow / Focus / Calm / Relax 指標直接可比；最大熵由 log2(5) ≈ 2.322 bits 降為 log2(3) ≈ 1.585 bits。

接著計算：

- `E_total = E_theta + E_alpha + E_beta`
- `p_k = E_k / E_total`
- `BandEn = -sum(p_k * log2(p_k))`（最大值 log2(3) ≈ 1.585 bits）

另外可選擇對一組左右通道加入**非零時滯互資訊**同步分析：

- `I(X(t); Y(t+tau))`，其中 `tau > 0`
- 預設 `tau = 5, 10, 15, 20 ms`
- 以雙向平均 `0.5 * [I(L(t); R(t+tau)) + I(R(t); L(t+tau))]` 作為該 `tau` 的左右腦同步量
- 再輸出各 `tau`、`lagged_mi_mean`、`lagged_mi_max`、`lagged_mi_best_tau_ms`

> **設計依據：** 容積傳導（volume conduction）為即時物理效應（時間差 ≈ 0），因此在 `tau = 0` 的互資訊中會被計入。引入 `tau > 0`（如 5–20 ms）可完全過濾這類偽同步訊號，抓到兩半球間真正的**動態資訊交換**（例如透過胼胝體的跨半球傳遞）。

### Joint probability distribution → mutual information（`--joint-mi`）

估計兩通道的 **2-D 聯合機率分布 `P(X, Y)`**，並由同一分布求互資訊：

- `I(X; Y) = H(X) + H(Y) - H(X, Y)`（bits），代數上等同於 `Σ P(x,y) log2[P(x,y)/(P(x)P(y))]`
- 加上 `--denoise` 時，兩通道為 **TinyUNetV4 神經網路去噪輸出**（4 raw ch 進 → 2 denoised ch 出 @ 200 Hz，與 `data_analysis.py` 推論流程一致），即「denoised ch1 vs denoised ch2」；否則使用 `--joint-pair` 指定的（bandpass 後）通道。
- 輸出：整段錄製的 `P(X,Y)` 熱圖（含邊際分布與 MI 標註）、滑動視窗的零時滯 MI 時序，以及單列摘要 CSV。

> **分箱策略（`--mi-binning`，重點）：** 預設為 **quantile（等機率分箱）**——各軸 bin 邊界放在資料分位數上，使每個 bin 樣本數 ≈ 相等。對乾電極 EEG 這類重尾、含假影的訊號至關重要：等寬（`uniform`）分箱會被假影撐大的振幅範圍稀釋，使分布塌縮到中央少數 bin（邊際熵遠低於 `log2(bins)` 上限），嚴重低估 MI（實測差約 5×：0.025 → 0.133 bits）。quantile 分箱讓邊際熵達到 `log2(bins)` 上限，並對重尾穩健。熱圖以等格的 **bin-index（copula / rank）空間** 呈現，邊際因等機率而呈平坦，對角線上的相依結構（MI 實際量到的部分）才看得見。

> **偏差與顯著性：** plug-in 直方圖 MI 為正偏；因此同時回報 **Miller–Madow 偏差校正 MI**，以及 **circular-shift surrogate 虛無分布**（保留各通道自相關、僅破壞跨通道耦合）給出的 p-value 與 z 分數。

### 獨立執行

```bash
# 基本使用
python spectral_entropy.py --csv <path.csv> [--fs 500] [--ch 1] [--win 2] [--step 2] [--out <dir>]

# 加入左右腦同步分析
python spectral_entropy.py --csv <path.csv> --sync-pair 1 2 [--tau-ms 5 10 15 20]

# 加入 iBrainCenter 活動標記，並將 x 軸轉為絕對時間（UTC+8）
python spectral_entropy.py --csv <path.csv> --ibrain-events

# 完整組合
python spectral_entropy.py --csv iBrainCenter/Ann(SN027)/merged.csv \
    --ch 1 --sync-pair 1 2 --tau-ms 5 10 15 20 --ibrain-events

# Joint probability distribution → mutual information（denoised ch1 vs ch2）
python spectral_entropy.py --csv <path.csv> --joint-mi --denoise

# 不去噪，直接比較 bandpass 後的 ch1 / ch2（可改分箱策略）
python spectral_entropy.py --csv <path.csv> --joint-mi --joint-pair 1 2 \
    --mi-binning quantile --mi-bins 16 --mi-surrogates 200

# Joint MI 對齊活動：時序圖疊加事件 + 每個活動「事件前 vs 起始」分布比較
python spectral_entropy.py --csv iBrainCenter/Hardy(SN036)/merged.csv \
    --joint-mi --denoise --ibrain-events
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--csv` | （必填） | 輸入 CSV（lilia 格式） |
| `--fs` | 500 | 取樣率 (Hz) |
| `--ch` | 1 | 1-based channel index |
| `--win` | 2 | 分析視窗長度（秒） |
| `--step` | `--win` | 滑動步長（秒） |
| `--sync-pair` | 無 | 可選，指定 1-based 左右通道配對，追加非零時滯互資訊同步分析 |
| `--tau-ms` | `5 10 15 20` | 可選，指定 lagged MI 的毫秒延遲列表，必須皆大於 0 |
| `--mi-bins` | 16 | 互資訊直方圖分箱數 |
| `--joint-mi` | 未設定 | 啟用 joint probability distribution → mutual information 模式（與 band-entropy / baseline 模式互斥，會直接執行並輸出後返回） |
| `--joint-pair` | `1 2` | `--joint-mi` 非去噪時的 1-based 通道配對；`--denoise` 時忽略（固定用兩個模型輸出） |
| `--denoise` | 未設定 | `--joint-mi` 限定：以 TinyUNetV4 產生兩個去噪通道（需 PyTorch 與 eeg_denoise 模型套件） |
| `--mi-binning` | `quantile` | `--joint-mi` 限定：`quantile`（等機率，建議）或 `uniform`（等寬） |
| `--mi-surrogates` | 200 | `--joint-mi` 限定：circular-shift surrogate 數量（0 關閉顯著性檢定） |
| `--out` | CSV 所在目錄 | 輸出目錄 |
| `--ibrain-events` | 未設定 | 啟用後：x 軸改為絕對本地時間（HH:MM:SS UTC+8），並疊加 iBrainCenter 活動色塊與起始標記；在 `--joint-mi` 模式下另外產生「事件前 vs 起始」聯合分布比較（見下節） |

### 活動標記說明（`--ibrain-events`）

加上 `--ibrain-events` 後，所有子圖（各 Band Proportion、BandEn、Sync）均會疊加下列視覺元素：

- **色塊**（`axvspan`）：活動持續期間的半透明背景，每個活動有獨立顏色
- **起始虛線**（`axvline`）：活動開始時間
- **旋轉文字標籤**：標示活動名稱，貼齊起始線左緣

活動定義來自 `plot_event_markers.EVENTS`（見 `plot_event_markers.py` 章節）。

在 **`--joint-mi` 模式**下，`--ibrain-events` 還會：

1. 將 **Zero-lag MI 時序圖** x 軸轉為絕對時間並疊加事件標記；
2. 對每個活動，分別計算 **事件前 `[onset − 30s, onset)`** 與 **起始 `[onset, onset + 30s)`** 兩段的聯合機率分布 `P(X,Y)` 與 MI，並輸出 `ΔMI = onset − pre`（活動起始時左右腦耦合的變化量）。事件起始時間以絕對 UTC µs 對齊錄製時間戳，落在錄製範圍外或視窗不足的活動會自動略過。

### 輸出檔案

| 檔名 | 說明 |
| --- | --- |
| `<basename>_band_entropy_ch<N>.csv` | 每個 window 的時間點、五個頻段能量、總能量、各頻段比例、BandEn |
| `<basename>_band_entropy_ch<N>.png` | 每個 band 各自獨立子圖的比例時序圖，並疊加 smooth 趨勢線；最下方附 BandEn |
| `<basename>_band_entropy_ch<N>_sync_ch<L>_ch<R>.csv` | 在原 BandEn 欄位外，追加各 `tau` 的 lagged MI、`lagged_mi_mean`、`lagged_mi_max`、`lagged_mi_best_tau_ms` |
| `<basename>_band_entropy_ch<N>_sync_ch<L>_ch<R>.png` | 在原 BandEn 圖下方追加左右腦非零時滯互資訊同步面板；指定 `--ibrain-events` 時 x 軸改為絕對時間並疊加活動標記 |
| `<basename>_joint_mi_<pair>_summary.csv` | （`--joint-mi`）單列摘要：通道、`fs`、`mi_bins`、`mi_binning`、有效 bin 數、樣本數、plug-in / Miller–Madow / 正規化 MI、各熵、surrogate 平均 / 標準差 / p-value / z；`<pair>` 為 `denoised_ch1_ch2` 或 `ch<X>_ch<Y>` |
| `<basename>_joint_mi_<pair>_timeseries.csv` / `.png` | （`--joint-mi`）滑動視窗零時滯 MI 時序（`time_s`、`joint_mi`、`joint_mi_norm`）及其圖 |
| `<basename>_joint_mi_<pair>_distribution.png` | （`--joint-mi`）聯合機率分布 `P(X,Y)` 熱圖（bin-index / copula 空間）＋邊際分布＋MI / 顯著性標註 |
| `<basename>_joint_mi_<pair>_events.csv` / `.png` | （`--joint-mi --ibrain-events`）每個活動「事件前 vs 起始」的聯合 MI（plug-in / Miller–Madow / 正規化）、樣本數與 `ΔMI`，及其成對長條圖 |

---

## merge_subject_csvs.py

掃描 `iBrainCenter/` 與 `YoGa/` 底下所有 `subject(***)` 資料夾，將同一 subject 的多個 CSV 依 `Abs Time Offset[us]` 轉換成絕對時間後合併，產生 `merged.csv`。

### 合併邏輯

1. 讀取每個 CSV 第 2 行的 `Abs Time Offset[us]`（容許 float / 含引號 / 含空白的數值）。
2. 將各 CSV 的相對 `Time[us]` 加上 offset → 絕對時間戳記；空白或非數值的時間列會被略過而非中斷合併。
3. 合併後依絕對時間穩定排序，僅去除「完全相同的整列」（避免不同檔案在同一時間戳的不同樣本被誤刪）。
4. 以最早 offset 對應的 CSV header 為基準，更新 `Abs Time Offset[us]` 欄位。
5. 寫出 `<subject_folder>/merged.csv`。

### 執行方式

```bash
# 預設掃描 iBrainCenter/ 與 YoGa/
python merge_subject_csvs.py

# 指定其他根目錄
python merge_subject_csvs.py --roots /path/to/dir1 /path/to/dir2

# 自訂輸出檔名（預設 merged.csv）
python merge_subject_csvs.py --outname combined.csv
```

### 合併結果（當前資料）

| Group | Subject | 來源檔數 | Total Points | 時間長度 |
| --- | --- | --- | --- | --- |
| iBrainCenter | Ann (SN027) | 1 | 1,965,633 | 3931 s |
| iBrainCenter | Hardy (SN036) | 6 | 2,126,712 | 4794 s |
| iBrainCenter | Hsin (SN032) | 1 | 1,978,896 | 3958 s |
| iBrainCenter | James (SN035) | 4 | 1,906,544 | 3990 s |
| iBrainCenter | TYY (SN041) | 1 | 2,009,328 | 4019 s |
| YoGa | James (SN035) | 1 | 2,459,472 | 4919 s |
| YoGa | Jammie (SN036) | 3 | 2,820,000 | 5691 s |
| YoGa | TYY (SN041) | 1 | 2,879,360 | 5758 s |

---

## plot_event_markers.py

將 `evt_time.docx` 紀錄的活動時間點疊加至 iBrainCenter 各 subject 的 `merged.csv`，輸出 PNG 驗證圖；YoGa subject 也會輸出 EEG overview。
每張圖分為 **N 個 EEG channel 子圖**、**EEG quality 子圖**、**qEEG heatmap**、**30s smooth summary**，若有活動標記則再加上 **event-level delta bar chart**。

### 圖表內容

| 子圖 | 說明 |
| --- | --- |
| ch1 … chN | 下採樣後的原始 EEG 波形，疊加活動色塊與起始虛線 |
| Quality | 每個 channel 的 `overall_quality`（0–1），以 5 s non-overlapping window @ 500 Hz 計算 |
| BP qEEG Δ heatmap | 以 5 s window 計算 Focus / Flow / Calm / Relaxation（bandpass 版），取 channel median 後再彙整成 30 s bin，顯示相對 baseline 的 delta |
| BP Summary (30s smooth) | 3 條重點線：Focus、Restfulness、Engagement（bandpass 版）|
| TFLite qEEG Δ heatmap | 同上，但資料來自 TFLite 模型輸出（200 Hz）|
| TFLite Summary (30s smooth) | 同上，但資料來自 TFLite 模型輸出 |
| BP Event Block Δ | 針對每個參與活動區段，顯示 bandpass 版 qEEG 相對 baseline 的平均 delta 與跨 channel 標準差 |
| TFLite Event Block Δ | 同上，但資料來自 TFLite 模型輸出 |

### 2026-05-19 繪圖優化

- `Index (30s smooth)` 面板改名為 `Summary (30s smooth)`。
- 趨勢線由 6 條減為 3 條：`Focus`、`Restfulness`、`Engagement`。
- `Restfulness = (Calm + Relaxation) / 2`；`Engagement = Focus - Restfulness`。
- `Flow`、`Calm`、`Relaxation` 的細節仍保留在 qEEG heatmap 與 event-level delta bar chart，不再重複塞進 smooth trend panel。
- `Restfulness` 與 `Engagement` 顏色調整為較清楚的紫色與青色，並微調線寬、legend 與 trend panel 高度，讓重點趨勢更容易掃讀。

### 活動時間表（2026-05-12，Asia/Taipei UTC+8）

| 活動 | 開始 | 時長 | 參與者 |
| --- | --- | --- | --- |
| Single Cycling | 14:13 | 3 min | Hsin, Hardy, James |
| Cycling Boxing | 14:17 | 3 min | Hsin, Hardy, James |
| Push-ups | 14:24 | 5 min | 全員 |
| Machine Chest Press | 14:35 | 3 min | 全員 |
| Agility Ladder | 14:43 | 7 min | 全員 |
| Color Agility Ladder | 14:50 | 5 min | Hardy, Ann, Hsin, James |
| Cone Rotation | 14:58 | 5 min | 全員（含 3 階段細分） |
| Mindfulness Meditation | 15:06 | 11 min | 全員 |

### Quality 計算參數

- 使用 `eeg_quality_v2.get_ibrain_device_eeg_quality_v2_params()`（裝置校準參數集）  
  → `kurtosis_weight=0, corr_weight=0`（只計算 flat + spectrum 兩項）
- 取樣率：**500 Hz**（原始取樣率，不做下採樣）
- 視窗：**5 秒 non-overlapping**
- 紅色虛線標示 threshold（`QUALITY_THRESHOLD=0.5`）

> **為何改用裝置校準參數集？** 原 `flat_spectrum_only` 參數集是為較乾淨的 EEG 調的，套在 iBrainCenter 4-ch 消費級裝置上，正常資料的品質中位數只有 ~0.5，導致每段錄製有 **40–60% 視窗**被誤判為低品質而捨棄。經五位受試者實測校準後（見下表），乾淨資料分數回到 ~0.7–0.9，0.5 門檻才能正確區隔「乾淨 (~0.7+)」與「假影 (~0.3)」，而非把乾淨資料一刀切兩半。

| 參數 | 原值 | 校準值 | 原因 |
| --- | --- | --- | --- |
| `flat_activity_k` | 0.8 | **0.35** | 活動度分數 `1−exp(−ratio/k)` 在 k=0.8 時，連完美穩態訊號上限也只有 ~0.71；調小 k 才讓乾淨訊號逼近 1.0 |
| `spectrum_fit_hi` | 45 Hz | **40 Hz** | 原本斜率擬合上緣壓在 `BP_HIGH=45` 的濾波器滾降上，使斜率被高估（偏陡） |
| `slope_center` | −2.0 | **−1.0** | 本裝置實測 1/f 斜率中位數 ≈ −0.9（非研究級 EEG 的 −2.0），原中心使乾淨資料落在帶緣只得 ~0.4 |
| `slope_good_low/high` | −3.5 / −0.5 | **−2.5 / −0.2** | 配合上述中心重新設定「良好」斜率帶 |
| `slope_edge_score` | 0.4 | **0.7** | 讓落在良好帶內的分數維持較高，而非從中心快速跌到 0.4 |

### qEEG 計算與摘要

- EEG 先經 0.5–45 Hz bandpass filter（500 Hz）。
- qEEG wellness indices 使用 `qeeg_indices.compute_qeeg_indices()` 計算 Focus / Flow / Calm / Relaxation。
- qEEG window：**5 秒 non-overlapping**。
- 趨勢圖使用 channel median，並套用 **30 秒 centered rolling mean**。
- 低品質 window（quality median < `QUALITY_THRESHOLD=0.5`）會在 qEEG trend / heatmap / event delta 中排除或遮罩。
- baseline 預設為第一個參與活動開始前的區段；若沒有活動標記，使用前 20% bins 作 baseline。

### TFLite 對照比較

若 `tiny_v4_optimized.tflite` 存在，會自動進行以下額外處理：

1. 將 bandpass 濾波後的資料從 **500 Hz → 200 Hz**（`scipy.signal.resample_poly`）。
2. 以非重疊 **400-sample window（= 2 s @ 200 Hz）** 送入 TFLite 模型。
3. 對模型輸出（2-ch, 200 Hz）同樣計算 qEEG indices、heatmap、30s smooth trend 與 event-level delta。
4. 在圖表中新增 4 個 TFLite 對照子圖，方便與 bandpass 版本直接比較。

使用 `--no-tflite` 旗標可跳過此步驟。

### 執行方式

```bash
python plot_event_markers.py [--ibrain-outdir <dir>] [--yoga-outdir <dir>] [--ds <factor>] [--no-tflite]
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--ibrain-outdir` | `iBrainCenter/event_verification/` | iBrainCenter PNG 輸出目錄 |
| `--yoga-outdir` | `YoGa/eeg_overview/` | YoGa PNG 輸出目錄 |
| `--ds` | 500 | EEG 波形下採樣倍率（500 → 1 pt/s） |
| `--no-tflite` | （未設定時啟用 TFLite 對照） | 跳過 TFLite 模型推論與對照面板 |

### 輸出檔案

每個 subject 輸出一張 PNG：

```
iBrainCenter/event_verification/
├── Ann_SN027_eeg.png
├── Hsin_SN032_eeg.png
├── Hardy_SN036_eeg.png
├── TYY_SN041_eeg.png
└── James_SN035_eeg.png

YoGa/eeg_overview/
├── James_SN035_eeg.png
├── Jammie_SN036_eeg.png
└── TYY_SN041_eeg.png
```

---

## plot_tflite_summary.py

以 `plot_event_markers.py` 中 `*_session_start.png` / `*_pre_event_rest.png` 的 **Summary 子圖風格**改寫，產生 **TFLite 專屬**的分析圖，並針對基線（baseline）建構導入更嚴謹、符合同行審查的流程。資料來源只取 **TFLite 模型處理後**的訊號計算 qEEG 指標。

### 圖表內容（由上而下）

| 子圖 | 說明 |
| --- | --- |
| Signal quality (before vs after) | TFLite 處理「前」與「後」的 channel-median 品質曲線疊圖，檢視模型是否改變/劣化訊號品質。**兩者皆在 200 Hz 評分**（前 = 僅降採樣未經 TFLite、後 = TFLite 重建），使唯一差別是模型重建本身、而非降採樣（apples-to-apples）|
| TFLite Summary (small multiples) | 每個指標各一條 strip（Relax / Calm / Flow / Focus），以「小倍數」呈現而非四線疊圖，較易讀且各自用滿縱軸。每條 strip：填色面積 = **原始**指標值（−1~+1，30 s smooth，**不減基線**）；黑色虛線 = 基線水準（填色與虛線的落差即 Δ，故單圖同時呈現絕對值與相對變化）。低品質/無訊號視窗（含時間斷點）一律捨棄並斷開，無效區段自然留白 |
| qEEG Δ heatmap (vs baseline) | 沿用 `plot_event_markers.py` 的 Δ-vs-baseline 熱圖（30 s bins、OrgPur colormap、固定 ±`HEATMAP_DELTA_VABS`=2.0 色階）；著色範圍依「heatmap 基線模式」而定（見下） |
| Per-event mean Δ (bar) | 各事件相對基線的平均 Δ，誤差棒為跨 channel 標準差 |

### Heatmap 基線模式（`--heatmap-baseline`）

heatmap 與長條圖的 Δ 可選兩種基線，解決「事件外/基線不足導致大量灰格」的問題：

| 模式 | 基線來源 | 著色範圍 | 適用 |
| --- | --- | --- | --- |
| `session`（預設） | 整段 session 隨機抽樣乾淨 1 s 微分段（固定種子）為單一基線 | **整條時間軸**都著色，灰格僅剩真正低品質的 bin | 想看「相對 session 平均狀態」的偏離、減少灰格 |
| `pre-event` | 每個事件各自的事件前微分段基線（含 Color Agility Ladder 特例） | 僅事件視窗內著色，事件之間維持灰色 | 聚焦「各任務 vs 其事前靜息」的變化 |
| `both` | 兩者皆建 | 各輸出一張圖 | 同時比較兩種觀點 |

> **為何 `session` 模式灰格大減？** `pre-event` 模式下，事件之間（佔約 40% 時間）依設計就是灰色，加上資料品質差的事件其微分段基線會建構失敗而整段留白。`session` 模式以整段隨機乾淨樣本為單一參照，每個品質足夠的 bin 都能算出 Δ，因此只有真正壞掉（低品質/斷點）的 bin 才會是灰色。

> **為何趨勢用 raw、heatmap/bar 用 Δ？** 基線相減（Δ）只在「比較事件 vs 靜息」時有意義；要判讀受試者「當下的絕對狀態」應看原始指標。qEEG 指標本身即為有界比值（−1~+1），可跨時段直接比較，故 Summary 趨勢採原始值，基線僅保留給本質上是「相對變化量」的 heatmap 與 bar。

### 事件前基線建構（核心方法）

`build_baseline_epochs()` 以下列步驟建立每個事件的乾淨基線，所有步驟皆有神經科學或方法學依據：

1. **緩衝區 Buffer Zone（`t_buffer = -3.0 s`）**：嚴格捨棄觸發點前 3 秒內的資料。受試者在「預期」任務開始時會出現預期焦慮波、關聯性負變化（CNV, Contingent Negative Variation）與 α 去同步化（alpha ERD），屬「任務預備」而非「靜息」狀態，納入會系統性污染基準值。
2. **微分段 Micro-epoching**：將搜尋窗 `[t_search_start, t_buffer]`（預設 −15 ~ −3 s）切成 1 秒非重疊微分段，逐段做假影/雜訊篩選，僅剔除含眨眼、EMG、移動假影的片段，而非整段全取或全棄。
3. **ADC 飽和篩選（原始訊號）**：在**未濾波**的原始視窗上偵測削波（任一通道貼在 ±2048 滿格、比例 > `SAT_FRAC_MAX=2%` 即剔除）。因為帶通濾波會把削波平滑掉（實測 64% 削波的視窗在濾波前評分 0.19、濾波後升到 0.42），單看濾波後分數會漏抓飽和；故先在原始訊號把關，再做品質評分。
4. **品質篩選**：每個微分段以 `eeg_quality_v2`（裝置校準參數集）評分，channel 中位數 ≥ `QUALITY_THRESHOLD` 才視為乾淨。
5. **盲抽樣 + 固定亂數種子**：從乾淨微分段中**隨機**抽樣至所需基線長度（`required_sec`），避免選樣偏差；固定 `random_seed` 確保結果可重現（reproducibility）。
6. **基線參考值**：抽樣串接後的基線資料同樣經 `500 → 200 Hz → TFLite → qEEG` 處理，得到每 channel 的指標基準值，與趨勢/heatmap 的計算鏈一致（apples-to-apples）。

### 特定模式的基線處理（Color Agility Ladder）

在 `pre-event-rest` 模式下，「Color Agility Ladder」緊接在「Agility Ladder」之後，**沒有獨立的事件前靜息段**，硬取其前一段會取到上一個高強度任務的尾段，使 delta 失真。因此透過 `BASELINE_ANCHOR = {'Color Agility Ladder': 'Agility Ladder'}` 改用標準「Agility Ladder」的事件前靜息區間作為共同基線。

### 錯誤處理

當乾淨微分段總時長 **小於** `required_sec` 時，`build_baseline_epochs()` 會丟出明確的 `ValueError`，並指示使用者三種調整方式：放寬品質門檻、擴大搜尋窗、或縮短所需基線長度。主流程預設 `on_insufficient='raise'`（嚴格中止）；設為 `'skip'` 則僅警告並略過該事件，方便批次掃描資料品質。

### 執行方式

```bash
# 對 iBrainCenter 全部受試者產圖（預設 pre-event-rest baseline）
python plot_tflite_summary.py

# 改用每事件事件前基線（僅事件內著色），或兩種模式都輸出
python plot_tflite_summary.py --heatmap-baseline pre-event
python plot_tflite_summary.py --heatmap-baseline both --on-insufficient skip

# 自訂緩衝區/搜尋窗/基線長度/亂數種子
python plot_tflite_summary.py --required-sec 8 --t-search -20 --seed 7

# 調整品質門檻：調高 → 更嚴格地捨棄疑似假影視窗（無效區段留白更多）
python plot_tflite_summary.py --quality-ratio 0.65
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--outdir` | `iBrainCenter/tflite_summary/` | PNG 輸出目錄 |
| `--heatmap-baseline` | `session` | heatmap/bar 的 Δ 基線模式：`session` / `pre-event` / `both`（見上表） |
| `--t-buffer` | −3.0 | 緩衝區秒數（負值；觸發前此區間一律捨棄；僅 `pre-event` 用） |
| `--t-search` | −15.0 | 基線搜尋窗起點（負值；觸發前幾秒開始找基線；僅 `pre-event` 用） |
| `--required-sec` | 10.0 | 所需乾淨基線總長度（秒） |
| `--quality-ratio` | 0.5 | EEG 品質分數門檻 (0~1)；低於此值的視窗視為無效並捨棄，亦作為基線微分段篩選門檻。調高更嚴格（捨棄更多疑似假影），調低保留更多 |
| `--seed` | 42 | 固定亂數種子（確保可重現） |
| `--on-insufficient` | `raise` | 基線不足時：`raise` 丟錯中止 / `skip` 警告略過 |

### 輸出檔案

每個 subject 依模式輸出 PNG（檔名以 `_session` 或 `_pre_event` 標記）：

```
iBrainCenter/tflite_summary/
├── Hardy_SN036_tflite_summary_session.png      # --heatmap-baseline session（預設）
└── Hardy_SN036_tflite_summary_pre_event.png    # --heatmap-baseline pre-event
```

> **品質面板的斜向長直線是什麼？** `merged.csv` 由多個錄製檔串接，檔間可能有數十~數百秒的時間斷點；繪圖時若直接連線，會在斷點兩端畫出「斜向長直線」假影。本腳本的品質面板已改用 `_series_with_gaps()` 在大缺口插入 NaN 自動斷開。若要進一步診斷這些異常線段的成因（連線假影 vs 真實壞訊號），請見下方 `check_quality_anomalies.py`。

---

## check_quality_anomalies.py

診斷工具：找出 EEG Quality 曲線上的「異常線段」，逐視窗分類並與**原始時域波形**對照，判別每段異常的真正成因。異常主要分兩類：

- **連線假影 (gap-bridging artifact)**：`merged.csv` 的檔間時間斷點被繪圖直線硬接，形成斜向長直線——並非真實品質變化。
- **真實壞訊號**：ADC 飽和削波（數值貼在 ±2048）、平線/斷線（連續樣本相同）、或大幅雜訊使品質驟降。

### 偵測項目（逐 5 s 視窗）

| 標記 | 條件（預設） | 意義 |
| --- | --- | --- |
| `time-gap` | 視窗內最大相鄰樣本時間差 > 1 s | 視窗橫跨資料斷點（檔案邊界）→ 連線假影來源 |
| `clip` | 削波樣本比例 > 1%（\|x\| ≥ 2047） | ADC 飽和削波（高動作任務常見的動作假影） |
| `flat` | 四通道同時零變化樣本比例 > 5% | 平線 / 斷線 |
| `low-Q` | 品質中位數 < 0.5 | 品質低落 |
| `jump` | 相鄰視窗品質中位數跳動 > 0.40 | 品質驟跳 |

### 輸出圖內容

- **上方**：品質時間軸（已用 `_series_with_gaps()` 斷開缺口，不再有假影直線），各類異常以不同標記疊在曲線上，並標註 `#1…#N` 對應下方原始波形。
- **下方**：依嚴重度挑選最多 6 個視窗的原始 4-ch 波形（含前後文），標出視窗範圍、±2048 ADC 滿格參考線與斷點位置，讓使用者一眼看出每段異常到底是「連線假影」還是「真實壞訊號」。

### 執行方式

```bash
# 預設檢查 Hardy（其 merged.csv 含 5 個時間斷點，最具代表性）
python check_quality_anomalies.py --subject Hardy

# 檢查全部 iBrainCenter 受試者
python check_quality_anomalies.py --all
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--subject` | `Hardy` | 受試者名稱 |
| `--all` | （未設定） | 對全部 iBrainCenter 受試者執行 |
| `--outdir` | `iBrainCenter/quality_anomalies/` | PNG 輸出目錄 |

偵測門檻（`RAIL_VALUE`、`CLIP_FRAC`、`FLAT_FRAC`、`GAP_SEC`、`JUMP_DELTA`、`MAX_RAW_PANELS`）定義於檔案頂端常數，可視資料情況調整。終端機亦會印出所有被標記視窗的時間與原因清單。

---

## plot_tyy_meditation.py

針對 TYY (SN041) 的 **Mindfulness Meditation** 區段輸出聚焦圖，由上而下四個面板：

1. Ch1 EEG 波形
2. Ch2 EEG 波形
3. BP-filtered qEEG Δ vs baseline heatmap（灰階）
4. TFLite qEEG Δ vs baseline heatmap（灰階，模型存在時）

X 軸限制在「pre-baseline + meditation」視窗內，同時輸出 PNG（150 dpi）與 SVG。

- baseline 取冥想開始前的區段（`BASELINE_END_HHMM = 14:24`，第一個 TYY 參與活動 Push-ups 之前）。
- 冥想區段：`MEDITATION_START_HHMM = 15:06`、時長 11 分鐘。
- TFLite 推論與 `plot_event_markers.py` / `data_analysis.py` 一致，採 per-window RMS 正規化（推論前除以 RMS、輸出再乘回）。

```bash
python plot_tyy_meditation.py [--outdir <dir>] [--ds <factor>] [--no-tflite]
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--outdir` | `iBrainCenter/TYY_meditation/` | PNG / SVG 輸出目錄 |
| `--ds` | 500 | EEG 波形下採樣倍率 |
| `--no-tflite` | （未設定時啟用 TFLite 對照） | 跳過 TFLite heatmap 面板 |

---

## compare_subjects.py

跨受試者比較 BP 與 TFLite qEEG delta index，兩組獨立分析：

- **iBrainCenter**（Ann、Hsin、Hardy、TYY、James）：以 8 個 event block 計算相對 baseline 的 delta。
- **YoGa**（James、Jammie、TYY）：無活動標記，計算整段 session 的 delta。

Quality gating 與 `plot_event_markers.py` 相同（5 s window @ 500 Hz、flat+spectrum 參數集），低品質 window 會在 delta 計算中遮罩；quality 旗標與 qEEG window 以重疊區段對齊（不再因視窗數不一致而整體失效）。

```bash
python compare_subjects.py [--ibrain-outdir <dir>] [--yoga-outdir <dir>]
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--ibrain-outdir` | `iBrainCenter/comparison/` | iBrainCenter 比較圖輸出目錄 |
| `--yoga-outdir` | `YoGa/comparison/` | YoGa 比較圖輸出目錄 |

輸出（每組）：`{group}_bp_delta_comparison.png`、`{group}_tflite_delta_comparison.png`、`{group}_combined_comparison.png`。

---

## plot_raw_eeg.py

對每份 iBrainCenter `merged.csv` 隨機取樣 `N_SEGS` 個非重疊 30 秒片段，繪製**未濾波、未下採樣**的 4-channel 原始 EEG 波形，用於快速目視檢查訊號品質。當錄製長度不足一個片段時會略過該檔（不會中斷）。

```bash
python plot_raw_eeg.py [--outdir <dir>] [--seed <int>] [--seg_sec <float>]
```

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--outdir` | `sample_quality/` | PNG 輸出目錄 |
| `--seed` | 42 | 隨機取樣種子 |
| `--seg_sec` | 30.0 | 每個片段長度（秒） |

---

## convert_to_tflite.py

將 `tiny_v4_optimized.pth` 轉換成 float32 TFLite 模型。用 TensorFlow/Keras 重新建立 TinyUNetV4 架構，從 PyTorch checkpoint 複製權重，驗證數值誤差後匯出。

### TFLite 模型 I/O

| 項目 | Shape | 說明 |
| --- | --- | --- |
| Input | `(1, 400, 4)` | batch, time, channels |
| Output | `(1, 400, 2)` | batch, time, output channels |

### 執行方式

```bash
python convert_to_tflite.py
```

### 轉換流程

1. 讀取 `tiny_v4_optimized.pth`。
2. 建立 TF/Keras 版 TinyUNetV4 並複製 PyTorch 權重。
3. 隨機 input 比較 PyTorch 與 TensorFlow 輸出（容許誤差 `5e-2`）。
4. 匯出 float32 TFLite 至 `tiny_v4_optimized.tflite`。
5. 使用 TFLite interpreter 做 smoke test。

---

## 注意事項

- 所有 script 均使用 500 Hz 取樣率的 lilia EEG CSV 格式（4-row header，時間欄位單位為 microseconds）。
- `data_analysis.py` 與 `convert_to_tflite.py` 依賴絕對路徑 `/home/bps-yichin/tommy`，移機時需修改。
- `convert_to_tflite.py` 匯出的是 float32 TFLite，尚未做 int8 quantization。
