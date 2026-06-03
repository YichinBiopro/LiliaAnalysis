# Lilia EEG Analysis

本專案包含多個 Python scripts / modules，涵蓋 EEG 訊號分析、qEEG wellness indices、band entropy、EEG 品質評分、受試者資料合併、活動標記驗證圖、以及模型轉檔。

## Python Scripts 總覽

| Script | 功能 |
| --- | --- |
| `data_analysis.py` | 比較 APP 與 NUC EEG 資料，進行前處理、TinyUNetV4 模型推論，並輸出分析圖（含 qEEG indices 圖） |
| `qeeg_indices.py` | 實作 Appendix J §3 的 qEEG wellness indices（Focus / Flow / Calm / Relaxation），可獨立執行或被其他 script 匯入 |
| `spectral_entropy.py` | 使用 Welch PSD 計算 delta / theta / alpha / beta / gamma 五個頻段能量、正規化成比例後計算 BandEn |
| `eeg_quality_v2.py` | 從 SleepStage 抽出的獨立 EEG quality v2 scorer，核心為 `get_eeg_quality_index_v2_parametric()` 與其參數 / wrapper |
| `eeg_utils.py` | 共用低階工具：`load_merged_csv()`（4-row header CSV 載入）、`bandpass_filter()`（零相位 Butterworth bandpass） |
| `merge_subject_csvs.py` | 將 `iBrainCenter/` 與 `YoGa/` 底下各 subject 的多個 CSV 合併為單一 `merged.csv` |
| `plot_event_markers.py` | 將 evt_time.docx 的活動時間點疊加至 iBrainCenter 各 subject 的 merged.csv，輸出驗證圖（含 EEG quality、qEEG heatmap、30s smooth summary、event-level delta，以及 TFLite 模型對照組） |
| `sample_quality_check.py` | 從每份 merged.csv 隨機取樣 2 個 30 秒片段，計算 EEG quality 與 qEEG indices，輸出品質檢查圖 |
| `convert_to_tflite.py` | 將 PyTorch checkpoint (`tiny_v4_optimized.pth`) 轉換成 float32 TFLite 模型 |

## 專案結構

```text
lilia_analysis/
├── data_analysis.py           # APP vs NUC 分析主程式
├── qeeg_indices.py            # qEEG wellness indices（Appendix J §3）
├── spectral_entropy.py        # Welch PSD + five-band entropy
├── eeg_quality_v2.py          # 獨立 EEG quality v2 scorer
├── eeg_utils.py               # 共用工具：CSV 載入、bandpass filter
├── merge_subject_csvs.py      # 合併各 subject 的多個 CSV
├── plot_event_markers.py      # 活動時間標記驗證圖（含 quality / qEEG / TFLite 對照面板）
├── sample_quality_check.py    # 隨機取樣品質檢查圖
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
│   └── event_verification/    # plot_event_markers.py 輸出的驗證圖
└── YoGa/
    ├── James(SN035)/ ...
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

### 分析流程

1. 讀取 APP 與 NUC CSV。
2. 訊號前處理：0.5–45 Hz bandpass → 60 Hz notch → 33.25 Hz bandstop。
3. MAD threshold artifact 偵測與 interpolation 修補。
4. TinyUNetV4 模型推論（4-ch in → 2-ch out）。
5. Cross-correlation 估計 APP/NUC lag 並對齊。
6. 輸出 time-domain、PSD、STFT 比較圖。
7. 對模型輸出 channel 計算 qEEG wellness indices 並輸出圖。

### 重要參數

```python
BASE_DIR     = '/home/bps-yichin/lilia_analysis'
FS           = 500       # Hz
N_CH         = 4         # 模型輸入 channels
N_CH_OUT     = 2         # 模型輸出 channels
MODEL_WINDOW = 400       # samples
MODEL_PATH   = os.path.join(BASE_DIR, 'tiny_v4_optimized.pth')
```

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
| `compute_relative_powers` | 3.1 | 計算 θ / α / β 相對功率（排除 Delta 與 Gamma） |
| `bounded_ratio(E, I)` | 3.2 | `clamp((E−I)/(E+I+ε), −1, 1)` |
| `focus_index` | 3.3 | 持續注意力：高 β（去 EMG）vs 抑制 α/θ |
| `flow_index` | 3.3 | 心流：α-θ 同步，加入 β flexibility 與 α-θ imbalance 懲罰項 |
| `calm_index` | 3.3 | 平靜清醒：θ+α vs β+excess-theta（避免將嗜睡誤判為平靜） |
| `relaxation_index` | 3.3 | 深度放鬆：高 α 主導，高 β 與過量 θ 均有懲罰 |

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

以 2 秒 EEG window 為單位，先用 Welch's method 計算 PSD，再整合五個 EEG 頻段能量：

- Delta: 0.5–4 Hz
- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz
- Gamma: 30–45 Hz

接著計算：

- `E_total = E_delta + E_theta + E_alpha + E_beta + E_gamma`
- `p_k = E_k / E_total`
- `BandEn = -sum(p_k * log2(p_k))`

另外可選擇對一組左右通道加入**非零時滯互資訊**同步分析：

- `I(X(t); Y(t+tau))`，其中 `tau > 0`
- 預設 `tau = 5, 10, 15, 20 ms`
- 以雙向平均 `0.5 * [I(L(t); R(t+tau)) + I(R(t); L(t+tau))]` 作為該 `tau` 的左右腦同步量
- 再輸出各 `tau`、`lagged_mi_mean`、`lagged_mi_max`、`lagged_mi_best_tau_ms`

> **設計依據：** 容積傳導（volume conduction）為即時物理效應（時間差 ≈ 0），因此在 `tau = 0` 的互資訊中會被計入。引入 `tau > 0`（如 5–20 ms）可完全過濾這類偽同步訊號，抓到兩半球間真正的**動態資訊交換**（例如透過胼胝體的跨半球傳遞）。

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
| `--out` | CSV 所在目錄 | 輸出目錄 |
| `--ibrain-events` | 未設定 | 啟用後：x 軸改為絕對本地時間（HH:MM:SS UTC+8），並在每個子圖疊加 iBrainCenter 活動色塊與起始標記 |

### 活動標記說明（`--ibrain-events`）

加上 `--ibrain-events` 後，所有子圖（Band Proportion × 5、BandEn、Sync）均會疊加下列視覺元素：

- **色塊**（`axvspan`）：活動持續期間的半透明背景，每個活動有獨立顏色
- **起始虛線**（`axvline`）：活動開始時間
- **旋轉文字標籤**：標示活動名稱，貼齊起始線左緣

活動定義來自 `plot_event_markers.EVENTS`（見 `plot_event_markers.py` 章節）。

### 輸出檔案

| 檔名 | 說明 |
| --- | --- |
| `<basename>_band_entropy_ch<N>.csv` | 每個 window 的時間點、五個頻段能量、總能量、各頻段比例、BandEn |
| `<basename>_band_entropy_ch<N>.png` | 每個 band 各自獨立子圖的比例時序圖，並疊加 smooth 趨勢線；最下方附 BandEn |
| `<basename>_band_entropy_ch<N>_sync_ch<L>_ch<R>.csv` | 在原 BandEn 欄位外，追加各 `tau` 的 lagged MI、`lagged_mi_mean`、`lagged_mi_max`、`lagged_mi_best_tau_ms` |
| `<basename>_band_entropy_ch<N>_sync_ch<L>_ch<R>.png` | 在原 BandEn 圖下方追加左右腦非零時滯互資訊同步面板；指定 `--ibrain-events` 時 x 軸改為絕對時間並疊加活動標記 |

---

## merge_subject_csvs.py

掃描 `iBrainCenter/` 與 `YoGa/` 底下所有 `subject(***)` 資料夾，將同一 subject 的多個 CSV 依 `Abs Time Offset[us]` 轉換成絕對時間後合併，產生 `merged.csv`。

### 合併邏輯

1. 讀取每個 CSV 第 2 行的 `Abs Time Offset[us]`。
2. 將各 CSV 的相對 `Time[us]` 加上 offset → 絕對時間戳記。
3. 合併後依絕對時間排序，去除重複時間點。
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

- 使用 `eeg_quality_v2.get_best_eeg_quality_v2_flat_spectrum_only_params()`  
  → `kurtosis_weight=0, corr_weight=0`（只計算 flat + spectrum 兩項）
- 取樣率：**500 Hz**（原始取樣率，不做下採樣）
- 視窗：**5 秒 non-overlapping**
- 紅色虛線標示 threshold（由參數集決定）

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
