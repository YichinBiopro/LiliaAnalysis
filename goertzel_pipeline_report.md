# Goertzel 單頻功率管線報告

**對應 commit：** `d9546ca` — "Add Goertzel pipeline with hard-gated quality and artifact filtering"
**範圍：** `lilia/goertzel.py`、`power_spectral_app.py` 與 6 支批次分析腳本，套用於 `iBrainCenter/` 6 位受試者、每人 2 通道的 dry-electrode EEG 資料。

---

## 0. 背景與目的

`power_spectral_app.c`（本次一併留存為未納入版控的參考檔）是韌體端用 Goertzel 演算法即時估算單一頻率功率的 C++ 片段，原始用途是嵌入式裝置上低成本地監測特定頻率（例如 60 Hz 電源雜訊）的強度，而不需要跑完整 FFT。

這次新增的管線做兩件事：

1. **把韌體演算法搬進 Python**（`lilia/goertzel.py`），維持與 C 版本一致的訊號處理順序（去均值 → Hann 窗 → Goertzel 遞迴），讓離線分析可以重現裝置端的量測方式，而不是用 Welch PSD 這類另一套估計法。
2. **在既有的 quality/artifact 前處理之上，建立一條「windowed Goertzel power + EEG quality + hard-artifact 判定」的批次分析管線**，用來回答一個具體問題：*60 Hz 頻段的功率讀數在什麼情況下可信、什麼情況下是雜訊/偽跡造成的假訊號？*

---

## 1. 管線架構

```
lilia/goertzel.py  (核心演算法)
        │
        ▼
plot_goertzel_vs_raw.py        ── 逐 subject/channel 滑動視窗，算出 goertzel_db + quality
        │  輸出: index_vs_raw_goertzel_ch{n}_{freq}Hz.csv / .png (每個 subject 資料夾內)
        │
        ├─▶ filter_and_plot_hard_artifacts.py   ── 依 quality + hard-artifact 篩選，畫時間軸
        │        輸出: iBrainCenter/comparison/hard_clip_filter/*
        │
        ├─▶ summarize_goertzel_distribution.py  ── 彙總統計量 (mean/std/分位數)
        │        輸出: iBrainCenter/comparison/goertzel_distribution_summary_*.csv
        │              iBrainCenter/comparison/goertzel_distribution_hist_*.csv
        │
        ├─▶ plot_goertzel_histograms.py         ── 依統計彙總畫直方圖 (mean, ±1σ 標註)
        │        輸出: iBrainCenter/comparison/goertzel_histograms/*.png
        │
        ├─▶ sample_segments_by_goertzel_db.py    ── 跨全體受試者，隨機抽樣接近目標 dB 的視窗
        │        輸出: iBrainCenter/comparison/goertzel_samples_67dB/*
        │
        └─▶ resample_clean_goertzel_samples.py   ── 逐受試者，排除 hard-artifact 後抽樣、畫小圖集
                 輸出: iBrainCenter/comparison/goertzel_samples_67dB_resampled/*
```

`plot_goertzel_vs_raw.py` 是整條管線的資料源頭；其餘 5 支腳本都是它輸出 CSV 的下游消費者，彼此互不依賴，可以獨立重跑。

---

## 2. 核心演算法：`lilia/goertzel.py`

`goertzel_power(data, target_freq, sample_rate, remove_dc=True, apply_hann=True)`：

1. **去 DC**：減去視窗內樣本平均值，避免直流分量污染功率估計。
2. **Hann 窗**：`0.5 * (1 - cos(2πn/(N-1)))`，抑制邊界不連續造成的頻譜洩漏（spectral leakage）。
3. **Goertzel 遞迴**：`s = sample + 2cos(ω)·s_prev - s_prev2`，其中 `ω = 2π·f_target/f_s`。
4. **功率**：`s_prev2² + s_prev² - 2cos(ω)·s_prev·s_prev2`。

`power_spectral_app.py` 是薄封裝，直接呼叫上述函式並提供一個 CLI（`--samples/--target-freq/--sample-rate`），docstring 註明「Converted from power_spectral_app.c」——即這支模組的存在理由就是與韌體端數值對齊，供交叉驗證用。

---

## 3. 六支批次腳本

### 3.1 `plot_goertzel_vs_raw.py`（管線入口）

對每位受試者的 `merged.csv`：
- 以 5 秒視窗（`--win-sec`）、5 秒步進（無重疊）在 bandpass（0.5–45 Hz）訊號上算 `goertzel_power` → 轉 dB。
- 同一視窗用 `lilia.quality.get_eeg_quality_index_v2_parametric`（iBrain 裝置校正過的 flat+spectrum 權重組合）算 EEG quality。
- **在原始（未濾波）訊號上**額外算三個 hard-artifact 特徵，理由是濾波會把削峰/階躍訊號的能量分散掉，用濾波後訊號判斷反而會漏偵：
  | 特徵 | 判定條件（預設值） |
  |---|---|
  | `sat_frac_1950` | 視窗內 \|振幅\|≥1950 µV 的樣本比例 ≥ 0.12 → 疑似飽和/削頂 |
  | `peak_to_peak_uv` | 視窗峰對峰值 ≥ 1000 µV → 疑似階躍/大幅雜訊 |
  | `bp_edge_shift_uv` | 視窗頭尾各 1 秒中位數差 ≥ 80 µV（濾波後訊號）→ 疑似 DC 跳動 |
- 三者任一成立則標記 `artifact_hard_clip=1`，並令 `quality_final = 0`（無論原始 quality 多高，一律視為不可用）。
- 輸出每視窗一列的 CSV（`goertzel_db, quality, quality_final, artifact_hard_clip` 等欄位）以及一張 4-panel 圖（Goertzel dB / quality / bandpass EEG / raw EEG，並標示事件與被遮蔽區間）。

### 3.2 `filter_and_plot_hard_artifacts.py`

讀取 3.1 的輸出，依 `quality_final > threshold（預設 0.5）且非 hard-artifact` 篩出「乾淨視窗」，另存 `artifact_hard_clip==1` 的視窗，並畫時間軸圖（Goertzel dB 走勢 + quality 走勢 + 原始波形，hard-artifact 用紅色 X 標示、疊上淡紅色時間帶）。

### 3.3 `summarize_goertzel_distribution.py`

彙總所有受試者/通道，計算 quality-filtered 後 Goertzel power（線性值與 dB）的 mean/std/min/max/五個分位數（p05/p25/p50/p75/p95），並自動附加「全通道彙總」與「全受試者+全通道彙總」列。同時輸出跨組別共用 bin 邊界的直方圖表（`goertzel_distribution_hist_*.csv`），確保各組直方圖可直接互相比較。

### 3.4 `plot_goertzel_histograms.py`

依每個 (subject, channel) 畫 dB 直方圖，並在圖上標出 `mean` 與 `mean±1σ` 的垂直線與對應散點，額外輸出「全受試者·單通道」與「全受試者·全通道」的彙總直方圖。

### 3.5 `sample_segments_by_goertzel_db.py`

跨全體受試者掃描，找出 `goertzel_db` 落在目標值 ±容忍度（預設 67.15±0.6 dB）且 quality>0.5 的視窗，隨機抽 N 筆（預設 6），各畫一張「raw vs bandpass」小圖，方便肉眼核對「67 dB 左右的訊號長什麼樣」。

### 3.6 `resample_clean_goertzel_samples.py`

邏輯與 3.5 類似，但**逐受試者**執行、且明確排除 hard-artifact（`artifact_hard_clip<0.5`），先取離目標 dB 最近的 30 筆（`--pool-size`）再隨機抽 10 筆（`--n`），輸出每位受試者一張多格小圖集（gallery）與抽樣清單 CSV，最後彙總成 `resample_report_*.csv`。

---

## 4. 執行結果

以下數字與圖片取自倉庫中既有的 `iBrainCenter/comparison/` 執行產出（60 Hz、quality 門檻 0.5、5 秒視窗，6 位受試者：Ann(SN027)、Hardy(SN036)、Hardy_2(SN036)、Hsin(SN032)、James(SN035)、TYY(SN041)，各 2 通道）。

> **單位說明：** 本節的功率數字與圖表全部改用 **線性（未取 log 的原始）Goertzel power**，不再是 dB。換算關係為 `power = 10^(dB/10)`；例如 4.3 節原本用來挑樣本的目標值 67.15 dB 換算後約等於 `5.19×10⁶`（線性單位）。原始 dB 版本的圖與統計數字仍保留在 `iBrainCenter/comparison/goertzel_histograms/`、`hard_clip_filter/` 等資料夾與 `goertzel_distribution_summary_thr_0p5.csv` 的 `db_*` 欄位中，供需要時對照。
>
> 圖片為本機執行後產生的 PNG，路徑對應 `.gitignore` 中的 `*.png` 規則，**未納入版控**。線性單位的圖表由一支獨立輔助腳本（未修改原本 6 支 pipeline 腳本）重新讀取既有的逐視窗 CSV 與 `merged.csv` 繪製而成；在這台機器 / 這個 checkout 上開啟本文件可正常顯示，若在別處重新 clone 此 repo 需重新產生。

### 4.1 整體分布（`goertzel_distribution_summary_thr_0p5.csv`，線性 power 欄位）

| 受試者 | 通道 | 總視窗數 | quality 篩選後保留數 | 保留率 | power 中位數(p50) | power p05–p95 | power 平均值(參考用) |
|---|---|---:|---:|---:|---:|---|---|
| Ann(SN027) | ch1/ch2 | 786 | 386 / 387 | ~49% | 2.51e6 / 3.55e6 | 3.93e4–2.35e7 / 3.22e4–3.43e7 | 6.32e6 / 8.67e6 |
| Hardy(SN036) | ch1/ch2 | 850 | 682 / 569 | 80% / 67% | 1.69e5 / 5.35e5 | 895–1.03e9 / 1.05e3–1.12e8 | 7.65e7 / 2.64e7 |
| Hardy_2(SN036) | ch1/ch2 | 383 | 340 / 339 | 89% / 89% | 257 / 294 | 16.2–9.57e9 / 12.5–2.49e9 | 1.61e9 / 3.38e8 |
| Hsin(SN032) | ch1/ch2 | 791 | 633 / 682 | 80% / 86% | 8.16e4 / 7.41e3 | 38.3–1.42e6 / 37.9–1.85e5 | 4.88e5 / 5.21e4 |
| James(SN035) | ch1/ch2 | 762 | 538 / 496 | 71% / 65% | 2.14e4 / 7.44e3 | 462–4.27e5 / 351–5.32e7 | 5.46e5 / 7.81e6 |
| TYY(SN041) | ch1/ch2 | 803 | 449 / 558 | 56% / 69% | 3.52e5 / 1.55e5 | 8.82e3–2.07e7 / 4.95e3–1.92e6 | 4.82e6 / 1.16e6 |
| **ALL_SUBJECTS** | **ALL_CHANNELS** | **8750** | **6059** | **69.2%** | **6.15e4** | **80.8–5.87e7** | **1.22e8**（std=8.96e8） |

（`power` 為 Goertzel 演算法輸出的線性量值，無實體單位／未校正絕對電壓，可視為任意單位 a.u.；換算 `dB = 10·log10(power)`。）

觀察：
- 整體保留率約 **69%**，但受試者間差異很大——Hardy_2 兩通道都有 ~89% 保留率，p05–p95 範圍卻橫跨 8–9 個數量級（16.2 到 95 億），代表該受試者訊號本身波動劇烈（不是被過度剔除，而是原始訊號真的很不穩定）；相對地 Ann(SN027) 保留率只有 ~49%，代表該受試者有將近一半視窗未達 quality 門檻。
- **改用線性單位後，「平均值 ± 標準差」不再是好的描述方式**：這批資料右偏極重（少數極端高值視窗把平均值和標準差都拉得很大，例如 ALL_SUBJECTS 的 mean=1.22e8 但中位數只有 6.15e4，相差超過 3 個數量級），所以本節統計與圖表改以**中位數 + p25/p75（IQR）**取代原本 dB 版本用的 mean±1σ；平均值僅留作參考。這也是為什麼原本在 dB（log）尺度上看起來平滑對稱的分布，換回線性尺度後，「平均」的代表性大幅下降。
- 全體彙總的 power 分布（下圖，x 軸為 log 刻度以便同時顯示跨數量級的資料）中位數落在 4.41×10⁴，IQR（p25–p75）約 2.78×10³–5.33×10⁵；右尾在 10⁷–10⁸ 附近有明顯的一群視窗（對應少數受試者如 Hardy_2 的極端高值視窗，quality 篩選並未把它們排除，值得留意是否為真實生理訊號還是殘留偽跡）。

![全受試者全通道 Goertzel 線性 power 分布](iBrainCenter/comparison/goertzel_histograms_linear/ALL_SUBJECTS_ALL_CHANNELS_60Hz_thr_0.5_linear.png)
*全體 6059 個通過 quality 篩選的視窗，power 中位數=4.41e4，p25=2.78e3，p75=5.33e5（x 軸為 log 刻度，僅用於容納跨數量級資料，圖上數字本身皆為線性 power 值）。*

**單一受試者分布形狀的對比**——並非所有受試者都是這種平滑單峰：

| Ann(SN027) ch1（分布集中，中位數=2.51e6） | Hardy_2(SN036) ch1（明顯多峰，中位數=257） |
|---|---|
| ![Ann ch1 直方圖](iBrainCenter/comparison/goertzel_histograms_linear/Ann(SN027)_ch1_60Hz_thr_0.5_linear.png) | ![Hardy_2 ch1 直方圖](iBrainCenter/comparison/goertzel_histograms_linear/Hardy_2(SN036)_ch1_60Hz_thr_0.5_linear.png) |

Hardy_2(SN036) ch1 的直方圖清楚呈現多個群聚：10¹–10² 量級一群（含中位數 257）、10⁵ 量級附近一群，以及 10⁹–10¹⁰ 量級一個陡峭尖峰（貼著資料上限，形狀類似量測飽和而非自然衰減的分布尾巴，換算回 dB 約落在 90–100 dB，與先前 dB 版本觀察一致）。這個尖峰即使在 quality>0.5 且排除 hard-artifact 後依然存在，是 §5 建議中「應人工核對右尾樣本」的直接依據。

### 4.2 Hard-artifact 排除結果（`hard_clip_filter/filter_report_60Hz_thr_0.5.csv`）

| 受試者 | 通道 | 總視窗數 | hard-artifact 視窗數 | 佔比 |
|---|---:|---:|---:|---:|
| Ann(SN027) | ch1/ch2 | 786 | 1 / 2 | 0.1% / 0.3% |
| Hardy(SN036) | ch1/ch2 | 850 | 18 / **179** | 2.1% / **21.1%** |
| Hardy_2(SN036) | ch1/ch2 | 383 | 32 / 13 | 8.4% / 3.4% |
| Hsin(SN032) | ch1/ch2 | 791 | 0 / 2 | 0% / 0.3% |
| James(SN035) | ch1/ch2 | 762 | 35 / 29 | 4.6% / 3.8% |
| TYY(SN041) | ch1/ch2 | 803 | 10 / 10 | 1.2% / 1.2% |
| **合計** | | **8750** | **331** | **3.8%** |

**Hardy(SN036) ch2 是明顯的離群案例**：21% 的視窗被判為 hard-artifact，遠高於其他所有受試者/通道（多數 <5%）。對照該通道的時間軸圖（power 軸改為線性、以 log 刻度顯示以容納跨數量級數值），可以看到在 t≈0–55 分鐘幾乎整段都夾雜密集的紅色 X 標記，power 在 hard-artifact 視窗中飆升到 10⁸–10⁹ 量級（原始波形則出現 ±2000 µV 滿量程尖峰），t≈56–66 分鐘則轉為一段乾淨、quality 穩定在 0.8–0.9、power 落在 10³ 量級的區間，之後又恢復高噪音：

![Hardy ch2 時間軸：密集 hard-artifact（線性 power）](iBrainCenter/comparison/hard_clip_filter_linear/Hardy(SN036)_ch2_60Hz_thr_0.5_timeline_linear.png)
*Hardy(SN036) ch2——紅色 X 為 artifact_hard_clip=1 的視窗，其 Goertzel power 明顯比周圍乾淨視窗高出 2–3 個數量級；原始波形（最下層）在 0–55 分與 66–80 分鐘反覆出現 ±2000 µV 滿量程尖峰。*

這與同一受試者 ch1（僅 2.1% hard-artifact）形成強烈對比，指向**單一通道的電極接觸問題**，而非受試者整體訊號品質差：

![Hardy ch1 時間軸：同受試者但乾淨得多（線性 power）](iBrainCenter/comparison/hard_clip_filter_linear/Hardy(SN036)_ch1_60Hz_thr_0.5_timeline_linear.png)
*同一受試者、同一時段的 ch1——原始波形平穩，power 走勢也連續得多，全程僅 5 個視窗被標記為 hard-artifact（集中在 66 分鐘附近的短暫尖峰）。*

`_hard_gallery_linear.png` 進一步把 ch2 被標記的視窗放大檢視（子圖標題已將 `db=` 換成 `power=` 的線性數值），可以看到典型的方波削頂 / 階躍雜訊型態（訊號在 ±2000 µV 間反覆跳動，而非漸進的生理訊號變化），佐證 hard-artifact 判定邏輯確實抓到的是量測偽跡，而非高振幅但真實的腦電活動：

![Hardy ch2 hard-artifact 片段集（線性 power）](iBrainCenter/comparison/hard_clip_filter_linear/Hardy(SN036)_ch2_60Hz_thr_0.5_hard_gallery_linear.png)
*12 個代表性 hard-artifact 視窗：灰線為原始訊號、藍線為 bandpass(0.5–45Hz)，標題中的 power 為線性 Goertzel power（量級約 6.6×10⁴–2.1×10⁷）。多數呈現方波狀削頂或瞬間大幅跳動，屬於典型的電極接觸/削波偽跡而非神經訊號。*

### 4.3 目標值重取樣（`goertzel_samples_67dB_resampled/resample_report_60Hz_67.15dB_thr_0.5.csv`）

原始抽樣腳本以 dB 為目標值（67.15 dB，±0.6 dB 容忍度）挑選視窗——換算成線性單位，目標約為 `power ≈ 5.19×10⁶`。抽樣本身（挑了哪些視窗）並未重跑，以下只是把同一批已選出的視窗，其 `goertzel_power` 欄位換算成線性數值後重新統計：

| 受試者 | 乾淨候選視窗數 | 抽樣數 | 實際 power 範圍 | 平均 power |
|---|---:|---:|---|---:|
| Ann(SN027) | 756 | 10 | 4.80e6–5.65e6 | 5.19e6 |
| Hardy(SN036) | 1115 | 10 | 3.91e6–6.88e6 | 4.92e6 |
| Hardy_2(SN036) | 529 | 10 | 2.65e5–1.00e8 | 2.28e7 |
| Hsin(SN032) | 1314 | 10 | 1.72e6–1.04e7 | 3.96e6 |
| James(SN035) | 992 | 10 | 6.11e5–4.81e7 | 1.85e7 |
| TYY(SN041) | 983 | 10 | 4.27e6–6.45e6 | 5.05e6 |

排除 hard-artifact 後，多數受試者能在目標值附近找到數百到千餘筆候選視窗，抽樣結果集中在 4×10⁶–7×10⁶ 這個量級（與目標 5.19×10⁶ 相符）。例外是 **Hardy_2(SN036)**，其抽樣池雖有 529 筆候選，實際抽到的 10 筆 power 範圍卻寬達 2.65×10⁵–1.00×10⁸（跨近 3 個數量級）——與 4.1 節觀察到的「該受試者分布極度右偏、p05–p95 橫跨 8–9 個數量級」一致，代表其訊號本身動態範圍就很大，即使排除 hard-artifact 也難以在單一目標值附近取得緊密聚集的樣本。

![Hardy_2 重取樣後的乾淨視窗集（線性 power）](iBrainCenter/comparison/goertzel_samples_67dB_resampled_linear/Hardy_2(SN036)_sampled_gallery_linear.png)
*Hardy_2(SN036) 通過 hard-artifact 排除、quality_final≥0.87 的 10 個抽樣視窗（qf 與線性 power 標於各子圖標題）。注意各子圖 y 軸尺度差異極大（±50 µV 到 ±400 µV 不等），部分視窗（如 #8、#9）呈現寬頻、無明顯節律的雜訊狀波形——即使通過了目前的 hard-artifact 與 quality 雙重篩選，仍可能混入非神經性的高頻雜訊，而非單純的削波/階躍偽跡，是現有判定邏輯尚未覆蓋的偽跡型態。*

---

## 5. 觀察與建議

1. **60 Hz 頻段功率的 quality-only 篩選不夠**：`plot_goertzel_vs_raw.py` 的原始 quality 分數是用濾波後訊號算的，飽和/階躍偽跡的能量在濾波後被抹平，quality 分數可能仍然偏高——這正是額外設計「hard-artifact 三特徵（皆取自原始訊號）」的原因，且已證實有效抓出 Hardy(SN036) ch2 這類肉眼可辨的電極接觸不良區段。
2. **Hardy(SN036) ch2** 建議標記為此裝置/受試者組合的已知問題通道，後續分析（例如 60 Hz 電源雜訊監測、跨受試者比較）應優先排除或單獨處理，而不是與其餘乾淨通道一起平均。
3. **Hardy_2(SN036)** 的高變異性目前無法單純用 hard-artifact 排除解決；後續若要在此受試者身上做「取固定 dB 附近樣本」的分析，可能需要縮小 `--tol-db` 或改用相對於該受試者自身分布的百分位數，而非跨受試者共用的絕對 dB 目標值。
4. 全體彙總直方圖右尾（power 約 10⁷–10⁸ 以上，換算回 dB 約為 70–90+ dB）的少量高值視窗，雖通過了 quality + hard-artifact 雙重篩選，仍建議用 `sample_segments_by_goertzel_db.py` 或 `resample_clean_goertzel_samples.py` 挑幾筆出來目視確認，排除是否為尚未涵蓋的偽跡型態。
5. **線性單位下報表統計量應以中位數/百分位數為主、平均值僅供參考**：dB（log）尺度下 mean±1σ 是合理的分布描述，但換回線性 power 後同一批資料變得極度右偏（見 4.1），算術平均值容易被少數極端值主導而失去代表性，後續若要建立門檻或警戒值，建議直接以百分位數（如 p95）而非平均值為基準。

---

## 6. 檔案清單

| 檔案 | 角色 |
|---|---|
| `lilia/goertzel.py` | Goertzel 核心演算法（新增至 `lilia` 套件） |
| `power_spectral_app.py` | CLI 封裝，對齊原始 C 實作 |
| `plot_goertzel_vs_raw.py` | 管線入口：逐視窗算 power/quality/hard-artifact，輸出 CSV+圖 |
| `filter_and_plot_hard_artifacts.py` | 篩選乾淨視窗，畫時間軸 |
| `summarize_goertzel_distribution.py` | 分布統計彙總 |
| `plot_goertzel_histograms.py` | 分布直方圖 |
| `sample_segments_by_goertzel_db.py` | 跨受試者目標 dB 抽樣 |
| `resample_clean_goertzel_samples.py` | 逐受試者目標 dB 抽樣（排除 hard-artifact） |
