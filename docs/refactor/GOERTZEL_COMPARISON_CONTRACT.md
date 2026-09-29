# M3 Goertzel 比較契約

日期：2026-09-17。在本輪實跑前固定；這是比較研究，沒有修改產品預設或新增 CLI profile。

| 比較 | 保存量與驗收規則 |
| --- | --- |
| Legacy 相容基準 | 固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed` 與工作區，獨立 process、相同來源／環境；不使用模型。全部 power／dB／品質／hard／索引／時間／NaN、legacy 平滑、實際圖線、CSV 與 metadata 精確一致，rtol=atol=0；空案例須核對空 shape／欄位，預期非有限來源拒絕須匹配。 |
| G-legacy-rolling | 品質 mask → pandas centered rolling median（5 窗、min_periods=1）→ 同一 mask；linear 與 dB 各自平滑。品質嚴格 `> .5`；hard 判定與 quality_final 不改。 |
| G-segment-rolling（比較用） | 僅將 rolling 限於連續相同來源 segment_id 的 run；保留原窗口位置、mask、5 窗中心與不足窗語義。無 segment 邊界須精確等於 legacy；任何跨段值不可影響另一段。低品質列仍參與窗口位置計數，不額外按有效列分段。保存全部差值／changed indices／NaN／接受排除列；不設定等價百分比，不自動採用。 |
| Goertzel 獨立核對 | 以 float64 複數指數直接求指定頻率的有限長 DTFT 幅值平方；先去 mean、對稱 Hann（N−1），不改成最近 FFT bin。與 recurrence 比較 rtol=1e−8、atol=1e−8；零／常數應為零。容差是實作誤差，不是方法等價門檻。 |
| Normalization（比較用） | 同一 power P，分別保存 P/N²、2P/(sum w)² 與 2P/(fs sum w²)，不與未正規化 P 混用。coherent 項只作 on-target、非 DC／Nyquist 單頻正弦的 mean-square 估計；density 是指定频率的單邊密度，不是整帶能量。已知振幅 2、60 Hz、500 Hz 採樣、至少 .5 秒的正弦，coherent 項相對 2 的容差 rtol=1e−3、atol=1e−6。 |
| dB／門檻 | 每個比較單位明示；正值遠離 floor 時檢查 dB 差等於 10log10(normalization factor)，atol=1e−10。顯示採同一數值 floor 1e−12，只便於比較，不能視為不同單位的同一物理 floor。67.15 dB ±.5 只作現有 sampler 範例的單位 witness：相同數字換單位會改變命中，轉換門檻的理想 offset 另存；不把任何比較列送入產品選取。 |
| 前端與窗長 | raw 與現有完整來源段 float32 0.5–45 Hz BP 分開保存；0.5／1／5 秒窗分開比較，不能把不同窗口列硬配成相同索引。60 Hz off-grid／noise／常數／已知正弦作控制；真實完整五來源、真實缺口副本、合成缺口／短段／品質邊界均涵蓋。保存原始格點與窗口數、排除與品質語義。 |

平滑 witness：dB `[0,0,100,100]`、segment `[0,0,1,1]`、全接受，legacy 為 `[0,50,50,100]`、分段為 `[0,0,100,100]`；另含單窗、全拒絕、NaN／品質等於 .5／hard 及同一 segment 值不連續重現的 run。這些是已知控制，不作真實效應標註。

物理數值控制：fs=500；duration=.5／1／5 秒；60 Hz 振幅2、60.3 Hz off-grid、10+60 Hz 混合、seed1923 noise、常數。相同輸入僅換 frontend／normalization；所有差異列出，不宣告哪個 profile 較佳。raw／filtered 差异包含現有濾波暫態，不把單一频点值稱為 EEG 行為效應。

方法來源：[SciPy symmetric Hann](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.windows.hann.html)、[SciPy spectrum／density scaling](https://docs.scipy.org/doc/scipy-1.15.3/tutorial/signal.html#spectral-analysis)、[Julius O. Smith 的 DTFT 定義](https://www.dsprelated.com/freebooks/sasp/Fourier_Transforms_Continuous_Discrete_Time_Frequency.html)。原 legacy 使用 symmetric Hann，不能因 Welch 預設 periodic Hann 而替換。

驗收需完整測試、真實舊新對照、reader／hash、平滑與 normalization／前端圖形目視；原始資料／模型／歷史產物不可覆寫。完整本機與 repository manifest 分開，真實衍生產物留 ignored `local/`。本輪比較不更改圖形預設平滑；若將來採用 G-segment-rolling，須另以具名產品選項與相容策略發佈。
