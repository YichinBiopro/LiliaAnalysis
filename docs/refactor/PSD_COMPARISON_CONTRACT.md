# M2 PSD 比較契約

日期：2026-09-16。先定契約再執行。這是比較研究與相容基準，沒有新增產品方法預設。

| 比較 | 保存量與事先門檻 |
| --- | --- |
| 固定版 `aa0df5f4` vs 工作區 | 同來源／模型／參數／環境，独立 process；keys、dtype、shape、頻率、PSD、顯示線值、raw／Before／After 索引、選取、品質及例外完全一致；rtol=atol=0，NaN 位置相同。 |
| Welch 獨立計算核對 | 用 NumPy FFT、逐窗去 mean、periodic Hann、單邊 density、50% overlap、逐窗 mean 重建；SciPy 頻率完全相同，PSD rtol=1e−10、atol=1e−12。此容差是浮點實作核對，不是方法差異容差。 |
| 已知振幅控制 | amplitude=2 的整週期 4／8／13／30 Hz 正弦，長度至少 1 秒；所有 PSD bins 的 `sum(PSD)×df` 應為 2，rtol=1e−10、atol=1e−12；常數經 constant detrend 後為 0。不得拿部分頻帶 trapezoid 當此能量等式。 |
| 跨方法敏感度 | 同一訊號用 1／4 秒窗、各自有效 nperseg／df；記峰值頻率、總矩形／梯形積分、三帶相對值、閉區間與半開區間差、單頻點與分母差。**沒有宣告等價的容許差異**；列出全部差異，不以任意百分比判定方法較佳。 |
| 短訊號／缺口 | Jenqwei 段長不足模型窗依原例外／排除；Before 保留尾端、After 裁尾、ch3/4 無 After。quality samples 不足兩段 30 秒依原 skip；不縮短正式抽樣參數。缺口資料不包成連續頻譜。 |

控制矩陣：fs=200／500；長度=.5／1／2／4／8 秒；4／8／13／30 Hz 邊界正弦、10.3 Hz off-grid、2/6/10/20/40/60 Hz 混合、seed 1922 noise、常數。保持同輸入，僅改名義窗長；短資料 nperseg=min(N,fs×窗長)。另以 fs=200、長度 .04 秒（8 點）觸發單頻點頻帶。

頻帶比較沿用 P-qeeg θ[4,8)、α[8,13)、β[13,30)／三帶和+1e−9；P-hardy 1–4、4–8、8–13、13–30、30–45 閉區間／五帶和+1e−12；P-entropy 單頻點 PSD×df。分母敏感度另固定同一半開三帶分子，比較三帶和、五帶閉區間和、[.5,45] 全帶積分（均加 1e−9，以隔離分母效果）；這些比較設定不是產品 profile。

真實捕獲：五份 Jenqwei 完整錄製／獨立四通道副本、實際 TFLite；P-jenqwei 全四個顯示通道逐段捕獲；P-quality-plot 捕獲入口實際 Welch 呼叫與 semilogy 線值、seed 42 starts、raw／filtered 品質表。表重讀不取代模型／PSD 重算。真實合成產物分開保存，原始來源與模型不覆寫。

Welch 的分段平均、density 單位、預設 detrend／overlap 依本環境 [SciPy 1.15.3 官方文件](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.welch.html)；窗函數與頻譜洩漏的解釋參照 [SciPy spectral analysis](https://docs.scipy.org/doc/scipy/tutorial/signal.html#spectral-analysis)。版本與預設另存 config，不把新版文件的預設直接移植到舊程式。這些來源不支持 EEG 行為效應或哪個窗長適合受試者的結論。
