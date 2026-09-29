# M4：MI 比較契約

日期：2026-09-22。先固定比較量再執行；只新增校準工具／證據，不改產品方法或預設。

- Legacy 固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed`；獨立舊／新 Python process、同來源與環境。數值 dtype／shape／值／NaN、索引、品質接受及 audit 精確一致（rtol=atol=0）；預期短段／非有限失敗須具名，不能把任意空結果當通過。
- Histogram：保留去均值、population std、clip ±5；uniform／quantile、ties 的 unique 邊界、plug-in／Miller–Madow 截零與 MI/min(Hx,Hy)。以保存 PMF 獨立重算 KL 與修正，rtol=atol=1e−12。退化 quantile 的全零 PMF 是 legacy sentinel，不宣稱有效機率分布。
- 已知 histogram 控制：平衡二元獨立／相同訊號（uniform 的理論 0／1 bit，atol=1e−12）、常數與 ties；Gaussian 獨立／rho=.8、獨立 AR(1) 及兩段共同均值變化。bins 8／16／32、樣本數 128／512／2048、uniform／quantile；記錄有限樣本偏差，不用單次 p<.05 當品質驗收門檻。
- Circular null：每個連續來源 run 獨立 shift，保存完整 surrogate 分布、seed、數量與 segment IDs；依既有 RNG 與移位範圍獨立重建，rtol=atol=1e−12。ddof=0、p=(1+#null≥observed)/(B+1)、零方差 z=NaN；B=0／短於16樣本 run 的停用保留。seed 0／7／42、B=19／99／200 分開比較。共同段均值控制另比較錯誤的全域 shift，僅示範 null 假設差異。
- Event KSG：實際 feature 是 θ／α／β 的 Hilbert **振幅包絡**子窗均值，不是平方功率；混合連續／離散 Ross estimator，輸出 nats→bits。保留 scale+jitter、Euclidean 距離、k 的小類別上限、joint 與各 band 加總兩種量。小型控制以暴力成對距離獨立核對，rtol=atol=1e−12；1D 與 sklearn 公開 API 同輸入／seed 比較。不得假設有限樣本估計必≤標籤熵。
- KSG 控制：iid 無標籤效應、已分離的兩類、三欄重複同一特徵；每類 16／64／128、k=1／3／5、seed 0／7／42。label shuffle 保存實際預處理輸入與每次估計，ddof=1（B=1 依 legacy 為0）、加一 p 校正；B=0／19／99／200，另以單一控制涵蓋 joint_mi per-event 的500。改參數時其餘設定固定。
- 真實：五份完整 Jenqwei 錄製的 ch1/2 histogram／lagged／品質 series；talk／move-head 鍵盤絕對微秒只減原 header offset 映射到相對來源時間，不修正既有1µs差。兩筆 trigger 各自是事件時點，不推斷行為開始／結束。KSG 原始完整段包絡，2／5／10秒事件半窗；k／seed／B／子窗重疊分別比較，事件接受及排除明示。
- 合成連續、缺口、短段、非有限、quality=.5／.49／NaN／disabled 明示；population summary 不套逐窗品質 mask，KSG quality disabled。缺口副本不是新錄製或行為標註；分段、raw／BP 與 lagged/zero-lag 不互換。未新增模型推論，denoised 路徑沿用 M0 證據，不能宣稱本輪重新驗收。
- 收尾：來源／工具／環境 hash、表重讀、獨立数值、接受／排除及 NaN 比較、完整測試、真實／合成圖目視。真實衍生物留 ignored local，repository manifest 僅提交合成與摘要。比較不能自動選出最佳 bins/k 或證明真實事件效應；未採新產品 profile。

方法依據：[Ross 2014 原論文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357)（混合變數估計及 k 的取捨）、[sklearn 公開 API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)（nats／jitter／seed；實跑版本另凍結）、[Schreiber & Schmitz 原作者綜述](https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/surropaper/Surrogates.html)（surrogate 依賴 null 假設）。重疊時間子窗的 label shuffle 不據此獲得獨立可交換樣本保證；p 值只作 legacy 比較，不作科學效應宣告。

## 首輪發現後的明示增補（release 前）

首輪 `m4_dev_2026-09-22/expected.log` 在真實事件暴力距離核對失敗：legacy 0、幾何參考 0.04269481 nats。另以固定 seed24／31×3／大 k 重現：NearestNeighbors 自動選 brute 時距離捨入與 KDTree 半徑計數會改變邊界點。原 1e−12 門檻不放寬；release 保存每次 observed 的獨立幾何值及誤差，超門檻明列為**方法限制**，不宣告幾何等價。另核對同 library 半徑的 digamma 公式（1e−12）以偵測捕獲錯誤。舊新精確相容仍為硬門檻；此輪比較不採用修正版估計器，方法修正／新 profile 留明示後續。

## M4-R1 預先契約（2026-09-29）

- `legacy_auto_kdtree` 為現有產品預設：保留自動鄰居搜尋＋KDTree 計數，不修改估計器、scale／jitter、各 band 加總、RNG、品質或索引。
- `ross_euclidean_direct_v1` 為**僅校準用**具名候選，不加入 CLI 或取代預設。float64 直接差平方加總開根號計算 Euclidean 距離；同一距離向量同時選類內第 k 鄰居與全體計數，避免混合兩種距離實作。逐列計算，空間 O(ND)、時間 O(N²D)，不宣稱適合大型 population。
- 候選沿用既有慣例：移除 singleton 類別後計數，k=min(requested, class_count−1)，半徑 nextafter(kth_distance, 0)，全體計數包括自己，負估計截零。零距離 ties 仍包含全部重複點；不追加 jitter、不依資料調整 tolerance。這是明示的有限精度／ties 規則，並非宣稱原論文對離散 ties 的有效性保證。
- 對獨立 dense reference，observed 與**每一個** label-shuffle null 的 nats／bits、mean／std／p／z 預先固定 rtol=atol=1e−12，NaN 位置必須相同。以完整既有78個有效 event呼叫（另2個無估計）重播 RNG，包括預處理消耗的常態亂數；不按 observed 差異篩選。legacy 對固定原碼與保存基準保持 rtol=atol=0。
- 小樣本、重複點／等距 ties、singleton、k=1/3/5/超過類別數、auto／brute／kd_tree／ball_tree 的差異作反例測試；候選幾何一致不等於估計無偏或 label shuffle 的可交換性成立。
- 產品事件圖使用 amplitude-envelope 用語，x 軸明示 pre/post half-window；由當次 pipeline context 提供 fs、額外前端BP、子窗／step、k、seed、surrogate數與品質設定。舊 CSV 缺 context 時明示 unknown，不猜值或讀全域 `_PROV`。CSV 既有欄位與數值不改。
- 方法核對依 [Ross 原文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357) 與 [NearestNeighbors 官方文件](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)；本機 sklearn 1.7.2，線上新版文件不作本機浮點行為證據。
