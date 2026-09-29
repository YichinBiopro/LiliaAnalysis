# 方法校準 M4-R1：KSG 半徑與事件圖語義

日期：2026-09-29。狀態：完成。範圍：校準候選、完整 null 核對、事件圖標示；產品估計器、CLI 預設、品質與索引政策不變。

## 本階段變更

- [預先契約](MI_COMPARISON_CONTRACT.md#m4-r1-預先契約2026-09-29) 固定 `legacy_auto_kdtree` 與僅校準用 `ross_euclidean_direct_v1`、半徑／ties 規則及 1e−12 門檻。[候選實作](../../tools/mi_radius_profiles.py) 以同一套直接 Euclidean 距離選類內第 k 鄰居及計數；沒有引入產品選項。
- [完整重播工具](../../tools/validate_mi_radius_profiles.py) 使用保存 features／labels／observed＋每筆 null，重播預處理消耗的 RNG 與所有 label permutations；逐筆核對候選、獨立 dense 幾何參考、legacy library 計數，以及 observed／null／mean／std／p／z。
- [事件圖](../../spectral_entropy.py) 與 [per-event 圖](../../joint_mi.py) 改稱 amplitude envelope，明示 pre/post half-window，當次 context 顯示前端、fs、子窗與 step、k、seed、surrogate 數、品質狀態。歷史 CSV 缺 context 時明示 unknown，不沿用 `_PROV`。
- 舊新捕獲在來源程式 hash 更新後只對允許的 hash 衍生欄位正規化；各版 table reader 仍驗證來源、參數、config ID、表及 sidecar hash。分析數值、設定、audit 必須精確相同。

## 驗證

- [完整檢查](validation/method_calibration/m4r1_full_checks_2026-09-29/summary.json)：384 tests／0 skipped；211 Python 編譯、Pyflakes、bundle、diff 通過。新反例涵蓋小樣本、等距／重複點、singleton、k=1/3/5/超上限、auto／brute／kd_tree／ball_tree、最後一筆 null 污染及圖形來源 context。
- [舊新入口](validation/method_calibration/m4r1_verified_release_2026-09-29/analysis.json)：18 入口＋1 控制套件，5,884 陣列精確一致（rtol=atol=0），48 reader、8 event 來源重載；索引、NaN、品質／事件接受排除與設定不變。[本機 570](validation/method_calibration/m4r1_evidence_local_2026-09-29/summary.json)／[repository 293](validation/method_calibration/m4r1_evidence_repository_2026-09-29/summary.json) 項證據通過。程式碼 hash 導致的 config ID／表 hash 必須各自正確綁定，兩版 hash 值本身不相等。
- [完整半徑比較](validation/method_calibration/m4r1_radius_2026-09-29/analysis.json)：4 event 案例 78 有效＋2 無估計，合成控制 46，有效共 124 呼叫／20,801 筆 null；候選對獨立參考的 observed／每筆 null／p／z 最大誤差 0。Legacy 的 observed 有 7 呼叫、null 有 1,026 筆超 1e−12；其中 5 呼叫 observed 相同但 null 不同。legacy 對幾何參考最大差 0.31075 bits（event gap 的 null），不宣告 legacy 幾何等價。[本機 23](validation/method_calibration/m4r1_radius_evidence_local_2026-09-29/summary.json)／[repository 17](validation/method_calibration/m4r1_radius_evidence_repository_2026-09-29/summary.json) 項產物 hash 通過。
- [圖形比較](validation/method_calibration/m4r1_final_plots_2026-09-29/analysis.json)：合成真實時間缺口／短窗事件 16 列及 audit 精確一致，五種現行／歷史 context 圖 84 組曲線、bar、scatter 資料與凍結版精確一致；[28 項圖形證據](validation/method_calibration/m4r1_plot_evidence_2026-09-29/summary.json) 通過。目視檢查 4 張真實／合成入口事件圖、5 張 context 圖及 5 張半徑圖：標示可讀，缺口／排除明確，null 與 p 差異圖不作行為效應宣稱。

## 限制與下一步

- 候選只證明定義好的浮點幾何規則與獨立參考一致；不證明估計無偏、重疊子窗可交換或鍵盤 trigger 是行為真值。產品仍採 legacy；若未來改預設，需另做具名 profile 與方法效用驗收。原 7 組差異之外發現的 5 組 null 差異已保存。
- 第一次捕獲因來源 hash 衍生 metadata 不同停下，第二次因誤用單一原碼 hash 判斷組合 code ID 停下；修正驗證工具後重新完整捕獲並通過。第一次補充圖測試漏讀 legend title，補上後重新執行通過；失敗產物留 ignored 開發目錄。
- 下一任務：[架構與 F19](TASKS.md#architecture-f19)。M3／M4 原增量與本次修改在同一工作區，發佈狀態依實際 Git 遠端為準。
