# 方法校準 M4：MI 基準、敏感度與 KSG 邊界發現

日期：2026-09-22。狀態：比較工作完成；KSG 幾何等價未通過，修正與產品圖形標示列 M4-R1。範圍：工具／測試／證據；產品方法、預設、品質／索引與 bundle 未修改。

## 本階段變更

- [獨立捕獲](../../tools/freeze_mi_profiles.py) 以固定 `aa0df5f4ebaefddae1fee8c9e50e95323e777aed` 在不同 process 跑舊／新原碼，保存實際圖線、histogram、lagged MI、event features／null、CSV／audit；來源／原碼／工具／環境綁定 hash。
- [比較工具](../../tools/compare_mi_methods.py) 依 [預先契約與明示增補](MI_COMPARISON_CONTRACT.md) 分開 histogram、circular null 與混合 continuous/discrete event KSG。PMF／KL／Miller–Madow、nats→bits、暴力距離／library 半徑、兩種 null 的 ddof／p／z 各自核對。
- [增量檢視工具](../../tools/review_mi_calibration.py) 對實際 observed 幾何差異重播同一預處理 RNG／label permutations，保存新舊 null 差異；另外提供振幅包絡標示及實際設定明確的事件圖。原始產品圖保留，不覆寫證據。

## 驗證

- [最終完整檢查](validation/method_calibration/m4_final_checks_2026-09-22/summary.json)：378 tests／0 skipped；207 Python 編譯、Pyflakes、bundle、diff 通過。[7 項新測試](../../tests/test_mi_method_comparison_regression.py) 涵蓋已知 0／1 bit、錯誤結果偵測、1D sklearn、幾何與半徑差異、singleton、null／品質邊界工具與意外空結果。
- [舊新數值](validation/method_calibration/m4_release_2026-09-22/analysis.json)：18 入口案例（8 真實／10 合成）＋1 控制套件；5,884 陣列與 metadata 精確相同，最大誤差 0、rtol=atol=0。5,880 非空陣列供一般 evidence，4 空陣列另逐 key／dtype／shape 核對。
- 舊新共 48 次來源表 reader（series＋summary）；4 event 案例各舊／新來源重載共 8 次，事件 CSV 精確回讀並核对重載來源的事件時間／索引／接受排除 audit。event 尚無產品版本化 table reader，不能把來源重載稱為現成 reader 驗收。
- [本機 569](validation/method_calibration/m4_evidence_local_2026-09-22/summary.json)／[repository 293](validation/method_calibration/m4_evidence_repository_2026-09-22/summary.json) 項證據通過。幾何／null 補充證據 [本機 17](validation/method_calibration/m4_review_evidence_local_2026-09-22/summary.json)／[repository 11](validation/method_calibration/m4_review_evidence_repository_2026-09-22/summary.json) 通過；這些檢查證明差異產物可追蹤，不代表幾何等價通過。
- 149 合成控制／邊界量測、180 真實 histogram 設定比較、80 event 參數呼叫（78 有估計、2 個缺口長窗口無估計）；bins 8/16/32、k 1/3/5、seed 0/7/42、surrogates 0/1/19/99/200/500 與樣本／事件窗／子窗重疊依契約各自比較。
- [29 張圖目視紀錄](validation/method_calibration/m4_acceptance_2026-09-22/visual_review.json)：連續／缺口斷線、全排除空圖、diagnostics unavailable、population unmasked 與數值曲線核對；4 張原 event 圖有下述標示限制，另供4張明示設定圖。目視產物／scope／原失敗紀錄 hash [本機35](validation/method_calibration/m4_visual_local_2026-09-22/summary.json)／[repository25](validation/method_calibration/m4_visual_repository_2026-09-22/summary.json) 通過。

## 觀測差異與未通過項目

- IID 獨立控制：128 樣本、quantile 32 bins 的 plug-in MI 為 3.125 bits，Miller–Madow 仍有 2.80377；2048 樣本時分別為 0.38428／0.09088。不可把正 MI 或修正後 MI 當依賴證據。
- 兩段共同均值控制：段內 circular null p=0.28856；錯誤全域 shift p=0.00498。不同 null 假設不能互換。重複三欄的分離類別控制，joint 約1.00566 bits、各 band 加總約3.01697；後者不是 joint information。
- [幾何差異明細](validation/method_calibration/m4_review_final_2026-09-22/findings.json)：78 個真實／合成 event observed 中 7 個超過原1e−12門檻；真實最大差0.07534 bits、全部最大0.20037 bits。小類別／較大 k 觸發自動 brute 距離與 KDTree 邊界計數的浮點不一致，library 半徑公式可重現 legacy；沒有放寬門檻或修改產品估計器。
- talk ch1 的2秒事件半窗，每類6子窗／k=3：legacy observed 0，幾何參考0.06160 bits；同一200次 label shuffle 的 p 從1變為0.33831。僅對7個 observed 有差異的呼叫完整重播 null，不能宣稱所有其他 null 都已幾何驗證；原幾何失敗 [log](validation/method_calibration/m4_acceptance_2026-09-22/failures/geometry_expected.log) 保留。
- 品質 .5／disabled 接受12窗；.49／NaN 接受0窗、MI全NaN；五種情況 population 仍為12,000樣本、0.90457 bits且 null 相同。品質政策／索引／NaN精確相容，不因方法比較調整。
- 原事件圖仍寫「power」，實際是 amplitude envelope；通用 footer 也不是 event 的窗口／前端設定，直接 helper 捕獲還會繼承上一入口 `_PROV`。原圖只作 legacy 形狀證據，方法判讀以 metadata 與補充 `event_review.png` 為準。部分 joint heatmap 相鄰刻度擁擠亦列目視限制。
- 首輪短案例工具預期了錯誤的拒絕位置，實際先由 window grid 拒絕；已限定正確例外字串並重跑 release，[原失敗 log](validation/method_calibration/m4_acceptance_2026-09-22/failures/short_outcome_expected.log) 保留。這是工具修正，未改產品失敗語義。

## 未解問題與下一步

- [scope](validation/method_calibration/m4_acceptance_2026-09-22/scope.json)：五份完整 Jenqwei 的 ch1/2；talk／move-head 只使用鍵盤 trigger，保留原1µs差，不推斷行為起訖。無新增模型推論，denoised 只沿用 M0，未宣稱本輪重驗；重疊子窗的 label shuffle 亦不等於獨立可交換樣本保證。
- 下一步 [M4-R1](TASKS.md#method-calibration)：明示 KSG 半徑相容策略／候選 profile，驗證 observed 與完整 null，再收斂事件圖語義。科學判讀依 [Ross 原論文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357) 與 [surrogate 原作者綜述](https://www.pks.mpg.de/tisean/TISEAN_2.1/docs/surropaper/Surrogates.html)，本輪不宣告估計器優劣或真實事件效應。
- Git：M2 `6eef079` 狀態沿用前輪遠端確認；M3／M4 修改與證據仍留工作區，未提交／推送。
