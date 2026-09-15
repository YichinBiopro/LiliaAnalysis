# 第十八階段 R3：補齊驗收產物

日期：2026-09-15。範圍：`40d2a23` 漏納入的 raw／TFLite 驗收表與圖。

## 本次變更

- 補回 51 CSV、19 PNG、3 SVG，共 73 檔；包含六張 visual manifests 引用的代表圖。
- 僅納入既有驗收產物，沒有重新生成表格、改寫數值或重跑模型。
- 兩個驗收目錄加入局部 ignore 例外；正式來源與一般生成產物的忽略規則保留。
- 本提交不包含工作區尚未提交的 R1／R2 程式及其驗收文件。

## 驗證

- 73 檔全部存在，逐一比對原有 `evidence.json`／`visual_evidence.json` 的預期 SHA-256，全部一致。
- [恢復 manifest](validation/stage18/artifact_restore/evidence.json) 保存原預期 hash 與可攜相對路徑。
- [檢查結果](validation/stage18/artifact_restore/checks/summary.json) 與同目錄 log 保存 73 項 hash 核對。
- 工作區既有 321 tests 完整檢查指紋仍匹配；此產物補齊沒有程式變更，不重跑相同科學分析。
- 本次未新增目視；既有三圖／三圖目視紀錄及其原始圖一併保留。
- 原 CSV 換行／SVG path 末尾空白會被 Git whitespace check 列出；保留原 bytes 與 hash，不為格式改寫歷史產物。新增文件另做 whitespace check。

## 限制與下一步

- 乾淨 checkout 可取得這些表與圖；source reader 重讀仍需要原始錄製、模型及先前暫存四通道副本，補產物不等於解除外部來源依賴。
- R3 補提交完成後可關閉；第十八階段仍需 R4 entropy、R5 Goertzel、R6 quality_check 與剩餘 helper。
- 下一增量沿用 `legacy_overall`，不可把 diagnostic validity 改當原門檻遮罩。
