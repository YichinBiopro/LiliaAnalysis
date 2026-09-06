# plot_index_vs_raw bundle

This bundle is for running `plot_index_vs_raw.py` as a self-contained package.

## Included files
- `plot_index_vs_raw.py`
- `plot_event_markers.py`
- `merge_subject_csvs.py`
- `lilia/` package (Python source files)
- `requirements.txt`

`merge_subject_csvs.py` can be used to create the per-subject `merged.csv`
files before running the plotting scripts.

## Example run command
```bash
python plot_index_vs_raw.py --subject Hsin --ch 1
```

To merge source CSV files under the bundle's `iBrainCenter/` and `YoGa/`
directories:

```bash
python merge_subject_csvs.py
```

For custom local-time markers and a 30-second pre-marker baseline:

```bash
python plot_index_vs_raw.py --subject test02 --ch 1 \
  --raw-csv ../iBrainCenter/test02/merged.csv \
  --marker-date 2026-09-04 \
  --custom-markers 15:11:48 15:49:00 16:05:37 \
  --baseline-sec 30 --post-sec 30 \
  --quality-threshold -1 --raw-ylim -450000 450000
```

The custom mode writes a PNG/SVG plot and a summary CSV. A marker before the
recording uses the first real recording 30 seconds as its baseline; its
post-marker value remains unavailable if no raw data exists after the marker.

## 維護與同步（2026-09-06）

主程式 `plot_index_vs_raw.py` 已包含本套件原有的 custom markers、日期、baseline fallback 與 summary 功能。請在 repo 根目錄修改主程式或 `lilia/`，再執行 `python build_bundles.py`；`python build_bundles.py --check` 可檢查副本是否一致。此工具保留 README、requirements 及資料檔。

custom marker 模式已支援缺口分段濾波，並以指定 `--fs` 對齊品質與 qEEG 視窗。舊 entropy CSV 繪圖模式目前要求原始錄製連續；有缺口或視窗數不符會明確報錯，需要完成上游時間 metadata 遷移後才能安全重畫。
