"""Hardy_2 special branch: numerical contracts, periods and source verification."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plot_event_markers as plot
from lilia.entropy_io import config_id
from lilia.hardy2 import analyze_hardy2, summarize_periods, smooth_metrics, METRIC_KEYS
from lilia.hardy2_io import load_hardy2_table
from lilia.io import bandpass_filter
from lilia.provenance import file_sha256


class Hardy2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.events = [plot.hhmm_to_us(t) for t in ('16:18','16:21','16:36')]
        self.epoch = plot.hhmm_to_us('16:17')

    def source(self, t, x):
        folder = self.root/'Hardy_2(SN036)'
        folder.mkdir(exist_ok=True)
        p = folder/'merged.csv'
        frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
        frame.insert(0, 'Time[us]', t)
        p.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\n'
                     'Channels,1,2,3,4\nSample Rate,500,500,500,500\n'+frame.to_csv(index=False))
        return p

    def run_cli(self, t, x):
        source = self.source(t, x)
        out = self.root/'out'
        def save(fig, path, **kwargs):
            Path(path).write_text('test plot placeholder')
        with patch.object(plot,'IBRAIN_DIR',str(self.root)), patch.object(Figure,'savefig',save), \
             contextlib.redirect_stdout(io.StringIO()):
            audit = plot.plot_hardy2_band_and_indices(str(out))
        return source, out, audit

    def gapped(self):
        t = self.epoch+np.r_[np.arange(100)*2000,10000000+np.arange(5001)*2000,
                             20008000+np.arange(5001)*2000]
        x = np.random.default_rng(4).normal(size=(len(t),4)).astype(np.float32)
        return t, x

    def test_saved_continuous_raw_and_bp_metrics_smoothing_and_period_means(self):
        ref = json.loads((Path(__file__).parent/'fixtures/hardy2_continuous_reference.json').read_text())
        x = np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32)
        t = ref['epoch_us']+np.arange(len(x))*2000
        for name in ('raw','bp'):
            r = analyze_hardy2(t,x,use_bandpass=name=='bp')
            expected = ref['results'][name]
            times = [plot.us_to_local_dt(v).isoformat() for v in r['grid'].columns['window_center_us']]
            self.assertEqual(times,expected['time_local'])
            for k in METRIC_KEYS:
                np.testing.assert_allclose(r['metrics'][k],expected['metrics'][k],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(r['smooth'][k],expected['smooth'][k],rtol=1e-12,atol=1e-12)
            summary = summarize_periods(r['grid'],r['metrics'],self.events)
            for row in summary['periods']:
                if row['name'] in expected['period_means']:
                    for k in METRIC_KEYS:
                        self.assertAlmostEqual(row['means'][k],expected['period_means'][row['name']][k],places=12)
            self.assertEqual(r['segments'][0]['tail_samples'],3)

    def test_gap_grid_and_filter_do_not_borrow_from_other_segment(self):
        t,x = self.gapped()
        r = analyze_hardy2(t,x)
        np.testing.assert_array_equal(r['grid'].starts,[100,2600,5101,7601])
        self.assertEqual(r['grid'].columns['segment_id'].tolist(),[1,1,2,2])
        self.assertEqual(r['grid'].time_s[0],12.5)
        self.assertEqual(r['segments'][0]['status'],'short_segment')
        other=x.copy();other[100:5101]*=1000
        changed=analyze_hardy2(t,other)
        for k in METRIC_KEYS:
            np.testing.assert_array_equal(r['metrics'][k][2:],changed['metrics'][k][2:])
            np.testing.assert_array_equal(r['smooth'][k][2:],changed['smooth'][k][2:])
        independent=analyze_hardy2(t[5101:],x[5101:])
        np.testing.assert_array_equal(r['metrics']['alpha'][2:],independent['metrics']['alpha'])

    def test_nonfinite_filtered_segment_and_raw_window_have_explicit_exclusions(self):
        t,x=self.gapped();x[101,0]=np.nan
        bp=analyze_hardy2(t,x)
        self.assertEqual(bp['valid'].tolist(),[False,False,True,True])
        self.assertEqual(bp['window_audit'][0]['status'],'nonfinite_segment')
        raw=analyze_hardy2(t,x,use_bandpass=False)
        self.assertEqual(raw['valid'].tolist(),[False,True,True,True])
        self.assertEqual(raw['window_audit'][0]['status'],'nonfinite_window')
        self.assertTrue(np.isnan(raw['smooth']['alpha'][0]))

    def test_filter_failure_and_invalid_metric_output_stay_nan(self):
        t,x=self.gapped()
        with patch('lilia.hardy2.bandpass_filter',side_effect=ValueError('failed filter')):
            r=analyze_hardy2(t,x)
        self.assertFalse(r['valid'].any())
        self.assertEqual(r['segments'][1]['status'],'filter_error')
        with patch('lilia.hardy2.compute_qeeg_indices',return_value=dict.fromkeys(METRIC_KEYS,np.inf)):
            r=analyze_hardy2(t,x,use_bandpass=False)
        self.assertFalse(r['valid'].any())
        self.assertTrue(np.isnan(r['metrics']['delta']).all())
        self.assertEqual(r['window_audit'][0]['status'],'metric_error')

    def test_period_boundary_crossing_is_excluded_instead_of_center_assigned(self):
        t=self.events[0]-8000000+np.arange(10000)*2000
        r=analyze_hardy2(t,np.ones((len(t),4)),use_bandpass=False)
        metrics={k:np.array([1.,99.,3.,5.]) for k in METRIC_KEYS}
        summary=summarize_periods(r['grid'],metrics,self.events)
        self.assertEqual(summary['boundary_crossing_rows'],[1])
        self.assertEqual(summary['periods'][0]['accepted_rows'],[0])
        self.assertEqual(summary['periods'][1]['accepted_rows'],[2,3])
        self.assertEqual(summary['periods'][0]['means']['alpha'],1.)
        times=np.array([plot.us_to_local_dt(v) for v in r['grid'].columns['window_center_us']])
        old_mask=plot._hardy2_period_masks(times)[0][1]
        self.assertEqual(float(np.mean(metrics['alpha'][old_mask])),50.)
        # An end exactly on an event belongs to the preceding half-open period.
        t=self.events[0]-5000000+np.arange(5000)*2000
        r=analyze_hardy2(t,np.ones((len(t),4)),use_bandpass=False)
        summary=summarize_periods(r['grid'],r['metrics'],self.events)
        self.assertEqual(summary['boundary_crossing_rows'],[])
        self.assertEqual(summary['periods'][0]['accepted_rows'],[0])
        self.assertEqual(summary['periods'][1]['accepted_rows'],[1])

    def test_period_spans_and_smoothing_break_at_invalid_rows_and_source_ids(self):
        t,x=self.gapped();r=analyze_hardy2(t,x)
        metrics={k:np.array([1.,np.nan,10.,20.]) for k in METRIC_KEYS}
        smoothed=smooth_metrics(metrics,r['grid'].columns['segment_id'])
        np.testing.assert_allclose(smoothed['alpha'],[1.,np.nan,15.,15.],equal_nan=True)
        s=summarize_periods(r['grid'],metrics,self.events)
        self.assertEqual(s['periods'][0]['excluded_rows'],[1])
        spans=s['periods'][0]['spans']
        self.assertEqual([r['metric_rows'] for r in spans],[[0],[2,3]])
        self.assertLess(spans[0]['end_us'],spans[1]['start_us'])
        self.assertIsNone(s['periods'][1]['means']['alpha'])

    def test_plot_preserves_small_gap_points_and_segment_shading(self):
        t,x=self.gapped();r=analyze_hardy2(t,x)
        r['summary']=summarize_periods(r['grid'],r['metrics'],self.events)
        captured=[]
        def save(fig,*args,**kwargs):
            if not captured:
                captured.append(fig.axes[0])
        with patch.object(Figure,'savefig',save):
            plot._plot_hardy2_result(r,self.root,5)
        ax=captured[0]
        self.assertTrue(np.isnan(ax.lines[0].get_ydata()[2]))
        self.assertEqual(ax.lines[1].get_marker(),'.')
        self.assertEqual(len(ax.patches),2)
        self.assertTrue(np.isfinite(ax.lines[0].get_ydata()[3]))

    def test_cli_round_trip_includes_all_candidates_and_period_audit(self):
        t,x=self.gapped();source,out,audit=self.run_cli(t,x)
        self.assertEqual(audit['status'],'complete')
        self.assertEqual(len(audit['artifacts']),14)
        f,m=load_hardy2_table(out/'Hardy_2_SN036_metrics.csv',source)
        self.assertEqual(f.window_start_idx.tolist(),[100,2600,5101,7601])
        self.assertEqual(f.quality_state.unique().tolist(),['disabled'])
        self.assertEqual(m['analysis']['summary']['periods'][0]['accepted_rows'],[0,1,2,3])
        self.assertEqual(m['analysis']['segments'][0]['tail_samples'],100)

    def test_source_and_rehashed_window_or_smoothing_tamper_rejected(self):
        t,x=self.gapped();source,out,_=self.run_cli(t,x)
        p=out/'Hardy_2_SN036_metrics.csv';side=Path(str(p)+'.meta.json')
        saved=p.read_text();saved_meta=side.read_text()
        for col in ('window_start_idx','alpha_smooth'):
            f=pd.read_csv(p);f.loc[0,col]+=1;f.to_csv(p,index=False)
            meta=json.loads(side.read_text());meta['table_sha256']=file_sha256(p);side.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'mapping differs|smoothing differs'):
                load_hardy2_table(p,source)
            p.write_text(saved);side.write_text(saved_meta)
        source.write_text(source.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):
            load_hardy2_table(p,source)

    def test_rehashed_period_or_segment_audit_tamper_rejected(self):
        t,x=self.gapped();source,out,_=self.run_cli(t,x)
        p=out/'Hardy_2_SN036_metrics.csv';side=Path(str(p)+'.meta.json');saved=side.read_text()
        for kind in ('period','segment'):
            meta=json.loads(saved)
            if kind=='period':meta['analysis']['summary']['periods'][0]['accepted_rows']=[0]
            else:meta['analysis']['segments'][1]['tail_start_idx']+=1
            meta['analysis_id']=config_id(meta['analysis']);side.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'period summary differs|segment mapping differs'):
                load_hardy2_table(p,source)

    def test_all_short_or_all_invalid_cli_persists_failure(self):
        for n,invalid in [(100,False),(2500,True)]:
            t=self.epoch+np.arange(n)*2000;x=np.ones((n,4))
            if invalid:x[0,0]=np.nan
            with self.assertRaisesRegex(ValueError,'No valid complete'):
                self.run_cli(t,x)
            audit=json.loads((self.root/'out/Hardy_2_SN036_analysis.json').read_text())
            self.assertEqual(audit['status'],'failed')
            self.assertEqual(audit['valid_windows'],0)
            self.assertTrue(audit['errors'])
            if invalid:
                f,_=load_hardy2_table(self.root/'out/Hardy_2_SN036_metrics.csv',self.root/'Hardy_2(SN036)/merged.csv')
                self.assertFalse(f.metric_valid.any())

    def test_invalid_settings_timestamps_and_compatibility_adapter(self):
        t=self.epoch+np.arange(2500)*2000;x=np.ones((len(t),4))
        for kwargs in ({'win_sec':0},{'fs':0},{'low':50,'high':45}):
            with self.assertRaises(ValueError):analyze_hardy2(t,x,**kwargs)
        with self.assertRaisesRegex(ValueError,'integer'):
            analyze_hardy2(t.astype(float),x)
        bad=t.copy();bad[20]=bad[19]
        with self.assertRaisesRegex(ValueError,'strictly increasing'):
            analyze_hardy2(bad,x)
        y=bandpass_filter(x)
        dt,b,i=plot._compute_hardy2_windowed_metrics(t,y)
        r=analyze_hardy2(t,y,use_bandpass=False)
        self.assertEqual(dt[0],plot.us_to_local_dt(t[1250]))
        np.testing.assert_array_equal(b['alpha'],r['metrics']['alpha'])
        np.testing.assert_array_equal(i['flow'],r['metrics']['flow'])

    def test_explicit_cli_source_and_missing_file_failure_audit(self):
        source=self.root/'missing.csv';out=self.root/'failed'
        args=['plot_event_markers.py','--hardy2-analysis','--hardy2-csv',str(source),
              '--hardy2-outdir',str(out)]
        with patch('sys.argv',args), self.assertRaises(FileNotFoundError):
            plot.main()
        audit=json.loads((out/'Hardy_2_SN036_analysis.json').read_text())
        self.assertEqual(audit['status'],'failed')
        self.assertEqual(audit['source_path'],str(source))
        self.assertEqual(audit['artifacts'],{})


if __name__=='__main__':
    unittest.main()
