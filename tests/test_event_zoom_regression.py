"""Event zoom preserves full-session policies, true rectangles and missing data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.dates as mdates
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_meditation_zoom as zoom
from lilia.entropy_io import config_id
from lilia.event_qeeg import analyze_recording, summarize_branch, INDEX_KEYS
from lilia.event_zoom import select_zoom, zoom_values
from lilia.event_zoom_io import load_zoom_table
from lilia.provenance import file_sha256
from lilia.windowing import build_window_grid


def good(data, **kwargs):
    return {'overall': np.ones(data.shape[0])}


class EventZoomTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.epoch = zoom.pem.hhmm_to_us('14:00')

    def event(self, start=60, end=150, label='Test', participates=True):
        return {'start_us': self.epoch+int(start*1e6), 'end_us': self.epoch+int(end*1e6),
                'label': label, 'color': 'red', 'participates': participates}

    def source(self, t, x):
        source = self.root/'merged.csv'
        f = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
        f.insert(0, 'Time[us]', t)
        source.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n'+f.to_csv(index=False))
        return source

    def gapped(self):
        t = self.epoch+np.r_[np.arange(100)*2000, 10000000+np.arange(35000)*2000,
                            80008000+np.arange(45000)*2000]
        return t, np.random.default_rng(1213).normal(size=(len(t),4)).astype(np.float32)

    def analyze(self, t, x, events=None, scorer=good):
        branch = analyze_recording(t, x, scorer=scorer, quality_params=zoom.pem.QUALITY_PARAMS)['bp']
        branch['summary'] = summarize_branch(branch, events or [self.event()], 'pre-event-rest')
        return branch

    def run_zoom(self, source, events=None, ds=17, scorer=good, save=None):
        def placeholder(fig, path, **kwargs):
            Path(path).write_text('plot placeholder')
        with patch.object(zoom.pem, 'get_eeg_quality_index_v2_parametric', scorer), \
                patch.object(Figure, 'savefig', save or placeholder), contextlib.redirect_stdout(io.StringIO()):
            return zoom.run_zoom('Test', {'sn':'SN000'}, source, self.root/'out', 'Test', ds,
                                 events=events or [self.event()])

    def audit(self):
        return json.loads(next((self.root/'out').glob('*_analysis.json')).read_text())

    def table(self):
        return next((self.root/'out').glob('*_bp_metrics.csv'))

    def test_continuous_real_quality_metrics_and_aligned_baseline_match_old(self):
        ref=json.loads((Path(__file__).parent/'fixtures/meditation_zoom_continuous_reference.json').read_text())
        x=np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32)
        t=ref['epoch_us']+np.arange(len(x))*2000
        b=self.analyze(t,x,[self.event(60,120)],zoom.pem.get_eeg_quality_index_v2_parametric)
        old=ref['continuous']
        np.testing.assert_allclose(b['quality'],old['quality'],rtol=1e-12,atol=1e-12)
        for key in INDEX_KEYS:np.testing.assert_allclose(b['scores'][key],old['scores'][key],rtol=1e-12,atol=1e-12)
        for key in ('heatmap_abs','heatmap_delta'):
            np.testing.assert_allclose(b['summary'][key],old[key],rtol=1e-12,atol=1e-12)
        centers=[r['center_us'] for r in b['summary']['bins']]
        np.testing.assert_array_equal(np.asarray(old['bin_centers_us'])-centers,[2500000]*4)
        s=select_zoom(t,b,[self.event(60,120)],'Test')
        np.testing.assert_allclose(zoom_values(b,s),old['plotted_heatmap'],atol=1e-12)

    def test_gap_crop_retains_first_last_indices_and_true_bin_edges(self):
        t,x=self.gapped();source=self.source(t,x)
        a=self.run_zoom(source)
        f,m=load_zoom_table(self.table(),source)
        self.assertEqual(a['status'],'complete')
        s=m['zoom']['selection']
        self.assertEqual([r['segment_id'] for r in s['display_runs']],[1,2])
        for run in s['display_runs']:
            self.assertEqual(run['display_raw_indices'][0],run['raw_start_idx'])
            self.assertEqual(run['display_raw_indices'][-1],run['raw_end_idx']-1)
        self.assertEqual(len(s['missing_raw_intervals']),1)
        gap=s['missing_raw_intervals'][0]
        self.assertEqual(gap['end_us']-gap['start_us'],8000)
        self.assertEqual(len(s['complete_event_bins']),2)
        self.assertEqual(len(s['boundary_bins']),2)
        self.assertEqual(m['zoom']['segments'][0]['status'],'short_segment')
        self.assertEqual(f.segment_id.iloc[0],1)
        for name,digest in a['artifacts'].items():self.assertEqual(file_sha256(self.root/'out'/name),digest)

    def test_draw_keeps_runs_disconnected_and_boundary_bins_missing(self):
        t,x=self.gapped();source=self.source(t,x);captured=[]
        b=self.analyze(t,x)
        s=select_zoom(t,b,[self.event()],'Test',17)
        def inspect(fig,path,**kwargs):
            self.assertEqual(len(fig.axes[0].lines),2)
            heat=fig.axes[4]
            for raw_ax in fig.axes[:4]:
                np.testing.assert_allclose([raw_ax.get_position().x0,raw_ax.get_position().x1],
                                           [heat.get_position().x0,heat.get_position().x1])
            self.assertEqual(len(heat.collections),len(s['display_bins']))
            for mesh,i in zip(heat.collections,s['display_bins']):
                expected=[zoom.pem.us_to_local_dt(b['summary']['bins'][i][k]) for k in ('start_us','end_us')]
                np.testing.assert_allclose(mesh.get_coordinates()[0,:,0],mdates.date2num(expected))
                if i in s['boundary_bins']:self.assertTrue(np.ma.getmaskarray(mesh.get_array()).all())
            captured.append(True);Path(path).write_text('plot')
        self.run_zoom(source,save=inspect)
        self.assertEqual(len(captured),2)
        self.assertEqual(plt.get_fignums(),[])

    def test_short_event_never_falls_back_to_other_bins(self):
        t,x=self.gapped();source=self.source(t,x)
        with self.assertRaisesRegex(ValueError,'no_complete_event_bins'):
            self.run_zoom(source,[self.event(65,67)])
        _,m=load_zoom_table(self.table(),source)
        s=m['zoom']['selection']
        self.assertEqual(s['complete_event_bins'],[])
        self.assertEqual(s['display_bins'],[1])
        self.assertEqual(len(self.audit()['artifacts']),4)

    def test_no_pre_event_rest_does_not_fallback(self):
        t,x=self.gapped();source=self.source(t,x)
        events=[self.event(10,60,'Earlier'),self.event()]
        with self.assertRaisesRegex(ValueError,'missing_baseline'):self.run_zoom(source,events)
        _,m=load_zoom_table(self.table(),source)
        self.assertEqual(m['zoom']['selection']['baseline']['baseline_bins'],[])
        self.assertEqual(self.audit()['status'],'excluded')

    def test_nonparticipation_and_outside_recording_are_audited(self):
        t,x=self.gapped();source=self.source(t,x)
        for event,reason in [(self.event(participates=False),'not_participating'),
                             (self.event(200,250),'no_raw_samples')]:
            with self.assertRaisesRegex(ValueError,reason):self.run_zoom(source,[event])
            _,m=load_zoom_table(self.table(),source)
            self.assertEqual(m['zoom']['selection']['status'],reason)

    def test_all_short_has_partial_raw_plot_and_audit_without_table(self):
        t=self.epoch+np.arange(100)*2000;source=self.source(t,np.ones((100,4)))
        with self.assertRaisesRegex(ValueError,'no_complete_metric_windows'):
            self.run_zoom(source,[self.event(0,1)])
        a=self.audit();self.assertEqual(a['status'],'excluded')
        self.assertEqual(len(a['artifacts']),2)
        self.assertEqual(a['segments'][0]['status'],'short_segment')

    def test_quality_nan_error_and_low_values_exclude_baseline(self):
        t,x=self.gapped();source=self.source(t,x)
        def broken(*a,**kw):raise ValueError('quality failure')
        for scorer in [lambda *a,**kw:{'overall':np.full(4,np.nan)},broken,
                       lambda *a,**kw:{'overall':np.zeros(4)}]:
            with self.assertRaisesRegex(ValueError,'missing_baseline'):self.run_zoom(source,scorer=scorer)
            f,_=load_zoom_table(self.table(),source)
            self.assertFalse(f.quality_valid.any())

    def test_nonfinite_source_segment_excludes_event_but_keeps_baseline(self):
        t,x=self.gapped();x[-1,0]=np.nan;source=self.source(t,x)
        with self.assertRaisesRegex(ValueError,'no_accepted_event_bins'):self.run_zoom(source)
        f,m=load_zoom_table(self.table(),source)
        self.assertTrue(f.loc[f.segment_id==1,'quality_valid'].all())
        self.assertFalse(f.loc[f.segment_id==2,'quality_valid'].any())
        self.assertEqual(m['zoom']['segments'][2]['status'],'nonfinite_segment')

    def test_invalid_ds_unknown_event_and_missing_source_save_failed_audit(self):
        t,x=self.gapped();source=self.source(t,x)
        for ds in (0,-1):
            with self.assertRaisesRegex(ValueError,'positive integer'):self.run_zoom(source,ds=ds)
            self.assertEqual(self.audit()['status'],'failed')
        with self.assertRaisesRegex(ValueError,'Unknown event'):self.run_zoom(source,[self.event(label='Other')])
        with self.assertRaises(FileNotFoundError):self.run_zoom(self.root/'missing.csv')
        self.assertEqual(self.audit()['status'],'failed')

    def test_rehashed_crop_summary_mapping_and_raw_changes_are_rejected(self):
        t,x=self.gapped();source=self.source(t,x);self.run_zoom(source)
        table=self.table();side=Path(str(table)+'.meta.json');saved=side.read_text()
        m=json.loads(saved);m['zoom']['selection']['display_runs'][0]['display_raw_indices'][0]+=1
        m['zoom_id']=config_id(m['zoom']);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'crop differs'):load_zoom_table(table,source)
        m=json.loads(saved);m['analysis']['summary']['heatmap_delta'][0][3]+=1
        m['analysis_id']=config_id(m['analysis']);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'summary differs'):load_zoom_table(table,source)
        side.write_text(saved)
        f=pd.read_csv(table);f.loc[0,'quality_raw_start_idx']+=1;f.to_csv(table,index=False)
        m=json.loads(saved);m['table_sha256']=file_sha256(table);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'mapping differs'):load_zoom_table(table,source)
        source.write_text(source.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):load_zoom_table(table,source)

    def test_complete_window_policy_changes_non_aligned_baseline_and_event(self):
        t=self.epoch+np.arange(90000)*2000
        grid=build_window_grid(t,500,5)
        scores={k:np.repeat(np.arange(36,dtype=float)[:,None],4,axis=1) for k in INDEX_KEYS}
        b={'grid':grid,'scores':scores,'valid':np.ones(36,dtype=bool)}
        events=[self.event(31,61,'Earlier'),self.event(100,170)]
        b['summary']=summarize_branch(b,events,'pre-event-rest')
        s=select_zoom(t,b,events,'Test')
        self.assertEqual(s['baseline']['baseline_bins'],[])
        self.assertEqual(s['complete_event_bins'],[4])
        self.assertEqual(s['status'],'missing_baseline')
        self.assertTrue(np.isnan(zoom_values(b,s)).all())


if __name__=='__main__':unittest.main()
