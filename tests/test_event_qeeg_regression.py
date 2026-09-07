import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import plot_event_markers as plot
from lilia.entropy_io import config_id
from lilia.event_qeeg import analyze_recording, summarize_branch, INDEX_KEYS
from lilia.event_qeeg_io import load_event_qeeg_table
from lilia.io import bandpass_filter
from lilia.provenance import file_sha256
from lilia.qeeg import compute_qeeg_indices
from lilia.windowing import build_window_grid, continuous_slices


def good_quality(data, **kwargs):
    return {'overall': np.ones(data.shape[0])}


class EventQEEGTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def analyze(self, t, x, **kwargs):
        return analyze_recording(t, x, scorer=kwargs.pop('scorer', good_quality),
                                 quality_params=plot.QUALITY_PARAMS, **kwargs)

    def source(self, t, x):
        folder = self.root / 'subject'
        folder.mkdir(exist_ok=True)
        source = folder / 'merged.csv'
        frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
        frame.insert(0, 'Time[us]', t)
        with source.open('w') as f:
            f.write('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n')
            frame.to_csv(f,index=False)
        return source

    def gapped(self):
        t = 1778565600000000 + np.r_[np.arange(17503)*2000, 50000000+np.arange(22502)*2000]
        x = np.random.default_rng(832).normal(size=(len(t),4)).astype(np.float32)
        return t, x

    def run_plot(self, source, use_tflite=False, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return plot.plot_subject('Test', {'dir':source.parent.name,'sn':'SN000'},str(self.root/'out'),500,
                base_dir=str(self.root),with_events=False,use_tflite=use_tflite,**kwargs)

    def test_real_continuous_bp_model_quality_and_summary_match_reference(self):
        ref=json.loads((Path(__file__).parent/'fixtures/event_qeeg_continuous_reference.json').read_text())
        self.assertEqual(file_sha256(plot.TFLITE_MODEL_PATH),ref['model_sha256'])
        x=np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32)
        t=ref['epoch_us']+np.arange(len(x))*2000
        result=self.analyze(t,x,scorer=plot.get_eeg_quality_index_v2_parametric,model_path=plot.TFLITE_MODEL_PATH)
        self.assertEqual(result['errors'],[])
        old=ref['results']
        for branch_name,prefix,score_key in [('bp','','qeeg_filt'),('tflite','tfl_','qeeg_tfl')]:
            branch=result[branch_name]
            np.testing.assert_allclose(branch['quality'],old['q_overall'],atol=1e-12,rtol=1e-12)
            for key in INDEX_KEYS:
                np.testing.assert_allclose(branch['scores'][key],old[score_key][key],atol=1e-12,rtol=1e-12)
            summary=summarize_branch(branch,[])
            for key in ['heatmap_abs','heatmap_delta']:
                np.testing.assert_allclose(summary[key],old[prefix+key],atol=1e-12,rtol=1e-12)
            for key in summary['smooth']:
                np.testing.assert_allclose(summary['smooth'][key],old[prefix+'smooth_trend'][key],atol=1e-12,rtol=1e-12)
            self.assertEqual(summary['bins'][0]['center_us'],int(t[0])+15000000)

    def test_gap_grids_raw_quality_mapping_and_independent_filter_numerics(self):
        t,x=self.gapped()
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda a,*_:a[:,:2]):
            result=self.analyze(t,x,model_path='unused')
        self.assertEqual(result['errors'],[])
        for name in ['bp','tflite']:
            branch=result[name]
            grid=branch['grid']
            for i,group in enumerate(grid.columns['segment_id']):
                sl=continuous_slices(t,500)[group]
                a=branch['quality_mapping']['quality_raw_start_idx'][i]
                b=branch['quality_mapping']['quality_raw_end_idx'][i]
                self.assertTrue(sl.start<=a<b<=sl.stop)
            second=np.flatnonzero(grid.columns['segment_id']==1)[0]
            self.assertEqual(grid.columns['window_start_us'][second],t[17503])
        filtered=bandpass_filter(x[17503:])
        expected=compute_qeeg_indices(filtered[:2500,0].astype(float))
        actual=result['bp']['scores']
        for k in INDEX_KEYS:self.assertAlmostEqual(actual[k][7,0],expected[k],places=12)
        bins=summarize_branch(result['bp'],[])['bins']
        self.assertEqual([(b['start_us']-t[0])/1e6 for b in bins],[0,50])
        self.assertEqual([(b['end_us']-t[0])/1e6 for b in bins],[30,80])

    def test_jitter_does_not_give_bp_quality_an_extra_sample(self):
        t=np.arange(7500)*2000
        t[2500]-=30
        x=np.random.default_rng(1).normal(size=(len(t),4))
        counts=[]
        def scorer(data,**kwargs):
            counts.append(data.shape[1]);return good_quality(data)
        result=self.analyze(t,x,scorer=scorer)
        self.assertEqual(counts,[2500,2500,2500])
        np.testing.assert_array_equal(result['bp']['quality_mapping']['quality_raw_end_idx'],[2500,5000,7500])

    def test_invalid_and_failed_quality_remain_missing_through_smoothing(self):
        t=np.arange(10000)*2000
        x=np.random.default_rng(1).normal(size=(len(t),4))
        q=[{'overall':np.ones(4)}, {'overall':np.full(4,np.nan)}, RuntimeError('scorer failed'), {'overall':np.ones(4)}]
        with patch('lilia.event_qeeg.compute_qeeg_indices',return_value=dict.fromkeys(INDEX_KEYS,.4)):
            branch=self.analyze(t,x,scorer=unittest.mock.Mock(side_effect=q))['bp']
        self.assertEqual(branch['valid'].tolist(),[True,False,False,True])
        self.assertEqual([r['quality_status'] for r in branch['window_audit']],['accepted','invalid_quality','quality_error','accepted'])
        summary=summarize_branch(branch,[])
        np.testing.assert_allclose(summary['smooth']['focus'],[.4,np.nan,np.nan,.4],equal_nan=True)

    def artificial_branch(self,n=24):
        grid=build_window_grid(np.arange(n*2500)*2000,500,5,reset_per_segment=True)
        return {'grid':grid,'valid':np.ones(n,dtype=bool),
                'scores':{k:np.repeat(np.arange(n,dtype=float)[:,None],2,axis=1) for k in INDEX_KEYS}}

    def event(self,label,start,end):
        return {'label':label,'start_us':int(start*1e6),'end_us':int(end*1e6),'color':'red','participates':True}

    def test_pre_event_no_rest_and_missing_baseline_never_fall_back_or_zero_fill(self):
        branch=self.artificial_branch()
        summary=summarize_branch(branch,[self.event('one',30,60),self.event('two',60,90)],'pre-event-rest')
        a=summary['baseline_audit'][1]
        self.assertEqual(a['baseline_metric_rows'],[])
        self.assertEqual(a['heatmap_status'],'missing_baseline')
        self.assertTrue(np.isnan(summary['block_deltas'][1][2]['focus']).all())
        self.assertTrue(np.isnan(summary['heatmap_delta'][:,2]).all())
        branch['valid'][:6]=False
        summary=summarize_branch(branch,[self.event('one',30,60)],'session-start')
        self.assertTrue(np.isnan(summary['heatmap_delta']).all())

    def test_session_baseline_is_shared_and_event_windows_are_fully_contained(self):
        summary=summarize_branch(self.artificial_branch(),[self.event('one',31,61),self.event('two',90,120)])
        first,second=summary['baseline_audit']
        self.assertEqual(first['baseline_metric_rows'],list(range(6)))
        self.assertEqual(first['event_metric_rows'],list(range(7,12)))
        self.assertEqual(second['baseline_metric_rows'],first['baseline_metric_rows'])
        self.assertEqual(first['event_bins'],[])

    def test_plot_exports_verifiable_tables_and_does_not_bridge_gaps(self):
        t,x=self.gapped();source=self.source(t,x);captured=[]
        def capture(fig,*args,**kwargs):
            captured.append(fig.axes[0].lines[0].get_ydata().copy())
        with patch.object(plot,'get_eeg_quality_index_v2_parametric',good_quality),patch.object(Figure,'savefig',capture):
            result=self.run_plot(source)
        self.assertEqual(result['branches']['bp']['total_windows'],16)
        self.assertEqual(np.isnan(captured[0]).sum(),1)
        table=next((self.root/'out').glob('*_bp_metrics.csv'))
        frame,meta=load_event_qeeg_table(table,source)
        self.assertEqual(frame.segment_id.tolist(),[0]*7+[1]*9)
        self.assertEqual(len(meta['analysis']['summary']['bins']),2)
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda a,*_:a[:,:2]),patch.object(plot,'get_eeg_quality_index_v2_parametric',good_quality),patch.object(Figure,'savefig'):
            self.run_plot(source,use_tflite=True)
        table=next((self.root/'out').glob('*_tflite_metrics.csv'))
        load_event_qeeg_table(table,source,plot.TFLITE_MODEL_PATH)

    def test_metadata_rejects_rehashed_quality_mapping_and_inference(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch.object(plot,'get_eeg_quality_index_v2_parametric',good_quality),patch.object(Figure,'savefig'):
            self.run_plot(source)
        table=next((self.root/'out').glob('*_bp_metrics.csv'));sidecar=Path(str(table)+'.meta.json')
        frame=pd.read_csv(table);frame.loc[0,'quality_raw_end_idx']+=1;frame.to_csv(table,index=False)
        meta=json.loads(sidecar.read_text());meta['table_sha256']=file_sha256(table);sidecar.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'quality_raw_end_idx mapping'):
            load_event_qeeg_table(table,source)
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda a,*_:a[:,:2]),patch.object(plot,'get_eeg_quality_index_v2_parametric',good_quality),patch.object(Figure,'savefig'):
            self.run_plot(source,use_tflite=True)
        model_table=next((self.root/'out').glob('*_tflite_metrics.csv'))
        model_sidecar=Path(str(model_table)+'.meta.json')
        model_original=model_sidecar.read_text()
        model_meta=json.loads(model_original)
        model_meta['inference']['segments'][0]['raw_end_idx']+=1
        model_sidecar.write_text(json.dumps(model_meta))
        with self.assertRaisesRegex(ValueError,'inference mapping'):
            load_event_qeeg_table(model_table,source)
        # Even rehashed analysis cannot substitute a different baseline selection.
        model_meta=json.loads(model_original)
        model_meta['analysis']['summary']['baseline_audit'][0]['baseline_bins']=[]
        model_meta['analysis_id']=config_id(model_meta['analysis'])
        model_sidecar.write_text(json.dumps(model_meta))
        with self.assertRaisesRegex(ValueError,'baseline mapping'):
            load_event_qeeg_table(model_table,source)
        source.write_text(source.read_text().replace('File Name,test','File Name,changed'))
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):
            load_event_qeeg_table(table,source)

    def test_all_excluded_and_no_complete_windows_save_audit(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch.object(plot,'get_eeg_quality_index_v2_parametric',return_value={'overall':np.full(4,np.nan)}),self.assertRaisesRegex(ValueError,'audit saved'):
            self.run_plot(source)
        table=next((self.root/'out').glob('*_bp_metrics.csv'))
        frame,_=load_event_qeeg_table(table,source)
        self.assertFalse(frame.quality_valid.any())
        source=self.source(np.arange(10)*2000,np.ones((10,4)))
        with self.assertRaisesRegex(ValueError,'audit saved'):self.run_plot(source)
        audit=json.loads(next((self.root/'out').glob('*analysis.json')).read_text())
        self.assertEqual(audit['segments'][0]['status'],'short_segment')
        self.assertEqual(audit['branches'],{})

    def test_requested_model_failure_is_reported_after_bp_audit(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch.object(plot,'TFLITE_MODEL_PATH',str(self.root/'missing.tflite')),patch.object(Figure,'savefig'),patch.object(plot,'get_eeg_quality_index_v2_parametric',good_quality),self.assertRaisesRegex(ValueError,'TFLite failed'):
            self.run_plot(source,use_tflite=True)
        audit=json.loads(next((self.root/'out').glob('*analysis.json')).read_text())
        self.assertIn('bp',audit['branches']);self.assertTrue(audit['errors'])

    def test_batch_continues_after_failure_and_exits_nonzero(self):
        args=SimpleNamespace(hardy2_analysis=False,ibrain_outdir=str(self.root/'out'),yoga_outdir=str(self.root/'yoga'),ds=500,no_tflite=True,ibrain_baseline_mode='session-start')
        with patch.object(plot,'parse_args',return_value=args),patch.object(plot,'SUBJECTS',{'bad':{},'good':{}}),patch.object(plot,'YOGA_SUBJECTS',{}),patch.object(plot,'plot_subject',side_effect=[ValueError('bad recording'),{}]) as run,contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):plot.main()
        self.assertEqual(run.call_count,2)
        self.assertTrue((self.root/'out/event_markers_failures.json').exists())

    def test_heatmap_rectangles_use_actual_bin_edges(self):
        import matplotlib.pyplot as plt
        t,x=self.gapped();branch=self.analyze(t,x)['bp'];summary=summarize_branch(branch,[])
        fig,ax=plt.subplots()
        try:
            plot._draw_segment_heatmap(ax,summary['bins'],np.ma.masked_invalid(summary['heatmap_delta']),plt.get_cmap('coolwarm'),.6)
            self.assertEqual(len(ax.collections),2)
            edges=[collection.get_coordinates()[0,:,0] for collection in ax.collections]
            self.assertAlmostEqual((edges[1][0]-edges[0][1])*86400,20,places=5)
        finally:plt.close(fig)

    def test_small_gap_ids_and_nonfinite_segments_are_retained_in_audit(self):
        t=np.r_[np.arange(3001)*2000,6008000+np.arange(3001)*2000]
        x=np.random.default_rng(6).normal(size=(len(t),4)).astype(np.float32)
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda a,*_:a[:,:2]):
            result=self.analyze(t,x,model_path='unused')
        self.assertEqual(result['tflite']['grid'].columns['segment_id'].tolist(),[0,1])
        x[1,0]=np.nan
        result=self.analyze(t,x)
        self.assertEqual(result['segments'][0]['status'],'nonfinite_segment')
        self.assertEqual(result['bp']['valid'].tolist(),[False,True])


if __name__=='__main__':unittest.main()
