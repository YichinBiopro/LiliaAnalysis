"""Fixed-baseline meditation: crop/global mappings and unscored BP/model outputs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_tyy_meditation as plot
from lilia.entropy_io import config_id
from lilia.meditation import analyze_meditation, crop_layout, model_timeline
from lilia.meditation_io import load_meditation_table
from lilia.event_qeeg import INDEX_KEYS
from lilia.io import bandpass_filter
from lilia.provenance import file_sha256
from lilia.qeeg import compute_qeeg_indices
from lilia.tflite import run_tflite_recording


class MeditationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
        self.epoch=plot.local_dt_to_utc_us(plot.hhmm_to_dt('14:10'),8)

    def params(self,**kwargs):
        p={'crop_start_us':32,'crop_end_us':185,'baseline_end_us':75,'view_start_us':60,'view_end_us':170,
           'meditation_start_us':90,'meditation_end_us':160}
        p.update(kwargs)
        return {k:self.epoch+int(v*1e6) for k,v in p.items()}

    def gapped(self):
        t=self.epoch+np.r_[np.arange(5000)*2000,20000000+np.arange(100)*2000,
            30000000+np.arange(35000)*2000,100008000+np.arange(45000)*2000]
        x=np.random.default_rng(1314).normal(size=(len(t),4)).astype(np.float32)
        return t,x

    def analyze(self,t,x,p=None,model=True):
        p=p or self.params()
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda data,*a:data[:,:2]):
            return analyze_meditation(t,x,**{k:v for k,v in p.items() if k not in ('meditation_start_us','meditation_end_us')},
                model_path=plot.TFLITE_PATH if model else None)

    def source(self,t,x):
        path=self.root/'source.csv';f=pd.DataFrame(x,columns=[f'ch{i+1}' for i in range(x.shape[1])]);f.insert(0,'Time[us]',t)
        path.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n'+f.to_csv(index=False))
        return path

    def run_plot(self,source,p=None,model=True,save=None):
        def placeholder(fig,path,**kw):Path(path).write_text('plot')
        with patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda data,*a:data[:,:2]), \
             patch.object(Figure,'savefig',save or placeholder),contextlib.redirect_stdout(io.StringIO()):
            return plot.plot_tyy_meditation(self.root/'out',ds=17,use_tflite=model,csv_path=source,parameters=p or self.params())

    def audit(self):return json.loads(next((self.root/'out').glob('*_analysis.json')).read_text())

    def test_continuous_actual_four_channel_bp_and_model_match_old_main(self):
        root=Path(__file__).parent/'fixtures';ref=json.loads((root/'tyy_meditation_continuous_reference.json').read_text())
        self.assertEqual(file_sha256(plot.TFLITE_PATH),ref['model_sha256'])
        x=np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32);t=ref['epoch_us']+np.arange(len(x))*2000
        p=self.params(crop_start_us=0,crop_end_us=121,baseline_end_us=60,view_start_us=0,view_end_us=120)
        r=analyze_meditation(t,x,**{k:v for k,v in p.items() if k not in ('meditation_start_us','meditation_end_us')},model_path=plot.TFLITE_PATH)
        self.assertEqual(r['errors'],[])
        for tag,nch in [('bp',4),('tflite',2)]:
            b=r[tag];old=ref['continuous'][tag]
            self.assertEqual(b['scores']['focus'].shape[1],nch)
            for k in INDEX_KEYS:np.testing.assert_allclose(b['scores'][k],old['scores'][k],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(b['grid'].columns['window_center_us'],old['center_us'])
            np.testing.assert_allclose(b['summary']['heatmap_delta'],old['heatmap_delta'],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(np.asarray(old['bin_center_us'])-[s['center_us'] for s in b['summary']['bins']],[2500000]*4)
        _,_,out=run_tflite_recording(t,bandpass_filter(x),plot.TFLITE_PATH)
        np.testing.assert_array_equal(out,np.load(root/'tyy_meditation_continuous_reference.npz')['model_output'])

    def test_crop_maps_global_raw_indices_segment_ids_and_epoch(self):
        t,x=self.gapped();r=self.analyze(t,x)
        self.assertEqual(r['errors'],[])
        self.assertEqual(r['crop']['raw_start_idx'],6100)
        for tag in ('bp','tflite'):
            b=r[tag];g=b['grid']
            self.assertEqual(sorted(set(g.columns['segment_id'])),[2,3])
            self.assertEqual(g.time_s[0],34.5)
            self.assertEqual(b['summary']['baseline_accepted_bins'],[0])
        self.assertEqual(r['bp']['grid'].starts[0],6100)
        self.assertEqual(r['timeline'].segments[0]['raw_start_idx'],6100)
        self.assertEqual(r['timeline'].source_samples,len(t))
        self.assertEqual(r['timeline'].source_epoch_us,self.epoch)
        self.assertEqual(r['timeline'].model_windows[0]['segment_id'],2)
        for row in r['timeline'].model_windows:
            self.assertGreaterEqual(row['raw_start_idx'],6100)
            self.assertLessEqual(row['raw_end_idx'],r['crop']['raw_end_idx'])

    def test_half_open_crop_excludes_endpoint_and_caps_model_support(self):
        t,x=self.gapped();p=self.params(crop_start_us=32.001,crop_end_us=42.001,baseline_end_us=50,view_start_us=32,view_end_us=42)
        crop=crop_layout(t,500,p['crop_start_us'],p['crop_end_us'])
        self.assertEqual(t[crop['raw_start_idx']],self.epoch+32002000)
        self.assertEqual(t[crop['raw_end_idx']],self.epoch+42002000)
        self.assertEqual(crop['segments'][0]['raw_end_us'],p['crop_end_us'])
        timeline=model_timeline(t,crop)
        self.assertLessEqual(timeline.grid(5).columns['window_end_us'][-1],p['crop_end_us'])
        p=self.params(crop_start_us=32,crop_end_us=42)
        crop=crop_layout(t,500,p['crop_start_us'],p['crop_end_us'])
        self.assertEqual(t[crop['raw_end_idx']],p['crop_end_us'])

    def test_filter_is_local_to_crop_and_segment(self):
        t,x=self.gapped();r=self.analyze(t,x,model=False)
        first=r['crop']['segments'][0]
        filtered=bandpass_filter(x[first['raw_start_idx']:first['raw_end_idx']])
        expected=compute_qeeg_indices(filtered[:2500,0].astype(float))
        for k in INDEX_KEYS:self.assertAlmostEqual(r['bp']['scores'][k][0,0],expected[k],places=12)
        changed=x.copy();changed[:6100]*=1000;changed[40100:]*=-100
        other=self.analyze(t,changed,model=False)
        for k in INDEX_KEYS:np.testing.assert_array_equal(r['bp']['scores'][k][:13],other['bp']['scores'][k][:13])

    def test_no_baseline_never_uses_zero_and_audits_unbinned_tails(self):
        t,x=self.gapped();r=self.analyze(t,x,self.params(baseline_end_us=40))
        for tag in ('bp','tflite'):
            s=r[tag]['summary'];self.assertEqual(s['status'],'missing_baseline')
            self.assertTrue(np.isnan(s['heatmap_delta']).all())
            self.assertTrue(any(row['unbinned_metric_rows'] for row in s['tails']))
        self.assertEqual(len(r['errors']),2)

    def test_polluted_crop_segment_excludes_bp_and_fails_model_explicitly(self):
        t,x=self.gapped();x[45000,0]=np.nan;r=self.analyze(t,x)
        self.assertIsNone(r['tflite'])
        self.assertTrue(any('Non-finite' in e for e in r['errors']))
        b=r['bp'];self.assertTrue(b['valid'][:13].all());self.assertFalse(b['valid'][13:].any())
        self.assertEqual(r['segments'][1]['status'],'nonfinite_segment')
        self.assertTrue(all(row['quality_state']=='disabled' for row in b['window_audit']))

    def test_outputs_reconstruct_crop_model_summary_and_display(self):
        t,x=self.gapped();source=self.source(t,x);a=self.run_plot(source)
        self.assertEqual(a['status'],'complete')
        for tag in ('bp','tflite'):
            table=next((self.root/'out').glob(f'*_{tag}_metrics.csv'))
            f,m=load_meditation_table(table,source,plot.TFLITE_PATH)
            self.assertTrue((f.quality_state=='disabled').all())
            self.assertEqual(m['parameters']['metric_channels'],4 if tag=='bp' else 2)
        for name,h in a['artifacts'].items():self.assertEqual(file_sha256(self.root/'out'/name),h)
        for run in a['display']['display_runs']:
            self.assertEqual(run['display_raw_indices'][0],run['raw_start_idx'])
            self.assertEqual(run['display_raw_indices'][-1],run['raw_end_idx']-1)

    def test_table_rejects_rehashed_crop_inference_summary_and_wrong_model(self):
        t,x=self.gapped();source=self.source(t,x);self.run_plot(source)
        p=next((self.root/'out').glob('*_tflite_metrics.csv'));side=Path(str(p)+'.meta.json');saved=side.read_text()
        for field,pattern in [('crop','crop mapping'),('summary','heatmap')]:
            m=json.loads(saved)
            if field=='crop':m['analysis']['crop']['raw_start_idx']+=1
            else:m['analysis']['summary']['baseline_reference'][0]+=1
            m['analysis_id']=config_id(m['analysis']);side.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,pattern):load_meditation_table(p,source)
        m=json.loads(saved);m['inference']['model_windows'][0]['raw_start_idx']-=1;side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'inference mapping'):load_meditation_table(p,source)
        side.write_text(saved);wrong=self.root/'wrong.tflite';wrong.write_text('wrong')
        with self.assertRaisesRegex(ValueError,'model differs'):load_meditation_table(p,source,wrong)
        m=json.loads(saved);m['analysis']['display']['display_runs'][0]['display_raw_indices'][0]+=1
        m['analysis_id']=config_id(m['analysis']);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'display crop'):load_meditation_table(p,source)
        side.write_text(saved)
        source.write_text(source.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):load_meditation_table(p,source)

    def test_requested_missing_model_keeps_bp_tables_and_failure_plot(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch.object(plot,'TFLITE_PATH',str(self.root/'missing.tflite')):
            with self.assertRaisesRegex(ValueError,'Requested model missing'):self.run_plot(source)
        a=self.audit();self.assertEqual(a['status'],'failed');self.assertEqual(len(a['artifacts']),4)
        self.assertEqual(a['branches']['bp']['status'],'computed')
        self.assertEqual(a['branches']['tflite']['status'],'unavailable')

    def test_disabled_model_is_successful_and_not_quality_accepted(self):
        t,x=self.gapped();source=self.source(t,x);a=self.run_plot(source,model=False)
        self.assertEqual(a['status'],'complete');self.assertEqual(a['branches']['tflite']['status'],'disabled')
        self.assertEqual(a['quality_state'],'disabled')
        self.assertNotIn('quality_threshold',a['branches']['bp']['parameters'])

    def test_model_runtime_failure_retains_bp_and_requested_checkpoint_mapping(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch('lilia.meditation.run_tflite_recording',side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(ValueError,'inference failed'):self.run_plot(source)
        a=self.audit()
        self.assertEqual(a['model']['sha256'],file_sha256(plot.TFLITE_PATH))
        self.assertEqual(a['branches']['bp']['status'],'computed')
        self.assertEqual(a['branches']['tflite']['status'],'unavailable')
        self.assertEqual(a['inference']['segments'][0]['raw_start_idx'],6100)
        load_meditation_table(next((self.root/'out').glob('*_bp_metrics.csv')),source)

    def test_allshort_or_empty_crop_preserves_failure_audit(self):
        t,x=self.gapped();source=self.source(t,x)
        for p in (self.params(crop_start_us=20,crop_end_us=21,view_start_us=20,view_end_us=21),
                  self.params(crop_start_us=200,crop_end_us=210,view_start_us=200,view_end_us=210)):
            with self.assertRaisesRegex(ValueError,'Meditation analysis failed'):self.run_plot(source,p)
            self.assertEqual(self.audit()['status'],'failed')
            self.assertEqual(len(self.audit()['artifacts']),2)

    def test_raw_heatmap_and_colorbar_keep_time_positions_aligned(self):
        t,x=self.gapped();source=self.source(t,x);seen=[]
        def inspect(fig,path,**kw):
            self.assertEqual(len(fig.axes[0].lines),3) # two source runs plus onset
            x0,x1=fig.axes[0].get_position().x0,fig.axes[0].get_position().x1
            for ax in fig.axes[1:4]:np.testing.assert_allclose([ax.get_position().x0,ax.get_position().x1],[x0,x1])
            self.assertEqual(len(fig.axes[2].collections),4)
            labels=[label.get_text() for label in fig.axes[3].get_xticklabels()]
            self.assertEqual(len(labels),len(set(labels)))
            seen.append(True);Path(path).write_text('plot')
        self.run_plot(source,save=inspect);self.assertEqual(len(seen),2);self.assertEqual(plt.get_fignums(),[])


if __name__=='__main__':unittest.main()
