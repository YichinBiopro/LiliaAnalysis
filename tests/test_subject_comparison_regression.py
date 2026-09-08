"""Inter-subject deltas: separate baselines, quality clocks and honest outputs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import compare_subjects as compare
from lilia.entropy_io import config_id
from lilia.event_qeeg import analyze_recording, INDEX_KEYS
from lilia.provenance import file_sha256
from lilia.subject_comparison import summarize_comparison, legacy_results
from lilia.subject_comparison_io import load_comparison_table
from lilia.windowing import build_window_grid


def good(data,**kwargs):
    return {'overall':np.ones(data.shape[0])}


class SubjectComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.epoch=1778565600000000

    def source(self,t,x,name='Subject'):
        folder=self.root/name;folder.mkdir(exist_ok=True)
        f=pd.DataFrame(x,columns=[f'ch{i+1}' for i in range(x.shape[1])]);f.insert(0,'Time[us]',t)
        path=folder/'merged.csv'
        path.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],0\nChannels,1,2,3,4\nSample Rate,500,500,500,500\n'+f.to_csv(index=False))
        return path

    def gapped(self):
        t=self.epoch+np.r_[np.arange(100)*2000,10000000+np.arange(5001)*2000,20008000+np.arange(10001)*2000]
        x=np.random.default_rng(1112).normal(size=(len(t),4)).astype(np.float32)
        return t,x

    def process(self,t=None,x=None,scorer=good,model=True):
        if t is None:t,x=self.gapped()
        source=self.source(t,x)
        with patch.object(compare,'get_eeg_quality_index_v2_parametric',scorer), \
             patch('lilia.tflite.apply_tflite_windowed',side_effect=lambda a,*_:a[:,:2]):
            result=compare.process_subject('Subject',{'dir':'Subject'},str(self.root),None,self.root/'out',model)
        return source,result

    def test_real_continuous_quality_bp_tflite_and_both_deltas_match_reference(self):
        ref=json.loads((Path(__file__).parent/'fixtures/subject_comparison_continuous_reference.json').read_text())
        self.assertEqual(file_sha256(compare.TFLITE_MODEL_PATH),ref['model_sha256'])
        x=np.random.default_rng(ref['seed']).normal(size=ref['shape']).astype(np.float32)
        t=ref['epoch_us']+np.arange(len(x))*2000
        r=analyze_recording(t,x,scorer=compare.get_eeg_quality_index_v2_parametric,
            quality_params=compare.QUALITY_PARAMS,model_path=compare.TFLITE_MODEL_PATH)
        self.assertEqual(r['errors'],[])
        events=[{'label':'Test','start_us':int(t[0])+20000000,'end_us':int(t[0])+50000000,'participates':True}]
        for tag in ('bp','tflite'):
            b=r[tag];old=ref['branches'][tag]
            np.testing.assert_allclose(b['quality'],ref['quality'],rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(b['valid'],old['valid'])
            for k in INDEX_KEYS:np.testing.assert_allclose(b['scores'][k],old['scores'][k],rtol=1e-12,atol=1e-12)
            for ev,expected in [(events,old['event_delta']), (None,{'Session':old['session_delta']})]:
                actual=legacy_results(summarize_comparison(b,ev))
                for label in expected:
                    for k in INDEX_KEYS:np.testing.assert_allclose(actual[label][k],expected[label][k],rtol=1e-12,atol=1e-12)

    def test_small_gap_grids_and_quality_raw_indexes_survive_tflite_trim(self):
        source,(results,_,audit)=self.process()
        self.assertEqual(audit['status'],'complete')
        self.assertIn('Session',results['tflite'])
        for tag in ('bp','tflite'):
            f,m=load_comparison_table(self.root/f'out/{tag}_metrics.csv',source,compare.TFLITE_MODEL_PATH)
            self.assertEqual(f.segment_id.tolist(),[1,1,2,2,2,2])
            self.assertEqual(f.quality_raw_start_idx.tolist(),[100,2600,5101,7601,10101,12601])
            self.assertEqual(f.time_s.iloc[0],12.5)
            self.assertEqual(m['analysis']['summary'][0]['baseline_candidate_rows'],[0])
        self.assertEqual(audit['inference']['segments'][0]['status'],'excluded')

    def test_quality_scored_against_each_actual_window_not_truncated_rows(self):
        calls=[]
        def scorer(data,**kwargs):
            calls.append(data.copy());return good(data)
        t,x=self.gapped();_,(_,_,audit)=self.process(t,x,scorer)
        self.assertEqual(len(calls),12)
        for i,start in enumerate([100,2600,5101,7601,10101,12601]):
            np.testing.assert_array_equal(calls[i],x[start:start+2500].T.astype(float))
            np.testing.assert_array_equal(calls[i+6],x[start:start+2500].T.astype(float))
        self.assertEqual(audit['branches']['tflite']['accepted_windows'],6)

    def test_model_trimmed_segment_does_not_shift_later_quality_rows(self):
        # Five seconds supports one BP window, but only four retained model
        # seconds. The next segment's first model row must use its own raw data.
        t=self.epoch+np.r_[np.arange(2500)*2000,10000000+np.arange(15000)*2000]
        x=np.random.default_rng(1113).normal(size=(len(t),4)).astype(np.float32)
        source,(_,_,audit)=self.process(t,x)
        self.assertEqual(audit['status'],'complete')
        bp,_=load_comparison_table(self.root/'out/bp_metrics.csv',source)
        model,_=load_comparison_table(self.root/'out/tflite_metrics.csv',source)
        self.assertEqual(len(bp),7)
        self.assertEqual(len(model),6)
        self.assertEqual(model.quality_raw_start_idx.tolist(),[2500,5000,7500,10000,12500,15000])
        self.assertEqual(model.segment_id.tolist(),[1]*6)
        self.assertEqual(model.time_s.iloc[0],12.5)

    def test_metric_exception_and_nonfinite_output_retain_excluded_windows(self):
        from lilia.qeeg import compute_qeeg_indices
        calls=0
        def broken_once(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==6:raise RuntimeError('metric failed on second channel')
            if calls==8:return dict.fromkeys(INDEX_KEYS,np.inf)
            return compute_qeeg_indices(*args,**kwargs)
        t=self.epoch+np.arange(15000)*2000
        x=np.random.default_rng(1114).normal(size=(len(t),4)).astype(np.float32)
        with patch('lilia.event_qeeg.compute_qeeg_indices',side_effect=broken_once):
            source,(_,_,audit)=self.process(t,x,model=False)
        self.assertEqual(audit['status'],'complete')
        f,m=load_comparison_table(self.root/'out/bp_metrics.csv',source)
        self.assertEqual(f.metric_valid.tolist(),[True,False,False,True,True,True])
        self.assertEqual(f.metric_status.iloc[1],'metric_error')
        self.assertEqual(f.metric_status.iloc[2],'nonfinite_metrics')
        self.assertTrue(f.filter(regex='_ch[1-4]$').drop(columns=[f'quality_ch{i}' for i in range(1,5)]).iloc[1].isna().all())
        self.assertEqual(m['analysis']['summary'][0]['comparison_excluded_rows'],[1,2])

    def test_quality_nan_shape_error_or_low_score_never_becomes_accepted(self):
        for scorer in (lambda *_a,**_k:{'overall':np.full(4,np.nan)},
                       lambda *_a,**_k:{'overall':np.ones(3)},lambda *_a,**_k:{'overall':np.zeros(4)}):
            source,(_,_,audit)=self.process(scorer=scorer)
            self.assertEqual(audit['status'],'failed')
            for tag in ('bp','tflite'):
                f,_=load_comparison_table(self.root/f'out/{tag}_metrics.csv',source)
                self.assertFalse(f.metric_valid.any())
                self.assertEqual(audit['branches'][tag]['summary'][0]['status'],'no_accepted_baseline')
        def broken(*args,**kwargs):raise RuntimeError('scorer failed')
        _,(_,_,audit)=self.process(scorer=broken)
        self.assertEqual(audit['branches']['bp']['window_audit'][0]['quality_status'],'quality_error')

    def branch(self,n=10,epoch=0):
        grid=build_window_grid(epoch+np.arange(n*2500)*2000,500,5)
        scores={k:np.column_stack([np.arange(n),np.arange(n)*2.]) for k in INDEX_KEYS}
        return {'grid':grid,'scores':scores,'valid':np.ones(n,dtype=bool)}

    def test_session_baseline_counts_candidates_before_quality_no_fallback(self):
        b=self.branch();b['valid'][0]=False
        row=summarize_comparison(b)[0]
        self.assertEqual(row['baseline_candidate_rows'],[0,1])
        self.assertEqual(row['baseline_accepted_rows'],[1])
        self.assertEqual(row['comparison_candidate_rows'],list(range(2,10)))
        self.assertEqual(row['metrics']['focus']['channel_delta'],[4.5,9.])
        self.assertEqual(row['metrics']['focus']['mean'],6.75)
        self.assertEqual(row['metrics']['focus']['channel_sd'],2.25)
        b['valid'][1]=False
        row=summarize_comparison(b)[0]
        self.assertEqual(row['status'],'no_accepted_baseline')
        self.assertIsNone(row['metrics']['focus']['channel_sd'])

    def test_complete_event_windows_and_nonparticipation_preserve_baseline_anchor(self):
        b=self.branch()
        events=[{'label':'Absent','start_us':4000000,'end_us':8000000,'participates':False},
                {'label':'First','start_us':12000000,'end_us':28000000,'participates':True},
                {'label':'Second','start_us':35000000,'end_us':45000000,'participates':True}]
        rows=summarize_comparison(b,events)
        self.assertEqual(rows[0]['status'],'not_participating')
        self.assertEqual(rows[1]['baseline_candidate_rows'],[0,1])
        self.assertEqual(rows[1]['comparison_candidate_rows'],[3,4])
        self.assertEqual(rows[2]['baseline_candidate_rows'],[0,1])
        self.assertEqual(rows[2]['comparison_candidate_rows'],[7,8])

    def test_no_baseline_or_event_rows_and_allshort_audit_are_explicit(self):
        b=self.branch()
        events=[{'label':'First','start_us':0,'end_us':5000000,'participates':True}]
        self.assertEqual(summarize_comparison(b,events)[0]['status'],'no_baseline_windows')
        events[0].update(start_us=60000000,end_us=70000000)
        self.assertEqual(summarize_comparison(b,events)[0]['status'],'no_comparison_windows')
        t=self.epoch+np.arange(100)*2000
        _,(_,_,audit)=self.process(t,np.ones((100,4)))
        self.assertEqual(audit['status'],'failed')
        self.assertEqual(audit['segments'][0]['status'],'short_segment')
        self.assertEqual(audit['artifacts'],{})

    def test_model_failure_or_missing_model_preserves_bp_and_reports_failure(self):
        t,x=self.gapped();source=self.source(t,x)
        with patch.object(compare,'get_eeg_quality_index_v2_parametric',good), \
             patch('lilia.tflite.apply_tflite_windowed',side_effect=RuntimeError('model failed')):
            results,_,audit=compare.process_subject('Subject',{'dir':'Subject'},str(self.root),None,self.root/'out')
        self.assertEqual(audit['status'],'failed')
        self.assertIn('Session',results['bp']);self.assertEqual(results['tflite'],{})
        load_comparison_table(self.root/'out/bp_metrics.csv',source)
        self.assertTrue(any('model failed' in e for e in audit['errors']))
        self.assertIn('inference',audit)
        with patch.object(compare,'TFLITE_MODEL_PATH',str(self.root/'missing.tflite')), \
             patch.object(compare,'get_eeg_quality_index_v2_parametric',good):
            results,_,audit=compare.process_subject('Subject',{'dir':'Subject'},str(self.root),None,self.root/'missing')
        self.assertIn('Session',results['bp'])
        self.assertEqual(audit['status'],'failed')

    def test_disabled_model_is_distinct_from_failure(self):
        _,(results,status,audit)=self.process(model=False)
        self.assertEqual(audit['status'],'complete')
        self.assertEqual(status['tflite']['Session'],'disabled')
        self.assertEqual(results['tflite'],{})

    def test_tables_reject_source_model_and_rehashed_quality_mapping_changes(self):
        source,_=self.process();p=self.root/'out/tflite_metrics.csv';side=Path(str(p)+'.meta.json')
        wrong=self.root/'wrong.tflite';wrong.write_text('wrong')
        with self.assertRaisesRegex(ValueError,'model differs'):load_comparison_table(p,source,wrong)
        f=pd.read_csv(p);f.loc[0,'quality_raw_start_idx']+=1;f.to_csv(p,index=False)
        m=json.loads(side.read_text());m['table_sha256']=file_sha256(p);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'mapping differs'):load_comparison_table(p,source)
        source.write_text(source.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):load_comparison_table(p,source)

    def test_rehashed_summary_and_inference_tamper_are_rejected(self):
        source,_=self.process();p=self.root/'out/tflite_metrics.csv';side=Path(str(p)+'.meta.json');saved=side.read_text()
        m=json.loads(saved);m['inference']['segments'][1]['raw_start_idx']+=1;side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'inference mapping'):load_comparison_table(p,source)
        m=json.loads(saved);m['analysis']['summary'][0]['baseline_accepted_rows']=[1]
        m['analysis_id']=config_id(m['analysis']);side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'baseline or delta summary'):load_comparison_table(p,source)

    def test_missing_zero_and_nonparticipating_bars_are_visually_distinct(self):
        fig,ax=plt.subplots()
        data={'Missing':{},'Zero':{'Event':{'focus':(0.,0.)}},'Absent':{'Event':{'focus':(1.,0.)}}}
        compare._bar_cell(ax,['Missing','Zero','Absent'],'Event','focus',data,{},
                          {'Missing':{'Event'},'Zero':{'Event'},'Absent':set()})
        self.assertEqual(len(ax.patches),1)
        self.assertEqual(ax.patches[0].get_height(),0.)
        self.assertEqual([t.get_text() for t in ax.texts],['NA','NP'])
        self.assertTrue(all(ax.get_xlim()[0]<t.get_position()[0]<ax.get_xlim()[1] for t in ax.texts))
        plt.close(fig)

    def test_all_missing_combined_annotations_stay_inside_subject_axis(self):
        inspected=[]
        def inspect(fig,*args,**kwargs):
            for ax in fig.axes:
                self.assertEqual(len(ax.patches),0)
                self.assertEqual([t.get_text() for t in ax.texts],['B:NA','T:NA'])
                self.assertTrue(all(ax.get_xlim()[0]<t.get_position()[0]<ax.get_xlim()[1] for t in ax.texts))
            inspected.append(True)
        with patch.object(Figure,'savefig',inspect),contextlib.redirect_stdout(io.StringIO()):
            compare.plot_combined_comparison('Test',['Missing'],['Event'],{}, {}, {},None,str(self.root/'empty.png'))
        self.assertEqual(len(inspected),2)

    def test_cli_continues_groups_after_missing_subject_and_returns_nonzero(self):
        t,x=self.gapped();self.source(t,x,'Good')
        args=SimpleNamespace(ibrain_outdir=str(self.root/'i'),yoga_outdir=str(self.root/'y'),no_tflite=True)
        def save(fig,p,**kw):Path(p).write_text('plot placeholder')
        with patch.object(compare,'parse_args',return_value=args), \
             patch.object(compare,'SUBJECTS',{'Missing':{'dir':'Missing'}}), \
             patch.object(compare,'YOGA_SUBJECTS',{'Good':{'dir':'Good'}}), \
             patch.object(compare,'IBRAIN_DIR',str(self.root)),patch.object(compare,'YOGA_DIR',str(self.root)), \
             patch.object(compare,'get_eeg_quality_index_v2_parametric',good),patch.object(Figure,'savefig',save), \
             contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'Comparison failures'):
            compare.main()
        i=json.loads((self.root/'i/comparison_analysis.json').read_text())
        y=json.loads((self.root/'y/comparison_analysis.json').read_text())
        self.assertEqual(i['status'],'failed');self.assertEqual(y['status'],'complete')
        self.assertEqual(len(y['artifacts']),6)
        self.assertEqual(y['subjects']['Good']['status'],'complete')


if __name__=='__main__':unittest.main()
