"""Quality caller audit propagation, source verification and legacy selection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lilia import quality
from lilia.entropy_io import config_id
from lilia.event_qeeg import analyze_recording, summarize_branch
from lilia.event_qeeg_io import write_event_qeeg_table, load_event_qeeg_table
from lilia.provenance import file_sha256
from lilia.quality_audit import plot_diagnostic_markers, diagnostic_label, COLUMNS
from lilia.subject_comparison import summarize_comparison
from lilia.subject_comparison_io import write_comparison_table, load_comparison_table


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.t = 1700000000000001 + np.arange(6003, dtype=np.int64)*2000
        self.x = np.random.default_rng(18019).normal(size=(len(self.t), 4)).astype(np.float32)
        self.source = self.root/'raw.csv'
        with self.source.open('w') as handle:
            handle.write('header\n'*4)
            pd.DataFrame({'Time[us]': self.t, **{f'ch{i+1}': self.x[:, i] for i in range(4)}}).to_csv(handle, index=False)
        self.params = quality.get_best_eeg_quality_v2_flat_spectrum_only_params()

    def analyze(self, scorer=None, **kwargs):
        return analyze_recording(self.t, self.x, scorer=scorer or quality.get_eeg_quality_index_v2_parametric,
                                 quality_params=self.params, **kwargs)

    def write(self, result, branch_name='bp', comparison=False):
        branch = result[branch_name]
        p = dict(fs=branch['grid'].fs, input_fs=500., win_sec=branch['grid'].win/branch['grid'].fs,
                 step_sec=branch['grid'].win/branch['grid'].fs,
                 index_space='raw_samples' if branch_name=='bp' else 'retained_tflite_output',
                 quality_params=self.params, quality_channels=4, metric_channels=4 if branch_name=='bp' else 2,
                 baseline_mode='session-start', quality_threshold=.5, model_window=400,
                 model_sha256=None, events=None if comparison else [])
        path = self.root / f'{branch_name}_{comparison}.csv'
        timeline = result['timeline'] if branch_name=='tflite' else None
        if comparison:
            branch['summary'] = summarize_comparison(branch)
            write_comparison_table(path, self.source, branch, p, 'test', result['segments'], timeline)
        else:
            branch['summary'] = summarize_branch(branch, [])
            write_event_qeeg_table(path, self.source, branch, p, 'test', timeline)
        return path

    def rehash(self, path, meta):
        meta['analysis_id'] = config_id(meta['analysis'])
        meta['table_sha256'] = file_sha256(path)
        Path(str(path)+'.meta.json').write_text(json.dumps(meta))

    def test_real_scorer_raw_context_and_both_branch_mappings(self):
        self.t[3003:] += 500000
        # Rewrite the source so the two small gap-local grids can be checked.
        with self.source.open('w') as handle:
            handle.write('header\n'*4)
            pd.DataFrame({'Time[us]':self.t, **{f'ch{i+1}':self.x[:,i] for i in range(4)}}).to_csv(handle,index=False)
        with patch('lilia.tflite.apply_tflite_windowed', side_effect=lambda x,*_:x[:,:2]):
            result=self.analyze(model_path='unused', win_sec=2.)
        for name in ('bp','tflite'):
            for comparison in (False, True):
                path=self.write(result,name,comparison)
                frame, meta=(load_comparison_table if comparison else load_event_qeeg_table)(path,self.source)
                self.assertEqual(set(frame.quality_stage), {'raw'})
                self.assertEqual(set(frame.quality_score_policy), {'legacy_overall'})
                for i, row in enumerate(meta['analysis']['window_audit']):
                    d=row['quality_diagnostics']
                    self.assertEqual(d['result']['context']['preset'],'flat_spectrum_only')
                    self.assertEqual(d['request']['n_samples'],int(frame.quality_raw_end_idx.iloc[i]-frame.quality_raw_start_idx.iloc[i]))
                    self.assertEqual(d['result']['context']['stage'],'raw')

    def test_legacy_injected_signature_is_unavailable_not_assumed_valid(self):
        def legacy(data, fs, params):
            return {'overall':np.full(data.shape[0],.8)}
        result=self.analyze(legacy)
        self.assertTrue(result['bp']['valid'].all())
        frame, meta=load_event_qeeg_table(self.write(result),self.source)
        self.assertEqual(set(frame.quality_diagnostic_state),{'unavailable'})
        self.assertIsNone(meta['analysis']['window_audit'][0]['quality_diagnostics']['result'])

    def test_finite_fallback_still_uses_legacy_selection_and_is_verifiable(self):
        self.params={key+'_weight':float(key=='spectrum') for key in ('flat','spectrum','kurtosis','corr')}
        def fallback(data, **kwargs):
            with patch.object(quality.sp_signal,'welch',side_effect=RuntimeError('forced')):
                return quality.get_eeg_quality_index_v2_parametric(data,**kwargs)
        result=self.analyze(fallback)
        self.assertTrue(result['bp']['valid'].all())
        self.assertTrue(np.all(result['bp']['quality']==.5))
        path=self.write(result)
        frame,meta=load_event_qeeg_table(path,self.source)
        self.assertEqual(set(frame.quality_diagnostic_state),{'invalid'})
        self.assertIn('spectrum:exception:RuntimeError',frame.quality_diagnostic_reason.iloc[0])
        meta['analysis']['window_audit'][0]['quality_diagnostics']['result']['detail']['spectrum'][0]=.9
        self.rehash(path,meta)
        with self.assertRaisesRegex(ValueError,'fallback score'):
            load_event_qeeg_table(path,self.source)

    def test_short_window_diagnostics_do_not_change_legacy_scores(self):
        result=self.analyze(win_sec=.5)
        self.assertTrue(all(r['quality_diagnostics']['state']=='invalid' for r in result['bp']['window_audit']))
        self.assertFalse(result['bp']['valid'].any())
        path=self.write(result)
        self.assertEqual(set(load_event_qeeg_table(path,self.source)[0].quality_diagnostic_state),{'invalid'})

    def test_scorer_error_is_distinct_and_legacy_invalid_output_is_unavailable(self):
        def raises(*args,**kwargs):
            raise ArithmeticError('forced')
        result=self.analyze(raises)
        frame,_=load_event_qeeg_table(self.write(result),self.source)
        self.assertEqual(set(frame.quality_diagnostic_state),{'error'})
        self.assertFalse(frame.quality_valid.any())
        result=self.analyze(lambda data,**_: {'overall':np.full(4,np.nan)})
        frame,_=load_event_qeeg_table(self.write(result),self.source)
        self.assertEqual(set(frame.quality_diagnostic_state),{'unavailable'})
        self.assertFalse(frame.quality_valid.any())

    def test_rehashed_stage_reason_context_and_coverage_tamper_rejected(self):
        path=self.write(self.analyze())
        original=json.loads(Path(str(path)+'.meta.json').read_text())
        for mutate in [lambda m:m['analysis']['window_audit'][0]['quality_diagnostics']['request'].update(stage='bandpass'),
                       lambda m:m['analysis']['window_audit'][0]['quality_diagnostics']['result']['context'].update(preset='ibrain_device'),
                       lambda m:m['analysis']['window_audit'].pop(),
                       lambda m:m['analysis']['window_audit'][0]['quality_diagnostics']['result']['component_valid']['spectrum'].__setitem__(0,False)]:
            meta=json.loads(json.dumps(original));mutate(meta);self.rehash(path,meta)
            with self.assertRaises(ValueError):load_event_qeeg_table(path,self.source)
        self.rehash(path,original)
        frame=pd.read_csv(path,dtype={'source_id':str,'config_id':str})
        frame.loc[0,'quality_diagnostic_state']='invalid'
        frame.to_csv(path,index=False);self.rehash(path,original)
        with self.assertRaisesRegex(ValueError,'differs from audit'):load_event_qeeg_table(path,self.source)

    def test_legacy_tables_explicitly_lack_diagnostic_version(self):
        path=self.write(self.analyze())
        meta=json.loads(Path(str(path)+'.meta.json').read_text())
        frame=pd.read_csv(path,dtype={'source_id':str,'config_id':str}).drop(columns=list(COLUMNS))
        frame.to_csv(path,index=False)
        del meta['quality_diagnostics_version']
        for row in meta['analysis']['window_audit']:del row['quality_diagnostics']
        self.rehash(path,meta)
        _,loaded=load_event_qeeg_table(path,self.source)
        self.assertNotIn('quality_diagnostics_version',loaded)

    def test_plot_label_uses_actual_components_and_preset(self):
        result=self.analyze()
        self.assertEqual(diagnostic_label(result['bp']['window_audit']), 'flat+spectrum [flat_spectrum_only]')
        self.params={key+'_weight':float(key=='spectrum') for key in ('flat','spectrum','kurtosis','corr')}
        result=self.analyze()
        self.assertEqual(diagnostic_label(result['bp']['window_audit']), 'spectrum [custom]')
        result=self.analyze(lambda data,**_: {'overall':np.ones(data.shape[0])})
        self.assertEqual(diagnostic_label(result['bp']['window_audit']), 'scorer diagnostics unavailable')

    def test_plot_marks_invalid_finite_scores_and_external_unavailable(self):
        rows=[{'quality_diagnostics':{'state':s}} for s in ('valid','invalid','error','unavailable')]
        fig,ax=plt.subplots()
        self.addCleanup(plt.close,fig)
        plot_diagnostic_markers(ax,np.arange(4),rows,np.array([[.2],[.5],[np.nan],[.8]]))
        self.assertEqual(len(ax.collections),2)
        np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,0],[1,2])
        np.testing.assert_array_equal(ax.collections[0].get_offsets()[:,1],[.5,.02])
        self.assertIn('Diagnostic invalid',ax.collections[0].get_label())


if __name__=='__main__':unittest.main()
