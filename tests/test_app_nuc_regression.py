"""APP/NUC CLI: preserved numerics, segment independence and verifiable clocks."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import signal
import torch

import data_analysis as da
from lilia.comparison import prepare_recording, pair_by_elapsed, align_recordings, segmented_psd, stft_parts
from lilia.comparison_io import load_signal_table, load_qeeg_table, load_pair_table
from lilia.neural import build_inference_timeline, model_provenance
from lilia.provenance import file_sha256
from lilia.windowing import continuous_slices


class Identity(torch.nn.Module):
    def forward(self, x):
        return x[:, :2]


class AppNucRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)

    def source(self, name, t, x):
        frame = pd.DataFrame(x, columns=[f'ch{i+1}' for i in range(x.shape[1])])
        frame.insert(0, 'Time[us]', t)
        p = self.root / name
        p.write_text('File Name,test\nAmp Gain,500,Abs Time Offset[us],1777000000000000\n'
                     'Channels,1,2,3,4\nSample Rate,500,500,500,500\n' + frame.to_csv(index=False))
        return p

    def cli(self, t=None, x=None, model=None):
        if t is None:
            t = np.r_[np.arange(100)*2000, 20000000+np.arange(3001)*2000,
                      40000000+np.arange(3001)*2000]
        if x is None:
            x = np.random.default_rng(77).normal(size=(len(t), 4))
        a = self.source('app.csv', t, x)
        b = self.source('nuc.csv', t+100000000, x)
        dest = self.root/'out'
        args = argparse.Namespace(group=None, app=str(a), nuc=str(b), outdir=str(dest))
        with patch.object(da, 'parse_args', return_value=args), \
             patch.object(da, 'load_model', return_value=model or Identity()), patch.object(Figure, 'savefig'):
            audit = da.main()
        return a, b, dest, audit

    def test_continuous_real_model_matches_saved_pre_refactor_values(self):
        fixtures = Path(__file__).parent/'fixtures'
        ref = json.loads((fixtures/'app_nuc_continuous_reference.json').read_text())
        self.assertEqual(model_provenance(), ref['model'])
        expected = np.load(fixtures/'app_nuc_continuous_reference.npz')
        x = np.random.default_rng(ref['seed']).normal(size=ref['shape'])
        r = prepare_recording(np.arange(len(x))*2000, x, da.load_model())
        for key, actual in [('filtered',r['filtered_raw']), ('cleaned',r['cleaned_raw']),
                            ('downsampled',r['input']), ('output',r['output'])]:
            np.testing.assert_allclose(actual, expected[key], atol=1e-6 if key=='output' else 1e-12, rtol=1e-6 if key=='output' else 1e-12)
        self.assertEqual(r['repair'][0]['counts'], ref['artifact_counts'])
        np.testing.assert_allclose(r['timeline'].time_us/1e6, expected['time_s'], atol=1e-12)
        for actual,key in zip(da.compute_psd(r['input'][:,0],fs=200), ['psd_freqs','psd_db']):
            np.testing.assert_allclose(actual, expected[key], atol=1e-12)
        for actual,key in zip(da.compute_stft(r['input'][:,0],fs=200), ['stft_freqs','stft_times','stft_db']):
            np.testing.assert_allclose(actual, expected[key], atol=1e-12)
        np.testing.assert_allclose(r['qeeg_grid'].time_s, expected['qeeg_time'], atol=1e-12)
        for key,actual in r['qeeg'].items():
            if key.endswith('_ch1'):
                np.testing.assert_allclose(actual, expected['qeeg_'+key[:-4]], atol=1e-12)

    def test_small_gap_filter_repair_and_ola_do_not_borrow_samples(self):
        t = np.r_[np.arange(1001)*2000, 2008000+np.arange(1201)*2000]
        x = np.random.default_rng(3).normal(size=(len(t),4))
        r = prepare_recording(t,x,Identity())
        altered = x.copy(); altered[:1001] *= 1000
        other = prepare_recording(t,altered,Identity())
        np.testing.assert_array_equal(r['output'][401:], other['output'][401:])
        np.testing.assert_array_equal(r['cleaned_raw'][1001:], other['cleaned_raw'][1001:])
        self.assertEqual(len(continuous_slices(r['timeline'].time_us,200)),1)
        self.assertEqual(len(continuous_slices(r['timeline'].time_us,200,segment_ids=r['timeline'].segment_ids)),2)
        for ch in r['repair'][1]['raw_ranges_by_channel']:
            self.assertTrue(all(a>=1001 and b<=len(t) for a,b in ch))
        self.assertIsNone(r['qeeg_grid'])

    def test_invalid_retained_segment_and_all_repair_anchors_fail(self):
        with self.assertRaisesRegex(ValueError,'interpolation anchors'):
            da.remove_artifacts(np.arange(20.)[:,None], thresh_mad=0, margin_ms=100)
        t = np.arange(1000)*2000
        x = np.ones((len(t),4)); x[20,0] = np.nan
        with self.assertRaisesRegex(ValueError,'Source segment 0 failed'):
            prepare_recording(t,x,Identity())
        with self.assertRaisesRegex(ValueError,'No continuous segment'):
            prepare_recording(t[:100],x[:100],Identity())

    def test_elapsed_pairing_preserves_gaps_and_ignores_absolute_epoch(self):
        a = build_inference_timeline(np.r_[np.arange(1500)*2000,10000000+np.arange(1500)*2000],500)
        b = build_inference_timeline(1777000000000000+np.r_[np.arange(1200)*2000,11000000+np.arange(1500)*2000],500)
        ia,ib,g = pair_by_elapsed(a,b)
        self.assertEqual(len(ia),880)
        self.assertEqual(np.unique(g).tolist(),[0,1])
        np.testing.assert_array_equal(a.time_us[ia]-a.source_epoch_us,b.time_us[ib]-b.source_epoch_us)
        self.assertTrue(np.all(np.diff(ib)>0))
        empty = build_inference_timeline(np.arange(10)*2000,500)
        self.assertEqual(len(pair_by_elapsed(a,empty)[0]),0)
        with self.assertRaisesRegex(ValueError,'integer'):
            pair_by_elapsed(a,b,.5)

    def test_planted_lag_and_continuous_alignment_match_legacy(self):
        tl = build_inference_timeline(np.arange(6000)*2000,500)
        a = np.random.default_rng(10).normal(size=(len(tl.time_us),4))
        for shift in (-17,0,23):
            b = np.roll(a,-shift,axis=0)
            ia,ib,groups,audit = align_recordings({'timeline':tl,'input':a},{'timeline':tl,'input':b})
            self.assertEqual(audit['lag_samples'],shift)
            old_a,old_b,old_t = da.align_for_comparison(a,tl.time_us/1e6,b,shift)
            np.testing.assert_array_equal(a[ia],old_a)
            np.testing.assert_array_equal(b[ib],old_b)
            np.testing.assert_array_equal(tl.time_us[ia]/1e6,old_t)
            self.assertTrue((groups==0).all())

    def test_lag_uses_only_longest_common_segment(self):
        tl = build_inference_timeline(np.r_[np.arange(1000)*2000,10000000+np.arange(1500)*2000],500)
        x = np.random.default_rng(4).normal(size=(1000,4))
        with patch.object(da,'estimate_lag',return_value=0) as lag:
            *_, audit = align_recordings({'timeline':tl,'input':x},{'timeline':tl,'input':x})
        np.testing.assert_array_equal(lag.call_args.args[0],x[400:])
        self.assertEqual(audit['reference_app_output_range'],[400,1000])
        with self.assertRaisesRegex(ValueError,'constant'):
            da.estimate_lag(np.ones((400,4)),np.ones((400,4)))
        with self.assertRaisesRegex(ValueError,'finite'):
            da.estimate_lag(np.empty((0,4)),np.empty((0,4)))

    def test_segment_psd_pooling_and_stft_centers(self):
        x = np.random.default_rng(1).normal(size=1600)
        slices = [slice(0,1000),slice(1000,1600)]
        f,db,audit = segmented_psd(x,200,slices)
        _,p1 = signal.welch(x[:1000],fs=200,nperseg=800,noverlap=400,nfft=800)
        _,p2 = signal.welch(x[1000:],fs=200,nperseg=600,noverlap=300,nfft=800)
        np.testing.assert_allclose(db,10*np.log10((p1+p2)/2+1e-12),atol=1e-12)
        self.assertEqual([r['welch_windows'] for r in audit],[1,1])
        self.assertEqual(len(f),401)
        t = np.r_[np.arange(1000)/200,20+np.arange(600)/200]
        parts = stft_parts(t,x,200)
        self.assertEqual(len(parts),2)
        self.assertLessEqual(parts[0][1][-1],t[999])
        self.assertEqual(parts[1][1][0],20)
        self.assertLessEqual(parts[1][1][-1],t[-1])

    def test_time_plot_contains_nan_break_at_small_source_gap(self):
        t = np.arange(8)/200
        x = np.ones((8,2))
        captured = []
        def save(fig,*args,**kwargs):
            captured.append(fig.axes[0].lines[0].get_ydata())
        with patch.object(Figure,'savefig',save):
            da.plot_model_before_after(t,x,x,'test',self.root/'plot.png','blue',segment_ids=np.repeat([0,1],4))
        self.assertTrue(np.isnan(captured[0][4]))

    def test_qeeg_plot_keeps_isolated_segment_visible(self):
        indices = {k:np.array([.1,np.nan,.2,.3]) for k in
                   ('theta','alpha','beta','focus','flow','calm','relaxation')}
        indices['time'] = np.array([2.5,np.nan,22.5,27.5])
        captured = []
        def save(fig,*args,**kwargs):
            captured.extend(fig.axes[0].lines)
        with patch.object(Figure,'savefig',save):
            da.plot_qeeg_indices(indices,'test',self.root/'q.png',marker='.')
        self.assertEqual(captured[0].get_marker(),'.')
        self.assertTrue(np.isnan(captured[0].get_ydata()[1]))

    def test_gapped_cli_tables_preserve_short_prefix_epoch_and_source_ids(self):
        a,b,dest,audit = self.cli()
        self.assertEqual(audit['status'],'complete')
        self.assertEqual(audit['alignment']['lag_samples'],0)
        self.assertEqual(audit['alignment']['paired_runs'],2)
        self.assertEqual(audit['sources']['app']['inference']['segments'][0]['status'],'excluded')
        f,_ = load_signal_table(dest/'app_signal.csv',a,da.MODEL_PATH)
        self.assertEqual(f.time_s.iloc[0],20)
        self.assertEqual(f.raw_fractional_idx.iloc[0],100)
        self.assertEqual(f.segment_id.unique().tolist(),[1,2])
        self.assertEqual(f.quality_state.unique().tolist(),['disabled'])
        q,_ = load_qeeg_table(dest/'app_qeeg.csv',a,da.MODEL_PATH)
        self.assertEqual(q.time_s.tolist(),[22.5,42.5])
        self.assertEqual(q.window_start_idx.tolist(),[0,1201])
        pairs,_ = load_pair_table(dest/'app_nuc_alignment.csv',a,b)
        self.assertTrue((pairs.residual_us==0).all())
        self.assertEqual(len(pairs),2402)

    def test_tables_reject_changed_source_checkpoint_and_rehashed_mapping(self):
        a,b,dest,_ = self.cli()
        wrong = self.root/'wrong.pth'; wrong.write_text('bad')
        with self.assertRaisesRegex(ValueError,'checkpoint differs'):
            load_signal_table(dest/'app_signal.csv',a,wrong)
        with self.assertRaisesRegex(ValueError,'Raw recording differs'):
            load_signal_table(dest/'app_signal.csv',b)
        for filename,col,loader in [('app_signal.csv','raw_fractional_idx',lambda p:load_signal_table(p,a)),
                ('app_qeeg.csv','window_start_idx',lambda p:load_qeeg_table(p,a)),
                ('app_nuc_alignment.csv','nuc_output_idx',lambda p:load_pair_table(p,a,b))]:
            p=dest/filename; f=pd.read_csv(p); f.loc[0,col]+=1; f.to_csv(p,index=False)
            side=Path(str(p)+'.meta.json'); meta=json.loads(side.read_text())
            meta['table_sha256']=file_sha256(p); side.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'mapping differs'):
                loader(p)

    def test_sidecars_reject_inference_and_lag_reference_changes(self):
        a,b,dest,_ = self.cli()
        p=dest/'app_signal.csv'; side=Path(str(p)+'.meta.json')
        m=json.loads(side.read_text()); m['inference']['segments'][1]['raw_start_idx']+=1
        side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'inference mapping'):
            load_signal_table(p,a)
        p=dest/'app_nuc_alignment.csv'; side=Path(str(p)+'.meta.json')
        m=json.loads(side.read_text()); m['alignment']['reference_app_output_range'][0]+=1
        side.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'reference mapping'):
            load_pair_table(p,a,b)

    def test_cli_failure_records_audit_and_raises(self):
        with self.assertRaisesRegex(ValueError,'No continuous segment'):
            self.cli(t=np.arange(100)*2000)
        audit=json.loads((self.root/'out/app_nuc_analysis.json').read_text())
        self.assertEqual(audit['status'],'failed')
        self.assertTrue(audit['errors'])
        self.assertEqual(audit['sources']['app']['inference']['segments'][0]['reason'],'shorter_than_model_window')
        class Bad(torch.nn.Module):
            def forward(self,x): return x[:,:2]*np.nan
        with self.assertRaisesRegex(ValueError,'output'):
            self.cli(model=Bad())
        audit=json.loads((self.root/'out/app_nuc_analysis.json').read_text())
        self.assertEqual(audit['status'],'failed')
        self.assertIn('Source segment 1 failed',audit['errors'][0])


if __name__ == '__main__':
    unittest.main()
