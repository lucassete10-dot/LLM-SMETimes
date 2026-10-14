"""Independent checks for CALCE extraction, split isolation and model artifacts."""
import json
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from battery_calce_validation import load_batteries, predict_vi
from prepare_calce import vi_features, IDS


class CalceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.root=Path('battery_calce_results')
        cls.batteries=load_batteries()
        cls.marks={k:v.numpy() for k,v in torch.load(next(cls.root.glob('vi_semantic_*.pt')),weights_only=True).items()}

    def test_prefix_cannot_read_future(self):
        t=np.arange(0,2000,30.0);v=4.2-t*.0002;i=np.full_like(t,-1.1)
        first=vi_features(t,v,i)
        v[t>1000]=999;i[t>1000]=999
        np.testing.assert_array_equal(first,vi_features(t,v,i))

    def test_capacity_is_per_cycle_not_cumulative(self):
        path=Path(r'D:\LLM\CALCE\raw\extracted\CS2_38\CS2_38\CS2_38_10_04_10.xlsx')
        df=pd.read_excel(path,sheet_name=1)
        cycle=df[df['Cycle_Index']==2]
        discharge=cycle[cycle['Step_Index']==7]
        expected=discharge['Discharge_Capacity(Ah)'].iloc[-1]-cycle['Discharge_Capacity(Ah)'].min()
        audit=pd.read_csv(r'D:\LLM\CALCE\processed\CS2_38_audit.csv')
        row=audit[audit['date']==str(discharge['Date_Time'].iloc[0])].iloc[0]
        self.assertAlmostEqual(expected,row.capacity_ah,places=8)
        self.assertGreater(discharge['Discharge_Capacity(Ah)'].iloc[-1],2)
        self.assertLess(row.capacity_ah,1.2)

    def test_audit_coverage_and_integral(self):
        for k in IDS:
            audit=pd.read_csv(Path(r'D:\LLM\CALCE\processed')/f'{k}_audit.csv')
            accepted=audit[audit.excluded_reason.isna()]
            self.assertEqual(len(accepted),len(self.batteries[k].soh))
            self.assertTrue((accepted.duration_s>=1000).all())
            self.assertTrue((accepted.capacity_discrepancy_fraction<=.03).all())
            self.assertTrue(pd.to_datetime(audit.date).is_monotonic_increasing)

    def test_saved_models_reproduce_each_fold(self):
        saved=pd.read_csv(self.root/'calce_held_out.csv')
        for k in IDS:
            pred=predict_vi(self.root/k,self.batteries[k].features,self.marks[k])
            np.testing.assert_allclose(pred[15:],saved[saved.cell==k].smetimes_fusion,atol=1e-6)
            checkpoint=torch.load(self.root/k/'smetimes_fusion_2026.pt',weights_only=True)
            self.assertNotIn(k,checkpoint['train_ids'])

    def test_future_cycles_cannot_change_past(self):
        k='CS2_35';x=self.batteries[k].features.copy();s=self.marks[k].copy()
        a=predict_vi(self.root/k,x,s)
        x[50:]=10;s[50:]=10
        b=predict_vi(self.root/k,x,s)
        np.testing.assert_allclose(a[:50],b[:50],atol=1e-7)

    def test_transfer_uses_only_nasa(self):
        folder=self.root/'nasa_only_transfer'
        checkpoint=torch.load(folder/'smetimes_fusion_2026.pt',weights_only=True)
        self.assertTrue(all(k.startswith('B00') for k in checkpoint['train_ids']))
        prep=joblib.load(folder/'preprocessing.joblib')
        x=np.concatenate([self.batteries[k].features for k in checkpoint['train_ids']])
        np.testing.assert_allclose(prep['scaler'].mean_,x.astype('float64').mean(0),atol=1e-10)
        saved=pd.read_csv(self.root/'nasa_only_transfer.csv')
        k='CS2_38'
        pred=predict_vi(folder,self.batteries[k].features,self.marks[k])
        np.testing.assert_allclose(pred[15:],saved[saved.cell==k].smetimes_fusion,atol=1e-6)

    def test_reported_mae_from_predictions(self):
        saved=pd.read_csv(self.root/'calce_held_out.csv')
        metrics=json.loads((self.root/'within_calce.json').read_text())
        for k in IDS:
            rows=saved[saved.cell==k]
            mae=np.abs(rows.smetimes_fusion-rows.true_soh).mean()
            self.assertAlmostEqual(mae,metrics[k]['metrics']['smetimes_fusion']['mae'],places=8)


if __name__=='__main__':unittest.main()
