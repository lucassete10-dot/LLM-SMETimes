"""Streaming waveform contracts, including invalid-input and boundary tests."""
import ctypes
from datetime import datetime,timezone
from pathlib import Path
import shutil
import subprocess
import unittest
import numpy as np
from scripts.verify_stm32_waveform import WaveStream
from prepare_mit import feature_vector,GRID
from adapt_nasa_mat_v2 import prefix_features
from scripts.prepare_same_type_forecast import save_json,sha


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.out=root/'experiments/stm32_stream_contract_tests'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        cls.out.mkdir(parents=True)
        cls.rows=[];shutil.copy2(__file__,cls.out/'test_source.py')
        for name in ('soh_waveform.c','soh_waveform.h'):shutil.copy2(root/'deployment/common'/name,cls.out/name)
        subprocess.run([shutil.which('gcc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-shared',str(cls.out/'soh_waveform.c'),'-o',str(cls.out/'wave.dll'),'-lm'],check=True,capture_output=True,text=True)
        cls.lib=ctypes.CDLL(str(cls.out/'wave.dll'))
        cls.lib.soh_wave_stream_init.argtypes=[ctypes.POINTER(WaveStream),ctypes.c_int]
        cls.lib.soh_wave_stream_push.argtypes=[ctypes.POINTER(WaveStream)]+[ctypes.c_double]*4
        cls.lib.soh_wave_stream_finish.argtypes=[ctypes.POINTER(WaveStream),ctypes.POINTER(ctypes.c_float)]
        cls.lib.soh_wave_stream_state_size.restype=ctypes.c_size_t

    def tearDown(self):
        result=self._outcome.result
        failures={test.id() for test,_ in result.failures+result.errors}
        self.rows.append(dict(test=self.id(),status='failed' if self.id() in failures else 'passed'))
        save_json(self.out/'results.json',dict(tests=self.rows,state_bytes=ctypes.sizeof(WaveStream),source_sha256={name:sha(self.out/name) for name in ('test_source.py','soh_waveform.c','soh_waveform.h')}))

    def start(self,family):
        s=WaveStream();self.assertEqual(self.lib.soh_wave_stream_init(ctypes.byref(s),family),0);return s

    def finish(self,s):
        out=np.empty({1:51,2:44,3:23}[s.family],dtype=np.float32)
        self.assertEqual(self.lib.soh_wave_stream_finish(ctypes.byref(s),out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))),0)
        return out

    def run_signal(self,family,t,v,i,temp):
        s=self.start(family)
        for row in zip(t,v,i,temp):self.assertGreaterEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),*row),0)
        return s,self.finish(s)

    def test_mit_irregular_duplicate_and_clamped_endpoint(self):
        t=np.r_[0.,0.,np.arange(1.,298.,1.7),310.];v=4.1-.002*t;v[1]=4.08
        i=-4.4+.01*np.sin(t);temp=25+.003*t
        _,out=self.run_signal(1,t,v,i,temp)
        expected=feature_vector(t/60,i,v,temp,GRID)
        np.testing.assert_allclose(out,expected,rtol=1e-6,atol=1e-5)

    def test_missing_temperature_allowed_only_for_mit(self):
        t=np.linspace(0,300,200);v=4-.001*t;i=np.full(len(t),-4.4);temp=np.full(len(t),25.);temp[5]=np.nan
        _,out=self.run_signal(1,t,v,i,temp)
        np.testing.assert_allclose(out,feature_vector(t/60,i,v,temp,GRID),rtol=1e-6,atol=1e-5,equal_nan=True)
        self.assertTrue(np.isnan(out[22:33]).all())
        for family in (2,3):
            s=self.start(family);self.assertLess(self.lib.soh_wave_stream_push(ctypes.byref(s),0,4,-1,float('nan')),0)

    def test_nasa_boundary_interpolation_and_ignored_future(self):
        t=np.arange(0.,1010.,7.);v=4.2-.001*t;i=-2+.01*np.sin(t);temp=25+.001*t
        s,out=self.run_signal(2,t,v,i,temp)
        expected=prefix_features(dict(Time=t,Voltage_measured=v,Current_measured=i,Temperature_measured=temp))
        np.testing.assert_allclose(out,expected,rtol=1e-6,atol=1e-5)
        s=self.start(2)
        for row in zip(t,v,i,temp):self.assertGreaterEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),*row),0)
        self.assertEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),2000,float('inf'),float('inf'),float('inf')),1)
        np.testing.assert_array_equal(self.finish(s),out)

    def test_invalid_time_is_sticky_until_reset(self):
        s=self.start(1);self.assertEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),1,4,-1,25),0)
        self.assertLess(self.lib.soh_wave_stream_push(ctypes.byref(s),.5,4,-1,25),0)
        self.assertLess(self.lib.soh_wave_stream_push(ctypes.byref(s),2,4,-1,25),0)
        self.assertEqual(self.lib.soh_wave_stream_init(ctypes.byref(s),1),0)
        self.assertEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),0,4,-1,25),0)

    def test_short_prefix_and_double_finish_rejected(self):
        for family in (1,2,3):
            s=self.start(family);self.lib.soh_wave_stream_push(ctypes.byref(s),0,4,-1,25)
            buf=(ctypes.c_float*51)();self.assertLess(self.lib.soh_wave_stream_finish(ctypes.byref(s),buf),0)
        t=np.arange(301,dtype=float);s,_=self.run_signal(1,t,4-.001*t,t*0-1,t*0+25)
        self.assertLess(self.lib.soh_wave_stream_finish(ctypes.byref(s),(ctypes.c_float*51)()),0)

    def test_xjtu_duplicate_first_and_minimum_count(self):
        s=self.start(3)
        for t in np.linspace(0,120,100):
            self.assertEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),t,3+t/300,2,25),0)
            self.assertEqual(self.lib.soh_wave_stream_push(ctypes.byref(s),t,1000,1000,1000),1)
        out=self.finish(s);np.testing.assert_allclose(out[:5],3+np.arange(0,121,30)/300,rtol=1e-6)
        s=self.start(3)
        for t in np.linspace(0,120,89):self.lib.soh_wave_stream_push(ctypes.byref(s),t,4,1,25)
        self.assertLess(self.lib.soh_wave_stream_finish(ctypes.byref(s),(ctypes.c_float*23)()),0)

    def test_state_abi_and_constant_memory_budget(self):
        self.assertEqual(self.lib.soh_wave_stream_state_size(),ctypes.sizeof(WaveStream))
        self.assertLessEqual(ctypes.sizeof(WaveStream),1024)


if __name__=='__main__':unittest.main()
