#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   measure_routine.py
@Time    :   2020/02/20 10:24:45
@Author  :   sk zhao 
@Version :   1.0
@Contact :   2396776980@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib
import numpy as np, time, pickle, datetime, scipy
# from easydl import clear_output
from qulab.yhs import waveform_new as wn
from tqdm import tqdm_notebook as tqdm
import asyncio, inspect
from qulab.yhs import imatrix as mx, dataTools as dt, optimize, measureroutine_population, computewave_wave
import functools
import imp, gc
op = imp.reload(optimize)
mrw = imp.reload(measureroutine_population)
cw = imp.reload(computewave_wave)

# cw.whichEnvelope = cw.whichEnvelope
# pulseTowave = cw.pulseTowave

# t_new = np.linspace(-45000,5000,125000)*1e-9
t_new = np.linspace(-15000,5000,50000)*1e-9
t_list = t_new*1e9 - np.min(t_new)*1e9
t_range = (-90e-6, 10e-6)
sample_rate = 2.5e9

################################################################################
# 单比特tomo波形
################################################################################

################################################################################
# 单比特tomo波形
################################################################################

async def singleQgate(envelopename=['square',1],pi_len=30e-9,amp=1,shift=0,delta_ex=110e6,axis='X',\
    DRAGScaling=None,phaseDiff=0,timing={'z>xy':0,'read>xy':0},phase=0.0,coordinatePhase=0,virtualPhase=0):
    # print(coordinatePhase,virtualPhase)
    wav_I, wav_Q = wn.zero(), wn.zero()
    shift += timing['read>xy']
    if envelopename[1] == 1:
        envelope_pi = cw.whichEnvelope(pi_len,*envelopename) * amp << shift
        envelope_half = cw.whichEnvelope(pi_len,*envelopename) * 0.5 * amp << shift
    if envelopename[1] == 2:
        envelope_pi = cw.whichEnvelope(pi_len,*envelopename) * amp << shift
        envelope_half = cw.whichEnvelope(pi_len,envelopename[0],1) * amp << shift

    # trig_wave = cw.whichEnvelope(3*pi_len,'square',envelopename[1]) << shift
    if axis == 'I':
        phi = 0
        wav_I, wav_Q = wn.zero(), wn.zero()
    if axis == 'X':
        phi = 0
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xn':
        phi = np.pi
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xhalf':
        phi = 0
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xnhalf':
        phi = np.pi
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Y':
        phi = np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Yn':
        phi = -np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Ynhalf':
        phi = -np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Yhalf':
        phi = np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Z':
        wav_I, wav_Q = wn.zero(), wn.zero()
    if axis == 'AnyRhalf':
        phi = phase
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'AnyRpi':
        phi = phase
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    return cw.pulseTowave((wav_I, wav_Q,np.array((wn.zero(),)),np.array((wn.zero(),))))

################################################################################
# 单比特tomo波形
################################################################################

async def tomoGate(envelopename=['square',1],pi_len=30e-9,amp=1,shift=0,delta_ex=110e6,axis='X',\
    DRAGScaling=None,phaseDiff=0,timing={'z>xy':0,'read>xy':0},phase=0.0,coordinatePhase=0,virtualPhase=0):
    # print(coordinatePhase,virtualPhase)
    
    shift += timing['read>xy']
    if envelopename[1] == 1:
        envelope_pi = cw.whichEnvelope(pi_len,*envelopename) * amp << shift
        envelope_half = cw.whichEnvelope(pi_len,*envelopename) * 0.5 * amp << shift
    if envelopename[1] == 2:
        envelope_pi = cw.whichEnvelope(pi_len,*envelopename) * amp << shift
        envelope_half = cw.whichEnvelope(pi_len,envelopename[0],1) * amp << shift
    if axis == 'I':
        phi = 0
        wav_I, wav_Q = wn.zero(), wn.zero()
    if axis == 'X':
        phi = 0
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xn':
        phi = np.pi
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xhalf':
        phi = 0
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Xnhalf':
        phi = np.pi
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Y':
        phi = np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Yn':
        phi = -np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Ynhalf':
        phi = -np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Yhalf':
        phi = np.pi/2
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'Z':
        wav_I, wav_Q = wn.zero(), wn.zero()
    if axis == 'AnyRhalf':
        phi = phase
        wav_I, wav_Q = wn.mixing(envelope_half,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    if axis == 'AnyRpi':
        phi = phase
        wav_I, wav_Q = wn.mixing(envelope_pi,phase=(phi+coordinatePhase+virtualPhase),freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    return cw.pulseTowave((wav_I, wav_Q,np.array((wn.zero(),)),np.array((wn.zero(),))))
