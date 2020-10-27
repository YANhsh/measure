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
import numpy as np, time, pickle
from qulab.waveform import step, gaussian, cosPulse
# from qulab import waveform as wn
from qulab import waveform as wn
from tqdm import tqdm_notebook as tqdm
import asyncio
# from qulab import imatrix as mx



################################################################################
# 收集变量
################################################################################

def saveStatus(fname='D:/status.obj'):
    import sys
    status = {}
    for k in filter(
        lambda s: s[0] != '_' and not callable(globals()[s]) and
        not isinstance(globals()[s], type(sys)) and not s in ['In', 'Out'],
        globals().keys()):
        try:
            status[k] = pickle.dumps(globals()[k])
        except:
            print(k, type(globals()[k]))
    with open(fname, 'wb') as f:
         pickle.dump(status, f)
    
def loadStatus(fname='D:/status.obj'):
    with open(fname, 'rb') as f:
        status = pickle.load(f)
    
    for k, v in status.items():
        globals()[k] = pickle.loads(v)



################################################################################
# ZPulse波形
################################################################################
'''
async def zWave(qubit, measure,volt=0.4,during=500e-9,shift=1000e-9,offset=0):
    pulse = (wn.square(during) << (during/2+shift)) * volt + offset
    pulselist = (pulse,) 
    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulselist, 'i' , 0, 0)
    return pulselist
   

'''
async def zWave(qubit, measure, volt=0.4, during=500e-9, shift=1000e-9, channel_output_delay=90e3,offset=0):
    channel_output_delay += -shift
    channel_output_delay += -during
    pulse = (wn.square(during) >> (during/2)) * volt + offset
    pulselist = (pulse,) 
    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulselist, 'i', channel_output_delay, 0)
    return pulselist


################################################################################
# 波包选择
################################################################################

def whichEnvelope(envelop):
    x = {'square':wn.square,'cospulse':wn.cosPulse,'gaussian':wn.gaussian}
    return x[envelop]


################################################################################
# 读出混频
################################################################################

def modulation_read(qubit, measure, delta, readlen=1100):
    t_list = measure.t_list
    measure.delta = delta
    # t_list=np.linspace(0,5000,10001)*1e-9
    twidth = readlen
    twidth = int(twidth) 
    ringup = 120
    pulse1 = wn.square((twidth-ringup)/1e9) >> ((twidth+ringup)/ 1e9 / 2)
    pulse2 = wn.square(ringup/1e9)>>(ringup/2/1e9)
    pulse = (0.4*pulse1+pulse2)*30000
    I, Q = wn.zero(), wn.zero()
    for i in delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    I, Q = I(t_list)/len(delta), Q(t_list)/len(delta)
    pulselist = I, Q
    measure.awg[qubit.inst['lo_awg']].da_write_wave(qubit.inst['lo_ch'][0], I, 'i' , 90100, 1, 0)
    measure.awg[qubit.inst['lo_awg']].da_write_wave(qubit.inst['lo_ch'][1], Q, 'i' , 90100, 1, 0)
    measure.ad.set_ad_freq(delta, measure.depth, window_start=8)  ##用于读出解模
    return pulselist



################################################################################
# 激励混频
################################################################################
'''
async def modulation_ex(qubit,measure,w=20000e-9,delta_ex=[0],shift=0):
    t_list = measure.t_list
    n = len(delta_ex)
    pulse = wn.square(w) << (w / 2 + 300e-9+shift)
    pulse = pulse(t_list)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], pulse, 'i' , 0, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], pulse, 'i' , 0, 0)

'''
def modulation_ex(qubit,measure,width=20000e-9,channel_output_delay=69e3):
    t_list = np.linspace(0,30000,60001)*1e-9
    pulse = (wn.square(width) >> (width / 2 ))*5000
    pulse = pulse(t_list)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], pulse, 'i' , channel_output_delay, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], pulse, 'i' , channel_output_delay, 1, 0)
    
#####################################################################################
# z_pulse
############################################################################################
def z_pulse(qubit,measure,width=20000e-9,amp=30000,channel_output_delay=59e3,offset=0):
    t_list = np.linspace(0,30000,60001)*1e-9
    pulse = (wn.square(width) >> (width / 2 ))*amp
    pulse = pulse(t_list) + offset
    # pulse = np.array([0]* len(t_list))+ offset
    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulse, 'i' , channel_output_delay, 1, 0)
    # print(qubit.inst['z_ch'])
    return pulse

def z_envelope(qubit,measure,width=20000e-9,amp=30000,channel_output_delay=59e3,envelopename='square', Delta_lo=200e6,t=0):
    t_list = np.linspace(0,30000,60001)*1e-9
    pulse = (wn.square(width) >> (width / 2 ))*amp
    pulse = pulse(t_list+t)*np.sin(2*np.pi*Delta_lo*t_list)
    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulse, 'i' , channel_output_delay, 1, 0)
    # print(qubit.inst['z_ch'])    

################################################################################
# Rabi波形
################################################################################
'''
async def rabiWave(qubit, measure, envelopename='square',nwave=1,amp=1,during=75e-9,shift=0,Delta_lo=200e6,phase=0,phaseDiff=0,DRAGScaling=None):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    wave = (((envelope(during) << (shift+during/2))) + ((envelope(during) << (shift+during/2*3)))) * amp
    # wave = ((wn.cosPulse(2*during) << (shift+during)))
    # mwav = (wn.square(2*during+380e-9) << (during+190e-9+10e-9)) * amp
    mwav = wn.square(2*during) << (during+shift)
    pulse, mpulse = wn.zero(), wn.zero()
    for i in range(nwave):
        pulse += (wave << 2*i*during)
        mpulse += (mwav << 2*i*during)
    wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , 0, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_I, 'i' , 0, 0)
    return wav_I, wav_Q, mpulse, mpulse
'''
def rabiWave(qubit, measure, envelopename='square', nwave=1, amp=20000, during=75e-9, shift=0, channel_output_delay=89.9e3, Delta_lo=200e6, phase=0, phaseDiff=0, DRAGScaling=None):
    t_list = np.linspace(0,20000,40001)*1e-9
    channel_output_delay += -shift*1e9
    channel_output_delay += -during*1e9*2
    print(shift,during,channel_output_delay)
    envelope = whichEnvelope(envelopename)
    pulse = ((envelope(during) >> (during/2))+(envelope(during) >> (during*3/2))) * amp
    # wave = ((wn.cosPulse(2*during) << (shift+during)))
    #pulse = wn.zero()
    wav_I1, wav_Q1 = wn.mixing(pulse, phase=phase,freq=Delta_lo, ratioIQ=-1.0, phaseDiff=phaseDiff, DRAGScaling=DRAGScaling)
    wav_I, wav_Q = wav_I1(t_list), wav_Q1(t_list)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , channel_output_delay, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , channel_output_delay, 1, 0)
    return wav_I1, wav_Q1


################################################################################
# Ramsey及SpinEcho,CPMG, PDD波形
################################################################################

def coherenceWave(qubit, measure, envelopename='square',t_run=0,during=75e-9,n_wave=0,seqtype='CPMG',detune=3e6,shift=0, channel_output_delay=89.9e3,Delta_lo=200e6,amp = 20000):
    #t_run pi/2之间的时间间隔
    
    t_list = np.linspace(0,30000,60001)*1e-9
    channel_output_delay += -(t_run+(2*n_wave+1.5)*during+shift)*1e9
    print(t_run,during,channel_output_delay)

    envelope = whichEnvelope(envelopename)
    pulse1 = envelope(during) >> (during/2+shift)
    wavI1, wavQ1 = wn.mixing(pulse1,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    pulse3 = envelope(during) >> (t_run+(2*n_wave+1.5)*during+shift)
    wavI3, wavQ3 = wn.mixing(pulse3,phase=2*np.pi*detune*t_run,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)

    if seqtype == 'CPMG':
        pulse2, step = wn.zero(), t_run / n_wave
        for i in range(n_wave):
            pulse = ((envelope(during) >> (during/2))+(envelope(during) >> (during*3/2))) 
            pulse2 += pulse >> ((i+0.5)*step+(i+0.5)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'PDD':
        pulse2, step = wn.zero(), t_run / (n_wave + 1)
        for i in range(n_wave):
            pulse = ((envelope(during) >> (during/2))+(envelope(during) >> (during*3/2)))
            pulse2 += pulse >> ((i+1)*step+(i+0.5)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = (wavI1 + wavI2 + wavI3)*amp, (wavQ1 + wavQ2 + wavQ3)*amp
    
    wav_I, wav_Q = wav_I(t_list), wav_Q(t_list)

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , channel_output_delay, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , channel_output_delay, 1, 0)
    
    return wav_I, wav_Q, wn.zero(), wn.zero()



################################################################################
# Ramsey波形
################################################################################

def ramseyWave(qubit, measure, delay, halfpi=75e-9,fdetune=3e6, amp = 20000,shift=0, Delta_lo=200e6, envelopename='square', channel_output_delay=89.9e3):
    t_list = np.linspace(0,30000,60001)*1e-9
    channel_output_delay += -(delay+shift+halfpi*2)*1e9
    print(delay,halfpi,channel_output_delay)

    envelope = whichEnvelope(envelopename)
    cosPulse1 = ((envelope(halfpi) >> (shift+halfpi/2)))
    wav_I1, wav_Q1 = wn.mixing(cosPulse1,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    
    cosPulse2 = ((envelope(halfpi) >> (delay+shift+halfpi/2*3)))
    wav_I2, wav_Q2 = wn.mixing(cosPulse2,phase=2*np.pi*fdetune*delay,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = (wav_I1 + wav_I2)*amp, (wav_Q1 + wav_Q2)*amp

    wav_I, wav_Q = wav_I(t_list), wav_Q(t_list)
    # init1 = wn.square(halfpi) >> (shift+halfpi/2)
    # init2 = wn.square(halfpi) >> (shift+halfpi*3/2+delay)
    # mrk = init1 + init2

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , channel_output_delay, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , channel_output_delay, 1, 0)

    return wav_I, wav_Q




################################################################################
# AC-Stark波形
################################################################################

async def ac_stark_wave(qubit, measure):
    pulse_read = modulation_read(qubit, measure,measure.delta,readlen=measure.readlen)
    width = 500e-9
    pulse = wn.square(width) << (width/2+3000e-9)
    I, Q = wn.zero(), wn.zero()
    for i in measure.delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    pulse_acstark = (I,Q,wn.zero(),wn.zero())
    pulselist = np.array(pulse_read) + np.array(pulse_acstark)
    measure.awg[qubit.inst['lo_awg']].da_write_wave(qubit.inst['lo_ch'][0], pulselist[0], 'i' , 0, 0)
    measure.awg[qubit.inst['lo_awg']].da_write_wave(qubit.inst['lo_ch'][1], pulselist[1], 'i' , 0, 0)
    

################################################################################
# RB波形
################################################################################

def genXY(during,shift=0e-9,tdelay=0e-9,pulse='pi',envelopename='square'):
    shift += 200e-9
    envelope = whichEnvelope(envelopename)
    if pulse == 'halfpi':
        pulse = envelope(during) << (shift+during/2)
    if pulse == 'pi':
        pulse = (envelope(during) << (shift+during/2)) + (envelope(during) << (shift+during/2*3))
    
    return pulse

def genParas(x):
    if x == 'I':
        paras = (0,'pi')
    elif x == 'X':
        paras = (0,'pi')
    elif x == 'Xhalf':
        paras = (0,'halfpi')
    elif x == 'Xnhalf':
        paras = (np.pi,'halfpi')
    elif x == 'Y':
        paras = (np.pi/2,'pi')
    elif x == 'Yhalf':
        paras = (np.pi/2,'halfpi')
    elif x == 'Ynhalf':
        paras = (3*np.pi/2,'halfpi')
    return paras

async def rbWave(measure,m,gate,pilen,Delta_lo=200e6,shift=0,phaseDiff=0.0,DRAGScaling=None):
    op = {'1':['I'],'2':['X'],'3':['Xhalf'],'4':['Xnhalf'],'5':['Y'],'6':['Yhalf'],'7':['Ynhalf'],
        '8':['X','Y'],'9':['Xhalf','Yhalf','Xnhalf'],'10':['Xhalf','Ynhalf','Xnhalf'],'11':['Ynhalf','X'],
        '12':['Yhalf','X'],'13':['Xhalf','Y'],'14':['Xnhalf','Y'],'15':['Xhalf','Yhalf','Xhalf'],'16':['Xnhalf','Yhalf','Xnhalf'],
        '17':['Xhalf','Yhalf'],'18':['Xnhalf','Yhalf'],'19':['Xhalf','Ynhalf'],'20':['Xnhalf','Ynhalf'],
        '21':['Ynhalf','Xnhalf'],'22':['Ynhalf','Xhalf'],'23':['Yhalf','Xnhalf'],'24':['Yhalf','Xhalf']}

    mseq = mx.cliffordGroup_single(m,gate)
    if mseq == []:
        return
    rotseq = []
    for i in mseq[::-1]:
        rotseq += op[i]
    waveseq_I, waveseq_Q, wav = wn.zero(), wn.zero(), wn.zero()
    # rotseq = ['Xhalf','Xnhalf','Yhalf','Ynhalf']*m
    # if rotseq == []:
    #     return
    # print(rotseq)
    for i in rotseq:
        paras = genParas(i)
        if i == 'I':
            waveseq_I += wn.zero()
            waveseq_Q += wn.zero()
            # continue
        else:
            pulse = genXY(during=pilen,pulse=paras[1],envelopename=measure.envelopename)
            cosPulse = pulse << shift
            phi = paras[0]
            wav_I, wav_Q = wn.mixing(cosPulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
            waveseq_I += wav_I
            waveseq_Q += wav_Q
            
        if paras[1] == 'pi':
            shift += 2*pilen
        if paras[1] == 'halfpi':
            shift += pilen

    return waveseq_I, waveseq_Q, wn.zero(), wn.zero()
    


################################################################################
# 单比特tomo波形
################################################################################

def tomoWave(envelopename='square',during=0,shift=0,Delta_lo=200e6,axis='X',amp=20000,DRAGScaling=None,phaseDiff=0):
    envelope = whichEnvelope(envelopename)
    if axis == 'X':
        phi = 0
        pulse = (((envelope(during) >> (shift+during/2))) + ((envelope(during) >> (shift+during/2*3))))
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = 2*during
    if axis == 'Xhalf':
        phi = 0
        pulse = envelope(during) >> (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = during
    if axis == 'Xnhalf':
        phi = np.pi
        pulse = envelope(during) >> (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = during
    if axis == 'Y':
        phi = np.pi/2
        pulse = (((envelope(during) >> (shift+during/2))) + ((envelope(during) >> (shift+during/2*3))))
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = during
    if axis == 'Ynhalf':
        phi = -np.pi/2
        pulse = envelope(during) >> (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = during
    if axis == 'Yhalf':
        phi = np.pi/2
        pulse = envelope(during) >> (shift+during/2)
        wav_I, wav_Q = wn.mixing(pulse,phase=phi,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
        shift = during
    if axis == 'Z':
        wav_I, wav_Q = wn.zero(), wn.zero()
        shift = 0
    return wav_I*amp, wav_Q*amp, shift

################################################################################
# 单比特tomo测试
################################################################################



def tomoTest(qubit,measure,t,halfpi,axis,amp=20000,channel_output_delay=89.9e3,DRAGScaling=None):
    t_list = np.linspace(0,20000,40000)*1e-9
    shift1 = (2*qubit.pi_len+10)/1e9
    
    wav_I1, wav_Q1 = rabiWave(qubit, measure, envelopename='cospulse', nwave=1, amp=amp, during=t/1e9, phase=0, phaseDiff=0, DRAGScaling=None)
    wav_I, wav_Q, shift2= tomoWave(envelopename='cospulse',during = halfpi/1e9,shift=shift1,Delta_lo=200e6,axis=axis,amp=amp,DRAGScaling=DRAGScaling)
    wav_I = wav_I1+wav_I
    wav_Q = wav_Q1+wav_Q
    wav_I, wav_Q = wav_I(t_list), wav_Q(t_list)

    channel_output_delay += -shift1*1e9
    channel_output_delay += -shift2*1e9
    # channel_output_delay += -during*1e9*2
    print(shift1,shift2,channel_output_delay)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , channel_output_delay, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , channel_output_delay, 1, 0)
   
    


