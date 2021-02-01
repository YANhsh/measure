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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
# 波包选择
################################################################################

def whichEnvelope(envelop):
    x = {'square':wn.square,'cospulse':wn.cosPulse,'gaussian':wn.gaussian}
    return x[envelop]

################################################################################
# 读出混频
################################################################################

def modulation_read(measure, delta, read_amp=0.4,readlen=1100,phlen=500e-9,phshift=3000e-9,read_delay=200,ac_stark=0):
    ##ac_stark=1,0对应扫或不扫ac_stark
    t_list = measure.t_list
    measure.delta = delta
    measure.read_amp = read_amp
    twidth = readlen + 600
    twidth = int(twidth) 
    ringup = 120

    readwave_start = measure.t_reference + read_delay
    pulse_photon = wn.square(phlen*ac_stark) << ((phlen/2+phshift)*ac_stark)

    pulse1 = wn.square((twidth-ringup)/1e9) >> ((twidth+ringup)/ 1e9 / 2)
    pulse2 = wn.square(ringup/1e9)>>(ringup/2/1e9)
    pulse = (read_amp*pulse1 + pulse2 + read_amp*pulse_photon)*32000    ###read_amp=[0,1]
    I, Q = wn.zero(), wn.zero()
    for i in delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    I, Q = I(t_list)/len(delta), Q(t_list)/len(delta)
    pulselist = I, Q
    measure.awg['awg1'].da_write_wave(1, I, 'i' , readwave_start, 1, 0)
    measure.awg['awg1'].da_write_wave(2, Q, 'i' , readwave_start, 1, 0)
    measure.ad.set_ad_freq(delta, measure.readlen, window_start=8)  ##用于读出解模
    return pulselist

################################################################################
# AC-Stark波形
################################################################################
def ac_stark_wave(measure,width=500e-9,toread=3000):
    t_list = measure.t_list
    delta = measure.delta# + measure.f_lo

    tlay = width*1e9 +toread
    pulse_photon = wn.square(width) << (width/2)

    twidth = measure.readlen +180
    twidth = int(twidth) 
    ringup = 120
    pulse1 = wn.square((twidth-ringup)/1e9) >> ((twidth+ringup)/ 1e9 / 2 + tlay*1e-9)
    pulse2 = wn.square(ringup/1e9)>>((ringup)/2/1e9 + tlay*1e-9)
    pulse = (0.4*pulse1+pulse2+pulse_photon)*30000

    I, Q = wn.zero(), wn.zero()
    for i in delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    I, Q = I(t_list)/len(delta), Q(t_list)/len(delta)

    acsWave_start = measure.t_reference
    measure.awg['awg1'].da_write_wave(1, I, 'i' , acsWave_start, 1, 0)
    measure.awg['awg1'].da_write_wave(2, Q, 'i' , acsWave_start, 1, 0)
    measure.ad.set_ad_freq(delta, measure.readlen, window_start=8)  ##用于读出解模


def ats_setup(measure,readlen, repeats=5000,mode=1):
    for i in measure.awg:
        measure.awg[i].da_trigg(repeats)
    measure.ad.set_ad(readlen, repeats, mode=mode)
################################################################################
# 激励混频
################################################################################

def modulation_ex(qubit,measure,ex_amp=10000, XYlen=20000e-9,shift=0):
    t_list = measure.t_list
    pulse = (wn.square(XYlen) << (XYlen / 2 +shift))*ex_amp   ###read_amp=[0,320]mV
    pulse = pulse(t_list)

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], pulse, 'i' , measure.t_reference, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], pulse, 'i' , measure.t_reference, 1, 0)
    
#####################################################################################
# z_pulse
############################################################################################
def z_pulse(qubit,measure,Zamp=30000,Zlen=3000e-9,Zamp2=0,Zlen2=0,offset=0,Crosstalk=0,shift=0):
    ##Crosstalk取值1或0,对应扫或不扫crosstalk
    t_list = measure.t_list
    pulse =(wn.square(Zlen) << (0.5*Zlen+shift))*Zamp -(wn.square(Zlen) << (1.5*Zlen+shift))*Zamp*Crosstalk
    pulse = pulse(t_list) + offset

    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulse, 'i' , measure.t_reference, 1, 0)
    return pulse  

def z_comwave(qubit,measure,Zamp=30000,Zlen=1000e-9,Zamp2=0.1,Zlen2=5,offset=0,shift=0):
    ##补偿台阶2的长度和幅值与偏置台阶的比值：Zamp2和Zlen2
    t_list = measure.t_list
    pulse1 = (wn.square(Zlen) << (-Zlen/2+Zlen2*Zlen+shift))*5000 - (wn.square(Zlen2*Zlen) << (Zlen2*Zlen/2+shift))*Zamp
    pulse2 = -(wn.square(Zlen) << (Zlen/2+Zlen2*Zlen+shift))*5000 + (wn.square(Zlen2*Zlen) << (3*Zlen2*Zlen/2+shift))*Zamp
    pulse = pulse1 + pulse2
    pulse = pulse(t_list) + offset
    measure.awg[qubit.inst['z_awg']].da_write_wave(qubit.inst['z_ch'], pulse, 'i' ,  measure.t_reference, 1, 0)
    return pulse

################################################################################
# Rabi波形
################################################################################
def rabiWave(qubit, measure, envelopename='square', nwave=1, amp=20000, during=75e-9, shift=0, Delta_lo=200e6, phase=0, phaseDiff=0, DRAGScaling=None):
    t_list = measure.t_list
    envelope = whichEnvelope(envelopename)
    wave = ((envelope(during) << (during/2+shift*1e-9))+(envelope(during) << (during*3/2+shift*1e-9))) * amp
    pulse = wn.zero()
    for i in range(nwave):
        pulse += (wave << 2*i*during)
    
    wav_I1, wav_Q1 = wn.mixing(pulse, phase=phase,freq=Delta_lo, ratioIQ=-1.0, phaseDiff=phaseDiff, DRAGScaling=DRAGScaling)
    wav_I, wav_Q = wav_I1(t_list), wav_Q1(t_list)

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , measure.t_reference, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , measure.t_reference, 1, 0)
    return wav_I1, wav_Q1


################################################################################
# Ramsey及SpinEcho,CPMG, PDD波形
################################################################################

def coherenceWave(qubit, measure, envelopename='square',t_run=0,during=75*1e-9,n_wave=0,seqtype='CPMG',detune=3e6,shift=0,  Delta_lo=200e6,amp = 20000):
    #t_run pi/2之间的时间间隔
    t_list = measure.t_list

    envelope = whichEnvelope(envelopename)
    pulse1 = envelope(during) << (during/2+shift)
    wavI1, wavQ1 = wn.mixing(pulse1,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    pulse3 = envelope(during) << (t_run+(2*n_wave+1.5)*during+shift)
    wavI3, wavQ3 = wn.mixing(pulse3,phase=2*np.pi*detune*t_run,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)

    if seqtype == 'CPMG':
        pulse2, step = wn.zero(), t_run / n_wave
        for i in range(n_wave):
            pulse = ((envelope(during) << (during/2))+(envelope(during) << (during*3/2))) 
            pulse2 += pulse << ((i+0.5)*step+(i+0.5)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'PDD':
        pulse2, step = wn.zero(), t_run / (n_wave + 1)
        for i in range(n_wave):
            pulse = ((envelope(during) << (during/2))+(envelope(during) << (during*3/2)))
            pulse2 += pulse << ((i+1)*step+(i+0.5)*2*during+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = (wavI1 + wavI2 + wavI3)*amp, (wavQ1 + wavQ2 + wavQ3)*amp
    wav_I, wav_Q = wav_I(t_list), wav_Q(t_list)

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , measure.t_reference, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , measure.t_reference, 1, 0)

    return wav_I, wav_Q, wn.zero(), wn.zero()

################################################################################
# Ramsey波形
################################################################################

def ramseyWave(qubit, measure, delay, halfpi=75*1e-9,fdetune=3e6, amp = 20000,shift=0, Delta_lo=200e6, envelopename='square'):
    t_list = measure.t_list

    envelope = whichEnvelope(envelopename)
    cosPulse1 = ((envelope(halfpi) << (shift+halfpi/2)))
    wav_I1, wav_Q1 = wn.mixing(cosPulse1,phase=0,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    
    cosPulse2 = ((envelope(halfpi) << (delay+shift+halfpi/2*3)))
    wav_I2, wav_Q2 = wn.mixing(cosPulse2,phase=2*np.pi*fdetune*delay,freq=Delta_lo,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wav_I, wav_Q = (wav_I1 + wav_I2)*amp, (wav_Q1 + wav_Q2)*amp

    wav_I, wav_Q = wav_I(t_list), wav_Q(t_list)
    # init1 = wn.square(halfpi) >> (shift+halfpi/2)
    # init2 = wn.square(halfpi) >> (shift+halfpi*3/2+delay)
    # mrk = init1 + init2

    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][0], wav_I, 'i' , measure.t_reference, 1, 0)
    measure.awg[qubit.inst['ex_awg']].da_write_wave(qubit.inst['ex_ch'][1], wav_Q, 'i' , measure.t_reference, 1, 0)
    return wav_I, wav_Q

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
   
    
################################################################################
# classify
################################################################################
def classify(measure,s_st,target=None,predictexe=True):
    
    num = measure.n//2+measure.n%2
    name = ''
    for i in measure.qubitToread:
        name += i
    if target is not None:
        name = f'q{target+1}'
    fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
    n = measure.n if target is None else 1
    if predictexe:
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            S = list(s_off) + list(s_on)
            x,z = np.real(S), np.imag(S)
            d = list(zip(x,z))
            kmeans = KMeans(n_clusters=2,max_iter=100,tol=0.001)
            kmeans.fit(d)
            measure.predict[measure.qubitToread[i]] = kmeans.predict
            y = kmeans.predict(d)
            print(list(y).count(1)/len(y))
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(x,z,c=y,s=10)
            ax.axis('equal')
        plt.savefig(r'D:\notebooks\HS20200827\crosstalk\%s.png'%(name+'predict'))
        plt.close()

        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_on, 0
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent1 = list(y).count(which)/len(y)
            measure.onwhich[measure.qubitToread[i]] = (which if percent1 > 0.5 else 1-which)
            measure.offwhich[measure.qubitToread[i]] = (1-which if percent1 > 0.5 else which)
            percent_on = list(y).count(measure.onwhich[measure.qubitToread[i]])/len(y)
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|1>pop={round(percent_on*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'D:\notebooks\HS20200827\crosstalk\%s.png'%(name+'e'))
        plt.close()

        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_off, measure.offwhich[measure.qubitToread[i]]
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent_off = list(y).count(which)/len(y)
            measure.readmatrix[measure.qubitToread[i]] = np.mat([[percent_off,1-percent_on],[1-percent_off,percent_on]])
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|0>pop={round(percent_off*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'D:\notebooks\HS20200827\crosstalk\%s.png'%(name+'g'))
        plt.close()
    else:
        fig, axes = plt.subplots(ncols=2,nrows=num,figsize=(9,4*num))
        for i in range(n):
            i = target if target is not None else i
            s_off, s_on = s_st[0,:,i], s_st[1,:,i]
            ss, which = s_on, measure.onwhich[measure.qubitToread[i]] 
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[measure.qubitToread[i]](d)
            percent_on = list(y).count(which)/len(y)
            ax = axes[i//2][i%2] if num>1 else axes[i]
            ax.scatter(np.real(ss),np.imag(ss),c=y,s=10)
            ax.set_title(f'|1>pop={round(percent_on*100,3)}%')
            ax.axis('equal')
        plt.savefig(r'D:\notebooks\HS20200827\crosstalk\%s.png'%(name+'classify'))
        plt.close()


################################################################################
# 读出校准
################################################################################

def readPop(measure,ss,readcali=True):
    matrix = {f'q{i+1}':np.mat(np.eye(2)) for i in range(8)}
    d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
    y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
    pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
    pop_on = np.array([pop[i] if measure.onwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    pop_off = np.array([pop[i] if measure.offwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    mat = measure.readmatrix if readcali else matrix
    pop_cali = [np.array(np.mat(mat[j]).I*np.mat([pop_off[i],pop_on[i]]).T)[:,0] for i, j in enumerate(measure.qubitToread)]
    return pop_cali

