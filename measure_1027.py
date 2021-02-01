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
import numpy as np, sympy as sy, serial, time, datetime
# from easydl import clear_output
from qulab.job import Job
# from qulab.wavepoint import WAVE_FORM as WF
# from collections import Iterable
from tqdm import tqdm_notebook as tqdm
from qulab import computewave_1027 as cww, optimize as op
import pandas as pd

t_list = np.linspace(0,100000,200001)
t_range = (-90e-6, 10e-6)
sample_rate = 2e9

################################################################################
# qubit
################################################################################

class qubit():
    def __init__(self,**kws):
        attribute = ['q_name','inst','T_bias','specfunc','bias','zpulse','f_lo','delta','f_ex','delta_ex','alpha',\
             'power_ex','power_rabi','pi_len','T1','state','timing','envelopename','nwave','amp',\
             'seqtype','detune','shift','phase','phaseDiff','DRAGScaling','volt','offset','vpp']
        for j in attribute:
            self.__setattr__(j,None)
        if len(kws) != 0:
            for i in kws:
                self.__setattr__(i,kws[i])
    def asdict(self):
        return self.__dict__

    def replace(self,**kws):
        for i in kws:
            # if self.__getattribute__(i):
            if hasattr(self,i):
                self.__setattr__(i,kws[i])
            else:
                raise(f'atrribute {i} is not existed')

class common():
    def __init__(self,ad,dc,psg,awg,XY,Z,n,qubits={},caliqubits={},dcstate={}):
        self.qubitToread = []
        self.freqall=[]
        self.ad = ad
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.XY = XY
        self.Z = Z
        self.qubits = {i.q_name:i for i in qubits} 
        self.caliqubits = {i.q_name:i for i in caliqubits} 
        self.wave = {}
        self.delta = np.array([80e6])
        self.com_lst = [f'com{i}' for i in np.arange(3,15)]
        self.inststate = 0
        self.t_list=np.linspace(-43000,5000,96000)*1e-9    ##采样率2G,一个数据点0.5ns,此处50us，100000个数据点
        self.delta_ex = np.array([200e6])
        self.t_reference = 0
        self.ex_toreade = 100
        self.zpulse_toreade = 300
        self.readlen = 2000
        self.trig_count = 500
        self.trig_interval = 260e-6
        self.n = n
        self.dcstate=dcstate
        self.predict={}
        self.onwhich={}
        self.offwhich={}
        self.readmatrix={}

################################################################################
# 比特位置
################################################################################

def q_index(qubits):
    q_index = []
    for qubit in (qubits):
        q_index.append(int(qubit.q_name[1])-1)
    return q_index

def qname_to_qubit(measure,qname):
    for qubit in measure.qubits:
        if qname ==qubit.q_name:
            return qubit
################################################################################
# 关闭通道输出，偏置设为零
################################################################################

def awg_stop(awg):
    for j in [1,2,3,4]:
        awg.da_stop_output_wave(j)

################################################################################
# 读出混频
################################################################################

def resn(f_list):
    f_list = np.array(f_list)
    f_lo = f_list.max() - 80e6
    delta =  -(f_lo - f_list)
    n = len(f_list)
    return f_lo, delta, n
################################################################################
# 将字典dict2合并到字典dict1
################################################################################
def merge(dict1,dict2):
    for k, v in dict2.items():
        dict1[k] = v
################################################################################
# 直流源设置
################################################################################
async def dcManage(measure,bias_mode='zpluse',dcstate={},readstate=None,calimatrix=None,qnum=0):
    matrix = np.mat(np.eye(len(measure.dcstate))) if np.all(calimatrix) == None else np.mat(calimatrix)

    # freqall = measure.freqall
    # if readstate==None:
    #     qubitToread = list(dcstate.keys())
    #     fread = [freqall[i] for i in qubitToread]
    # else:
    #     if len(readstate)==0:
    #         qubitToread = list(dcstate.keys())
    #         fread = [freqall[i] for i in qubitToread]
    #     else:
    #         qubitToread=readstate
    #         fread = [freqall[i] for i in qubitToread]
    # measure.qubitToread = qubitToread
    # if readstate != None:
    #     f_lo, delta, n = resn(np.array(fread))
    #     measure.n, measure.delta, measure.f_lo = n, delta, f_lo
    #     cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 

    dcst={**measure.dcstate,**dcstate}   ##
    bias=np.array([dcst[i] for i in dcst])
    current = matrix.I * np.mat(bias).T
    # print('%s_bias = '%bias_mode,'\n',current)
    # print('dcstate=',dcst)
    for i,j in enumerate(dcst):
        c = current[i,0]
        if bias_mode=='zpluse':
            cww.z_pulse(measure.qubits[j],measure,Zlen=4300e-9,Zamp=c)
        if bias_mode=='DC':
            measure.dc[j].dc(c)
        
################################################################################
# 激励频率计算
################################################################################

async def exMixing(f):
    if f == {}:
        return 
    qname = [i for i in f]
    f_ex = np.array([f[i] for i in f])
    ex_lo = f_ex.max() + 110e6
    delta =  ex_lo - f_ex
    delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
    # n = len(f_ex)
    return ex_lo, delta_ex


# ################################################################################
# # 激励源设置
# ################################################################################

async def exManage(measure,dcstate={},exstate={},calimatrix=None,qnum=8):
    qubits = measure.qubits
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    f_ex1, f_ex2, f_ex3 = {}, {}, {}
    delta_ex1, delta_ex2, delta_ex3 = {}, {}, {}
    for i,j in enumerate(qubits):
        bias[i] = dcstate[j] if j in dcstate else 0
        if j in exstate:
            # Att_Setup(measure,qubits[j].inst['com']).Att(exstate[j])
            if qubits[j].inst['ex_lo'] == 'psg_ex1':
                f_ex1[j] = qubits[j].f_ex[0]
            if qubits[j].inst['ex_lo'] == 'psg_ex2':
                f_ex2[j] = qubits[j].f_ex[0]
            if qubits[j].inst['ex_lo'] == 'psg_ex3':
                f_ex3[j] = qubits[j].f_ex[0]
    if f_ex1 != {}:
        ex_lo1, delta_ex1 = await exMixing(f_ex1)
        await measure.psg['psg_ex1'].setValue('Frequency',ex_lo1)
        await measure.psg['psg_ex1'].setValue('Output','ON')
    if f_ex2 != {}:
        ex_lo2, delta_ex2 = await exMixing(f_ex2)
        await measure.psg['psg_ex2'].setValue('Frequency',ex_lo2)
        await measure.psg['psg_ex2'].setValue('Output','ON')
    if f_ex3 != {}:
        ex_lo3, delta_ex3 = await exMixing(f_ex3)
        await measure.psg['psg_ex3'].setValue('Frequency',ex_lo3)
        await measure.psg['psg_ex3'].setValue('Output','ON')
    delta_ex = {**delta_ex1,**delta_ex2,**delta_ex3}
    current = matrix.I * np.mat(bias).T
    current = {f'q{i+1}':current[i,0] for i in range(qnum)}
    return delta_ex, current

################################################################################
# S21
################################################################################

async def S21(qubit,measure,modulation=False,f_lo=None,f=None):
    #await jpa_switch(measure,state='OFF')
    if f_lo == None:
        f_lo, delta, n = resn(np.array(qubit.f_lo))
        print('f_lo = ',f_lo )
        print(f_lo, delta, n )
        freq = np.linspace(-5,5,201)*1e6 + f_lo
    else:
        freq = np.linspace(-5,5,201)*1e6 + f_lo
        delta, n = measure.delta, measure.n
    if modulation:
        cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 
    await measure.psg['psg_lo'].setValue('Output','ON')
    if f is not None:
        freq = f
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B= measure.ad.getIQ()

        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        yield i+delta, s
    
################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False,flo=None,freq=None):
    length = len(freq) if freq is not None else 201
    job = Job(S21, (qubit,measure,modulation,flo,freq),auto_save=True,max=length,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    print('shape(f_s21)=',np.shape(f_s21),'\n','shape(s_s21)=' ,np.shape(s_s21))
    index = np.abs(s_s21).argmin(axis=0)
    print(index)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = resn(np.array(f_res))
    print(f_lo, delta, n)
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 
        base = 0
        for i in range(15):
            ch_A, ch_B= measure.ad.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 15
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    # yield f_lo, delta, n, f_res, base,f_s21, s_s21
    return f_lo, delta, n, f_res, base,f_s21, s_s21

################################################################################
# S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,freq,bias_mode='zpluse',modulation=False):
    for i in current:
        await dcManage(measure,bias_mode,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)   
        # measure.dc[qubit.q_name].dc(i)
        # print('want_current=%f'%i)
        #z_pulse(qubit,measure,Zamp=30000,Zlen=3000e-9,Zamp2=0,Zlen2=0,offset=0,Crosstalk=0)

        job = Job(S21, (qubit,measure,modulation,measure.f_lo,freq),auto_save=True, no_bar=False)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    await dcManage(measure,bias_mode,dcstate={},readstate=None,calimatrix=calimatrix)

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att_value,freq,modulation=False,bias_mode='zpluse',calimatrix=None):
    l = measure.readlen
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 
    measure.readlen = l
    for i in att_value:
        await measure.att[qubit.inst['att']].set_att(i)
        print(i)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo,freq),auto_save=True, no_bar=False)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# SingleSpec
################################################################################

async def singlespec(qubit,measure,ex_freq,modulation=False,f_read=None,readponit=True,freq=None):
    print(measure.delta)
    if readponit:
       f_lo, delta, n, f_res, base,f_s21, s_s21 = await again(qubit,measure,modulation,f_read,freq)
    else:
        n, base = measure.n, measure.base
    print( measure.delta)
    cww.modulation_ex(qubit,measure)
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in ex_freq:
        await measure.psg['psg_ex'].setValue('Frequency',i)
        ch_A, ch_B= measure.ad.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*n, s-base
    await measure.psg['psg_ex'].setValue('Output','OFF')
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=None)
    awg_stop(measure.awg[qubit.inst['ex_awg']])

################################################################################
# Spec2d  
################################################################################

async def spec2d(qubit,measure,freq,current,calimatrix,bias_mode='zpulse',modulation=False):
    # current = np.linspace(-qubit.T_bias[0]*0.35,qubit.T_bias[0]*0.35,36) + qubit.T_bias[1] 
    n = len(measure.freqall)
    # current = np.linspace(-qubit.bias*0.35,qubit.bias*0.35,8) + qubit.bias
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    print('calimatrix=',calimatrix)
    for i in current:
        dcstate={'q1':3, 'q2':3, 'q3':3, 'q4':3}
        await dcManage(measure,bias_mode,dcstate=dcstate,readstate=None,calimatrix=calimatrix)
        # await dcManage(measure,bias_mode,dcstate={qubit.q_name:i},readstate=None,calimatrix=calimatrix)
        measure.dc[qubit.q_name].dc(i)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,True),auto_save=True,max=len(freq))
        f_ss, s_ss = await job.done()
        print(np.shape(f_ss),np.shape(s_ss))
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

###############################################################################
# Crosstalk
################################################################################

async def CS21(target_qubit,bias_qubit,measure,clist,z_len):
    #波形write
    
    for i in clist:
        cww.z_pulse(bias_qubit,measure,Zamp=i,Zlen=z_len*1e-9,offset=0,Crosstalk=1)
        #采数
        res = measure.ad.getIQ()
        if isinstance(res, int):
           break 
        a = np.mean(res[0], axis=0)  # 对各列求均值
        b = np.mean(res[1], axis=0)
        s = a + 1j*b
        yield [i]*8, s - measure.base
    
async def Crosstalk(target_qubit,bias_qubit,measure,compenlist,biaslist,dcstate={},calimatrix=None):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)   ###  各比特的z偏置均设为零
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)  ## 固定所有比特的直流偏置
    await measure.psg['psg_ex'].setValue('Frequency',target_qubit.f_ex[0])
    z_len = 3400  ##激励微波的长度为z_len-400ns，z_pulse长z_len，且均等包围激励微波,单位为ns
    await measure.psg['psg_ex'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Output','ON')

    for i in compenlist:
        cww.modulation_ex(target_qubit,measure,XYlen=3000e-9,shift=200*1e-9)
        cww.z_pulse(target_qubit,measure,Zamp=i,Zlen=z_len*1e-9,offset=0,Crosstalk=1)

        job = Job(CS21, (target_qubit,bias_qubit,measure,biaslist,z_len),auto_save=True,max=len(biaslist),tags=[bias_qubit.q_name])
        f_ss, s_ss = await job.done()
        yield [i]*8, f_ss, s_ss
    cww.modulation_ex(target_qubit,measure)
    cww.z_pulse(target_qubit,measure,Zamp=0,Zlen=z_len*1e-9,offset=0,Crosstalk=1)
    cww.z_pulse(bias_qubit,measure,Zamp=0,Zlen=z_len*1e-9,offset=0,Crosstalk=1)
    await measure.psg['psg_ex'].setValue('Output','OFF')
    cww.modulation_ex(target_qubit,measure)

################################################################################
# RabiTime
################################################################################

async def Rabi(qubit,measure,t_rabi,calimatrix,dcstate,nwave=1):
    await dcManage(measure,'zpulse',dcstate={qubit.q_name:0},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex-qubit.delta_ex[0]
    print(freq,qubit.delta_ex[0])
    amp = qubit.power_ex
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_rabi:
        cww.rabiWave(qubit, measure,amp=amp, envelopename='gaussian',nwave=nwave,\
            during=i*1e-9,Delta_lo=qubit.delta_ex[0],phase=0,phaseDiff=0,DRAGScaling=None)
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s - measure.base
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# RabiPower
################################################################################

async def RabiPower(qubit,measure, power, calimatrix,dcstate={}):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex - qubit.delta_ex[0]
    print('freq=%s'%freq)
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in power:
        cww.rabiWave(qubit, measure, envelopename='square',nwave=1,\
            during=qubit.pi_len/1e9/2,Delta_lo=qubit.delta_ex[0],amp=i,phase=0,phaseDiff=0,DRAGScaling=None)
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# T1
################################################################################
async def T1(qubit, measure, t_t1, calimatrix,dcstate={}):
    mt = measure.t_list
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex-qubit.delta_ex[0]
    
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cww.rabiWave(qubit, measure, envelopename='cospulse',during=qubit.pi_len/2/1e9,\
            shift=i,Delta_lo=qubit.delta_ex[0],amp=qubit.power_ex)
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        # yield [i]*8, s-measure.base
        ss = res[0] + 1j*res[1]
        pop = cww.readPop(measure,ss)
        yield [i]*8, np.array(pop)[:,1]

    measure.t_list = mt
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# T1_2d
################################################################################
async def T1_2d(qubit, measure, t_rabi, v_rabi,calimatrix,dcstate={}):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit[0].f_ex-qubit[0].delta_ex[0]
    await measure.psg['psg_ex'].setValue('Frequency',freq)

    for j in (v_rabi):
        job = Job(single_tRabi, (qubit, measure, j, t_rabi),auto_save=True,max=len(t_rabi),tags=[qubit[0].q_name])
        t_ss, s_ss = await job.done()
        yield [j]*8, t_ss, s_ss
    await measure.psg['psg_ex'].setValue('Output','OFF')
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    cww.z_pulse(qubit[0],measure,Zamp=0)
    cww.z_pulse(qubit[1],measure,Zamp=0)
################################################################################
# Ramsey
################################################################################

async def Ramsey(qubit, measure, t_t1, calimatrix,dcstate={}):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex-qubit.delta_ex[0]
    amp = qubit.power_ex
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cww.ramseyWave(qubit, measure,envelopename='cospulse', delay=i/1e9, halfpi=qubit.pi_len/2/1e9,fdetune=1e6,amp=amp)
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s - measure.base
        # ss = res[0] + 1j*res[1]
        # pop = cww.readPop(measure,ss)
        # yield [i]*8, np.array(pop)[:,1]
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')

###############################################################################
# Spin echo
################################################################################

async def SpinEcho(qubit, measure, t_t1, seqtype,calimatrix,dcstate={}):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)

    await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Output','ON')
    amp = qubit.power_ex
    freq = qubit.f_ex-qubit.delta_ex[0]
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cww.coherenceWave(qubit, measure, envelopename='cospulse',t_run=i*1e-9,during=qubit.pi_len/2*1e-9,amp=amp,\
            n_wave=1,seqtype=seqtype,detune=1e6,shift=0, Delta_lo=qubit.delta_ex[0])
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s - measure.base
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')


################################################################################
# 真空拉比
################################################################################
async def single_vRabi(qubit, measure, v_rabi, t_rabi):
    for i in (v_rabi):
        cww.z_pulse(qubit[0],measure,Zamp=i,Zlen=t_rabi*1e-9,shift=qubit[0].pi_len*1e-9+200e-9)
        cww.rabiWave(qubit[0], measure, envelopename='cospulse',during=qubit[0].pi_len/2*1e-9,\
            shift=t_rabi*1e-9,Delta_lo=qubit[0].delta_ex[0],amp=qubit[0].power_ex )
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-measure.base

async def single_tRabi(qubit, measure, v_rabi, t_rabi):
    for i in (t_rabi):
        cww.z_pulse(qubit[0],measure,Zamp=v_rabi,Zlen=i*1e-9,shift=qubit[0].pi_len*1e-9+200e-9)
        cww.rabiWave(qubit[0], measure, envelopename='cospulse',during=qubit[0].pi_len/2*1e-9,\
            shift=i*1e-9,Delta_lo=qubit[0].delta_ex[0],amp=qubit[0].power_ex )
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-measure.base

async def vRabi(qubit, measure, t_rabi, v_rabi,calimatrix,dcstate={}):
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit[0].f_ex-qubit[0].delta_ex[0]
    await measure.psg['psg_ex'].setValue('Frequency',freq)

    for j in (t_rabi):
        job = Job(single_vRabi, (qubit, measure, v_rabi, j),auto_save=True,max=len(v_rabi),tags=[qubit[0].q_name])
        t_ss, s_ss = await job.done()
        yield [j]*8, t_ss, s_ss
    awg_stop(measure.awg[qubit[0].inst['ex_awg']])
    awg_stop(measure.awg[qubit[1].inst['ex_awg']])
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# timing  
################################################################################
async def z_timing(qubit,measure,t_rabi,v_rabi):
    for i in (t_rabi):
        cww.z_pulse(qubit,measure,Zamp=v_rabi,Zlen=qubit.pi_len*1e-9,shift=400e-9)
        cww.rabiWave(qubit, measure, envelopename='gaussian',during=qubit.pi_len/2*1e-9,\
            shift=i,Delta_lo=qubit.delta_ex[0],amp=qubit.power_ex)
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-measure.base
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    cww.z_pulse(qubit,measure,Zamp=0)

async def read_timing(measure,qubit,t_list,ex_delta=0,nwave=1):
    for i in t_list:
        cww.modulation_read(measure, measure.delta,readlen=measure.readlen,phlen=100*1e-9,ac_stark=1)
        cww.rabiWave(qubit, measure, envelopename='gaussian',during=qubit.pi_len/2*1e-9,\
            shift=i,Delta_lo=qubit.delta_ex[0]+ex_delta,amp=qubit.power_ex )
        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-measure.base
    awg_stop(measure.awg[qubit.inst['ex_awg']])
    cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 


################################################################################
# Z_pulse补偿  
################################################################################
async def z_comwave_single(qubit,measure,t_rabi,v_rabi,calimatrix,dcstate={}):
        for i in t_rabi:
            cww.rabiWave(qubit, measure, envelopename='cospulse',during=qubit.pi_len/2*1e-9,\
                shift=i,Delta_lo=qubit.delta_ex[0],amp=qubit.power_ex)
            res = measure.ad.getIQ()
            Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
            s = Am + 1j*Bm
            ss = res[0] + 1j*res[1]
            pop = cww.readPop(measure,ss)
            yield [i]*8, np.array(pop)[:,1]
            # yield [i]*8, s-measure.base

async def z_comwave(qubit,measure,t_rabi,v_rabi,calimatrix,dcstate={}):
    await dcManage(measure,'DC',dcstate=dcstate,readstate=None,calimatrix=None)

    freq = qubit.f_ex-qubit.delta_ex
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in (v_rabi):
        cww.z_comwave(qubit,measure,Zamp=i,shift=qubit.pi_len*1e-9+200e-9)
        # cww.z_pulse(qubit,measure,Zamp=i,Zlen=1000*1e-9,shift=400e-9)
        job = Job(z_comwave_single, (qubit,measure,t_rabi,v_rabi,calimatrix,dcstate), max=len(t_rabi),auto_save=True)
        t_ss, s_ss = await job.done()
        yield [i]*8, t_ss, s_ss
    awg_stop(measure.awg[qubit.inst['ex_awg']])
################################################################################
# Spec2d_ex  
################################################################################

async def spec2d_ex(qubit,measure,freq,current,calimatrix,modulation=False):
    # await measure.dc[qubit.q_name].DC(0.4)
    await dcManage(measure,'zpulse',dcstate={},readstate=None,calimatrix=calimatrix)
    print('calimatrix=',calimatrix)
    for i in current:
        await measure.psg['psg_ex'].setValue('Power',i)
        # await measure.dc[qubit.q_name].DC(i)
        job = Job(singlespec, (qubit,measure,freq,modulation,measure.f_lo,True),auto_save=True,max=len(freq))
        f_ss, s_ss = await job.done()
        print(np.shape(f_ss),np.shape(s_ss))
        yield [i]*8, f_ss, s_ss

################################################################################
# 对比度
################################################################################

async def visibility(n,s0,s1):
    theta = np.arange(0, 2*np.pi, 0.01)
    data = []
    for i in range(n):
        c0, c1 = np.mean(s0), np.mean(s1)
        s0 = s0 / ((c1-c0)/np.abs(c1-c0))
        s1 = s1 / ((c1-c0)/np.abs(c1-c0))
        s0 = np.real(s0)
        s1 = np.real(s1)
        bins = np.linspace(np.min(np.r_[s0,s1]), np.max(np.r_[s0,s1]), 61)
        y0,_ = np.histogram(s0, bins=bins)
        y1,_ = np.histogram(s1, bins=bins)
        inte0 = np.cumsum(y0)/np.sum(y0)
        inte1 = np.cumsum(y1)/np.sum(y0)
        inte_diff = np.cumsum(y0)/np.sum(y0) - np.cumsum(y1)/np.sum(y1)
        offstd, onstd = np.std(s0), np.std(s1)
        roff = np.real(c0) + offstd * np.cos(theta)
        ioff = np.imag(c0) + offstd * np.sin(theta)
        ron = np.real(c1) + onstd * np.cos(theta)
        ion = np.imag(c1) + onstd * np.sin(theta)
        data.append([inte0,inte1,inte_diff,(roff,ioff),(ron,ion)])
    return data

################################################################################
# 临界判断
################################################################################

async def threshHold(measure,qubit,modulation=True):
    cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 
    cww.ats_setup(measure,measure.readlen,repeats=5000)
    
    if modulation:
        cww.rabiWave(qubit, measure, envelopename='cospulse',during=qubit.pi_len/2*1e-9,\
            shift=0*1e-9,Delta_lo=qubit.delta_ex[0],amp=qubit.power_ex)

    for j, i in enumerate(['OFF','ON']):
        await measure.psg['psg_ex'].setValue('Output',i)
        time.sleep(2)
        res = measure.ad.getIQ()
        Am, Bm = res[0],res[1]
        s = Am + 1j*Bm
        yield j, s
    cww.ats_setup(measure,measure.readlen,repeats=measure.trig_count) 
    awg_stop(measure.awg[qubit.inst['ex_awg']])

################################################################################
# 优化pi脉冲
################################################################################
async def pipulseOpt(qubit,measure,calimatrix,dcstate,nwave,wavlen):
    pilen = qubit.pi_len
    t = np.linspace(0.5*pilen-10,0.5*pilen+10,wavlen)
    for i in range(nwave):
        job = Job(Rabi, (qubit,measure,t,calimatrix,dcstate,4*i+1), max=500,auto_save=True)
        t_r, s_r = await job.done()
        yield [4*i+1]*measure.n, t_r, s_r

################################################################################
# 优化读出点 
################################################################################

async def readOp(measure,qubit,modulation=True):
    if modulation:
        cww.rabiWave(qubit, measure, envelopename='cospulse',during=qubit.pi_len/2*1e-9,\
            shift=0*1e-9,Delta_lo=qubit.delta_ex[0],amp=qubit.power_ex )
    for j, i in enumerate(['OFF','ON']):
        await measure.psg['psg_ex'].setValue('Output',i)
        time.sleep(2)
        job = Job(S21, (qubit,measure,True,measure.f_lo),auto_save=True, no_bar=False)
        f_s21, s_s21 = await job.done()
        yield [i]*8, f_s21, s_s21
    awg_stop(measure.awg[qubit.inst['ex_awg']])
################################################################################
# 优化读出功率 （待调整）
################################################################################

async def readpowerOpt(measure,which,readamp):
    n = len(measure.delta)
    for k in readamp:
        measure.readamp = k
        cww.modulation_read(measure, measure.delta, readlen=measure.readlen) 
        cww.ats_setup(measure,measure.readlen,repeats=5000)

        res = measure.ad.getIQ()
        Am, Bm = res[0],res[1]
        s = Am + 1j*Bm
        d = [list(zip(np.real(s)[:,i],np.imag(s)[:,i])) for i in range(n)]
        y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
        # d = list(zip(np.real(ss[:,0]),np.imag(ss[:,0])))
        # y = measure.predict[measure.qubitToread[0]](d)
        # pop = list(y).count(which[0])/len(y)
        yield [k]*n, pop

################################################################################
# ac_stark （待调整）
################################################################################
async def single_Ac_stark(measure,qubit,t_list,ex_delta=0,nwave=1):
    for i in t_list:
        cww.rabiWave(qubit, measure, envelopename='gaussian',during=qubit.pi_len/2*1e-9,\
            shift=i,Delta_lo=qubit.delta_ex[0]+ex_delta,amp=qubit.power_ex)

        res = measure.ad.getIQ()
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-measure.base

async def Ac_stark(measure,qubit,t_list,ex_delta):
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    cww.modulation_read(measure, measure.delta,readlen=measure.readlen,phlen=500e-9,phshift=3000e-9,ac_stark=1)
    freq = qubit.f_ex-qubit.delta_ex[0]
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in ex_delta:
        job = Job(single_Ac_stark, (measure,qubit,t_list,i), max=len(t_list),auto_save=True)
        t_ac, s_ac = await job.done()
        yield [i]*8+qubit.f_ex[0], t_ac, s_ac
    cww.modulation_read(measure, measure.delta,readlen=measure.readlen,ac_stark=0)
    await measure.psg['psg_ex'].setValue('Output','OFF')
    awg_stop(measure.awg[qubit.inst['ex_awg']])

async def ZZ_coupler(measure,qubits,z_len,z_amp,on_off):
    # cww.z_pulse(qubits[1],measure,width=i*1e-9,amp=v_rabi,channel_output_delay=89.9e3-i)
    if on_off == True:
        cww.rabiWave(qubits[0], measure, envelopename='cospulse',during=qubits[0].pi_len/2*1e-9,\
            shift=0*1e-9,Delta_lo=qubits[0].delta_ex[0],amp=qubits[0].power_ex)
    
     

    

