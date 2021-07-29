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
from itertools import chain, combinations, count, product, repeat
from collections import Counter, OrderedDict, defaultdict, deque, namedtuple, Iterable

import numpy as np, sympy as sy, serial, time, datetime, pickle
from numpy.lib.function_base import iterable
# from easydl import clear_output
from qulab.job import Job
from qulab.tomo import qst, tomography
# from qulab.wavepoint import WAVE_FORM as WF
from collections.abc import Callable
from functools import reduce
from tqdm import tqdm_notebook as tqdm
from qulab.yhs import waveform_new as wn
from qulab.yhs import computewave_wave, optimize, dataTools, qGate_basewave, imatrix
import pandas as pd
from qulab.storage.utils import save
import asyncio, imp, scipy, logging, time
cww = imp.reload(computewave_wave)
op = imp.reload(optimize)
dt = imp.reload(dataTools)
mx = imp.reload(imatrix)
qgw = imp.reload(qGate_basewave)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=r'D:\skzhao\log\qubit.log', level=logging.DEBUG, format=LOG_FORMAT)

# logging.debug("This is a debug log.")
# logging.info("This is a info log.")
# logging.warning("This is a warning log.")
# logging.error("This is a error log.")
# logging.critical("This is a critical log.")

t_new = np.linspace(-43000,5000,96000)*1e-9
# t_new = np.linspace(-15000,5000,40000)*1e-9
t_list = t_new*1e9 - np.min(t_new)*1e9
t_range = (-90e-6, 10e-6)
sample_rate = 2e9
mTrig_ex_lo = wn.square(len(t_new)/sample_rate-10e-6) << (len(t_new)/sample_rate/2-4.5e-6)

################################################################################
# qubit
################################################################################

class qubit():
    def __init__(self,**kws):
        attribute = ['index','q_name','iqm','inst','T_bias','T_z','specinterp','specfunc','specfuncz','specfunc_cavity','photonnum_func','chi','acstark_shift','bias','zpulse','f_lo','delta','f_ex','delta_ex','alpha',\
             'power_ex','power_rabi','pi_len','pi_len2','during','T1','state','timing','envelopename','nwave','readamp','ringup','ringupamp',"virtualPhase",\
                 'weight','readmatrix','readvolt','amp','amp2','zCali','dressenergy','bessel','volt_swap','rise','voltn','during_swap','coordinatePhase',\
             'seqtype','detune','shift','phaseim','imAmp','imAmpn','imAmp_off','phase','ampDiff','phaseDiff','DRAGScaling','volt','offset','vpp','volt_zgate','volt_zgate_b','gateduring',\
                 'during_cz','volt_cz','g_cq','f_max','f_cross','f_cc','pi_lenq','alphaq']
        for j in attribute:
            self.__setattr__(j,None)
        if len(kws) != 0:
            for i in kws:
                self.__setattr__(i,kws[i])

    def asdict(self):
        return self.__dict__

    def replace(self,**kws):
        for i in kws:
            assert hasattr(self,i)
            self.__setattr__(i,kws[i])

    def getvalue(self,name):
        return self.__getattribute__(name)

    def setvalue(self,name,value):
        self.__setattr__(name,value)


################################################################################
# IQ-Mixer
################################################################################

class IQMixer():
    def __init__(self,**kws):
        attribute = ['index','q_name','inst','lo','image']
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
    global t_list, t_new
    def __init__(self,freqall={},ats=None,dc=None,psg=None,awg=None,attinst=None,jpa=None,qubits={},ats2=None,iqmixers={}):
    
        self.freqall = freqall
        self.ats = ats
        self.ats2 = ats2
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = {i.q_name:i for i in qubits} 
        self.iqmixers = {i.q_name:i for i in iqmixers} 
        self.wave = {}
        self.attinst = attinst
        self.com_lst = [f'com{i}' for i in np.arange(3,15)]
        self.inststate = 0
        self.t_list = t_list
        self.t_new = t_new
        self.t_range = (-90e-6,10e-6)
        self.sample_rate = 2e9
        self.readamp = [0.08]*10
        self.ringup = [100]*10
        self.ringupamp = [0.1]*10
        self.mode = 'hbroadcast'
        self.steps = 101
        self.onwhich = {} 
        self.offwhich = {}
        self.predict = {}
        self.readmatrix = {}
        self.postSle = {}
        self.ad_trig_delay= 43348e-9
        
class qubitCollections():
    def __init__(self,qubits,q_target=None):
        self.qubits = {i.q_name:i for i in qubits}
        if q_target != None:
            qasdict = self.qubits[q_target]._asdict()
            for i in qasdict:
                # if not hasattr(self,i):
                self.__setattr__(i,qasdict[i])
        else:
            self.inst = {i.q_name:i.inst for i in qubits}
            self.q_name = 'allqubits'

        self.f_lo = np.array([i.f_lo[0] for i in qubits])
        self.f_ex = np.array([i.f_ex[0] for i in qubits])

    def qubitExecute(self,q_target=[]):
        q = []
        for i in q_target:
            q.append(self.qubits[i])
        return q

################################################################################
# 设置衰减器
################################################################################

class Att_Setup():
    
    def __init__(self,measure=None,com='com7'):
        if measure == None:
            measure = simpleclass()
            measure.att = {}
        self.com = com
        try:
            ser = serial.Serial(self.com,baudrate=115200, parity='N',bytesize=8, stopbits=1, timeout=1)
            self.ser = ser
        except:
            self.ser = measure.att[com]
        measure.att[com] = self.ser
        if self.ser.isOpen():    # make sure port is open     
            print(self.ser.name + ' open...')
            # self.ser.write(b'*IDN?\n')
            # x = self.ser.readline().decode().split('\r''\n')
            # print(x[0])
            self.ser.write(b'ATT?\n')
            y = self.ser.readline().decode().split('\r''\n')
            print('last ATT',y[0])

    def close(self):
        self.ser.close()

    def Att(self,att,closeinst=True):
        self.ser.write(b'ATT %f\n'%att)
        time.sleep(1)
        self.ser.write(b'ATT?\n')
        y = self.ser.readline().decode().split('\r''\n')
        print('now ATT',y[0])

        if closeinst:
            self.close()

################################################################################
# 调度器
################################################################################

async def dispatcher(measure,whichexe,processlist,paraslist,tags=[],maxlist=[500],update_state=[False],auto_save=True):
    for j in whichexe:
        title = processlist[j].__name__
        job = Job(processlist[j], (measure,),paraslist[j], tags=tags,max=maxlist[j], auto_save=auto_save)
        data = await job.done()
        res_op = op.exeFit(measure,title,data,paraslist[j])
        for k in measure.qubits:
            if k in list(res_op.keys()):
                attr = res_op[k] 
                measure.qubits[k].replace(**attr)
            if update_state[j]:
                if k in measure.qubitToread:
                    state = await cww.QueryInst(measure)
                    measure.qubits[k].replace(state=[state,measure.delta,measure.base,measure.readlen])
        # clear_output()
    return title, data
################################################################################
# Index
################################################################################

def index_max(f,rg,N,dM):   ###f一维元组，rg剪掉的范围，N寻找的次数，极值点的占最大值的比率
    f1 = [0]*(N+1)
    f1[0] = f.copy()
    f0 = f.copy()
    index = []
    f_max = max(abs(f0))
    for i in np.arange(N):
        MP = np.argmax(abs(f1[i]))
        fM = max(abs(f1[i]))
        if fM < dM*f_max:
            break
        index.append(MP)
        start = max(0,MP-rg)
        end = min(len(f)-1,MP+rg)
        # L = len(f1[i][start:end])
        for j in np.arange(start,end+1,1):
            f1[i][j] = 0 
        f1[i+1] = f1[i].copy()
    index=sorted(index)
    return index
    
def index_merge(index1,index2,length,rg=100):
    a = []
    for i in index1:
        for k in index2:
            start0 = k
            end0 = 2*k-i
            start = max(min(start0.copy(),end0.copy()),0)
            end = min(max(start0.copy(),end0.copy()),length-1)
            if start0 == end0:
                start = max(start0.copy()-rg,0)
                end = min(end0.copy()+rg,length-1)
            a.append([start,end])
    t = sorted(a.copy())
    b=[t.copy()[0]]
    for i in np.arange(len(a)):
        if b[-1][1]<t[i][0] or b[-1][0]>t[i][1]:
            b.append(t[i])
        else:
            b0=b.copy()
            b[-1][1]=max(t[i][1],b0[-1][1])
            b[-1][0]=min(t[i][0],b0[-1][0])
    return b

################################################################################
# 打印日志
################################################################################
        
def printLog(info,mode='w'):
    with open(r'D:\skzhao\log\qubit.log', mode=mode) as filename:
        for i in info:
            filename.write(str(i)+':')
            filename.write(str(info.get(i)))
            filename.write('\n')

async def printState(measure,title,mode='w'):
    state = await cww.QueryInst(measure)
    t = time.strftime('%Y') / time.strftime('%m%d') / f"{time.strftime('%Y%m%d%H%M%S')}"
    with open(r'D:\skzhao\log\state.log', mode=mode) as filename:
        filename.write('measure'+':'+title)
        filename.write('\n')
        filename.write('measure time'+':'+t)
        filename.write('\n')
        for i in state:
            filename.write(str(i)+':')
            filename.write(str(state.get(i)))
            filename.write('\n')

################################################################################
# 并行处理
################################################################################
'''
    此处代码及其啰嗦，有空修整
'''
def namepack(task):
    # task_run1 = []
    for i in task:
        keyname = list(i.keys())
        task1, task2, task3, task4 = [], [], [], []
        for j in keyname:
            if 'awg131' in j:
                task1.append(i[j])
            if 'awg132' in j:
                task2.append(i[j])
            if 'awg133' in j:
                task3.append(i[j])
            if 'awg134' in j:
                task4.append(i[j])
#         print(task1, task2, task3, task4)
        for k in range(8):
            task_run1 = []
            if len(task1) == len(task2) == len(task3) == len(task4) == 0:
                continue
            if len(task1) != 0:
                task_run1.append(task1.pop(0))
            if len(task2) != 0:
                task_run1.append(task2.pop(0))
            if len(task3) != 0:
                task_run1.append(task3.pop(0))
            if len(task4) != 0:
                task_run1.append(task4.pop(0))
#             print(task1, task2, task3, task4)
            yield task_run1

        # task_run1.append((task1,rask2,task3,task4))
    


async def main(task):
    # loop = asyncio.get_running_loop()
    # loop.run_until_complete(asyncio.wait(task))

    for i in task:
        await i
    #     task_run = asyncio.create_task(i)
    #     await task_run

    # try:
    #     result = await asyncio.gather(*task,loop=loop,return_exceptions=True)    ###最初用的这个
    #     # result = await asyncio.wait(task,loop=loop)
    #     # print(result)
    # except IndexError:
        # pass
            
async def concurrence(task):
    # asyncio.run(main([cww.openandcloseAwg(measure,'OFF')])) 
    # await cww.openandcloseAwg('ON')
    f = namepack(task)
    
    for i in f:
        if len(i) < 1:
            continue
        else:
            await main(i)
            # asyncio.run(main(i))    ###最初用的这个
 
    # asyncio.run(main([cww.openandcloseAwg(measure,'ON')])) 
    # loop = asyncio.get_event_loop()
    # for i in task:
    #     if i == []:
    #         continue
    #     else:
    #         # print('loop')
    #         # # loop = asyncio.get_event_loop()
    #         # try:
    #         loop.run_until_complete(asyncio.wait(i))
    #         # except Exception as e:
    #         #     pass
    #         #     print(e)
    #         #     # 引发其他异常后，停止loop循环
    #         #     lp.stop()
    #         # finally:
    #         #     # 不管是什么异常，最终都要close掉loop循环
    #         #     loop.close()
    # # loop.close()

################################################################################
# 获取数据
################################################################################

def getSamplesPerRecode(numOfPoints):
    samplesPerRecord = numOfPoints 
    return samplesPerRecord

def getExpArray(f_list, numOfPoints, weight=None, sampleRate=1e9):
    samplesPerRecord = getSamplesPerRecode(numOfPoints)
    t = np.arange(0, samplesPerRecord, 1) / sampleRate
    e = []
    for i,f in enumerate(f_list):
        if weight is None:
            weight_m = np.ones(samplesPerRecord)
        else:
            weight_m = np.asarray(weight[i])
        e.append(weight_m * np.exp(-1j * 2 * np.pi * f * t))
    return np.asarray(e).T

async def yieldData(measure,avg=False,fft=True,offset=True,hilbert=False,is2ch=False,filter=None):

    A_lst, B_lst = measure.ats.getTraces(avg=True,fft=False)
    # if filter is not None:
    #     A_lst = op.RowToRipe().smooth(A_lst,f0=filter,axis=1)
    #     B_lst = op.RowToRipe().smooth(B_lst,f0=filter,axis=1)
    # weight = []
    # for j, i in enumerate(measure.qubitToread):
    #     qubit = measure.qubits[i]
    #     weight.append(qubit.weight)
    e = getExpArray(measure.delta,measure.readlen,None)
    n = e.shape[0]
    # if hilbert:
    #     Analysis_cos = scipy.signal.hilbert(A_lst,axis=1)
    #     Analysis_sin = scipy.signal.hilbert(B_lst,axis=1)
    #     # theta = np.angle(Analysis_cos) - np.angle(Analysis_sin)
    #     # Analysis_sin *= np.exp(1j*(theta))
    #     # A_lst, B_lst = (np.real(Analysis_cos) + np.real(Analysis_sin)), (np.imag(Analysis_cos) + np.imag(Analysis_sin)) 
    #     if is2ch:
    #         A_lst, B_lst = (np.real(Analysis_cos) - np.imag(Analysis_sin)), (np.imag(Analysis_cos) + np.real(Analysis_sin))
    #     else: 
    #         A_lst, B_lst = np.real(Analysis_cos), np.imag(Analysis_cos)
    #         A_lst_Q, B_lst_Q = np.real(Analysis_sin), np.imag(Analysis_sin)
    if fft:
        # print(np.shape(A_lst),np.shape(e))
        A_lst = (A_lst[:n]).dot(e)
        B_lst = (B_lst[:n]).dot(e)
        # if hilbert and is2ch == False:
        #     A_lst_Q = (A_lst_Q[:, :n]).dot(e)
        #     B_lst_Q = (B_lst_Q[:, :n]).dot(e)
    # if avg:
    #     if hilbert and is2ch == False:
    #         return A_lst.mean(axis=0), B_lst.mean(axis=0), A_lst_Q.mean(axis=0), B_lst_Q.mean(axis=0)
    #     else:
    #         return A_lst.mean(axis=0), B_lst.mean(axis=0)
    # else:
    #     if hilbert and is2ch == False:
    #         return A_lst, B_lst, A_lst_Q, B_lst_Q
    #     else:
    #         return A_lst, B_lst
    return A_lst, B_lst
    

################################################################################
# 读出校准
################################################################################

def coRead(measure,ss,readcali=True):
    state = product(range(2), repeat=len(measure.qubitToread))
    matrix = {f'q{i+1}':np.mat(np.eye(2)) for i in range(10)}
    mat = measure.readmatrix if readcali else matrix

    d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
    y = np.array([measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]).T
    repeats = np.shape(y)[0]

    coMatrix = 1
    onwhich = []
    for i in measure.qubitToread:
        coMatrix = np.kron(coMatrix,mat[i])
        onwhich.append(measure.onwhich[i])
    
    onwhich = np.array([onwhich]*repeats)
    index = ((y+onwhich) % 2 == 0) * 1

    c = Counter(map(tuple, index))
    pop = []
    for i in state:
        pop.append(c[tuple(i)]/repeats)

    pop_cali = coMatrix.I*np.mat(pop).T
    return np.array(pop_cali)[:,0]


def readPop(measure,ss,readcali=True):
    matrix = {f'q{i+1}':np.mat(np.eye(2)) for i in range(10)}
    d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
    y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]

    pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
    pop_on = np.array([pop[i] if measure.onwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    pop_off = np.array([pop[i] if measure.offwhich[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
    mat = measure.readmatrix if readcali else matrix
    pop_cali = [np.array(np.mat(mat[j]).I*np.mat([pop_off[i],pop_on[i]]).T)[:,0] for i, j in enumerate(measure.qubitToread)]
    return np.array(pop_cali)


async def popRead(measure,pop=False):

    if pop:
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop1=readPop(measure,ss,readcali=True)
        return np.array(pop1)[:,1]
    else:
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        return s-measure.base
    
################################################################################
# AWG恢复设置
################################################################################

async def resetAwg(awg):
    global t_new
    for i in awg:  
        await awg[i].setValue('Run Mode','Triggered')   # Triggered, Continuous
        for j in range(8):
            await cww.genwaveform(awg[i],[f'ch{j+1}'],[j+1])
            await awg[i].update_waveform(np.zeros((len(t_new),)),f'ch{j+1}')
            await awg[i].setValue('Vpp',1.5,ch=j+1)
            await awg[i].setValue('Offset',0,ch=j+1)
            await awg[i].write('OUTPUT%d:WVALUE:ANALOG:STATE %s'%(j+1,'FIRST'))  #FIRST, ZERO
            await awg[i].write('SOUR%d:TINP %s'%((j+1),'ATR'))        
            for k in range(4):
                await awg[i].write('OUTPUT%d:WVALUE:MARKER%d %s'%(j+1,(k+1),'FIRST'))            #FIRST, LOW, HIGH
        # if i == 'awg_trig':
        #     await cww.genwaveform(awg[i],['Readout_Q'],[5])
        #     await cww.couldRun(measure,awg[i])
        #     await awg[i].write('TRIGg:SOUR %s'%'INT')   
        #     await awg[i].write('TRIGGER:INTERVAL %f'%260e-6) 
        #     for m in range(8):
        #         await awg[i].write('SOUR%d:TINP %s'%((m+1),'ITR'))

################################################################################
# 等效磁通计算
################################################################################

def voltTophi(qubit,bias,distance=0,freq=0):
    tm = np.arange(-qubit.T_bias[0]/2,0,0.001)
    t = tm + qubit.T_bias[1] if qubit.T_bias[1] > bias else -tm + qubit.T_bias[1]
    x, func = sy.Symbol('x'), qubit.specfunc
    y = sy.lambdify(x,func,'numpy')
    spec = y(t)
    bias_deviate = t[np.abs(spec-freq).argmin()] - bias
    f_bias = y(bias)
    f_distance = y(bias+distance)
    f_deviate = f_distance - f_bias
    return f_bias, f_deviate, bias_deviate

################################################################################
# 腔频率设置
################################################################################

async def resn(f_cavity):
    # f_lo = f_cavity.min() - 20e6 
    f_lo = f_cavity.mean() - 80e6
    if len(f_cavity) == 1:
        f_lo = f_cavity.mean() - 80e6
    delta =  -(f_lo - f_cavity )
    n = len(f_cavity)
    return f_lo, delta, n

################################################################################
# 直流源设置  IQ
################################################################################

async def dcManage(measure,dcstate={},readstate=[],calimatrix=None,qnum=10):
    matrix = np.mat(np.eye(qnum)) if np.all(calimatrix) == None else np.mat(calimatrix)
    bias = [0] * qnum
    fread = []
    qubitToread = []
    for i,j in enumerate(measure.freqall):
        bias[i] = dcstate[j] if j in dcstate else 0
        if readstate != None:
            if readstate == []:
                if j in dcstate:
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
            else:
                if j in readstate: 
                    qubitToread.append(j)
                    fread.append(measure.freqall[j])
    if readstate != None:
        measure.qubitToread = qubitToread
        f_lo, delta, n = await resn(np.array(fread))
        measure.n, measure.delta, measure.f_lo = n, delta, f_lo
        await cww.modulation_read(measure,delta,readlen=measure.readlen,repeats=measure.repeats)
        await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    current = matrix.I * np.mat(bias).T
    for i,j in enumerate(measure.freqall):
        await measure.dc[j].DC(float(round(current[i,0],3)))

    # return f_lo, delta, n

################################################################################
# 激励频率计算
################################################################################

async def exMixing(f):
    if len(f) <=1:
        return 
    if len(f)>1:
        qname = [i for i in f]
        f_ex = np.array([f[i] for i in f])
        # ex_lo = f_ex.mean() - 100e6  #161e6
        ex_lo = f_ex.mean() - 180e6
        delta =  -(ex_lo - f_ex)
        delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
        # n = len(f_ex)
        return ex_lo, delta_ex

################################################################################
# 激励源设置
################################################################################

async def exManage(measure,exstate=[],qnum=10):
    qubits = measure.qubits
    if len(exstate) == 1:
        q_target = qubits[exstate[0]]
        ex_lo = q_target.f_ex - q_target.delta_ex
        delta_ex = {exstate[0]:q_target.delta_ex}
        await measure.psg[q_target.inst['ex_lo']].setValue('Frequency',ex_lo)
        # x = await measure.psg['psg_ex1'].getValue('Frequency')
        await measure.psg[q_target.inst['ex_lo']].setValue('Output','ON')
    else:
        f_ex1, f_ex2, f_ex3 = {}, {}, {}
        delta_ex1, delta_ex2, delta_ex3 = {}, {}, {}
        for i,j in enumerate(qubits):
            if j in exstate:
                if qubits[j].inst['ex_lo'] == 'psg_ex1':
                    f_ex1[j] = qubits[j].f_ex
                if qubits[j].inst['ex_lo'] == 'psg_ex2':
                    f_ex2[j] = qubits[j].f_ex
                if qubits[j].inst['ex_lo'] == 'psg_ex3':
                    f_ex3[j] = qubits[j].f_ex
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

    return delta_ex

async def zManage(measure,dcstate={},calimatrix=None,qnum=10):
    len_data = 0
    for i,j in enumerate(dcstate):
        x = dcstate[j]
        if not isinstance(dcstate[j],Iterable):
            x = np.array([dcstate[j]])
        dcstate[j] = x
        len_data = len(x)
            
    bias_dict = {}
    qubits = measure.qubits
    if qnum < 10:
        qubits = {}
        for i in measure.qubits:
            if i in dcstate:
                qubits[i] = measure.qubits[i]
    for i,j in enumerate(qubits):

        bias_dict[j] = np.array(dcstate[j]) if j in dcstate else np.array([0]*len_data)

    bias = np.array(list(bias_dict.values()))

    matrix = np.mat(np.eye(qnum)) if calimatrix is None else np.mat(calimatrix)
        
    current = np.array(matrix.I * np.mat(bias))
    current = {j:current[i,:] for i,j in enumerate(qubits)}
    return current

################################################################################
# 执行激励并行处理
################################################################################

async def executeEXwave(measure,update_wave,exstate=[],output=False,**paras):
    delta_ex = await exManage(measure,exstate=exstate)
    bit = measure.qubits
    task1, task2, task3, namelist, awglist = {}, {}, {}, [], []
    for i in delta_ex:
        qubit = bit[i]
        taskname = ''.join((qubit.inst['ex_awg'],'_ch',str(qubit.inst['ex_ch'][0])))
        qubit.delta_ex = delta_ex[i]
        chlist = qubit.inst['ex_ch']
        awg = measure.awg[qubit.inst['ex_awg']]
        # pulselist = await cww.funcarg(update_wave,qubit,**paras)
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        task_ready = await cww.writeWave(measure,awg,ch=chlist,pulse=pulselist)
        # task1.append(task_ready)
        task1[taskname] = task_ready
        # if output:
        #     task_manageawg = cww.couldRun(measure,awg,chlist,namelist)
        #     # task2.append(task_manageawg)
        #     task2[taskname] = task_manageawg
        # else:
        #     task_manageawg = cww.couldRun(measure,awg)
        #     # task2.append(task_manageawg)
        #     task2[taskname] = task_manageawg
    # taskread = ''.join(('awg133','_read'))
    # task_read = cww.Trig_wave(qubit,measure,**paras)
    # task1[taskread] = task_read
    # read_run = cww.couldRun(measure,measure.awg['awgread'])
    # task2[taskread] = read_run

    return [task1]

################################################################################
# 执行z并行处理
################################################################################

async def executeZwave(measure,update_wave,dcstate={},qnum=10,calimatrix=None,output=True,args='volt',**paras):
    current = await zManage(measure,dcstate=dcstate,calimatrix=calimatrix,qnum=qnum) if calimatrix is not None else dcstate
    if dcstate == {}:
        current = {f'q{i+1}':0 for i in range(qnum)}
    # print(current)
    bit = measure.qubits
    task1, task2, awglist = {}, {}, []
    for i in current:
        qubit = bit[i]
        # print(qubit.q_name)
        taskname = ''.join((qubit.inst['z_awg'],'_ch',str(qubit.inst['z_ch'][0])))
        zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
        awg = measure.awg[qubit.inst['z_awg']]
        # if 'volt' in paras:
        #     pulselist = await cww.funcarg(update_wave,qubit,**paras)
        # else:
        paras[args] = current[i]
        pulselist = await cww.funcarg(update_wave,qubit,**paras)
        task_ready = await cww.writeWave(measure,awg,zch,pulse=pulselist)
        # task1.append(task_ready)
        task1[taskname] = task_ready
        # if output:
        #     task_manageawg = await cww.couldRun(measure,awg,zch,zname)
        #     # task2.append(task_manageawg)
        #     task2[taskname] = task_manageawg
        # else:
        #     task_manageawg = await cww.couldRun(measure,awg)
        #     # task2.append(task_manageawg)
        #     task2[taskname] = task_manageawg
    return [task1]

################################################################################
# 开关JPA
################################################################################

async def jpa_switch(measure,state='OFF'):
    if state == 'ON':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','ON')
        await measure.psg[measure.jpa.inst['pump']].setValue('Frequency',(measure.jpa.f_ex))
        await measure.psg[measure.jpa.inst['pump']].setValue('Power',measure.jpa.power_ex)
        await measure.dc[measure.jpa.q_name].DC(measure.jpa.bias)
    if state == 'OFF':
        await measure.psg[measure.jpa.inst['pump']].setValue('Output','OFF')
        await  measure.dc[measure.jpa.q_name].DC(0)


################################################################################
# S21
################################################################################

async def S21(qubit,measure,modulation=False,f_lo=None,f=None):
    #await jpa_switch(measure,state='OFF')
    if f_lo == None:
        f_lo, delta, n = await resn(np.array(qubit.f_lo))
        freq = np.linspace(-3,3,91)*1e6 + f_lo
    else:
        freq = np.linspace(-3,3,91)*1e6 + f_lo
        delta, n = measure.delta, measure.n
    if modulation:
        await cww.modulation_read(measure,delta,readlen=measure.readlen,repeats=measure.repeats)
    await measure.psg['psg_lo'].setValue('Output','ON')
    if f is not None:
        freq = f
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ()
        # ch_A, ch_B = await yieldData(measure)
        # print(np.shape(ch_A))
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        yield i+delta, s

################################################################################
# satpower
################################################################################

async def satpower(measure,readamp):
    delta = measure.delta
    for i in readamp:
        measure.readamp = [i]*measure.n
        await cww.modulation_read(measure,delta,readlen=measure.readlen,repeats=measure.repeats)
        await measure.psg['psg_lo'].setValue('Output','ON')
        s_on_off = []
        for j in ['OFF','ON']:
            await jpa_switch(measure,state=j)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            s_on_off.append(Am + 1j*Bm)
        yield [i]*measure.n, s_on_off[1]/s_on_off[0]


async def test(measure,n):
    for i in range(n):
        time.sleep(0.1)
        # await measure.psg['psg_lo'].setValue('Frequency', i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 读出信号相位噪声
################################################################################

async def rPhase(measure,phase):
    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in phase:
        await cww.modulation_read(measure,measure.delta,readlen=1200,phase=i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s

################################################################################
# 优化读出IQMixer
################################################################################

async def readIQMixer(measure,var1,var2,optkind='LO',mode=0):
    awg = measure.awg['awgread']
    ch = [1,2]
    f_list = np.array(await measure.ats.getValue('f_list'))
    if optkind == 'LO':
        await awg.setValue('Offset',var1,ch=ch[0])
        await awg.setValue('Offset',var2,ch=ch[1])
        time.sleep(1)
        # await cww.couldRun(measure,awg)
        I, Q = 0, 0
        for i in range(10):
            chA, chB = await measure.ats.getTraces()
            I += chA
            Q += chB
        I, Q = I / 10, Q / 10
    if optkind == 'Imag':
        pulse = await cww.rabiWave(envelopename=['square',1],nwave=1,amp=1,phaseDiff=var2,pi_len=2000e-9,shift=-2000e-9,delta_ex=f_list[0])
        wav_I, wav_Q, mrk1, mrk2 = pulse
        # await cww.writeWave(awg,['opt_I','opt_Q'],(wav_I, wav_Q, mrk1, mrk2))
        await cww.writeWave(measure,awg,ch,pulse,mode=mode)
        # await cww.couldRun(measure,awg)
        I, Q = 0, 0
        for i in range(10):
            chA, chB = await measure.ats.getTraces()
            I += chA
            Q += chB
        I, Q = I / 10, Q / 10
    f = np.fft.fftshift(np.fft.fftfreq(len(I)))*1e9
    Pxx = np.abs(np.fft.fftshift(np.fft.fft(I + 1j*Q)))
    yield  dt.nearest(f,0,np.abs(Pxx)) if optkind == 'LO' else dt.nearest(f,-f_list[0],np.abs(Pxx))

################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,modulation=False,flo=None,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,121)*1e6+f_lo
    for i in measure.psg:
        if i != 'psg_lo' and i != 'psg_pump':
            await measure.psg[i].setValue('Output','OFF')
    length = len(freq) if freq is not None else 91
    job = Job(S21, (qubit,measure,modulation,flo,freq),auto_save=True,max=length,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    # print(np.shape(f_s21),np.shape(s_s21),index)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = await resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cww.ats_setup(measure.ats,delta)
        await cww.modulation_read(measure,delta,readlen=measure.readlen,repeats=measure.repeats)
        base = 0
        for i in range(15):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 15
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base,f_s21, s_s21

################################################################################
# S21vsFlux
################################################################################

async def S21vsFlux(qubit,measure,current,calimatrix,readstate=[],modulation=False):
    await dcManage(measure,dcstate={},readstate=readstate,calimatrix=calimatrix)
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:float(i)},readstate=None,calimatrix=calimatrix)
        # await measure.dc['q9'].DC(i)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsFlux_awgoffset
################################################################################

async def S21vsFlux_awgoffset(qubit,measure,current,calimatrix,readstate=[],modulation=False):
    awg = measure.awg[qubit.inst['z_awg']]
    zch = qubit.inst['z_ch']
    # await cww.genwaveform(awg,zname,qubit.inst['z_ch'])
    await dcManage(measure,dcstate={},readstate=readstate,calimatrix=calimatrix)
    for i in current:
        # pulse = await cww.funcarg(cww.zWave,qubit,offset=i,volt=0,during=0,shift=-3000e-9)
        # await cww.writeWave(measure,awg,zch,pulse)
        # await cww.couldRun(measure,awg)
        task = await executeZwave(measure,cww.zWave,dcstate={qubit.q_name:0},offset=i,during=0,shift=-3000e-9)
        await concurrence(task)
        job = Job(S21, (qubit,measure,modulation,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

################################################################################
# S21vsPower
################################################################################

async def S21vsPower(qubit,measure,att):
    l = measure.readlen
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=None)
    for i in att:
        # await measure.attinst['com8'].set_att(i)
        measure.readamp[0] = i
        await cww.modulation_read(measure,measure.delta,readlen=2000)
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        yield [i]*measure.n, f_s21, s_s21
    measure.readlen = l

################################################################################
# SingleSpec
################################################################################

async def singlespec(measure,freq,modulation=False,f_read=None,readponit=True,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    if readponit:
        f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,f_read)
    else:
        n, base = measure.n, measure.base
    await measure.psg['psg_trans'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.writeWave(measure,measure.awg['awg_trig'],[3],[mTrig_ex_lo(measure.t_new)],False)
    # task = await executeEXwave(measure,cww.rabiWave,exstate=[qubit.q_name],amp=0.5,envelopename=['square',1],pi_len=10000e-9,shift=200e-9,delta_ex=0.001)
    # await concurrence(task)
    for i in freq:
        await measure.psg['psg_trans'].setValue('Frequency',i)
        # await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',i)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*n, s-base
    await measure.psg['psg_trans'].setValue('Output','OFF')
################################################################################
# specbias
################################################################################

async def specbias(qubit,measure,ftarget,bias,modulation=False):
    # await measure.dc[qubit.q_name].DC(round(np.mean(bias),3))
    await measure.psg['psg_trans'].setValue('Frequency',ftarget)
    # f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    await measure.psg['psg_trans'].setValue('Output','ON')
    for i in bias:
        # i -= np.mean(bias)
        await measure.dc[qubit.q_name].DC(i)
        f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        
        yield [i]*measure.n, s-measure.base
    await measure.psg['psg_trans'].setValue('Output','OFF')

async def specbias_awg(measure,qubit,ftarget,bias,dcstate={},calimatrix=None,modulation=True):
    # dcstate[qubit.q_name] = qubit.volt
    task = await executeZwave(measure,cww.zWave,dcstate=dcstate,\
    calimatrix=calimatrix,during=(len(measure.t_new)/2.5/2e9+2000e-9),offset=0,shift=0e-9)
    await concurrence(task)
    await measure.psg['psg_trans'].setValue('Frequency',ftarget)
    f_lo, delta, n, f_res,base,f_s21, s_s21 = await again(qubit,measure,modulation,measure.f_lo)
    await measure.psg['psg_trans'].setValue('Output','ON')

    for j, i in enumerate(bias):
        dcstate[qubit.q_name] = i
        task = await executeZwave(measure,cww.zWave,dcstate=dcstate,\
        calimatrix=calimatrix,during=(len(measure.t_new)/2.5/2e9+2000e-9),offset=0,shift=0e-9)
        await concurrence(task)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        
        yield [i]*measure.n, s-measure.base
    await measure.psg['psg_trans'].setValue('Output','OFF')

################################################################################
# Spec2d
################################################################################

async def spec2d(qubit,measure,freq,calimatrix,modulation=False,readstate=[]):
    current = np.linspace(-qubit.T_bias[0]*0.31,qubit.T_bias[0]*0.31,32) 
    # await dcManage(measure,dcstate={},readstate=[f'q{i+1}' for i in range(10)],calimatrix=calimatrix)
    await dcManage(measure,dcstate={},readstate=readstate,calimatrix=calimatrix)
    f_ex = qubit.f_max
    freq = np.arange(f_ex-1.2,f_ex+0.1,0.001)*1e9
    
    for i in current:
        await dcManage(measure,dcstate={qubit.q_name:i+ qubit.T_bias[1] },readstate={qubit.q_name},calimatrix=calimatrix)
        # await measure.dc[qubit.q_name].DC(i)
        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,True,[qubit.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss
    await measure.dc[qubit.q_name].DC(0)

async def spec2d2(qubit,measure,calimatrix,modulation=False,readstate=[],rg=10,N=6,dM=0.5):
    current = np.linspace(-qubit.T_bias[0]*0.32,qubit.T_bias[0]*0.32,33)
    await dcManage(measure,dcstate={},readstate={qubit.q_name},calimatrix=calimatrix)
    f_ex = qubit.f_max
    freq = np.arange(f_ex-1.4,f_ex+0.1,0.001)*1e9
    freq_shape=[]
    for k in freq:
        freq_shape.append([k]*measure.n)

    start0 = np.argmax(freq)
    end0 = np.argmax(freq)
    length = len(freq)
    index0=[[0,start0-700]]
    index1=[[start0-200,end0]]
    index2=[[start0-330,start0-120]]
    Indexp = []
    Indexm = []
    for j,i in enumerate(current):
        pop = np.zeros((length,measure.n))
        if i<=-qubit.T_bias[0]*0.05:
            if len(Indexm)<=1:
                Index = index0.copy()
            else:
                Index = index_merge(Indexm.copy()[-2],Indexm.copy()[-1],length)
        
        if abs(i)<qubit.T_bias[0]*0.05:
            Index = index1.copy()
        if i>=qubit.T_bias[0]*0.05:
            if len(Indexp)<=1:
                Index = index2.copy()
            else:
                Index = index_merge(Indexp.copy()[-2],Indexp.copy()[-1],length)
    
        await dcManage(measure,dcstate={qubit.q_name:i+qubit.T_bias[1]},readstate={qubit.q_name},calimatrix=calimatrix)
        for I in Index:
            if i<=-qubit.T_bias[0]*0.05:
                I[1]+=20
            if i >=qubit.T_bias[0]*0.05:
                I[0]+=(-20)
            job = Job(singlespec, (measure,freq[I[0]:I[1]],modulation,measure.f_lo,True,[qubit.q_name]),auto_save=False,max=len(freq[I[0]:I[1]]))
            f_ss, s_ss = await job.done()
            pop[I[0]:I[1]] = np.abs(s_ss)
        indexs = index_max(pop.copy(),rg,N,dM) 
        if i<=-qubit.T_bias[0]*0.05:
            Indexm.append(indexs)
        if i>=qubit.T_bias[0]*0.05:
            Indexp.append(indexs)

        yield [i+qubit.T_bias[1]]*measure.n, freq_shape, pop
    await dcManage(measure,dcstate={},readstate={qubit.q_name},calimatrix=calimatrix)

################################################################################
# Spec2d_awg
################################################################################

async def spec2d_awg(qubit,measure,current,freq,calimatrix,modulation=False,readstate=[]):
    
    qnum = len(measure.qubits)
    task = await executeZwave(measure,cww.zWave,dcstate={},qnum=qnum,\
            calimatrix=calimatrix,offset=0,during=0/1e9,shift=100e-9)
    await concurrence(task)
    dcstate = {i: round(measure.qubits[i].T_bias[1]+measure.qubits[i].T_bias[0]/2,3) for i in measure.qubits}
    dcstate[qubit.q_name] = round(qubit.T_bias[1],3)
    # dcstate = {i: measure.qubits[i].T_bias[1] for i in measure.qubits }
    await dcManage(measure,dcstate=dcstate,readstate=readstate,calimatrix=None)
    res = await again(qubit,measure,False,measure.f_lo)
    
    for j,i in enumerate(current):
        output = True if j== 0 else False
        task = await executeZwave(measure,cww.zWave,dcstate={qubit.q_name:i},\
            calimatrix=calimatrix,output=output,during=(len(measure.t_new)/2.5/2e9+2000e-9),offset=0,shift=200e-9)
        await concurrence(task)
        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,False,[qubit.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss
    await cww.OffEx([qubit.q_name],measure)
    await cww.OffZ([qubit.q_name],measure)
################################################################################
# Spec2d_awg2
################################################################################
async def spec2d_awg2(qubit,measure,current,calimatrix,modulation=False,rg=10,N=6,dM=0.5):
    dcstate = {i: round(measure.qubits[i].T_bias[1]+measure.qubits[i].T_bias[0]/2,3) for i in measure.qubits}
    dcstate[qubit.q_name] = round(qubit.T_bias[1],3)
    # dcstate = {i: measure.qubits[i].T_bias[1] for i in measure.qubits }
    await dcManage(measure,dcstate=dcstate,readstate=[qubit.q_name],calimatrix=None)

    qnum = len(measure.qubits)
    task = await executeZwave(measure,cww.zWave,dcstate={},qnum=qnum,\
            calimatrix=calimatrix,offset=0,during=0/1e9,shift=100e-9)
    await concurrence(task)

    res = await again(qubit,measure,False,measure.f_lo)
    f_ex = qubit.f_max
    freq = np.arange(f_ex-1.4,f_ex+0.1,0.001)*1e9
    freq_shape=[]
    for k in freq:
        freq_shape.append([k]*measure.n)

    start0 = np.argmax(freq)
    end0 = np.argmax(freq)
    length = len(freq)
    index0=[[0,start0-700]]
    index1=[[start0-200,end0]]
    index2=[[start0-330,start0-120]]
    Indexp = []
    Indexm = []
    for j,i in enumerate(current):
        output = True if j== 0 else False
        task = await executeZwave(measure,cww.zWave,dcstate={qubit.q_name:i},\
            calimatrix=calimatrix,output=output,during=(len(measure.t_new)/2.5/2e9+2000e-9),offset=0,shift=200e-9)
        await concurrence(task)

        if i<=-qubit.T_bias[0]*0.05:
            if len(Indexm)<=1:
                Index = index0.copy()
            else:
                Index = index_merge(Indexm.copy()[-2],Indexm.copy()[-1],length)
        
        if abs(i)<qubit.T_bias[0]*0.05:
            Index = index1.copy()
        if i>=qubit.T_bias[0]*0.05:
            if len(Indexp)<=1:
                Index = index2.copy()
            else:
                Index = index_merge(Indexp.copy()[-2],Indexp.copy()[-1],length)
    
        pop_m = np.zeros((length,measure.n))
        for I in Index:
            if i<=-qubit.T_bias[0]*0.05:
                I[1]+=20
            if i >=qubit.T_bias[0]*0.05:
                I[0]+=(-20)
            job = Job(singlespec, (measure,freq[I[0]:I[1]],modulation,measure.f_lo,False,[qubit.q_name]),auto_save=True,max=len(freq[I[0]:I[1]]))
            f_ss, s_ss = await job.done()
            pop_m[I[0]:I[1]] = np.abs(s_ss)
        indexs = index_max(pop_m,rg,N,dM) 
        if i<= -qubit.T_bias[0]*0.05:
            Indexm.append(indexs)
        if i>= qubit.T_bias[0]*0.05:
            Indexp.append(indexs)

        yield [i]*measure.n, freq_shape, pop_m
    await cww.OffEx([qubit.q_name],measure)
    await cww.OffZ([qubit.q_name],measure)
    await dcManage(measure,dcstate={},readstate=[qubit.q_name],calimatrix=None)


###############################################################################
# Crosstalk
################################################################################

async def CS21(exstate,measure,clist):
    for j in(clist):
        task_z2 = await executeZwave(measure,cww.zWave,dcstate={exstate[0]:j},qnum=len(exstate),calimatrix=None,\
        offset=0,during=3400e-9,shift=100*1e-9,args='volt')
        await concurrence(task_z2)
        s = await popRead(measure,pop=False)
        yield [j]*measure.n, s
    
async def Crosstalk(measure,exstate,dcstate,exbias,dcbias):
    task = await executeZwave(measure,cww.zWave,dcstate={},qnum=measure.n,calimatrix=None,offset=0,during=0/1e9,shift=100e-9)
    await concurrence(task)
    qubit=measure.qubits[exstate[0]]
    # await dcManage(measure,dcstate=dcstate,readstate=exstate+dcstate,calimatrix=None)
    # await measure.psg['psg_ex'].setValue('Output','ON')
    # await measure.psg['psg_lo'].setValue('Output','ON')
    arg = 'pi_len'
    qubit.f_ex = qubit.f_cross
    para = {arg:1500e-9,}
    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,shift=300e-9,**para)
    await concurrence(task)
    
    for i in dcbias:
        task_z2 = await executeZwave(measure,cww.zWave,dcstate={dcstate[0]:i},qnum=len(dcstate),calimatrix=None,\
            offset=0,during=3400e-9,shift=100*1e-9,args='volt')
        await concurrence(task_z2)

        job = Job(CS21, (exstate,measure,exbias),auto_save=True,max=len(exbias),tags=exstate)
        f_ss, s_ss = await job.done()
        yield [i]*measure.n, f_ss, s_ss


################################################################################
# 用谱验证crosstalk矫正
################################################################################

async def spec2d_crosstalk(measure,current,freq,calimatrix,modulation=False,dcstate=[],exstate=[]):
    qubit_ex, qubit_z = measure.qubits[exstate[0]], measure.qubits[dcstate[0]]
    task = await executeZwave(measure,cww.zWave,dcstate={},\
            calimatrix=calimatrix,offset=0,during=0/1e9,shift=100e-9)
    await concurrence(task)
    # dcstate = {i: round(measure.qubits[i].T_bias[1]+measure.qubits[i].T_bias[0]/2,3) for i in measure.qubits}
    # dcstate[qubit_ex.q_name] = round(qubit_ex.T_bias[1],3)
    # # dcstate = {i: measure.qubits[i].T_bias[1] for i in measure.qubits }
    # await dcManage(measure,dcstate=dcstate,readstate=[qubit_ex.q_name],calimatrix=None)
    res = await again(qubit_ex,measure,False,measure.f_lo)
    
    
    for j,i in enumerate(current):
        output = True if j== 0 else False
        task = await executeZwave(measure,cww.zWave,dcstate={qubit_z.q_name:i},\
            calimatrix=calimatrix,output=output,during=(len(measure.t_new)/2.5/2e9+2000e-9),offset=0,shift=200e-9)
        await concurrence(task)

        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,False,[qubit_ex.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss


################################################################################
# 坠饰态能谱
################################################################################

async def spec2d_dress(qubit,measure,current,freq,f_target,modulation=False):
    
    namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    chlist = qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]

    for j,i in enumerate(current):

        pulselist, _ = await cww.funcarg(cww.zWave_im,qubit,f_ex=f_target,during=(len(measure.t_new)/2.5/2e9),volt=(qubit.volt+qubit.offset),\
            delta_im=120e6,imAmp=i,shift=400e-9)
        await cww.writeWave(measure,z_awg,chlist,pulse=pulselist)
        # await cww.couldRun(measure,z_awg,chlist,namelist)
        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,False,[qubit.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i/2/np.pi]*n, f_ss, s_ss

################################################################################
# Rabi  
################################################################################

async def rabi(measure,amp,arg='amp',exstate=[]):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    # qubit = measure.qubits[exstate[0]]
    # namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    # chlist = qubit.inst['ex_ch']
    # awg_ex = measure.awg[qubit.inst['ex_awg']]
    # await cww.couldRun(measure,awg_ex,chlist,namelist)
    await measure.psg['psg_lo'].setValue('Output','ON')
    
    for j,i in enumerate(amp):
        para = {arg:i} if arg == 'amp' else {arg:i/1e9}
        output = True if j == 0 else False
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,**para)
        await concurrence(task)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        s = Am + 1j*Bm
        n = np.shape(s)[-1]
        yield [i]*n, s-measure.base

        # ch_A, ch_B = await measure.ats.getIQ()
        # Am, Bm = ch_A,ch_B
        # ss = Am + 1j*Bm
        # pop = readPop(measure,ss,readcali=True)
        # yield [i]*measure.n, np.array(pop)[:,0]


################################################################################
# T1  
################################################################################
async def T1(measure,t_rabi,exstate=[], pop=False):
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)

    T_max = np.max(t_rabi)
    ad_trig_delay0 = measure.ad_trig_delay+T_max*1e-9
    await measure.awg['awg1'].da_set_trig_delay(float(ad_trig_delay0))
    await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=measure.repeats,channel_output_delay=T_max)

    for i,j in enumerate (t_rabi):
        pulse = await cww.rabiWave(envelopename=qubit.envelopename,nwave=1,amp=qubit.amp,\
            pi_len=qubit.pi_len,shift=qubit.pi_len+100e-9,delta_ex=qubit.delta_ex)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse=pulse,channel_output_delay=T_max-j)
        s = await popRead(measure,pop=pop)
        yield [j]*measure.n, s
    await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=measure.repeats)
    await measure.awg['awg1'].da_set_trig_delay(measure.ad_trig_delay)

################################################################################
# 优化pi脉冲
################################################################################
    
async def pipulseOpt(measure,nwave,wavlen,optwhich='pi_len',exstate=[]):
    qubit = measure.qubits[exstate[0]]
    if optwhich == 'pi_len':
        pilen = qubit.pi_len*1e9
        start = 0.5*pilen if 0.5*pilen > 10 else 1
        end = 2.5*pilen if 0.5*pilen > 10 else pilen+30
        x = np.linspace(start,end,wavlen)
        # func = rabiTime_seq
    if optwhich == 'amp':
        amp = qubit.amp
        start = 0.5*amp
        end = 1.5*amp if 1.5*amp <=1 else 1
        x = np.linspace(start,end,wavlen)
        # func = rabiPower_seq
    for j,i in enumerate(range(nwave)):
        readseq = True if j == 0 else False
        qubit.nwave = 8*i+1
        job = Job(rabi, (measure,x,optwhich,exstate),auto_save=False,max=len(x))
        t_r, s_r = await job.done()

        # job = Job(rabi_many, (measure,x,optwhich,exstate), max=300,avg=True,auto_save=False)
        # t_r, s_r = await job.done()
        yield [4*i+1]*measure.n, t_r, s_r
    qubit.nwave = 1

################################################################################
# 双光子能级
################################################################################
    
async def fLevel(measure,exstate=[],delta_ex=230e6,pop=False,zstate={},calimatrix=None,rang=50):

    qubit = measure.qubits[exstate[0]]
    freq = qubit.delta_ex - delta_ex + np.linspace(-rang,rang,4*rang+1)*1e6
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)

    for i in freq:
        pi_len=50e-9
        amp = qubit.amp
        interval = 2e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9+2*qubit.pi_len+2*pi_len+2*interval)
        pulse2 = await cww.funcarg(cww.rabiWave,qubit,pi_len=pi_len,delta_ex=i,shift=100e-9+2*qubit.pi_len+interval,amp=amp)
        pulse3 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)

        task_z = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
        offset=0,during=i/1e9*2,shift=100e-9+qubit.pi_len+interval,args='volt')
        await concurrence(task_z)

        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)
        s = await popRead(measure,pop=pop)
        yield [qubit.delta_ex-i]*measure.n, s


################################################################################
# 双光子rabi
################################################################################

async def fRabi(measure,t_rabi,exstate=[],pop=False):

    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)
    print('qubit.f_ex-qubit.delta_ex=',qubit.f_ex-qubit.delta_ex)
    for i in t_rabi:
        pi_len = i*1e-9
        amp = qubit.amp
        interval = 2e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9+2*qubit.pi_len+2*pi_len+2*interval)
        pulse2 = await cww.funcarg(cww.rabiWave,qubit,pi_len=pi_len,delta_ex=(qubit.delta_ex-qubit.alpha),shift=100e-9+2*qubit.pi_len+interval,amp=amp)
        pulse3 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)

        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s

################################################################################
# 双光子Ramsey
################################################################################
    
async def fRamsey(measure,t_rabi,exstate=[],pop=False):
    
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)

    for i in t_rabi:
        amp = qubit.amp
        interval=2e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=(100e-9+2*qubit.pi_len+2*qubit.pi_len2+2*interval+i/1e9))

        pulse2 = await cww.coherenceWave(envelopename=qubit.envelopename,t_run=i/1e9,amp=amp,pi_len=qubit.pi_len2,nwave=0,seqtype='PDD',\
            detune=qubit.detune,shift=100e-9+2*qubit.pi_len+interval,delta_ex=qubit.delta_ex-qubit.alpha,phaseDiff=0.0,DRAGScaling=qubit.DRAGScaling,timing=qubit.timing)

        pulse4 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse4)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)

        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s
    qubit.replace(nwave=1,seqtype='CPMG')

################################################################################
# 比特能级差
################################################################################
async def QLevel(measure,z_volt,exstate,zqubit,pop=False,calimatrix=None):
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)
    
    for i in z_volt:
        pi_len = 200*1e-9
        amp = qubit.amp
        interval = 4e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9+2*qubit.pi_len+2*pi_len+2*interval)
        pulse2 = await cww.funcarg(cww.rabiWave,qubit,pi_len=pi_len,delta_ex=(qubit.delta_ex-qubit.alphaq),shift=100e-9+2*qubit.pi_len+interval,amp=amp)
        pulse3 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)
        
        zstate={zqubit[0]:i}
        task_z = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
            offset=0,during=2*pi_len+interval,shift=100e-9+2*qubit.pi_len+0.5*interval,args='volt')  
        await concurrence(task_z)

        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s

################################################################################
# 比特能级差rabi
################################################################################

async def QRabi(measure,t_rabi,exstate,zstate={},pop=False,calimatrix=None):
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)
    
    for i in t_rabi:
        pi_len = i*1e-9
        amp = qubit.amp
        interval = 4e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9+2*qubit.pi_len+2*pi_len+2*interval)
        pulse2 = await cww.funcarg(cww.rabiWave,qubit,pi_len=pi_len,delta_ex=(qubit.delta_ex-qubit.alphaq),shift=100e-9+2*qubit.pi_len+interval,amp=amp)
        pulse3 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)
        
        task_z = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
            offset=0,during=2*pi_len+interval,shift=100e-9+2*qubit.pi_len+0.5*interval,args='volt')  
        await concurrence(task_z)

        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s

################################################################################
# 比特能级差Ramsey
################################################################################
    
async def QRamsey(measure,t_rabi,exstate=[],zstate={},pop=False,calimatrix=None):
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)

    for i in t_rabi:
        amp = qubit.amp
        interval=2e-9
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=(100e-9+2*qubit.pi_len+2*qubit.pi_lenq+3*interval+i/1e9))
        pulse2 = await cww.coherenceWave(envelopename=qubit.envelopename,t_run=i/1e9+interval,amp=amp,pi_len=qubit.pi_lenq,nwave=0,seqtype='PDD',\
            detune=qubit.detune,shift=100e-9+2*qubit.pi_len+interval,delta_ex=qubit.delta_ex-qubit.alphaq,phaseDiff=0.0,DRAGScaling=qubit.DRAGScaling,timing=qubit.timing)
        pulse4 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=100e-9)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse4)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)

        task_z = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
            offset=0,during=i/1e9,shift=100e-9+qubit.pi_len2+qubit.pi_lenq+1.5*interval,args='volt')   
        await concurrence(task_z)

        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s
    qubit.replace(nwave=1,seqtype='CPMG')
################################################################################
# 动力学相位
################################################################################
    
async def Dphase0(measure,z_volt,z_len,vphase=[],exstate=[],calimatrix=None,w=0):
    qubit = measure.qubits[exstate[0]]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',qubit.f_ex-qubit.delta_ex)
    flux={exstate[0]:z_volt}
    task_z = await executeZwave(measure,cww.zWave,dcstate=flux,qnum=len(flux),calimatrix=calimatrix,\
            offset=0,during=z_len/1e9,shift=100e-9+2*qubit.pi_len,args='volt')
    await concurrence(task_z)

    for i in vphase:
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='Xhalf',shift=(100e-9+4*qubit.pi_len + z_len/1e9))
        pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis='Xhalf',shift=100e-9,virtualPhase=i+w*z_len/1e9*2*np.pi)
        pulse = np.array(pulse1) + np.array(pulse2)
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop = readPop(measure,ss,readcali=True)
        yield [i]*measure.n, np.array(pop)[:,1]

async def Dphase(measure,z_volt,z_len,vphase=[],exstate=[],calimatrix=None,w=0):
    for i in z_len:
        job = Job(Dphase0, (measure,z_volt,i,vphase,exstate,calimatrix,w),max=len(z_len),auto_save=False)
        phase, s_phase = await job.done()
        yield [i]*measure.n, phase, s_phase

################################################################################
# IQ-Mixer 线性度
################################################################################
    
async def lineIQMixer(measure,amp,t_rabi,exstate=[],mode='vbroadcast'):
    qubit = measure.qubits[exstate[0]]
    for j,i in enumerate(amp):
        readseq = True if j == 0 else False
        qubit.amp = i
        numrepeat, avg = (len(t_rabi),False) if mode == 'hbroadcast' else (300,True)
        job = Job(rabi, (measure,t_rabi,'pi_len',exstate),max=numrepeat,auto_save=False)
        t_r, s_r = await job.done()

        # job = Job(rabi_many, (measure,x,optwhich,exstate), max=300,avg=True,auto_save=False)
        # t_r, s_r = await job.done()
        yield [i]*measure.n, t_r, s_r
    qubit.amp = 1

################################################################################
# ZZ coupling
################################################################################

async def ZZ_coupling(measure,t_Ramsey,exstate=[],control=[],comwave=True,readseq=True,mode='vbroadcast'):
    qubit_target, qubit_control = measure.qubits[exstate[0]], measure.qubits[control[0]]
    delta_copy = [np.copy(qubit_target.delta_ex),np.copy(qubit_control.delta_ex)]
    qubit_target.replace(nwave=0,seqtype='PDD',detune=2e6)

    delta_ex = await exManage(measure,exstate=[qubit_target.q_name,qubit_control.q_name])
    qubit_target.delta_ex, qubit_control.delta_ex = delta_ex[qubit_target.q_name], delta_ex[qubit_control.q_name]
    

    len_data, t = len(t_Ramsey), np.array([t_Ramsey]*measure.n).T
    if comwave:
        task = await executeEXseq(measure,cww.Rabi_sequence,len_data,comwave,control,readseq,mode,v_or_t=(t_Ramsey/1e9+2*qubit_target.pi_len),arg='shift')
        await concurrence(task)
    # await cww.openandcloseAwg('ON')
    await measure.psg['psg_lo'].setValue('Output','ON')
    numrepeat, avg = (len(t_Ramsey),False) if mode == 'hbroadcast' else (300,True)
    job = Job(Ramsey_seq, (measure,t_Ramsey,exstate,comwave,readseq,mode), tags=exstate, max=numrepeat,avg=avg,auto_save=False)
    t_ram, s_ram = await job.done()
    yield t_ram, s_ram
    qubit_target.replace(nwave=1,delta_ex=float(delta_copy[0]))
    qubit_control.replace(delta_ex=float(delta_copy[1]))


################################################################################
# 优化读出点
################################################################################

async def readOp(measure,exstate=[]):

    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    await concurrence(task)
    qubit = measure.qubits[exstate[0]]
    ex_name, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    # for j, i in enumerate(['OFF','ON']):
    #     for k in exstate:
    #         await measure.psg[measure.qubits[k].inst['ex_lo']].setValue('Output',i)
    # for j, i in enumerate(['OFF','ON','ON2']):
    for j, i in enumerate(['OFF','ON']):
        if i != 'ON2':
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            time.sleep(0.1)
        else:
            task = await executeEXwave(measure,cww.eWave,exstate=exstate,shift=200e-9)
            await concurrence(task)
            # pass
        job = Job(S21, (qubit,measure,False,measure.f_lo),auto_save=False,max=81)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [j]*n, f_s21, s_s21

################################################################################
# 优化读出点占比
################################################################################

async def readOpweight(measure,exstate=[]):

    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate)
    await concurrence(task)

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    for j, i in enumerate(['OFF','ON']):
        for k in exstate:
            await measure.psg[measure.qubits[k].inst['ex_lo']].setValue('Output',i)
        chA, chB, ch_I, ch_Q = await measure.ats.getTraces()
        yield chA + 1j*chB

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

async def threshHold(measure,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    ex_name, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,shift=200e-9)
    await concurrence(task)

    await cww.ats_setup(measure,measure.delta,readlen=measure.readlen,repeats=2000)
    # await measure.awg[qubit.inst['ex_awg']].da_trigg(5000)
    # await measure.ats.set_ad(measure.delta, measure.wavelen, window_start=[8]*len(measure.delta), trig_count=5000)
    for j, i in enumerate(['OFF','ON']):
        if i != 'ON2':
            await measure.psg[qubit.inst['ex_lo']].setValue('Output',i)
            time.sleep(0.1)
        else:
            pass
            # pulse_rabi2 = await cww.funcarg(qgw.singleQgate,qubit,pi_len=qubit.pi_len2,amp=qubit.amp2,\
            #     delta_ex=(qubit.delta_ex + qubit.alpha),shift=200e-9)
            # pulse_pi1 = await cww.funcarg(qgw.singleQgate,qubit,shift=(qubit.envelopename[1]*qubit.pi_len2 + 210e-9))
            # pulse = np.array(pulse_pi1) + np.array(pulse_rabi2)
            # await cww.writeWave(measure,ex_awg,ex_name,pulse,mark=False)
            # await cww.couldRun(measure,ex_awg,exch)
            # task = await executeEXwave(measure,cww.eWave,exstate=exstate,shift=200e-9)
            # await concurrence(task)
            # pass
    
        # ch_A, ch_B = await measure.ats.getIQ(hilbert=False)
        ch_A, ch_B = await measure.ats.getIQ()
        # ch_A, ch_B = await yieldData(measure,avg=False,fft=True,hilbert=False,filter=None)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        s = Am + 1j*Bm
        # sq = I + 1j*Q
        yield j, s #, sq

#############################################################################
# 优化JPA
################################################################################

async def singleJpa(measure,pumppower,exstate=[]):
    for j in pumppower:
        await measure.psg['psg_pump'].setValue('Power',j)
        job = Job(threshHold, (measure,exstate), tags=exstate,no_bar=True,auto_save=False)
        st, s_st = await job.done()
        # print(np.shape(st))
        s_off, s_on = s_st[0,:,0], s_st[1,:,0]
        mean_off, mean_on = np.mean(s_off), np.mean(s_on)
        d = np.abs(mean_on-mean_off)
        std = (np.std(s_off)+np.std(s_on))/2
        yield [j], [d/std]
async def optJpa(measure,current,pumppower,exstate=[]):
    for i in current:
        await measure.dc['jpa'].DC(i)
        job = Job(singleJpa, (measure,pumppower,exstate), tags=exstate,max=len(pumppower))
        pump,snr = await job.done()
        yield [i],pump,snr

################################################################################
# 优化读出功率 
################################################################################

async def readpowerOpt(measure,which,readamp):
    # qubit = measure.qubits[exstate[0]]
    n = len(measure.delta)
    for k in readamp:
        measure.readamp = [k]
        await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=5000)
        await cww.couldRun(measure,measure.awg['awgread'])
        
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + Bm
        ss = Am + 1j*Bm
        d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(n)]
        y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
        # d = list(zip(np.real(ss[:,0]),np.imag(ss[:,0])))
        # y = measure.predict[measure.qubitToread[0]](d)
        # pop = list(y).count(which[0])/len(y)
        yield [k]*n, pop


        # S = list(s_off) + list(s_on)
        # x,z = np.real(S), np.imag(S)
        # d = list(zip(x,z))
        # y = measure.predict[qubit.q_name](d)
        # pop = list(y).count(1)/len(y)
        # yield [i]*n, [pop]

    #     offmean, onmean = np.mean(x), np.mean(y)
    #     offstd, onstd = np.std(x), np.std(y)
    #     # doff, don = np.abs(x-offmean), np.abs(y-onmean)
    #     d = np.abs(offmean-onmean)
    #     # popoff, popon = d/offstd, d/onstd
    #     snr = 2*d**2/(offstd+onstd)**2
    #     # popoff, popon = list(doff<offstd).count(True)/len(x), list(don<onstd).count(True)/len(y)
    #     # yield [i]*n, [popoff]*n, [popon]*n
    #     yield [i]*n, [snr]*n
    # att_setup.close()

################################################################################
# AllXY drag detune
################################################################################
    
async def AllXYdragdetune(measure,which,exstate=[]):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    coef = np.linspace(-5,5,101)

    qubit = measure.qubits[exstate[0]]
    alpha = qubit.alpha*2*np.pi
    # ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]

    # await cw.modulation_read(measure,measure.delta,readlen=measure.readlen)
    # await cw.genwaveform(ex_awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for j in [['X','Yhalf'],['Y','Xhalf']]:
        for i in coef:
            pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis=j[0],DRAGScaling=i/alpha,shift=(qubit.pi_len+110e-9))
            pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis=j[1],DRAGScaling=i/alpha,shift=100e-9)
            pulse = np.array(pulse1) + np.array(pulse2)
            await cww.writeWave(measure,ex_awg,ex_ch,pulse)
            # await cww.couldRun(measure,ex_awg)

            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            yield [i]*measure.n, s-measure.base

################################################################################
# Opt pi detune
################################################################################

async def optPidetune(measure,exstate=[]):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    qubit = measure.qubits[exstate[0]]
    deltaf = np.linspace(-4,4,201)*1e6 + qubit.f_ex

    ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    pulse = 0
    for i in range(10):
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=((2*i+1)*qubit.envelopename[1]*qubit.pi_len+(3*i+3)*1e-9))
        pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis='Xn',shift=(2*i*qubit.envelopename[1]*qubit.pi_len+(3*i+3)*1e-9))
        pulse += (np.array(pulse1) + np.array(pulse2)) 
    await cww.writeWave(measure,ex_awg,ex_ch,pulse)
    # await cw.modulation_read(measure,measure.delta,readlen=measure.readlen)
    # await cw.genwaveform(ex_awg,qname,qubit.inst['ex_ch'])
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for j in deltaf:
        await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(j+qubit.delta_ex))
        ch_A, ch_B = await measure.ats.getIQ(hilbert=False)
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        pop = readPop(measure,ss,readcali=False)
        yield [j-qubit.f_ex]*measure.n, np.array(pop)

################################################################################
# Opt Drag alphas
################################################################################

async def optDragalpha(measure,exstate=[],rang=3):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    qubit = measure.qubits[exstate[0]]
    alpha = qubit.alpha*2*np.pi
    dragcoef = np.linspace(-rang,rang,61)

    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(qubit.f_ex-qubit.delta_ex))
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for j in dragcoef:
        pulse = 0
        for i in range(10):
            pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='X',shift=((2*i+1)*qubit.envelopename[1]*qubit.pi_len+(3*i+3)*1e-9),DRAGScaling=j/alpha)
            pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis='Xn',shift=(2*i*qubit.envelopename[1]*qubit.pi_len+(3*i+3)*1e-9),DRAGScaling=j/alpha)
            pulse += (np.array(pulse1) + np.array(pulse2)) 
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)
        # await cww.couldRun(measure,ex_awg)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        pop = readPop(measure,ss,readcali=True)
        yield [j]*measure.n, np.array(pop)

async def optDragalpha_many(measure,exstate=[]):
    
    task = await executeEXwave(measure,cww.optDragalpha_wave,exstate=exstate,output=False)
    await concurrence(task)
    ch_A, ch_B = await measure.ats.getIQ(hilbert=False)
    Am, Bm = ch_A,ch_B
    # theta0 = np.angle(Am) - np.angle(Bm)
    # Bm *= np.exp(1j*theta0)
    ss = Am + 1j*Bm
    pop = readPop(measure,ss,readcali=False)
    yield np.array(pop)[:,0]

################################################################################
# AllXY_kind
################################################################################

async def AllXY_kind(measure,auxqubit=None,exstate=[],kind='reflection'):

    delta_ex = await exManage(measure,exstate=exstate)
    indexlist_AllXY_singlequbit = mx.AllXY(kind)

    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    qubit = measure.qubits[exstate[0]]

    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    if auxqubit is not None:
        delta_ex = await exManage(measure,exstate=(exstate+auxqubit))
        auxqubit = measure.qubits[auxqubit[0]]
        aux_ch = auxqubit.inst['ex_ch']
        aux_awg = measure.awg[auxqubit.inst['ex_awg']]
        pulse0 = await cww.funcarg(qgw.singleQgate,auxqubit,axis='X',shift=2*(qubit.envelopename[1]*qubit.pi_len+3e-9),delta_ex=delta_ex[auxqubit[0]])
        await cww.writeWave(measure,aux_awg,aux_ch,pulse0)
        # await cww.couldRun(measure,aux_awg,namelist=aux_name,chlist=aux_ch)

    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(qubit.f_ex-qubit.delta_ex))
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for i,j in enumerate(indexlist_AllXY_singlequbit):
        pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis=j[0],shift=(qubit.envelopename[1]*qubit.pi_len+3e-9),delta_ex=delta_ex[exstate[0]])
        pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis=j[1],delta_ex=delta_ex[exstate[0]])
        pulse = (np.array(pulse1) + np.array(pulse2)) 
        await cww.writeWave(measure,ex_awg,ex_ch,pulse)
        # await cww.couldRun(measure,ex_awg)
        pop = 0
        for rep in range(10):
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A,ch_B
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            pop_m = readPop(measure,ss,readcali=True)
            pop += np.array(pop_m)
        yield [reduce(lambda x,y:x+y,j)]*measure.n, np.array(pop)/10

async def AllXY_auto(measure,auxqubit=None,exstate=[],ref={'I':'I','X':['pi',0],'Xhalf':['half',0],'Y':['pi',np.pi/2],'Yhalf':['half',np.pi/2]}):
    
    delta_ex = await exManage(measure,exstate=exstate)
    indexlist_AllXY_singlequbit = mx.AllXY('reflection')

    await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    qubit = measure.qubits[exstate[0]]

    ex_name = [f'ch{i}' for i in qubit.inst['ex_ch']]
    ex_ch = qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    if auxqubit is not None:
        delta_ex = await exManage(measure,exstate=(exstate+auxqubit))
        auxqubit = measure.qubits[auxqubit[0]]
        aux_name = [f'ch{i}' for i in auxqubit.inst['ex_ch']]
        aux_ch = auxqubit.inst['ex_ch']
        aux_awg = measure.awg[auxqubit.inst['ex_awg']]
        pulse0 = await cww.funcarg(qgw.singleQgate,auxqubit,axis='X',shift=2*(qubit.envelopename[1]*qubit.pi_len+3e-9),delta_ex=delta_ex[auxqubit[0]])
        await cww.writeWave(measure,aux_awg,aux_name,pulse0)
        await cww.couldRun(measure,aux_awg,namelist=aux_name,chlist=aux_ch)

    await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',(qubit.f_ex+qubit.delta_ex))
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    for i,j in enumerate(indexlist_AllXY_singlequbit):
        axis0, axis1 = ref[j[0]], ref[j[1]]
        pulse1 = await cww.funcarg(qgw.singleQgate_new,qubit,axis=axis0,shift=(qubit.envelopename[1]*qubit.pi_len+3e-9),delta_ex=delta_ex[exstate[0]])
        pulse2 = await cww.funcarg(qgw.singleQgate_new,qubit,axis=axis1,delta_ex=delta_ex[exstate[0]])
        pulse = (np.array(pulse1) + np.array(pulse2)) 
        await cww.writeWave(measure,ex_awg,ex_name,pulse)
        await cww.couldRun(measure,ex_awg)
        pop = 0
        for rep in range(10):
            ch_A, ch_B = await measure.ats.getIQ(hilbert=False)
            Am, Bm = ch_A,ch_B
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            ss = Am + 1j*Bm
            pop_m = readPop(measure,ss,readcali=True)[:,0]
            pop += np.array(pop_m)
        yield [reduce(lambda x,y:x+y,j)]*measure.n, np.array(pop)/10


################################################################################
# AC-Stark（XY与Readout时序矫正)
################################################################################

async def singleacStark(measure,t_shift,power=1,exstate=[],comwave=True):

    pulselist = await cww.ac_stark_wave(measure,power)
    await cww.writeWave(measure,measure.awg['awgread'],['Readout_I','Readout_Q'],pulselist,False,mark=False)
    job = Job(rabi, (measure,t_shift,'shift',exstate,comwave,True,'vbroadcast'), max=300)
    # job = Job(T1_seq, (measure,t_shift,exstate,comwave), tags=(exstate+['acstark']), max=300,avg=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def acStark(measure,t_T1,power,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    freq = np.linspace(-260,40,51)*1e6 + qubit.f_ex 
    f_m = qubit.f_ex
    # await cww.ac_stark_wave(measure)
    # await cww.couldRun(measure,measure.awg['awgread'])

    for i,j in enumerate(freq):
        # await measure.psg[qubit.inst['ex_lo']].setValue('Frequency',j)
        qubit.f_ex = j
        comwave = True if i == 0 else False
        job = Job(singleacStark, (measure,t_T1,power,exstate,comwave), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_ac = await job.done()
        yield [j]*measure.n, t_shift, s_ac
    qubit.f_ex = f_m
    
################################################################################
# 腔内光子数
################################################################################

async def singleNum(measure,power,comwave=True):
    
    len_data, t = len(power), np.array([power]*measure.n).T
    kind = 'Read'
    awg = measure.awg['awgread']
    await cww.gen_packSeq(measure,kind,awg,['I','Q'],len_data,readseq=False)
    if comwave:
        await cww.acstarkSequence(measure,kind=kind,v_or_t=power,arg='power')
    await cww.awgchmanage(measure,awg,kind,[1,5])

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex1'].setValue('Output','ON')

    for i in range(300):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base
    
async def photonNum(measure,power,end,exstate=[]):

    qubit = measure.qubits[exstate[0]]
    freq = np.linspace(-260,40,51)*1e6 + qubit.f_ex 
    f_m = qubit.f_ex

    for i,j in enumerate(freq):
        qubit.f_ex = j
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output='True',shift=end/1e9)
        await concurrence(task)
        comwave = True if i == 0 else False
        job = Job(singleNum, (measure,power,comwave), no_bar=True,auto_save=False,avg=True)
        t_shift, s_ac = await job.done()
        yield [j]*measure.n, t_shift, s_ac
    qubit.f_ex = f_m

################################################################################
# Z_Pulse（Z与Readout时序矫正)
################################################################################

async def singleZpulse(measure,t_shift,exstate=[],updatez=True):
    qubit = measure.qubits[exstate[0]]
    if updatez:
        task = await executeZwave(measure,cww.zWave,dcstate={exstate[0]:0.05},output=True,\
            during=((qubit.pi_len*qubit.envelopename[1])),shift=3000e-9)
        await concurrence(task)

    job = Job(rabi, (measure,t_shift,'shift',exstate),tags=exstate, max=len(t_shift), no_bar=True,auto_save=True)
    t_t, s_t = await job.done()
    yield t_t, s_t

async def zPulse(measure,t_T1,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=0.2,shift=3000e-9)
    volt = np.linspace(-0.2,0.1,40)
    for i,j in enumerate(volt):
        pulse2 = await cww.funcarg(cww.zWave,qubit,during=4000/1e9,volt=j,shift=1500e-9)
        pulse = np.array(pulse1) + np.array(pulse2)
        await cww.writeWave(measure,z_awg,zch,pulse,mark=False)
        # await cww.couldRun(measure,z_awg,zch,zname)
        comwave = True if i == 0 else False
        job = Job(singleZpulse, (measure,t_T1,exstate,False), tags=exstate, no_bar=True,auto_save=False)
        t_shift, s_z = await job.done()
        yield [j]*measure.n, t_shift, s_z


# async def singleZ(measure,t_T1,which=0,exstate=[]):
#     qubit = measure.qubits[exstate[0]]
#     exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
#     ex_awg = measure.awg[qubit.inst['ex_awg']]
#     await cww.couldRun(measure,ex_awg,exch,exname)
#     for k in t_T1:
#         pulselist = await cww.funcarg(cww.rabiWave,qubit,shift=k/1e9)
#         await cww.writeWave(measure,ex_awg,name=exname,pulse=pulselist)
#         await cww.couldRun(measure,ex_awg)
#         ch_A, ch_B = await measure.ats.getIQ()
#         Am, Bm = ch_A,ch_B
#         # theta0 = np.angle(Am) - np.angle(Bm)
#         # Bm *= np.exp(1j*theta0)
#         ss = Am + 1j*Bm
#         # d = list(zip(np.real(ss),np.imag(ss)))
#         d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
#         y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
#         pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
#         pop = np.array([pop[i] if j == 1 else 1-pop[i] for i, j in enumerate(which)])
#         yield [k], pop

# async def zPulse_pop(measure,t_T1,which=0,exstate=[]):

#     qubit = measure.qubits[exstate[0]]
#     zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
#     z_awg = measure.awg[qubit.inst['z_awg']]
#     pulse1 = await cww.funcarg(cww.zWave,qubit,pi_len=2000/1e9,volt=0.8,shift=3000e-9)
#     await cww.couldRun(measure,z_awg,zch,zname)

#     volt = np.linspace(-0.21,-0.01,51)
#     await cww.ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=1000)
#     for j in volt:
#         pulse2 = await cww.funcarg(cww.zWave,qubit,pi_len=4000/1e9,volt=j,shift=1500e-9)
#         pulse = np.array(pulse1) + np.array(pulse2)
#         await cww.writeWave(measure,z_awg,zname,pulse,mark=False)
#         await cww.couldRun(measure,z_awg)
#         job = Job(singleZ,(measure,t_T1,which,exstate),max=len(t_T1))
#         t, pop = await job.done()
#         yield [j], t, pop

################################################################################
# Z_Pulse_XY（Z与Readout时序矫正) population zPulse_pop
################################################################################

async def zPulse_XY(measure,tdelay,t_int,tshift,which,exstate=[]):
    
    qubit = measure.qubits[exstate[0]]
    ex_name, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
    ex_awg = measure.awg[qubit.inst['ex_awg']]
    
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    await cww.couldRun(measure,z_awg,zch,zname)

    pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis='Xhalf',shift=tshift*1e-9)
    pulse2 = await cww.funcarg(qgw.singleQgate,qubit,axis='Yhalf',shift=(tshift*1e-9-t_int/1e9-qubit.pi_len))
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,exch,ex_name)
    
    for i in tdelay:

        pulse = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=0.8,shift=(tshift*1e-9+i*1e-9))
        await cww.writeWave(measure,z_awg,zname,pulse,mark=False)
        await cww.couldRun(measure,z_awg)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        # d = list(zip(np.real(ss),np.imag(ss)))
        # d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
        # y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
        # pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
        # pop = np.array([pop[i] if which[j] == 1 else 1-pop[i] for i, j in enumerate(which)])
        pop = readPop(measure,ss)
        yield [i], np.array(pop)[:,1] 

################################################################################
#T1_2d or vRabi
################################################################################

async def T1_2d_s(measure,t_rabi,flux={},exstate=[],calimatrix=None,pop=False):
    for j in t_rabi:
        task_z = await executeZwave(measure,cww.zWave,dcstate=flux,qnum=len(flux),calimatrix=calimatrix,\
            offset=0,during=j/1e9,shift=100e-9,args='volt')
        await concurrence(task_z)

        task_ex = await executeEXwave(measure,cww.rabiWave,exstate=exstate,shift=(j/1e9+200e-9))
        await concurrence(task_ex)

        s = await popRead(measure,pop=pop)
        yield [j]*measure.n, s

async def T1_2d(measure,t_rabi,v_rabi,dcstate={'q1':0},exstate=[],pop=False,calimatrix=None):
    len_data = len(t_rabi)
    qubit_ex = measure.qubits[exstate[0]]
    # printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    for i in v_rabi:
        if dcstate is not None:
            flux = {qubit_ex.q_name:i}
        else:
            dcstate[qubit_ex.q_name] = i
            flux = dcstate
        job = Job(T1_2d_s, (measure,t_rabi,flux,exstate,calimatrix,pop),tags=exstate, no_bar=False, auto_save=True)
        t_t, s_t = await job.done()
        yield [i]*measure.n, t_t, s_t

################################################################################
# 纵场调制
################################################################################

async def vModulation_single(measure,f_target,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    # flux = {dcstate[0]:0,exstate[0]:v_rabi}
    flux = {dcstate[0]:0,exstate[0]:v_rabi}
    # task_z = await executeZseq(measure,cww.Z_sequence_im,len_data,dcstate=flux,readseq=False,calimatrix=calimatrix,qnum=2,\
    #     v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,kws='volt')
    # await concurrence(task_z)
    await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=2,\
        v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=200e-9)
    if comwave:
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+200)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

async def vModulation_pop(measure,f_target,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    for i, j in enumerate(v_rabi):
        output = True if i == 0 else False
        job = Job(vModulation_single, (measure,f_target,t_rabi,j,dcstate,exstate,calimatrix,output),auto_save=False,no_bar=True)
        t, sv = await job.done()
        yield [j/2/np.pi]*measure.n, t, sv

################################################################################
# 根号iswap
################################################################################

async def iswap_sqrt(measure,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,repeats=600,auto_save=False):
    qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    # flux = {dcstate[0]:0,exstate[0]:v_rabi}
    flux = {dcstate[0]:qubit_z.volt_swap,exstate[0]:qubit_ex.volt_swap}
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,True,mode='vbroadcast',dcstate=flux,\
        calimatrix=calimatrix,qnum=2,v_or_t=t_rabi/1e9,arg='during',shift=20e-9)
    await concurrence(task_z)
    if comwave:
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+40)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data,repeats=repeats,auto_save=auto_save)
    yield t, pop_m

################################################################################
# 纵场调制
################################################################################

async def scanZ_single(measure,f_target,imAmp,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    # flux = {dcstate[0]:0,exstate[0]:v_rabi}
    flux = {dcstate[0]:0,exstate[0]:imAmp}
    # task_z = await executeZseq(measure,cww.Z_sequence_im,len_data,dcstate=flux,readseq=False,calimatrix=calimatrix,qnum=2,\
    #     v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,kws='volt')
    # await concurrence(task_z)
    await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=2,\
        v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=200e-9)
    if comwave:
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+200)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

async def scanZ_pop(measure,f_target,imAmp,t_rabi,dcstate=[],exstate=[],calimatrix=None):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    v1 = np.copy(qubit_ex.volt)
    v_rabi = np.linspace(-0.006,0.006,7) + v1
    for i, j in enumerate(v_rabi):
        qubit_ex.volt = j
        output = True if i == 0 else False
        job = Job(scanZ_single, (measure,f_target,imAmp,t_rabi,dcstate,exstate,calimatrix,output),auto_save=False,no_bar=True)
        t, sv = await job.done()
        yield [j]*measure.n, t, sv
    qubit_ex.volt = v1

################################################################################
# 动力学局域化
################################################################################

async def dLocalization(measure,f_target,center,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp

    if comwave:
        await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
            v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=100e-9)
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+150)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    # await measure.psg['psg_ex1'].setValue('Output','OFF')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

async def dLocalization_wave(measure,f_target,center,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp

    for l,i in enumerate(t_rabi):
        output = True if l == 0 else False
        flux_im = {}
        shiftlist = {}
        for k in flux:
            q = measure.qubits[k]
            # paras['imAmp']=dcstate[k]
            pulse_im,shifttime = await cww.funcarg(cww.zWave_im,q,during=i/1e9,f_ex=f_target,delta_im=120e6,shift=100e-9)
            flux_im[k] = pulse_im[0] 
            shiftlist[k] = shifttime
        current_im = await zManage(measure,flux_im,calimatrix=calimatrix,qnum=np.shape(calimatrix)[0])
        for c in current_im:
            l = len(measure.t_new)
            x = np.zeros(l)
            num = int(shiftlist[c]/0.4e-9)
            index = l-num
            x[:index] = current_im[c][num:]
            current_im[c] = x
        for k in flux:
            name_z = [f'ch{i}' for i in measure.qubits[k].inst['z_ch']]
            awg_z = measure.awg[measure.qubits[k].inst['z_awg']]
            pulse = current_im[k] 
            # chlst, namelst = measure.qubits[k].inst['z_ch']
            await cww.writeWave(measure,awg_z,name_z,[pulse])
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,shift=(i+150)/1e9)
        await concurrence(task)

        await cww.openandcloseAwg(measure,'ON')
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop = readPop(measure,ss)
        yield [i]*measure.n, np.array(pop)[:,1] 

################################################################################
# 动力学局域化_check
################################################################################

async def dLocalization_check(measure,f_target,center,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,volt=0):
    v_m = np.copy(measure.qubits[exstate[0]].volt)
    for i in volt:
        measure.qubits[exstate[0]].volt = i
        job = Job(dLocalization,(measure,f_target,center,t_rabi,dcstate,exstate,calimatrix,comwave),\
            no_bar=True,auto_save=False)
        td, pop_d = await job.done()
        yield [i]*measure.n, td, pop_d
    measure.qubits[exstate[0]].volt = float(v_m)

################################################################################
# 多体Echo
################################################################################

async def mEcho(measure,f_target,center,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,readseq=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp
    await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
            v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=100e-9,center=center/1e9)
    if comwave:
        # await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
        #     v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=100e-9,center=center/1e9)
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=readseq,v_or_t=(t_rabi+150)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data,readcali=True)
    yield t, pop_m

################################################################################
# OTOC
################################################################################

async def oToc(measure,f_target,gateduring,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,readseq=True,repeats=600,auto_save=False):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp

    await executeLseq(measure,cww.Z_sequence_otoc,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
            v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=30e-9,gateduring=gateduring/1e9)
    if comwave:
        
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=readseq,v_or_t=(t_rabi*2+gateduring+100)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data,repeats=repeats,auto_save=auto_save)
    yield t, pop_m

################################################################################
# gateDuring
################################################################################

async def gateDuring(measure,f_target,gateduring,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,readseq=True):

    len_data, t = len(gateduring), np.array([gateduring]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp

    await executeLseq(measure,cww.Z_sequence_otoc,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
            v_or_t=gateduring/1e9,arg='gateduring',f_ex=f_target,delta_im=120e6,shift=100e-9,during=t_rabi/1e9)
    if comwave:
        
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=readseq,v_or_t=(gateduring+160)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

################################################################################
# oToc_tomo
################################################################################

async def oToc_wave_single(measure,f_target,gateduring,t_rabi,volt_zgate,qubit,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(volt_zgate), np.array([volt_zgate]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp
    volt_m = np.copy(qubit.volt_zgate)
    gateduring_m = np.copy(gateduring)
    task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=True,shift=((t_rabi+gateduring)/1e9+qubit.pi_len))
    await concurrence(task)
    for l,i in enumerate(volt_zgate):
        if l == 0:
            gateduring = 0
        else:
            gateduring = float(gateduring_m)
            qubit.volt_zgate = i
        output = True if l == 0 else False
        # task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,shift=((i+gateduring)/1e9+qubit.pi_len))
        # await concurrence(task)
        flux_im = {}
        shiftlist = {}
        for k in flux:
            q = measure.qubits[k]
            # paras['imAmp']=dcstate[k]
            pulse_im,shifttime = await cww.funcarg(cww.zWave_otoc,q,during=t_rabi/1e9,gateduring=gateduring/1e9,\
                f_ex=f_target,delta_im=120e6,shift=qubit.pi_len)
            flux_im[k] = pulse_im[0] 
            shiftlist[k] = shifttime
        current_im = await zManage(measure,flux_im,calimatrix=calimatrix,qnum=np.shape(calimatrix)[0])
        for c in current_im:
            l = len(measure.t_new)
            x = np.zeros(l)
            num = int(shiftlist[c]/0.4e-9)
            index = l-num
            x[:index] = current_im[c][num:]
            current_im[c] = x
        for k in flux:
            name_z = [f'ch{i}' for i in measure.qubits[k].inst['z_ch']]
            awg_z = measure.awg[measure.qubits[k].inst['z_awg']]
            pulse = current_im[k] 
            # chlst, namelst = measure.qubits[k].inst['z_ch']
            await cww.writeWave(measure,awg_z,name_z,[pulse])
        poplist = []
        popm = []
        for axis in ['Ynhalf','Xhalf','Z']:
            task_tomo = await executeEXwave(measure,cww.oToc_tomo_wave,exstate=[qubit.q_name],output=output,\
                axis=axis,during=t_rabi/1e9,gateduring=gateduring/1e9)
            await concurrence(task_tomo)

            await cww.openandcloseAwg(measure,'ON')
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A,ch_B
            ss = Am + 1j*Bm
            pop = readPop(measure,ss)
            popm.append(ss)
            poplist.append(np.array(pop)[:,1])
        yield [i]*measure.n, poplist, 1
    qubit.volt_zgate = float(volt_m)

async def oToc_tomo(measure,f_target,gateduring,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True,readseq=True):




    len_data, t = len(gateduring), np.array([gateduring]*measure.n).T
    flux = {i:measure.qubits[i].imAmp for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].imAmp

    await executeLseq(measure,cww.Z_sequence_otoc,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=np.shape(calimatrix)[0],\
            v_or_t=gateduring/1e9,arg='gateduring',f_ex=f_target,delta_im=120e6,shift=100e-9,during=t_rabi/1e9)
    if comwave:
        
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=readseq,v_or_t=(gateduring+160)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

################################################################################
# quantum walk
################################################################################

async def qWalk(measure,t_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):
    
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {i:measure.qubits[i].volt for i in dcstate}
    flux[exstate[0]] = measure.qubits[exstate[0]].volt
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,
        qnum=np.shape(calimatrix)[0],v_or_t=t_rabi/1e9,arg='during',shift=20e-9)
    await concurrence(task_z)
    if comwave:
        
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+40)/1e9,arg='shift')
        # await concurrence([{**task_ex[0],**task_z[0]},{**task_ex[1],**task_z[1]},{**task_ex[2],**task_z[2]}])
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    # await measure.psg['psg_ex1'].setValue('Output','OFF')
    pop_m = await seqReadpop(measure,len_data,readcali=True)
    yield t, pop_m

################################################################################
# 纵场拉比
################################################################################

# async def lRabi(measure,f_target,t_rabi,freq,dcstate=[]):
#     qubit = measure.qubits[dcstate[0]]
#     len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
#     flux = {i:measure.qubits[i].imAmp for i in dcstate}
#     await executeLseq(measure,len_data,comwave=True,dcstate=flux,readseq=False,qnum=1,\
#         v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=0e-9)
#     await cww.openandcloseAwg(measure,'ON')
#     f = np.copy(qubit.f_ex)
#     for j,i in enumerate(freq):
#         qubit.f_ex = i
#         comwave = True if j == 0 else False
#         numrepeat, avg, measure.repeat = (300,True,len(t_rabi))
#         job = Job(rabi_seq, (measure,t_rabi,'pi_len',dcstate,comwave,comwave), tags=dcstate, max=numrepeat,avg=avg,auto_save=False)
#         v_rp, s_rp = await job.done()
#         yield [i]*measure.n, v_rp, s_rp
#     qubit.f_ex = f

# async def lRabi(measure,f_target,t_rabi,volt,dcstate=[]):
#     qubit = measure.qubits[dcstate[0]]
#     namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
#     chlist = qubit.inst['z_ch']
#     awg_z = measure.awg[qubit.inst['z_awg']]
            
#     for j,i in enumerate(volt):
#         # qubit.f_ex = i
#         pulse_im,shifttime = await cww.funcarg(cww.zWave_im,qubit,during=(len(t_new)/2.5/2e9+2000e-9),volt=i,delta_im=120e6)
#         l = len(measure.t_new)
#         x = np.zeros(l)
#         num = int(shifttime/0.4e-9)
#         index = l-num
#         x[:index] = pulse_im[0][num:]
#         pulselist = x
#         await cww.writeWave(measure,awg_z,namelist,[pulselist])
#         await cww.couldRun(measure,awg_z,chlist,namelist)

#         comwave = True if j == 0 else False
#         numrepeat, avg, measure.repeat = (300,True,len(t_rabi))
#         job = Job(rabi_seq, (measure,t_rabi,'pi_len',dcstate,comwave,comwave), tags=dcstate, max=numrepeat,avg=avg,auto_save=False)
#         v_rp, s_rp = await job.done()
#         yield [i]*measure.n, v_rp, s_rp

async def lRabi(measure,f_target,t_rabi,imAmp,dcstate=[]):
    qubit = measure.qubits[dcstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    chlist = qubit.inst['z_ch']
    awg_z = measure.awg[qubit.inst['z_awg']]
    # len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    for j,i in enumerate(imAmp):
        # flux = {dcstate[0]:i}
        pulse_im,shifttime = await cww.funcarg(cww.zWave_im,qubit,during=(len(t_new)/2.5/2e9+2000e-9),volt=0,offset=0,delta_im=120e6,imAmp=i)
        l = len(measure.t_new)
        x = np.zeros(l)
        num = int(shifttime/0.4e-9)
        index = l-num
        x[:index] = pulse_im[0][num:]
        pulselist = x
        await cww.writeWave(measure,awg_z,namelist,[pulselist])
        await cww.couldRun(measure,awg_z,chlist,namelist)
        comwave = True if j == 0 else False
        numrepeat, avg, measure.repeat = (300,True,len(t_rabi))
        job = Job(rabi_seq, (measure,t_rabi,'pi_len',dcstate,comwave,comwave), tags=dcstate, max=numrepeat,avg=avg,auto_save=False)
        v_rp, s_rp = await job.done()
        yield [i/2/np.pi]*measure.n, v_rp, s_rp


################################################################################
# 纵场T1
################################################################################

async def lT1(measure,f_target,t_rabi,imAmp,dcstate=[]):
    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    qubit = measure.qubits[dcstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    chlist = qubit.inst['z_ch']
    awg_z = measure.awg[qubit.inst['z_awg']]
    # len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    for j,i in enumerate(imAmp):
        flux = {dcstate[0]:i}

        await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=None,readseq=False,qnum=1,\
            v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=100e-9)
        await cww.openandcloseAwg(measure,'ON')
        # pulse_im,shifttime = await cww.funcarg(cww.zWave_im,qubit,during=(len(t_new)/2.5/2e9+2000e-9),volt=0,offset=0,delta_im=120e6,imAmp=i)
        # l = len(measure.t_new)
        # x = np.zeros(l)
        # num = int(shifttime/0.4e-9)
        # index = l-num
        # x[:index] = pulse_im[0][num:]
        # pulselist = x
        # await cww.writeWave(measure,awg_z,namelist,[pulselist])
        # await cww.couldRun(measure,awg_z,chlist,namelist)
        comwave = True if j == 0 else False
        numrepeat, avg, measure.repeat = (300,True,len(t_rabi))
        job = Job(rabi_seq, (measure,(t_rabi+200),'shift',dcstate,comwave,comwave), tags=dcstate, max=numrepeat,avg=avg,auto_save=False)
        v_rp, s_rp = await job.done()
        yield [i/2/np.pi]*measure.n, v_rp, s_rp


# ###################################################
# ###################################################

# async def vModulation_single(measure,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,which={}):
#     dc = {i:0 for i in (dcstate+exstate)}
#     for i, k in enumerate(v_rabi):
#         output = True if i == 0 else False
#         dc[dcstate[0]] = k
#         task_z = await executeZwave(measure,cww.zWave_im,dcstate=dc,qnum=len(dc),calimatrix=calimatrix,output=output,args='imAmp',\
#             during=t_rabi/1e9,delta_im=120e6)
#         await concurrence(task_z)
#         ch_A, ch_B = await measure.ats.getIQ(hilbert=True)
#         Am, Bm = ch_A,ch_B
#         ss = Am + 1j*Bm
#         # d = list(zip(np.real(ss),np.imag(ss)))
#         d = [list(zip(np.real(ss)[:,i],np.imag(ss)[:,i])) for i in range(np.shape(ss)[1])]
#         y = [measure.predict[j](d[i]) for i, j in enumerate(measure.qubitToread)]
#         pop = np.count_nonzero(y,axis=1)/np.shape(y)[1]
#         pop = np.array([pop[i] if which[j] == 1 else 1-pop[i] for i, j in enumerate(measure.qubitToread)])
#         yield [k]*measure.n, pop

# async def vModulation_pop(measure,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,which={}):
    
#     # qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
#     # z_awg = measure.awg[qubit.inst['z_awg']]
#     # ex_awg = measure.awg[qubit.inst['ex_awg']]
#     # zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
#     # exname, exch = [f'ch{i}' for i in qubit.inst['ex_ch']], qubit.inst['ex_ch']
#     # await cww.couldRun(measure,ex_awg,exch,exname)
#     # # await cww.couldRun(measure,z_awg,zch,zname)

#     for i, j in enumerate(t_rabi):
#         output = True if i == 0 else False
#         task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,output=output,shift=j/1e9)
#         await concurrence(task)
#         job = Job(vModulation_single, (measure,j,v_rabi,dcstate,exstate,calimatrix,which),max=len(v_rabi))
#         vv, sv = await job.done()
#         yield [j]*measure.n, vv, sv     

################################################################################
# 纵场相位校准
################################################################################

async def phaseCali_single(measure,f_target,t_rabi,v_rabi,dcstate=[],exstate=[],calimatrix=None,comwave=True):

    len_data, t = len(t_rabi), np.array([t_rabi]*measure.n).T
    flux = {dcstate[0]:0,exstate[0]:v_rabi}
    # task_z = await executeZseq(measure,cww.Z_sequence_im,len_data,dcstate=flux,readseq=False,calimatrix=calimatrix,qnum=2,\
    #     v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,kws='volt')
    # await concurrence(task_z)
    await executeLseq(measure,cww.Z_sequence_im,len_data,comwave=True,dcstate=flux,calimatrix=calimatrix,readseq=False,qnum=2,\
        v_or_t=t_rabi/1e9,arg='during',f_ex=f_target,delta_im=120e6,shift=200e-9)
    if comwave:
        task_ex = await executeEXseq(measure,cww.Rabi_sequence,len_data,True,exstate,readseq=True,v_or_t=(t_rabi+200)/1e9,arg='shift')
        await concurrence(task_ex)
    await cww.openandcloseAwg(measure,'ON')
    pop_m = await seqReadpop(measure,len_data)
    yield t, pop_m

async def phaseCali(measure,f_target,t_rabi,imAmp,phase,dcstate=[],exstate=[],calimatrix=None):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    for i, j in enumerate(phase):
        output = True if i == 0 else False
        qubit_ex.phaseim = j
        job = Job(phaseCali_single, (measure,f_target,t_rabi,imAmp,dcstate,exstate,calimatrix,output),auto_save=False,no_bar=True)
        t, sv = await job.done()
        yield [j]*measure.n, t, sv

# async def phaseCali(measure,phase,mode='vbroadcast',dcstate=[],exstate=[],calimatrix=None):
#     len_data, t = len(phase), np.array([phase]*measure.n).T
#     qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
#     exname, ex_awg = [f'ch{i}' for i in qubit_ex.inst['z_ch']], measure.awg[qubit_ex.inst['z_awg']]
#     exch = qubit_ex.inst['z_ch']

#     dc = {i:0 for i in (dcstate+exstate)}
#     dc[dcstate[0]] = 0.8
#     current = await zManage(measure,dcstate=dc,qnum=len(dc),calimatrix=calimatrix)

#     task_ex = await executeEXwave(measure,cww.phaseCaliwave,exstate=exstate,output=True,t_run=300e-9)
#     await concurrence(task_ex)
#     task_zex = await executeZseq(measure,cww.Z_sequence_cos,len_data,True,mode=mode,dcstate=dcstate,\
#         v_or_t=phase,arg='phaseim',during=200e-9,shift=20e-9,volt=0,delta_im=120e6,imAmp=current[dcstate[0]][0])
#     await concurrence(task_zex)

#     pulse = await cww.funcarg(cww.zWave_cos,qubit_ex,during=200e-9,shift=20e-9,delta_im=120e6,volt=0,imAmp=current[exstate[0]][0])
#     await cww.writeWave(measure,ex_awg,exname,pulse)
#     await cww.couldRun(measure,ex_awg,exch,namelist=exname)
#     await measure.psg['psg_lo'].setValue('Output','ON')

#     for i in range(300):
#         ch_A, ch_B = await measure.ats.getIQ()
#         Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
#         # theta0 = np.angle(Am) - np.angle(Bm)
#         # Bm *= np.exp(1j*theta0)
#         s = Am + 1j*Bm
#         yield t, s - measure.base  

################################################################################
# 纵场频率校准
################################################################################

async def freqCali(measure,t_run,mode='vbroadcast',dcstate=[],exstate=[],calimatrix=None):
    len_data, t = len(t_run), np.array([t_run]*measure.n).T
    qubit_ex = measure.qubits[exstate[0]]
    flux = {dcstate[0]:measure.qubits[dcstate[0]].volt,exstate[0]:measure.qubits[exstate[0]].volt}
    task_ex = await executeEXseq(measure,cww.Coherence_sequence,len_data,True,exstate,True,mode,\
        v_or_t=(t_run+100)/1e9,arg='t_run')
 
    task_zex = await executeZseq(measure,cww.Z_sequence,len_data,True,mode=mode,dcstate=flux,calimatrix=calimatrix,\
        v_or_t=t_run/1e9,arg='during',shift=(10e-9+qubit_ex.pi_len),kws='volt',qnum=2)
    await concurrence([{**task_ex[0],**task_zex[0]},{**task_ex[1],**task_zex[1]},{**task_ex[2],**task_zex[2]}])

    await measure.psg['psg_lo'].setValue('Output','ON')

    for i in range(300):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base  


################################################################################
# 共振处频率
################################################################################

async def anticross(qubit_bias,qubit_aux,measure,current,freq,calimatrix,modulation=False):
    
    # namelist = [f'ch{i}' for i in qubit.inst['z_ch']]
    # chlist = qubit.inst['z_ch']
    # z_awg = measure.awg[qubit.inst['z_awg']]
    
    for j,i in enumerate(current):
        output = True if j== 0 else False
        task = await executeZwave(measure,cww.zWave,dcstate={qubit_bias.q_name:i,qubit_aux.q_name:qubit_aux.volt},\
            calimatrix=calimatrix,output=output,qnum=2,during=(len(measure.t_new)/2.5/2e9+2000e-9),shift=100e-9)
        await concurrence(task)
        
        job = Job(singlespec, (measure,freq,modulation,measure.f_lo,False,[qubit_bias.q_name]),auto_save=False,max=len(freq))
        f_ss, s_ss = await job.done()
        n = np.shape(s_ss)[1]
        yield [i]*n, f_ss, s_ss

################################################################################
# Ramsey
################################################################################

async def Ramsey(measure,t_run,exstate=[],pop=False):
    for i in (exstate):
        measure.qubits[i].replace(nwave=0,seqtype='PDD',detune=4e6)
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    qubit = measure.qubits[exstate[0]]
    # namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,awg_ex,chlist,namelist=namelist)

    for i in t_run:
        # pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        # await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        # await cww.couldRun(measure,awg_ex)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_run=i/1e9)
        await concurrence(task)
        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s
    for i in (exstate):
        measure.qubits[i].replace(nwave=1,seqtype='CPMG')
    qubit.replace(nwave=1,seqtype='CPMG')

################################################################################
# Ramsey2q
################################################################################

async def Ramsey2q(measure,t_run,exstate=[],zstate={},pop=False,calimatrix=None):
    for i in (exstate):
        measure.qubits[i].replace(nwave=0,seqtype='PDD',detune=4e6)
    qubit = measure.qubits[exstate[0]]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]
    await measure.psg['psg_lo'].setValue('Output','ON')

    for i in t_run:
        task_z1 = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
            offset=0,during=(i)*1e-9,args='volt')
        await concurrence(task_z1)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_run=i/1e9)
        await concurrence(task)
        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s
    for i in (exstate):
        measure.qubits[i].replace(nwave=1,seqtype='CPMG')

async def Ramsey2qZ(measure,t_run,exstate=[],z_off=[],pop=False,calimatrix=None):
    q1 =  measure.qubits[exstate[0]]
    q2 =  measure.qubits[exstate[1]]
    
    for i in z_off:
        zstate={exstate[0]:q1.volt_swap+i,exstate[1]:q2.volt_swap}
        job = Job(Ramsey2q,(measure,t_run,[exstate[0]],zstate,pop,calimatrix), tags=exstate, max=len(t_run))
        t_ram2, s_ram2 = await job.done()
        yield [i+q1.volt_swap]*measure.n, t_ram2, s_ram2
    await cww.OffEx(exstate,measure)
    await cww.OffZ(exstate,measure)

################################################################################
# SpinEcho
################################################################################

async def SpinEcho(measure,t_run,exstate=[],pop=False):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])
    qubit = measure.qubits[exstate[0]]
    # namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    # await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')
    # await cww.couldRun(measure,awg_ex,chlist,namelist=namelist)
    for i in t_run:
        # pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=i/1e9)
        # await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        # await cww.couldRun(measure,awg_ex)
        task = await executeEXwave(measure,cww.coherenceWave,exstate=exstate,t_run=i/1e9)
        await concurrence(task)
        s = await popRead(measure,pop=pop)
        yield [i]*measure.n, s

################################################################################
# spec crosstalk cali
################################################################################

async def single_cs(measure,t,len_data):
    for i in range(300):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s 

async def crosstalkSpec(measure,v_rabi,dcstate=[],exstate=[],comwave=False):
    qubit_zz = measure.qubits[dcstate[0]]
    zz_awg = measure.awg[qubit_zz.inst['z_awg']]
    zname, zch = [f'ch{i}' for i in qubit_zz.inst['z_ch']], qubit_zz.inst['z_ch']
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    qubit_ex = measure.qubits[exstate[0]]
    await measure.psg['psg_trans'].setValue('Frequency',qubit_ex.f_ex)
    await measure.psg['psg_trans'].setValue('Output','ON')
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave,dcstate=exstate,\
        v_or_t=v_rabi,arg='volt',offset=0,during=(len(measure.t_new)/2.5/2e9+2000e-9),shift=200e-9)
    await concurrence(task_z)

    await measure.psg['psg_lo'].setValue('Output','ON')
    for i in np.linspace(-1,1,11):

        pulse = await cww.funcarg(cww.zWave,qubit_zz,during=(len(measure.t_new)/2.5/2e9+2000e-9),volt=i,shift=200/1e9)
        await cww.writeWave(measure,zz_awg,zname,pulse,mark=False)
        await cww.couldRun(measure,zz_awg,namelist=zname,chlist=zch)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=300,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

async def crosstalkSpec_second(measure,v_rabi,dcstate=[],exstate=[],comwave=False,calimatrix=None):
    qubit_zz = measure.qubits[dcstate[0]]
    # zz_awg = measure.awg[qubit_zz.inst['z_awg']]
    # zname, zch = [f'ch{i}' for i in qubit_zz.inst['z_ch']], qubit_zz.inst['z_ch']
    len_data, t = len(v_rabi), np.array([v_rabi]*measure.n).T
    
    qubit_ex = measure.qubits[exstate[0]]
    await measure.psg['psg_trans'].setValue('Frequency',qubit_ex.f_ex)
    await measure.psg['psg_trans'].setValue('Output','ON')

    await measure.psg['psg_lo'].setValue('Output','ON')
    for j,i in enumerate(np.linspace(-0.9,0.9,11)):
        output = True if j == 0 else False
        flux = {qubit_zz.q_name:[i]*len(v_rabi),qubit_ex.q_name:v_rabi}
        task = await executeZseq(measure,cww.Z_sequence,len_data,dcstate=flux,\
            calimatrix=calimatrix,comwave=True,readseq=output,arg='volt',qnum=2,offset=0,during=(len(measure.t_new)/2.5/2e9+2000e-9),shift=100e-9)
        await concurrence(task)
        job = Job(single_cs,(measure,t,len_data),avg=True,max=300,auto_save=False,no_bar=True)
        v_bias, s =await job.done()
        n = np.shape(s)[1]
        yield [i]*n, v_bias, s

################################################################################
# ramsey crosstalk cali
################################################################################

async def Rcscali_single(measure,v_or_t,during=0,arg='volt',exstate=[],dcstate=[],calimatrix=None,comwave=False):
    qubit_ex = measure.qubits[exstate[0]]
    ex_name = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    ex_ch = qubit_ex.inst['ex_ch']
    ex_awg = measure.awg[qubit_ex.inst['ex_awg']]
    len_data, t = len(v_or_t), np.array([v_or_t]*measure.n).T
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')

    pulse1 = await cww.funcarg(qgw.singleQgate,qubit_ex,axis='Xhalf',shift=(qubit_ex.pi_len+900e-9))
    pulse2 = await cww.funcarg(qgw.singleQgate,qubit_ex,axis='AnyRhalf',phase=2*np.pi*0.25e6*900e-9)
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)
    v_or_t = v_or_t if arg == 'volt' else v_or_t/1e9
    task_z = await executeZseq(measure,cww.Z_sequence,len_data,comwave,dcstate=dcstate,calimatrix=calimatrix,\
        v_or_t=v_or_t,arg=arg,during=during*1e-9,shift=(qubit_ex.pi_len+150/1e9))
    await concurrence(task_z)
    for i in range(300):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

################################################################################
# Phase Gate
################################################################################

async def Phase_gate(measure,v_cali,gateduring=0,arg='volt',num_gate=[1],exstate=[],dcstate=[],calimatrix=None,comwave=False):
    qubit_ex = measure.qubits[exstate[0]]
    ex_name = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    ex_ch = qubit_ex.inst['ex_ch']
    ex_awg = measure.awg[qubit_ex.inst['ex_awg']]
    len_data, t = len(v_cali), np.array([v_cali]*measure.n).T
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Frequency',(qubit_ex.f_ex+qubit_ex.delta_ex))

    pulse1 = await cww.funcarg(qgw.singleQgate,qubit_ex,axis='Xhalf',shift=(qubit_ex.pi_len+100e-9+num_gate*(gateduring+10)*1e-9))
    pulse2 = await cww.funcarg(qgw.singleQgate,qubit_ex,axis='Xhalf')
    pulse = np.array(pulse1) + np.array(pulse2)
    await cww.writeWave(measure,ex_awg,ex_name,pulse)
    await cww.couldRun(measure,ex_awg,namelist=ex_name,chlist=ex_ch)

    task_z = await executeZseq(measure,cww.Z_sequence_pg,len_data,comwave,dcstate=dcstate,calimatrix=calimatrix,\
        v_or_t=(v_cali-qubit_ex.volt),arg=arg,num_gate=num_gate,during=gateduring*1e-9,shift=(qubit_ex.pi_len+50/1e9))
    await concurrence(task_z)
    for i in range(300):
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A[1:len_data+1,:],ch_B[1:len_data+1,:]
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield t, s - measure.base

async def Phase_gate_2d(measure,v_cali,gateduring=0,arg='volt',num_gate=1,exstate=[],dcstate=[],calimatrix=None,comwave=False):
    for i in range(num_gate):
        num = 2*i+1
        job = Job(Phase_gate, (measure,v_cali,gateduring,arg,num,exstate,dcstate,calimatrix,comwave), \
                tags=exstate,avg=True, max=300,auto_save=False)
        v_ex, s_sc = await job.done()
        yield [num]*measure.n, v_ex, s_sc

################################################################################
# Phase Gate tomo
################################################################################

async def Phase_gate_tomo(measure,gateduring=0,v_cali=0,num_gate=1,exstate=[],dcstate=[],calimatrix=None,comwave=False):
    qubit_ex = measure.qubits[exstate[0]]
    ex_name = [f'ch{i}' for i in qubit_ex.inst['ex_ch']]
    ex_ch = qubit_ex.inst['ex_ch']
    ex_awg = measure.awg[qubit_ex.inst['ex_awg']]
    
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Output','ON')
    await measure.psg[qubit_ex.inst['ex_lo']].setValue('Frequency',(qubit_ex.f_ex+qubit_ex.delta_ex))

    for v_scan in v_cali:

        task_z = await executeZwave(measure,cww.zWave,dcstate=dcstate,calimatrix=calimatrix,\
           arg='volt',volt=(v_scan+qubit_ex.volt),during=gateduring/1e9,shift=(qubit_ex.pi_len+50/1e9))
        await concurrence(task_z)

        poplist = []
        popm = []
        for axis in ['Ynhalf','Xhalf','Z']:
            task_tomo = await executeEXwave(measure,cww.phasegate_tomo_wave,exstate=[qubit_ex.q_name],output=output,\
                axis=axis,during=100/1e9,gateduring=num_gate*gateduring/1e9)
            await concurrence(task_tomo)

            await cww.openandcloseAwg(measure,'ON')
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A,ch_B
            ss = Am + 1j*Bm
            pop = readPop(measure,ss)
            popm.append(ss)
            poplist.append(np.array(pop)[:,1])
        yield [i]*measure.n, poplist, 1



################################################################################
# qqTiming
################################################################################

async def qqTiming(measure,t_rabi,exstate=[],dcstate=[],calimatrix=None,volt=None,pop=False):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    printLog(qubit_z.asdict())
    printLog(qubit_ex.asdict(),mode='a')
    f_mean = (qubit_ex.f_ex+qubit_z.f_ex)/2
    volt_ex, vtarget = dt.biasshift(qubit_ex.specfuncz,qubit_ex.f_ex/1e9,(f_mean-qubit_ex.f_ex)/1e9,side='higher') 
    volt_z, vtarget = dt.biasshift(qubit_z.specfuncz,qubit_z.f_ex/1e9,(f_mean-qubit_z.f_ex)/1e9,side='higher') 

    if volt !=None:
        volt_ex, volt_z =volt[0],volt[-1]

    task_z1 = await executeZwave(measure,cww.zWave,dcstate={dcstate[0]:volt_z},qnum=len(dcstate),calimatrix=calimatrix,\
        offset=0,during=25e-9,shift=1000e-9,args='volt')
    await concurrence(task_z1)

    for j in t_rabi:

        task_z2 = await executeZwave(measure,cww.zWave,dcstate={exstate[0]:volt_ex},qnum=len(exstate),calimatrix=calimatrix,\
        offset=0,during=25e-9,shift=j*1e-9,args='volt')
        await concurrence(task_z2)
        para ={'shift':np.max(t_rabi+100)/1e9}
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,**para)
        await concurrence(task)

        s = await popRead(measure,pop=pop)
        yield [j]*measure.n, s


################################################################################
# RTO_Notomo
################################################################################

async def RTO_Notomo(measure,t_run,exstate=[]):
    await cww.couldRun(measure,measure.awg['awgread'],[1,5])
    qubit = measure.qubits[exstate[0]]
    namelist = [f'ch{i}' for i in qubit.inst['ex_ch']]
    chlist = qubit.inst['ex_ch']
    awg_ex = measure.awg[qubit.inst['ex_awg']]

    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg[qubit.inst['ex_lo']].setValue('Output','ON')

    for i in t_run:
        pulselist = await cww.funcarg(cww.coherenceWave,qubit,t_run=500/1e9)
        await cww.writeWave(measure,awg_ex,name=namelist,pulse=pulselist)
        await cww.couldRun(measure,awg_ex,chlist)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A.mean(axis=0),ch_B.mean(axis=0)
        # Im, Qm = I.mean(axis=0), Q.mean(axis=0)
        # sq = Im + 1j*Qm
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        s = Am + 1j*Bm
        yield [i], s-measure.base


################################################################################
# RB waveform
################################################################################

async def RB_waveform(measure,mlist,len_data,gate=None,exstate=[]):
    
    # pulse = await cww.modulation_read(measure,measure.delta,readlen=measure.readlen,repeats=2000)
    await cww.ats_setup(measure,measure.delta,readlen=measure.readlen,repeats=2000)
    # awg, qname = measure.awg[qubit.inst['ex_awg']], [qubit.q_name+'_I', qubit.q_name+'_Q']

    await measure.psg['psg_lo'].setValue('Output','ON')

    for k,j in enumerate(tqdm(mlist,desc='RB')):
        pop = []
        for i in range(len_data):
            output = True if k == 0 else False
            task = await executeEXwave(measure,cww.rbWave,exstate=exstate,output=output,m=j,gate=gate)
            await concurrence(task)
            ch_A, ch_B = await measure.ats.getIQ()
            Am, Bm = ch_A,ch_B
            ss = Am + 1j*Bm
            pop.append(np.array(readPop(measure,ss))[:,0])
        yield [j]*measure.n, pop


################################################################################
# XYZ_timing
################################################################################

async def XYZ_timing(measure,t_rabi,exstate=[],calimatrix=None,pop=False,arg='shift',updatez=True):
    flux={exstate[0]:0.05}
    qubit = measure.qubits[exstate[0]]
    if updatez:
        task_z = await executeZwave(measure,cww.zWave,dcstate=flux,qnum=len(flux),calimatrix=calimatrix,\
            offset=0,during=((qubit.pi_len*qubit.envelopename[1])),shift=3000e-9,args='volt')
        await concurrence(task_z)
    for j in t_rabi:
        para = {arg:j} if arg == 'amp' else {arg:j/1e9}
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,**para)
        await concurrence(task)

        s = await popRead(measure,pop=pop)
        yield [j]*measure.n, s

################################################################################
# Z_step
################################################################################

async def singleZ(measure,volt_list,height=0,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    zname, zch = [f'ch{i}' for i in qubit.inst['z_ch']], qubit.inst['z_ch']
    z_awg = measure.awg[qubit.inst['z_awg']]
    pulse1 = await cww.funcarg(cww.zWave,qubit,during=2000/1e9,volt=height,shift=5000e-9,readvolt=0)
    pulse3 = await cww.funcarg(cww.zWave,qubit,during=0,volt=0,shift=0)
    # await cww.couldRun(measure,z_awg,zch,zname)
    
    for k in volt_list:
        pulse2 = await cww.funcarg(cww.zWave,qubit,during=7500/1e9,volt=k,shift=1500e-9,readvolt=0)
        pulse = np.array(pulse1) + np.array(pulse2)
        pulse = np.array(pulse1) + np.array(pulse2) + np.array(pulse3)
        await cww.writeWave(measure,z_awg,zch,pulse)
        # await cww.couldRun(measure,z_awg)
        
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        ss = Am + 1j*Bm
        pop = readPop(measure,ss)
        yield [k]*measure.n, np.array(pop)[:,1]

async def zPulse_pop(measure,t_T1,height=0,exstate=[]):
    qubit = measure.qubits[exstate[0]]
    printLog(qubit.asdict())

    # func = lambda t: -(scipy.special.erf((t-5000)/2)+1)/2*0.8
    sampleStep = 0.0008
    volt_init = np.arange(-0.06,0.06,sampleStep)
    length = len(volt_init)
    index = int(length//2)
    start, end = index-40, index+40
    volt = volt_init[start:end]

    volt_shape=[]
    for k in volt_init:
        volt_shape.append([k]*measure.n)
    
    for i, j in enumerate(t_T1):
        pop_m = np.zeros((length,measure.n))

        para = {'shift':j/1e9}
        task = await executeEXwave(measure,cww.rabiWave,exstate=exstate,**para)
        await concurrence(task)

        # await cww.couldRun(measure,ex_awg)
        job = Job(singleZ,(measure,volt,height,exstate),max=len(volt))
        v, pop = await job.done()
        pop_m[start:end] = pop
        
        if np.max(pop_m[:,qubit.index]) > 0.5 :
            index = np.argmax(pop_m[:,qubit.index])
            start, end = index-12, index+12
            if start < 0:
                start = 0
            if end > length-1:
                end = length-1
            volt = volt_init[start:end]
        else:
            start = 0
            end = length-1
            volt = volt_init[start:end]
            index = len(volt)//2 

        yield [5000-qubit.pi_len*qubit.envelopename[1]/2e-9-j]*measure.n, volt_shape, pop_m


################################################################################
# iswap_optzpas
################################################################################

async def iswap_optzpa_pop(measure,v_other,zpa,dcstate=[],exstate=[],calimatrix=None):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]

    paras = {'shift':(qubit_ex.during_swap+qubit_ex.envelopename[1]*qubit_ex.pi_len+100e-9)}
    paras['output'] = True
    task = await executeEXwave(measure,qgw.singleQgate,exstate=exstate,**paras)
    await concurrence(task)
    # flux = {i:v_other for i in measure.qubits}
    # flux[qubit_z.q_name] = qubit_z.volt_swap
    flux = {qubit_z.q_name:qubit_z.volt_swap,qubit_ex.q_name:qubit_ex.volt_swap}
    for i in zpa:
        flux[qubit_ex.q_name] = i
        task = await executeZwave(measure,cww.swapzWave,dcstate=flux,args='volt_swap',\
        offset=0,qnum=len(flux),output=True,shift=(qubit_ex.envelopename[1]*qubit_ex.pi_len+30e-9),calimatrix=calimatrix,during_swap=qubit_ex.during_swap)
        await concurrence(task)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        # pop = np.array(coRead(measure,ss,readcali=True))
        # yield [i]*(2)**measure.n, pop
        pop = np.array(readPop(measure,ss,readcali=True))
        yield [i]*measure.n, pop

################################################################################
# iswap_optduring
################################################################################

async def iswap_optduring_pop(measure,during,dcstate=[],exstate=[],calimatrix=None):
    qubit_ex = measure.qubits[exstate[0]]
    qubit_z = measure.qubits[dcstate[0]]
    # during_swap = float(np.copy(qubit_ex.during_swap))
    pop_save = []
    for j,i in enumerate(during):
        # qubit_ex.during_swap = i
        # qubit_z.during_swap = i
        paras = {'shift':(qubit_ex.during_swap+100e-9)}
        paras['output'] = True if j == 0 else False
        task = await executeEXwave(measure,qgw.singleQgate,exstate=exstate,**paras)
        await concurrence(task)
        task = await executeZwave(measure,cww.swapzWave,dcstate={qubit_z.q_name:qubit_z.volt_swap,qubit_ex.q_name:qubit_ex.volt_swap},args='volt_swap',\
        offset=0,qnum=2,output=True,shift=(30e-9),calimatrix=calimatrix,during_swap=i)
        await concurrence(task)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        # pop = np.array(coRead(measure,ss,readcali=True))
        # yield [i]*(2)**measure.n, pop
        pop = np.array(readPop(measure,ss))
        pop_save.append(ss)
        yield [i]*measure.n, pop
    status = [measure.onwhich,measure.offwhich,measure.predict,measure.readmatrix,measure.postSle]
    fname = save('iswap_optduring_pop',base_path=r'D:\skzhao\Vmodulation',value=pop_save,status=status,allow_pickle=True)

    # qubit_ex.during_swap = during_swap
    # qubit_z.during_swap = during_swap

################################################################################
# 比特坐标系相位矫正
################################################################################

async def coordinatePhase(measure,phase,dcstate=[],exstate=[],calimatrix=None):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    await measure.psg['psg_lo'].setValue('Output','ON')

    qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]
    task = await executeZwave(measure,cww.zWave,dcstate={qubit_z.q_name:qubit_z.volt_swap,qubit_ex.q_name:qubit_ex.volt_swap},\
        offset=0,qnum=2,output=True,shift=0,calimatrix=calimatrix,during=qubit_ex.during_swap)
    await concurrence(task)
    coordinatePhase_z = float(np.copy(qubit_z.coordinatePhase))
    for j,i in enumerate(phase):
        qubit_z.coordinatePhase = (i+coordinatePhase_z)
        paras = {'shift':(qubit_ex.during_swap),'axis':'Xhalf','virtualPhase':0}
        paras['output'], ex = (True, (exstate+dcstate)) if j ==0 else (False, dcstate)
        task = await executeEXwave(measure,qgw.singleQgate,exstate=ex,**paras)
        await concurrence(task)
        ch_A, ch_B= await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop = np.array(coRead(measure,ss,readcali=True))
        yield [i]*(2)**measure.n, pop
    qubit_z.coordinatePhase = coordinatePhase_z


################################################################################
# Tomography
################################################################################

async def tomo(measure,func,paras,exstate=[],tomostate=[],qpt_init=None,dotomo=True):
    paras_copy = paras.copy()
    ex_will_m = set(tomostate) | set(exstate)
    ex_will = []
    for i in measure.qubits:
        if i in ex_will_m:
            ex_will.append(i)
        else:
            continue
    gate, datastart = (qst.transformList(len(tomostate)),0) if dotomo else ([['I']*len(tomostate)],0)
    pop = []
    for k,j in enumerate(gate):
        qnum = -1
        for i in ex_will:
            if i in tomostate:
                qnum += 1
                if qpt_init is not None:
                    paras['init'] = qpt_init[qnum]
                paras['axis'] = j[qnum]
                if i in exstate:
                    task = await executeEXwave(measure,func,exstate=[i],**paras)
                else:
                    task = await executeEXwave(measure,cww.tomoWave,exstate=[i],**paras)
            else:
                # paras_copy['axis'] = 'X'
                task = await executeEXwave(measure,qgw.singleQgate,exstate=[i],**paras_copy)
            # print(i,paras)
            
            await concurrence(task)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop.append(np.array(coRead(measure,ss,readcali=True))[datastart:])
    yield pop

################################################################################
# singletomo
################################################################################
async def singletomo(measure,exstate,init='I',delay=0e-9):
    gate = ['I','Xhalf', 'Ynhalf']
    for k,j in enumerate(gate):
        task = await executeEXwave(measure,cww.qptomoWave,exstate=exstate,init=init,axis=j,shift=delay*1e-9)
        await concurrence(task)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop = np.array(readPop(measure,ss,readcali=True))
        yield [j]*measure.n, pop
################################################################################
# multitomo
################################################################################
async def multitomo(measure,exstate,tomostate,init=['I'],delay=0e-9,dotomo=True):
    gate, datastart = (qst.transformList(len(tomostate)),0) if dotomo else ([['I']*len(tomostate)],0)
    for k,j in enumerate(gate):
        jb = str()
        for kj,jj in enumerate(j):
            jb +=jj
            task = await executeEXwave(measure,cww.qptomoWave,exstate=[exstate[kj]],init=init[kj],axis=jj,shift=delay*1e-9)
            await concurrence(task)

        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        pop = np.array(readPop(measure,ss,readcali=True))
        yield [jb]*measure.n, pop

################################################################################
# qwalk
################################################################################
async def qwalk(measure,exstate,zstate,init=['I'],axis=['Xhalf'],delay=[],calimatrix=None):
    for during_z in delay:
        # for i,j in enumerate(zstate):
        task_z1 = await executeZwave(measure,cww.zWave,dcstate=zstate,qnum=len(zstate),calimatrix=calimatrix,\
            offset=0,during=(during_z)*1e-9,shift=0e-9,args='volt')
        await concurrence(task_z1)
        for j,i in enumerate(exstate):
            # task = await executeEXwave(measure,cww.qptomoWave,exstate=[exstate[j]],init=init[j],axis=axis[j],shift=during_z*1e-9)
            # await concurrence(task)
            qubit = measure.qubits[i]
            ex_ch = qubit.inst['ex_ch']
            ex_awg = measure.awg[qubit.inst['ex_awg']]
            pulseex = await cww.qptomoWave(qubit,init=init[j],axis=axis[j],shift=(during_z)*1e-9)
            await cww.writeWave(measure,ex_awg,ex_ch,pulseex)
        ch_A, ch_B = await measure.ats.getIQ()
        Am, Bm = ch_A,ch_B
        ss = Am + 1j*Bm
        # pop = np.array(coRead(measure,ss,readcali=True))
        # yield [during_z]*(2)**measure.n, pop
        pop = np.array(readPop(measure,ss,readcali=True))
        yield [during_z]*measure.n, pop

################################################################################
# sqrtiswap_Tomography
################################################################################

async def sqrtiswapTomo(measure,dcstate=[],exstate=[],tomostate=[],calimatrix=None,comwave=True):
    # await cww.couldRun(measure,measure.awg['awgread'],[1,5],['Readout_I','Readout_Q'])

    await measure.psg['psg_lo'].setValue('Output','ON')

    qubit_z, qubit_ex = measure.qubits[dcstate[0]], measure.qubits[exstate[0]]

    # flux = {qubit_z.q_name:qubit_z.volt_swap,qubit_ex.q_name:qubit_ex.volt_swap}
    # for i in measure.qubits:
    #     if measure.qubits[i] != qubit_ex and measure.qubits[i] != qubit_z:
    #         flux[i] = measure.qubits[i].volt + v_other

    # task = await executeZwave(measure,cww.zWave,dcstate=flux,\
    # calimatrix=calimatrix,output=True,during=(qubit_ex.during_swap+30e-9),offset=0,shift=(qubit_ex.envelopename[1]*qubit_ex.pi_len+30e-9))
    # await concurrence(task)
    if comwave:
        task = await executeZwave(measure,cww.swapzWave,dcstate={qubit_z.q_name:qubit_z.volt_swap,qubit_ex.q_name:qubit_ex.volt_swap},args='volt_swap',\
            offset=0,qnum=2,output=True,shift=(qubit_ex.pi_len+0e-9),calimatrix=calimatrix)
        await concurrence(task)

    paras = {'shift':0e-9}
    paras['output'] = True
    job = Job(tomo, (measure,cww.sqrtiswaptomoWave,paras,exstate,tomostate), tags=exstate, no_bar=True)
    poplist = await job.done()
    # index = measure.
    # v = qst.acquireVFromData(len(exstate), np.array(poplist))
    # rho = qst.vToRho(v)
    yield poplist

################################################################################
# iswap_optdvirtualz
################################################################################

async def iswap_optdvirtualz(measure,rho_real,phase,dcstate=[],exstate=[],tomostate=[],calimatrix=None):
    qubit_z = measure.qubits[dcstate[0]]
    virtualPhase = float(np.copy(qubit_z.virtualPhase))
    for j,i in enumerate(phase):
        ##during is not definded
        comwave = True if j == 0 else False
        qubit_z.virtualPhase = i
        job = Job(sqrtiswapTomo, (measure,dcstate,exstate,tomostate,calimatrix,comwave), tags=tomostate,no_bar=True,auto_save=False)
        pop = await job.done()
        v = qst.acquireVFromData(2, np.array(pop)[0,:,1:].flatten())
        # rho = qst.vToRho(v)
        rho = tomography.maximumLikelihoodRho(v)
        fid = qst.fidelity(rho,rho_real)
        yield [i]*measure.n, [fid]*measure.n
    qubit_z.virtualPhase = virtualPhase
