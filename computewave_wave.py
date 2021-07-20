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
from qulab.storage.utils import save
# from easydl import clear_output
# from qulab.wavepoint import WAVE_FORM as WF
# from qulab.waveform import CosPulseDrag, Expi, DC, Step, Gaussian, CosPulse
from qulab.yhs import waveform_new as wn
from tqdm import tqdm_notebook as tqdm
import asyncio, inspect
from qulab.yhs import imatrix as mx, dataTools as dt, optimize, measureroutine_population, qGate_basewave
import functools
import imp, gc
op = imp.reload(optimize)
mrw = imp.reload(measureroutine_population)
qgw = imp.reload(qGate_basewave)


sample_rate = 2  #####################GHz为单位
trig_shift = 10e-9
amp_norm = 32700 
# samplingRate = 2e9

t_start, t_end = -43000, 5000
numofpoints = (t_end-t_start) * sample_rate

t_new = np.linspace(t_start,t_end,numofpoints)*1e-9
t_list = t_new*1e9 - np.min(t_new)*1e9
t_range = (-90e-6, 10e-6)

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
# 保存Qubit状态
################################################################################

def saveQubitstate(measure,tags,qubits=None):
    if qubits is None:
        qubits = [measure.qubits[i] for i in measure.qubits]
    qstate = []
    def tuple2dict(q):
        for i in q:
            qstate.append(i.asdict())
        return qstate
    filepath = save(tags,'qubitstate',state = tuple2dict(qubits),base_path=r'D:\skzhao\file_name\wave')
    with open(r'D:\skzhao\file_name\qubit_state_wave.txt', mode='a') as filename:
        filename.write(str(filepath))
        filename.write('\n')
    return filepath

def loadQubitstate(f=-1,measure=None):
    qubits = []
    def dict2tuple(q):
        for i,k in enumerate(q):
            attr = {}
            for j in mrw.qubit().asdict():
                if j in k:
                    attr[j] = k[j]
                else:
                    attr[j] = 0
            q = mrw.qubit(**attr)
            qubits.append(q)
            if measure is not None:
                measure.qubits[k['q_name']] = q
    
    fl = np.loadtxt(r'D:\skzhao\file_name\qubit_state_wave.txt',dtype='str')
    if isinstance(f,str):
        f = f
    else:
        f = str(fl[f])   
    data = np.load(f,allow_pickle=True)
    dict2tuple(data['state'])
    return f, qubits

################################################################################
# 函数参数解析及赋值
################################################################################

async def funcarg(f,qubit,**kws):

    bit = dict(qubit.asdict())
    insp = inspect.getfullargspec(f)
    # paras, defaults = insp[0][::-1], insp[3]
    paras, defaults = insp[0], insp[3]
    status = {}
    for j,i in enumerate(paras):
        if i in kws:
            status[i] = kws[i]
        else:
    #         if defaults and j < len(defaults):
    #             ptype = type(defaults[-j-1])
    #             status[i] = bit[i]
            if i in bit:
                status[i] = bit[i]
    # print(status)
    if 'qubit' in paras:
        status['qubit'] = qubit
    pulse = await f(**status)
    return pulse
    
################################################################################
# 收集awg波形名字
################################################################################

def Collect_Waveform(dictname,kind):
    
    def decorator(func):
        def wrapper(*args, **kw):
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.get_event_loop()
                name_list = loop.run_until_complete(func(*args, **kw))
                dictname[kind] = name_list
            else:
                return func(*args, **kw)
        return wrapper
    return decorator

################################################################################
# 创建波形及sequence
################################################################################

async def create_wavelist(measure,kind,para):
    awg,name,n_wave,length,mode = para
    # if kind in measure.wave:
        # print('Warning: kind has already existed')
    @Collect_Waveform(measure.wave,kind)
    async def create_waveformlist(awg,name,n_wave,length,mode):
        name_list = []
        name_sub_list = []
        for k, i in enumerate(name):
            name_collect = []
            for j in range(1,n_wave+1):
                if mode == 'hbroadcast':
                    if k == 0:
                        name_sub = ''.join((kind,'_sub','%d'%j))
                        await awg.create_sequence(name_sub,2,2)
                        name_sub_list.append(name_sub)
                name_w = ''.join((kind,'_',i,'%d'%j))
                name_collect.append(name_w)
                await awg.create_waveform(name=name_w, length=length, format=None) 
            name_list.append(name_collect)
        name_list.append(name_sub_list)
        return name_list
    return create_waveformlist(*para)

################################################################################
# 关闭仪器  
################################################################################

async def close(measure):
    inst = {**measure.dc , **measure.psg}
    for i in inst:
        await inst[i].setValue('Output','OFF')

async def OffEx(Qubit,measure):
    for i in Qubit:
        for ch in measure.qubits[i].inst['ex_ch']:
            await measure.awg[measure.qubits[i].inst['ex_awg']].output_off(ch=ch)

async def OffZ(Qubit,measure):
    for i in Qubit:
        for ch in measure.qubits[i].inst['z_ch']:
            await measure.awg[measure.qubits[i].inst['z_awg']].output_off(ch=ch)
################################################################################
# 查询仪器状态
################################################################################

async def QueryInst(measure):
    inst = {**measure.dc , **measure.psg, **measure.awg}
    state = {}
    for i in inst:
        try:
            if 'psg' in i:
                freq = await inst[i].getValue('Frequency')
                power = await inst[i].getValue('Power')
                Output = await inst[i].getValue('Output')
                Moutput = await inst[i].getValue('Moutput')
                Mform = await inst[i].getValue('Mform')
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'freq':'%fGHz'%(freq/1e9),'power':'%fdBm'%power,'output':Output.strip('\n\r'),\
                    'moutput':Moutput.strip('\n\r'),'mform':Mform.strip('\n\r'),'error':err[0]}
                state[i] = sm
            
            # elif 'awg' in i:
            #     if i == 'awg_trig100' or i== 'awgread' or i=='awg_trig':
            #         continue
            #     # print(i)
            #     err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
            #     x = await inst[i].run_state()
            #     if x == 1 or x == 2:
            #         output = 'RUN'
            #     else:
            #         output = 'OFF'
            #     sm = {'error':err[0],'output':output}
            #     for j in range(8):
            #         output_state = await inst[i].output_state(ch=(j+1))   
            #         sm[f'ch{j+1}'] = 'ON' if output_state else 'OFF'        
            #     state[i] = sm
            else:
                current = await inst[i].getValue('Offset')
                load = await inst[i].getValue('Load')
                load = eval((load).strip('\n\r'))  
                load = 'high Z' if load != 50 else 50
                err = (await inst[i].query('syst:err?')).strip('\n\r').split(',')
                sm = {'offset':current,'load':load,'error':err[0]}
                state[i] = sm
        finally:
            pass
    measure.inststate = state
    return state

################################################################################
# 初始化仪器
################################################################################

async def InitInst(measure,psgdc=True,awgch=True,clearwaveseq=None):
    if psgdc:
        await close(measure)
    if awgch:
        for i in measure.awg:
            for j in range(4):
                await measure.awg[i].output_off(ch=j+1)
    # if clearwaveseq != None:
    #     for i in clearwaveseq:
    #         await measure.awg[i].stop()
    #         #await measure.awg[i].query('*OPC?')
    #         await measure.awg[i].clear_waveform_list()
    #         await measure.awg[i].clear_sequence_list()
    #         measure.wave = {}

################################################################################
# 恢复仪器最近状态
################################################################################

async def RecoverInst(measure,state=None):
    if state is None:
        state = measure.inststate
    for i in state:
        if 'psg' in i:
            await measure.psg[i].setValue('Frequency',eval(state[i]['freq'].strip('GHz'))*1e9)
            await measure.psg[i].setValue('Power',eval(state[i]['power'].strip('dBm')))
            if state[i]['output'] == '1':
                await measure.psg[i].setValue('Output','ON')
            else:
                await measure.psg[i].setValue('Output','OFF')
        # elif 'awg' in i:
        #     if i == 'awg_trig100':
        #         continue
        #     awg = measure.awg[i]
        #     output = state[i]['output']
        #     if output == 'RUN':
        #         await awg.run()
        #     else:
        #         await awg.stop()
        #     for j in range(8):
        #         output_state = state[i][f'ch{j+1}']
        #         if output_state == 'ON':
        #             await awg.output_on(ch=(j+1))
        #             time.sleep(0.1)
        #         if output_state == 'OFF':
        #             await awg.output_off(ch=(j+1))
        else:
            await measure.dc[i].DC(state[i]['offset'])

################################################################################
# 询问AWG状态
################################################################################

async def couldRun(measure,awg,chlist=None,namelist=None):
    await awg.run()
    if chlist != None:
        for j, i in enumerate(chlist):
            if namelist != None:
                await awg.use_waveform(name=namelist[j],ch=i)
            await awg.output_on(ch=i)
            time.sleep(0.1)
    time.sleep(1)
    while True:
        x = await awg.run_state()
        time.sleep(0.4)
        if x == 1 or x == 2:
            await measure.awg['awg_trig100'].run()
            for i in range(4):
                await measure.awg['awg_trig100'].output_on(ch=(i+1))
                time.sleep(0.1)
            time.sleep(0.1)
            break

async def openandcloseAwg(measure,state,chlist=None,namelist=None):
    if state == 'ON':
        await couldRun(measure,measure.awg['awg131'],chlist=chlist,namelist=namelist)
        await couldRun(measure,measure.awg['awg132'],chlist=chlist,namelist=namelist)
        await couldRun(measure,measure.awg['awg134'],chlist=chlist,namelist=namelist)
        await couldRun(measure,measure.awg['awg133'],chlist=chlist,namelist=namelist)
    if state == 'OFF':
        await measure.awg['awg131'].stop()
        await measure.awg['awg132'].stop()
        await measure.awg['awg134'].stop()
        await measure.awg['awg133'].stop()
################################################################################
# awg清理sequence
################################################################################

async def clearSeq(measure,awg):
    if isinstance(awg, list):
        for i in awg:
            await measure.awg[i].stop()
            await measure.awg[i].write('*WAI')
            for j in range(8):
                await measure.awg[i].output_off(ch=j+1)
                time.sleep(0.1)
            await measure.awg[i].clear_sequence_list()
    elif isinstance(awg, str):
        await measure.awg[awg].stop()
        await measure.awg[awg].write('*WAI')
        for j in range(8):
            await measure.awg[awg].output_off(ch=j+1)
            time.sleep(0.1)
        await measure.awg[awg].clear_sequence_list()
    else:
        await awg.stop()
        await awg.write('*WAI')
        for j in range(8):
            await awg.output_off(ch=j+1)
            time.sleep(0.1)
        await awg.clear_sequence_list()
    # measure.wave = {}

################################################################################
# awg载入sequence
################################################################################

async def awgchmanage(measure,awg,seqname,ch):
    await awg.stop()
    await awg.use_sequence(seqname,channels=ch)
    time.sleep(5)
    # await awg.query('*OPC?')
    for i in ch:
        await awg.output_on(ch=i)
        time.sleep(0.1)
    # await couldRun(measure,awg)
    return

################################################################################
# awg生成并载入waveform
################################################################################

async def genwaveform(awg,wavname,ch,t_list=t_list):
    # t_list = measure.t_list
    await awg.stop()
    for j, i in enumerate(wavname):
        await awg.create_waveform(name=i, length=len(t_list), format=None)
        await awg.use_waveform(name=i,ch=ch[j])
        await awg.write('*WAI')
        await awg.output_on(ch=ch[j])
        time.sleep(0.1)

################################################################################
# awg生成子sequence
################################################################################

async def subSeq(measure,awg,subkind,wavename):
    await awg.set_sequence_step(subkind,wavename,1,wait='OFF')
    await awg.set_sequence_step(subkind,['zero','zero'],2,wait='OFF',goto='FIRST',repeat=160) #####repeat乘以zero的长度就是pad的时间长度

################################################################################
# 生成Sequence
################################################################################

async def genSeq(measure,awg,kind,mode='vbroadcast'):
    await awg.stop()
    await awg.remove_sequence(kind)
    time.sleep(1)
    # await awg.create_waveform(name='zero', length=2500, format=None)
    
    if mode == 'vbroadcast' or 1:
        await awg.create_sequence(kind,(len(measure.wave[kind][0])+1),2)
        # await awg.create_sequence(kind,400,2)
        await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=len(measure.wave[kind][0]),mode='vbroadcast')
        wait = 'ATR' if awg == measure.awg['awg_trig'] else 'ATR'
        for j,i in enumerate(measure.wave[kind][:2],start=1):
            await awg.set_seq(i,1,j,seq_name=kind,wait=wait,firstwait='BTR')
            # for k in np.arange((len(i)+2),1001):
            #     await awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' %(k, j, kind, 'zero'))

    # if mode == 'hbroadcast':
    #     repeat = measure.repeat
    #     # repeat = measure.repeat if measure.repeat%64==0 else (measure.repeat//64+1)*64
    #     wait = 'ATR' if awg == measure.awg['awg_trig'] else 'ATR'
    #     # await awg.create_sequence(kind,1000,2)
    #     await awg.create_sequence(kind,(len(measure.wave[kind][0])+1),2)
    #     await ats_setup(measure.ats,measure.delta,l=measure.readlen,repeats=(repeat*(len(measure.wave[kind][0])+1)),mode='hbroadcast')
    #     # await awg.create_sequence('zero_sub',1,2)
    #     # await awg.set_sequence_step('zero_sub',['zero','zero'],1)
    #     name_sub_list = measure.wave[kind][-1]
    #     wavenameIQ = np.array(measure.wave[kind][0:2])

    #     await subSeq(measure,awg,name_sub_list[0],wavenameIQ[:,0])
    #     await awg.set_sequence_step(kind,name_sub_list[0],1,wait=wait,goto='NEXT',repeat=repeat,jump=('OFF','NEXT'))

    #     for j, i in enumerate(name_sub_list,start=2):
    #         await subSeq(measure,awg,i,wavenameIQ[:,(j-2)])
    #         # goto = 'FIRST' if j == len(name_sub_list) else 'NEXT'
    #         goto = 'FIRST' if j == (len(name_sub_list)+1) else 'NEXT'
    #         await awg.set_sequence_step(kind,i,j,wait=wait,goto=goto,repeat=repeat,jump=('OFF',goto))
        # for k in np.arange((len(name_sub_list)+1),1001):
        #     await awg.set_sequence_step(kind,'zero_sub',int(k))
            # for l in range(2):
            #     await awg.write('SLIS:SEQ:STEP%d:TASS%d:WAV "%s","%s"' %(k, (l+1), kind, 'zero'))

################################################################################
# 打包sequence
################################################################################

async def gen_packSeq(measure,kind,awg,name,steps,readseq=True,mode='vbroadcast'):
    # if measure.mode != mode or measure.steps != steps:
    #     measure.mode = mode
    #     measure.steps = steps
    #     state = await QueryInst(measure)
    #     await clearSeq(measure,[f'awg13{i+1}' for i in range(4)])
    #     await RecoverInst(measure,state)
    #     await clearSeq(measure,'awgread')
    # await create_wavelist(measure,'Read',(measure.awg['awgread'],['Readout_I','Readout_Q'],steps,len(measure.t_list),mode))
    await create_wavelist(measure,kind,(awg,name,steps,len(measure.t_list),mode))
    # await genSeq(measure,measure.awg['awgread'],'Read',mode=mode)
    await genSeq(measure,awg,kind,mode=mode)
    # if readseq:
    #     measure.wave['Read'] = [['Readout_I']*steps,['Readout_Q']*steps,['Read_sub']*steps]
    #     # await measure.awg['awgread'].create_sequence('Read_sub',2,2)

    #     await genSeq(measure,measure.awg['awgread'],'Read',mode=mode)
    #     await awgchmanage(measure,measure.awg['awgread'],'Read',[1,5])

################################################################################
# Z波形矫正
################################################################################

def expcaliZ(waveform,paras):
    print(paras)

    def fitfunc(t, p):
        #p[2::2] = p[2::2] 
        return (t > 0) * (np.sum(p[1::2,None]*np.exp(-p[2::2,None]*t[None,:]), axis=0))

    if isinstance(paras, dict):
        return waveform
    waveform = waveform[0]
    length = len(waveform)
    paras[0] = 0
    x = np.arange(length) / sample_rate
    response = fitfunc(x,paras)
    f_cali = 1/(1+1j*np.pi*np.fft.fftfreq(len(x),1/sample_rate)*np.fft.fft(response))
    f_step = np.fft.fft(waveform)
    signal = np.real(np.fft.ifft(f_step*f_cali))
    return (signal,)


def polycaliZ(waveform,paras):

    # def smoothfilt(f,h):
    #     f0, sigma = 0.05, 0.02
    #     return 1+0.5*(1/h-1)*(np.tanh((f-f0)/sigma)+1)

    waveform = waveform[0]
    length = len(waveform)
    # height = np.max(waveform)-np.min(waveform)

    nrfft = length/2+1
    nfft = 2*(nrfft-1)
    # freqs = np.linspace(0, nrfft*1.0/nfft*samplingRate,nrfft, endpoint=False)
    freqs = np.fft.rfftfreq(int(nfft),1/sample_rate)
    # freqs = np.fft.fftfreq(length,0.4)
    i_two_pi_freqs = 2j*np.pi*freqs

    precalc = 1.0
    for i in paras:
        timeFunData = 0.0
        if paras[i] != {}:
            
            expAmpRates, polyParas, delayParasAND2nd = np.array(paras[i]['pexp']), np.array(paras[i]['ppoly']),\
                 np.array(paras[i]['time'])
        else:
            continue
        pExp0 = expAmpRates
        pExp1 = polyParas[:2]
        pPoly = polyParas[2:]
        tCut,tShift,tstart,sigma1 = delayParasAND2nd
        # tlist = np.arange(2*(nrfft-1),dtype=float)/sample_rate
        tlist = np.arange(length,dtype=float)/sample_rate
        if len(pExp0) > 1:
            timeFunData += np.sum(pExp0[1::2,None]*np.exp(-pExp0[2::2,None]*tlist[None,:]), axis=0)
        timeFunData += pExp1[0]*np.exp(-pExp1[1]*tlist)*np.polyval(pPoly,tlist)*(tlist<=tCut+20)*\
        (0.5-0.5*scipy.special.erf(sigma1*(tlist-tCut+tShift)))*(0.5+0.5*scipy.special.erf(4.0*(tlist-tstart+0.5)))
        timeFunDataf = np.fft.rfft(timeFunData)
        # timeFunDataf *= smoothfilt(freqs,timeFunDataf)
        precalc /= (1.0+timeFunDataf*i_two_pi_freqs/sample_rate)
        

    f_cali = precalc
    f_step = np.fft.rfft(waveform)
    signal = np.fft.irfft(f_step*f_cali)
    # signal = op.RowToRipe().smooth(np.fft.irfft(f_step*f_cali),0.4)
    return (np.real(signal),)

def caliZ(waveform,paras):
    if isinstance(paras, dict):
        return polycaliZ(waveform,paras)
    else:
        return expcaliZ(waveform,paras)

################################################################################
# 更新波形  mTrig_ex_lo
################################################################################
# @functools.lru_cache()
async def writeWave(measure,awg,ch,pulse,channel_output_delay=0,norm=False,update=True):

    if len(pulse) == 2 or len(pulse) == 4:
        I, Q = pulse[:2]
        if np.max(I)>1 or np.max(Q)>1:
            print(np.max(I),np.max(Q))
            raise 'Too big'
        if norm:
            I, Q = I / np.max(np.abs(I)), Q / np.max(np.abs(Q))
        if update:
            await awg.da_write_wave(ch[0], I*amp_norm, 'i' , channel_output_delay, 0, 0)
            await awg.da_write_wave(ch[1], Q*amp_norm, 'i' , channel_output_delay, 0, 0)
    if len(pulse) == 1:
        signal, = pulse
        if np.max(signal)>1:
            raise 'Too big'
        await awg.da_write_wave(ch[0], np.real(signal)*amp_norm, 'i' , channel_output_delay, 0, 0)


################################################################################
# 波形数值化
################################################################################

def pulseTowave(pulse,t=t_new):
    # wavelist = []
    # if isinstance(pulse, (list,tuple,np.ndarray,set,dict)):
    #     for i in pulse:
    #         if isinstance(i, (list,tuple,np.ndarray,set,dict)):
    if len(pulse) == 4:
        wav_I, wav_Q, mrk1, mrk2 = pulse
        I, Q = wav_I(t), wav_Q(t)
        mrk1 = np.array([i(t) for i in mrk1])
        mrk2 = np.array([i(t) for i in mrk2])
        wavelist = np.array([I, Q, mrk1, mrk2],dtype=object)
    if len(pulse) == 1:
        wave = pulse[0]
        wavelist = np.array((wave(t),),dtype=object)
    if len(pulse) == 2:
        wave1,wave2 = pulse
        wavelist = np.array((wave1(t),wave2(t)),dtype=object)
    return wavelist

################################################################################
# 波包选择
################################################################################

def whichEnvelope(pi_len,envelope,num):
    # trig1 = (wn.square(20e-9+pi_len) << (pi_len/2+trig_shift))
    # trig2 = (wn.square(20e-9+2*pi_len) << (pi_len+trig_shift))
    if envelope == 'hanning':
        if num == 1:
            return (wn.hanning(pi_len) << pi_len/2)
        if num == 2:
            return (wn.hanning(pi_len) << pi_len/2) + (wn.hanning(pi_len) << (1.5*pi_len))
    if envelope == 'hamming':
        if num == 1:
            return wn.hamming(pi_len) << pi_len/2
        if num == 2:
            return (wn.hamming(pi_len) << pi_len/2) + (wn.hamming(pi_len) << (1.5*pi_len))
    if envelope == 'gaussian':
        if num == 1:
            return wn.gaussian(pi_len) << pi_len/2
        if num == 2:
            return (wn.gaussian(pi_len) << pi_len/2) + (wn.gaussian(pi_len) << (1.5*pi_len))
    if envelope == 'square':
        if num == 1:
            return wn.square(pi_len,2e-9) << pi_len/2
        if num == 2:
            return wn.square(2*pi_len,2e-9) << pi_len


################################################################################
# 设置采集卡
################################################################################

async def ats_setup(measure,delta,readlen=1000,repeats=300,ATS=True):
    if ATS==True:
        await measure.awg['awgread'].da_trigg(repeats)
        await measure.ats.set_ad(delta, readlen, window_start=[8]*len(delta), trig_count=repeats)
    if ATS==False:
        await measure.awg['awgread2'].da_trigg(repeats)
        await measure.ats2.set_ad(delta, readlen, window_start=[8]*len(delta), trig_count=repeats)
    # await ats.set(n=l,repeats=repeats,mode=mode,
    #                        f_list=delta,
    #                        weight=weight,
    #                        maxlen=512,
    #                        ARange=1.0,
    #                        BRange=1.0,
    #                        trigLevel=0.5,
    #                        triggerDelay=0,
    #                        triggerTimeout=0,
    #                        bufferCount=512)

################################################################################
# 读出混频
################################################################################

async def readWave(measure,delta,readlen=1100,phase=0.0,pi_len=0,shift=0):
    readamp = measure.readamp
    ringup = measure.ringup
    ringupamp = measure.ringupamp
    twidth, n, measure.readlen = readlen, len(delta), readlen
    wavelen = twidth 
    # if wavelen < twidth:
    #     wavelen += 64
    measure.wavelen = int(wavelen) 
    
    wavelen = wavelen 

    I, Q = wn.zero(), wn.zero()
    for j,i in enumerate(delta):
        pulse_ringup = (whichEnvelope(ringup[j]/1e9,*['gaussian',1]) >> ringup[j]/1e9) * ringupamp[j]
        pulse_read = (whichEnvelope(wavelen/1e9,*['square',1]) >> (wavelen/1e9+ringup[j]/1e9)) * readamp[j]
        pulse = pulse_ringup + pulse_read
        wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    I, Q = I , Q 

    mrkp1 = wn.square(len(measure.t_new)/sample_rate/2e9) << (len(measure.t_new)/sample_rate/2/2e9 + 500e-9)
    mTrig_ex_lo = wn.square(len(measure.t_new)/sample_rate/1e9-5e-6) << (len(measure.t_new)/sample_rate/2/1e9-2.5e-6-1000e-9)
    mrkp2 = wn.square(wavelen/1e9) >> (wavelen / 1e9 / 2 + 300 / 1e9 + 2*np.max(ringup) / 1e9)
    # if pi_len != 0:
    #     mTrig_ex_lo = wn.square(3*pi_len) << (pi_len*1.5+shift-6e-9)
    
    pulselist = I, Q#, np.array((mrkp2,mrkp1)), np.array((mTrig_ex_lo,))
    wave = pulseTowave(pulselist)
    # measure.mrk_ats = wave[2][0]
    # measure.mrk_exlo = wave[3][0]
    # measure.mrk_trans = wave[2][1]
    measure.Readout_I = wave[0]
    measure.Readout_Q = wave[1]

    return pulseTowave(pulselist)

async def modulation_read(measure,delta,readlen=1100,repeats=300,phase=0.0,weight=None,ch=[1,2],window_start=[8],channel_output_delay=0,ATS=True):
    measure.delta = delta
    measure.repeats = repeats
    if ATS==True:
        ats, awg = measure.ats, measure.awg['awgread']
    if ATS==False:
        ats, awg = measure.ats2, measure.awg['awgread2']
    pulselist = await readWave(measure,delta,readlen,phase)
    await writeWave(measure,awg,ch,pulselist,channel_output_delay=channel_output_delay,norm=False)
    # await ats.set_ad(delta, measure.wavelen, window_start=[8]*len(delta), trig_count=repeats)
    await ats_setup(measure,delta,readlen=readlen,repeats=repeats,ATS=ATS)
    # ats.set_ad_freq(delta, measure.wavelen, window_start=8)
    # return pulselist

################################################################################s
# 生成Trig序列
################################################################################

async def Trig_sequence(qubit,measure,v_or_t,arg,**paras):
    paras['shift'], paras['pi_len'] = 0, qubit.pi_len*qubit.envelopename[1]
    vort = np.copy(v_or_t)
    # print(v_or_t,arg,paras)
    if arg == 'shift':
        vort += qubit.timing['read>xy']
    else:
        paras['shift'] += qubit.timing['read>xy']
    awg= measure.awg['awgread']
    kind = 'Read'
    await awg.stop()
    # paras['pi_len'] = len(measure.t_new)/samplingRate/1e9-10e-6
    for j,i in enumerate(tqdm(vort,desc=''.join(('Trig_',kind)))):
    # for j,i in enumerate(v_or_t):
        if arg == 'pi_len' or arg == 'shift':
            paras[arg] = i 
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await readWave(measure,measure.delta,measure.readlen,phase=0.0,**paras)
        await writeWave(measure,awg,name_ch,pulse,mark=True)

async def Trig_wave(qubit,measure,**paras):
    # shift0, pi_len0 = 0, 0
    if 'pi_len' in paras:
        paras['shift'] = qubit.timing['read>xy']
    if 'shift' in paras:
        paras['shift'] += qubit.timing['read>xy']
        paras['pi_len'] = qubit.pi_len*qubit.envelopename[1]
    if ('pi_len' not in paras) and ('shift' not in paras):
        paras['shift'] = qubit.timing['read>xy']
        paras['pi_len'] = qubit.pi_len*qubit.envelopename[1]
    paras['pi_len'] = len(measure.t_new)/samplingRate/1e9-10e-6
    kws = {'shift':paras['shift'],'pi_len':paras['pi_len']}
    # print(kws)
    awg= measure.awg['awgread']
    await awg.stop()
    name_ch = ['Readout_I','Readout_Q']
    pulse = await readWave(measure,measure.delta,measure.readlen,phase=0.0,**kws)
    await writeWave(measure,awg,name_ch,pulse,mark=True)

################################################################################s
# Rabi波形
################################################################################

async def rabiWave(envelopename=['square',1],nwave=1,amp=1,pi_len=75e-9,shift=0,delta_ex=110e6,phase=0,\
    phaseDiff=0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):
    shift += timing['read>xy']
    wave = whichEnvelope(pi_len,*envelopename) << shift
    wave *= amp
    mpulse = whichEnvelope((nwave*pi_len*2+(nwave-1)*10e-9+100e-9),'square',envelopename[1]) << (trig_shift+shift)
    pulse = wn.zero()
    for i in range(nwave):
        pulse += (wave << (envelopename[1]*i*pi_len + i*10e-9))
        # mpulse += (mwav << (envelopename[1]*i*pi_len + i*10e-9))
    wav_I, wav_Q = wn.mixing(pulse,phase=phase,freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    
    return pulseTowave((wav_I, wav_Q, [mpulse], [mpulse]))

################################################################################s
# Rabi_seq
################################################################################

async def Rabi_sequence(qubit,measure,kind,v_or_t,arg,**paras):
    # print(v_or_t,arg,paras)
    awg= measure.awg[qubit.inst['ex_awg']]
    await awg.stop()
    for j,i in enumerate(tqdm(v_or_t,desc=''.join(('Rabi_',kind)))):
    # for j,i in enumerate(v_or_t):
        # if arg == 'shift':
        #     paras[arg] = i + qubit.shift_delay
        # else:
        paras[arg] = i
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        name_read = [measure.wave['Read'][0][j],measure.wave['Read'][1][j]]
        pulse = await funcarg(rabiWave,qubit,**paras)
        await writeWave(measure,awg,name_ch,pulse,mark=False,markname=name_read)

################################################################################s
# 双光子能级
################################################################################

# async def fLevel(envelopename=['square',1],nwave=1,amp=1,pi_len=75e-9,shift=0,delta_ex=110e6,phase=0,\
#     phaseDiff=0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):

#     pulse1 = await cww.funcarg(qgw.singleQgate,qubit,axis=j[0],DRAGScaling=i/alpha,shift=(qubit.pi_len+5e-9))
#     pulse_pi1 = await funcarg(rabiWave,qubit,nwave=1,shift=shift_copy)
#     shift1 =  qubit.envelopename[1]*qubit.pi_len + 10e-9 + shift_copy 
#     paras['shift'] = shift1 
#     pulse_rabi2 = await funcarg(rabiWave,qubit,**paras)
#     # pulse_pi1 = await funcarg(rabiWave,qubit)
#     pulse_pi1_1 = await funcarg(rabiWave,qubit,nwave=1,shift=(shift1 + qubit.nwave*(qubit.envelopename[1]*qubit.pi_len2 + 10e-9)))
#     pulse = np.array(pulse_pi1) + np.array(pulse_rabi2) + np.array(pulse_pi1_1)

################################################################################s
# Rabi2_seq
################################################################################

async def Rabi2_sequence(qubit,measure,kind,v_or_t,arg,**paras):
    paras['delta_ex'] = qubit.delta_ex + qubit.alpha
    paras['pi_len'] = qubit.pi_len2
    # print(v_or_t,arg,paras)
    shift_copy = np.copy(paras['shift']) if 'shift' in paras else 0
    awg= measure.awg[qubit.inst['ex_awg']]
    await awg.stop()
    for j,i in enumerate(tqdm(v_or_t,desc=''.join(('Rabi_',kind)))):
        paras[arg] = i
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse_pi1 = await funcarg(rabiWave,qubit,nwave=1,shift=shift_copy)
        shift1 =  qubit.envelopename[1]*qubit.pi_len + 10e-9 + shift_copy 
        paras['shift'] = shift1 
        pulse_rabi2 = await funcarg(rabiWave,qubit,**paras)
        # pulse_pi1 = await funcarg(rabiWave,qubit)
        pulse_pi1_1 = await funcarg(rabiWave,qubit,nwave=1,shift=(shift1 + qubit.nwave*(qubit.envelopename[1]*qubit.pi_len2 + 10e-9)))
        pulse = np.array(pulse_pi1) + np.array(pulse_rabi2) + np.array(pulse_pi1_1)
        await writeWave(measure,awg,name_ch,pulse,mark=False)

################################################################################s
# 激发2态
################################################################################

async def eWave(envelopename=['square',1],nwave=1,amp=1,pi_len=75e-9,amp2=1,pi_len2=75e-9,shift=0,delta_ex=110e6,phase=0,\
    alpha=0e9,phaseDiff=0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):

    shift += timing['read>xy']
    wave_pi = whichEnvelope(pi_len,*envelopename) << shift
    wave_pi2 = whichEnvelope(pi_len2,*envelopename) << (shift + envelopename[1]*pi_len + 10e-9)
    wave_pi *= amp
    wave_pi2 *= amp2

    mpulse = whichEnvelope((nwave*pi_len+(nwave-1)*10e-9+200e-9),'square',envelopename[1]) >> (100e-9*envelopename[1])
    pulse = wn.zero()
    shiftm = envelopename[1]*pi_len + envelopename[1]*pi_len2 + 20e-9
    for i in range(nwave):
        pulse += (wave_pi2 << (envelopename[1]*i*pi_len2 + i*10e-9))
        shiftm += (envelopename[1]*i*pi_len2 + i*10e-9)
        # mpulse += (mwav << (envelopename[1]*i*pi_len + i*10e-9))
    wave_pi_start = wave_pi << shiftm
    pulse_pi = (wave_pi_start)
    wav_I_pi, wav_Q_pi = wn.mixing(pulse_pi,phase=phase,freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    wav_I_pi2, wav_Q_pi2 = wn.mixing(pulse,phase=phase,freq=(delta_ex+alpha),ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=DRAGScaling)
    wav_I, wav_Q = wav_I_pi + wav_I_pi2, wav_Q_pi + wav_Q_pi2
    return pulseTowave((wav_I, wav_Q, [mpulse]*4, [mpulse]*4))

    # paras['delta_ex'] = qubit.delta_ex + qubit.alpha
    # paras['pi_len'] = qubit.pi_len2
    # # print(v_or_t,arg,paras)
    # shift_copy = np.copy(paras['shift']) if 'shift' in paras else 0
    # awg= measure.awg[qubit.inst['ex_awg']]
    # await awg.stop()
    # for j,i in enumerate(tqdm(v_or_t,desc=''.join(('Rabi_',kind)))):
    #     paras[arg] = i
    #     name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
    #     pulse_pi1 = await funcarg(rabiWave,qubit,nwave=1,shift=shift_copy)
    #     shift1 =  qubit.envelopename[1]*qubit.pi_len + 10e-9 + shift_copy 
    #     paras['shift'] = shift1 
    #     pulse_rabi2 = await funcarg(rabiWave,qubit,**paras)
    #     # pulse_pi1 = await funcarg(rabiWave,qubit)
    #     pulse_pi1_1 = await funcarg(rabiWave,qubit,nwave=1,shift=(shift1 + qubit.nwave*(qubit.envelopename[1]*qubit.pi_len2 + 10e-9)))
    #     pulse = np.array(pulse_pi1) + np.array(pulse_rabi2) + np.array(pulse_pi1_1)
    #     await writeWave(measure,awg,name_ch,pulse,mark=False)

################################################################################
# pipulseDetune波形
################################################################################

# async def pipulseDetunewave(measure,awg,pilen,n,alpha,delta,name_ch,phaseDiff=0):
#     shift = 200/1e9
#     pilen = pilen / 1e9
#     envelope = whichEnvelope(measure.envelopename)
#     I, Q, mrk1, mrk2 = wn.zero(), wn.zero(), wn.zero(), wn.zero()
#     for i in [np.pi,0]*n:
#         pulse = (envelope(pilen) << (0.5*pilen+shift)) + (envelope(pilen) << (1.5*pilen + shift))
#         shift += 2*pilen
#         wav_I, wav_Q = wn.mixing(pulse,phase=i,freq=delta,phaseDiff=phaseDiff,ratioIQ=-1.0,DRAGScaling=alpha)
#         I, Q = I+wav_I, Q+wav_Q
#     await writeWave(measure,awg,name_ch,pulseTowave(I,Q,mrk1,mrk2))
#     return I,Q,mrk1,mrk2

################################################################################
# optDragalpha_wave
################################################################################
async def optDragalpha_wave(envelopename=['square',1],pi_len=30e-9,amp=1,shift=0,delta_ex=110e6,axis='X',\
    DRAGScaling=None,phaseDiff=0,timing={'z>xy':0,'read>xy':0},phase=0.0): 
    pulse = 0
    for i in range(5):
        pulse1 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,delta_ex=delta_ex,phaseDiff=phaseDiff,\
            timing=timing,phase=phase,axis='X',shift=((2*i+1)*envelopename[1]*pi_len+(3*i+3)*1e-9),DRAGScaling=DRAGScaling)

        pulse2 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,delta_ex=delta_ex,phaseDiff=phaseDiff,\
            timing=timing,phase=phase,axis='Xn',shift=(2*i*envelopename[1]*pi_len+(3*i+3)*1e-9),DRAGScaling=DRAGScaling)
        pulse += (np.array(pulse1) + np.array(pulse2)) 
    return pulse

################################################################################
# ALLXY_kind
################################################################################
async def AllXY_kind_wave(envelopename=['square',1],pi_len=30e-9,amp=1,shift=0,delta_ex=110e6,axis='X',\
    DRAGScaling=None,phaseDiff=0,timing={'z>xy':0,'read>xy':0},phase=0.0): 
    pulse = 0
    for i in range(5):
        pulse1 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,delta_ex=delta_ex,phaseDiff=phaseDiff,\
            timing=timing,phase=phase,axis='X',shift=((2*i+1)*envelopename[1]*pi_len+(3*i+3)*1e-9),DRAGScaling=DRAGScaling)

        pulse2 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,delta_ex=delta_ex,phaseDiff=phaseDiff,\
            timing=timing,phase=phase,axis='Xn',shift=(2*i*envelopename[1]*pi_len+(3*i+3)*1e-9),DRAGScaling=DRAGScaling)
        pulse += (np.array(pulse1) + np.array(pulse2)) 
    return pulse

################################################################################
# Ramsey及SpinEcho,CPMG, PDD波形
################################################################################

async def coherenceWave(envelopename=['square',1],t_run=0,amp=1,pi_len=75e-9,nwave=0,seqtype='CPMG',\
    detune=3e6,shift=0e-9,delta_ex=110e6,phaseDiff=0.0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):
    shift += timing['read>xy']
    if envelopename[1] == 1:
        envelope_pi = whichEnvelope(pi_len,*envelopename) * amp 
        envelope_half = whichEnvelope(pi_len,*envelopename) * 0.5 * amp 
    if envelopename[1] == 2:
        envelope_pi = whichEnvelope(pi_len,*envelopename) * amp 
        envelope_half = whichEnvelope(pi_len,envelopename[0],1) * amp 
    pulse1 = envelope_half << shift
    wavI1, wavQ1 = wn.mixing(pulse1,phase=2*np.pi*detune*t_run,freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=None)
    pulse3 = envelope_half << (t_run+(2*nwave+1)*pi_len+shift)
    wavI3, wavQ3 = wn.mixing(pulse3,phase=0,freq=delta_ex,ratioIQ=-1.0,phaseDiff=phaseDiff,DRAGScaling=None)

    if seqtype == 'CPMG':
        pulse2, step = wn.zero(), t_run / nwave
        for i in range(nwave):
            # pulse2 += ((envelope(during) << (during*0.5)+envelope(during) << (during*1.5)) << ((i+0.5)*step+(i+0.5)*2*during+shift))
            pulse2 += envelope_pi << ((i+0.5)*step+(i+0.5)*2*pi_len+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=delta_ex,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    if seqtype == 'PDD':
        pulse2, step = wn.zero(), t_run / (nwave + 1)
        for i in range(nwave):
            # pulse2 += envelope(2*pi_len) << ((i+1)*step+(i+1)*2*pi_len+shift)
            # pulse2 += ((envelope(pi_len) << (pi_len*0.5)+envelope(pi_len) << (pi_len*1.5)) << ((i+0.5)*step+(i+0.5)*2*pi_len+shift))
            pulse2 += envelope_pi << ((i+1)*step+(i+0.5)*2*pi_len+shift)
        wavI2, wavQ2 = wn.mixing(pulse2,phase=np.pi/2,freq=delta_ex,ratioIQ=-1.0,phaseDiff=0.0,DRAGScaling=None)
    wavI, wavQ = wavI1 + wavI2 + wavI3, wavQ1 + wavQ2 + wavQ3
    
    return pulseTowave((wavI, wavQ,( wn.zero(),)*4, (wn.zero(),)*4))

async def Coherence_sequence(qubit,measure,kind,v_or_t,arg,**paras):
    awg= measure.awg[qubit.inst['ex_awg']]
    await awg.stop()
    for j,i in enumerate(tqdm(v_or_t,desc=''.join(('Coherence_',kind)))):
        paras[arg] = i
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await funcarg(coherenceWave,qubit,**paras)
        await writeWave(measure,awg,name_ch,(pulse),mark=False)

################################################################################s
# Ramsey2_seq
################################################################################

async def Coherence2_sequence(qubit,measure,kind,v_or_t,arg,**paras):
    paras['delta_ex'] = qubit.delta_ex + qubit.alpha
    paras['pi_len'] = qubit.pi_len2
    paras['amp'] = qubit.amp2
    # print(v_or_t,arg,paras)
    shift_copy = np.copy(paras['shift']) if 'shift' in paras else 0
    awg= measure.awg[qubit.inst['ex_awg']]
    await awg.stop()
    for j,i in enumerate(tqdm(v_or_t,desc=''.join(('c2_',kind)))):
        paras[arg] = i
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse_pi1 = await funcarg(rabiWave,qubit,nwave=1,shift=shift_copy)
        shift1 =  qubit.envelopename[1]*qubit.pi_len + 10e-9 + shift_copy 
        paras['shift'] = shift1 
        # pulse_rabi2 = await funcarg(rabiWave,qubit,**paras)
        pulse_r2 = await funcarg(coherenceWave,qubit,**paras)
        pulse_pi1_1 = await funcarg(rabiWave,qubit,nwave=1,shift=(shift1 + i + (1+qubit.nwave)*(qubit.envelopename[1]*qubit.pi_len2) + 10e-9))
        pulse = np.array(pulse_pi1) + np.array(pulse_r2) + np.array(pulse_pi1_1)
        await writeWave(measure,awg,name_ch,pulse,mark=False)

################################################################################
# Z_cross波形
################################################################################

async def Z_cross_sequence(measure,kind_z,kind_ex,awg_z,awg_ex,v_rabi,halfpi):
    #t_range, sample_rate = measure.t_range, measure.sample_rate
    for j,i in enumerate(tqdm(v_rabi,desc='Z_cross_sequence')):
        name_ch = [measure.wave[kind_ex][0][j],measure.wave[kind_ex][1][j]]
        pulse_ex = await coherenceWave(measure.envelopename,300/1e9,halfpi/1e9,0,'PDD')
        await writeWave(measure,awg_ex,name_ch,(pulse_ex))
        pulse_z = (wn.square(200/1e9) << (50+halfpi+200)/1e9) * i
        await writeWave(measure,awg_z,np.array(measure.wave[kind_z])[:,j],pulseTowave(pulse_z))

################################################################################
# AC-Stark波形
################################################################################

async def ac_stark_wave(measure,power=1):
    # awg = measure.awg['awgread']
    pulse_read = await readWave(measure,measure.delta,readlen=measure.readlen)
    width = 50e-9
    pulse = (wn.square(width) << (width/2+3000e-9)) * power
    I, Q = wn.zero(), wn.zero()
    for i in measure.delta:
        wav_I, wav_Q = wn.mixing(pulse,phase=0.0,freq=i,ratioIQ=-1.0)
        I, Q = I + wav_I, Q + wav_Q
    pulse_acstark = pulseTowave([I,Q,(wn.zero(),)*4,(wn.zero(),)*4])
    pulselist = np.array(pulse_read) + np.array(pulse_acstark)
    # await writeWave(measure,awg,['Readout_I','Readout_Q'],pulseTowave(pulselist))
    return pulselist

async def acstarkSequence(measure,kind,v_or_t,arg,**paras):

    awg = measure.awg['awgread']
    for j,i in enumerate(tqdm(v_or_t,desc='acStark')):
        paras[arg] = i
        # pulse_z = await funcarg(ac_stark_wave,qubit,**paras)
        pulse_z = await ac_stark_wave(measure,**paras)
        name_z = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        await writeWave(measure,awg,name_z,(pulse_z),False,mark=True)
    
################################################################################
# ZPulse波形
################################################################################


async def zWave(volt=0.4,during=0e-9,shift=0e-9,offset=0,timing={'z>xy':0,'read>xy':0},zCali=None,readvolt=0):
    shift += timing['read>xy'] - timing['z>xy']
    # print(shift,volt,during,offset)
    pulse = (wn.square(during,0e-9) << (during/2 + shift)) * volt 
    pulse_offset = (wn.square((len(t_new)/sample_rate/2e9+2000e-9),0e-9) << ((len(t_new)/sample_rate/2e9+2000e-9)/2 + shift)) * offset 
    pulse += pulse_offset
    # if during >= 2e-9 and volt !=0:
    #     pulse_overshot = (-(wn.square(0.8e-9) << (0.4e-9 + shift)) + (wn.square(0.8e-9) << (shift + during - 0.4e-9)))*amp_overshot
    #     if volt >=0:
    #         pulse += pulse_overshot
    #     else:
    #         pulse -= pulse_overshot
    pulse_read = (wn.square(3000e-9,2e-9) >> (3000e-9/2)) * readvolt 
    pulse += (pulse_read << shift)

    pulselist = (pulse,) 
    # t = np.linspace(-90000,10000,100000)*1e-9
    return pulseTowave(pulselist) if zCali is None else caliZ(pulseTowave(pulselist),zCali)
    # return pulseTowave(pulselist)


async def swapzWave(qubit,volt_swap=0,during_swap=0e-9,shift=0e-9,offset=0,readvolt=0,gateduring=0e-9,volt_zgate=0,volt_zgate_b=0):
    # print(qubit.q_name,volt_swap,during_swap)
    pulse_swap = await funcarg(zWave,qubit,volt=volt_swap,during=during_swap,shift=(shift+gateduring),offset=offset,readvolt=readvolt)
    pulse_phase_a = await funcarg(zWave,qubit,volt=volt_zgate,during=gateduring,shift=(shift),offset=0,readvolt=0)
    pulse_phase_b = await funcarg(zWave,qubit,volt=volt_zgate_b,during=gateduring,shift=(shift+gateduring+during_swap),offset=0,readvolt=0)
    pulse = np.array(pulse_swap) + np.array(pulse_phase_b) + np.array(pulse_phase_a)
    return pulse

async def specf2v(specfuncz,f_ex,volt=0,during=0e-9,shift=0e-9,timing={'z>xy':0,'read>xy':0},zCali=None,delta_im=0,imAmp=0):
    shift += timing['read>xy'] - timing['z>xy']
    pulse_im = (wn.square(during,0e-9) << (during/2 + shift))

    if delta_im != 0:
        wav_I, wav_Q = wn.mixing(imAmp*pulse_im,phase=0.0,freq=delta_im,ratioIQ=-1.0)
    # fc,vtarget = dt.specshift(specfuncz,f_ex/1e9,volt)
    
    # f, v = qubit.specinterp
    fshift, = pulseTowave((wav_I,))
    # vwindow, = pulseTowave((pulse_im,))
    bias = dt.biasshift(specfuncz,f_ex/1e9,fshift) 
    return (bias,)
    # return bias if zCali is None else caliZ(bias,zCali)

async def zWave_im(specfuncz,dressenergy,f_ex,volt=0,during=0e-9,shift=0e-9,timing={'z>xy':0,'read>xy':0},\
    zCali=None,delta_im=0,imAmp=0,phaseim=0,readvolt=0,center=1,imAmpn=0):
    shift += timing['read>xy'] - timing['z>xy']
    shift1 = np.copy(shift) 
    shift = 0
    during1 = during if during<center else center
    during2 = 0 if during<center else during-center
    pulse_im = (wn.square(during1,2e-9) << (during1/2 + shift + during2))
    pulse_imn = (wn.square(during2,2e-9) << (during2/2 + shift))
    # pulse_im_g = (wn.hanning(during1) << (during1/2 + shift + during2))
    # pulse_imn_g = wn.zero()
    # if during2 != 0:
    #     pulse_imn_g = (wn.hanning(during2) << (during2/2 + shift))
    if delta_im != 0:
        wav_I1, wav_Q1 = wn.mixing(imAmp*pulse_im,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
        wav_I2, wav_Q2 = wn.mixing(imAmpn*pulse_imn,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
    # fc,vtarget = dt.specshift(specfuncz,f_ex/1e9,volt)
    wav_I = wav_I1 + wav_I2
    fshift, = pulseTowave((wav_I,))
    # vwindow, = pulseTowave((pulse_im,))
    # bias = fshift
    bias = dt.biasshift(specfuncz,f_ex/1e9,fshift/2/np.pi) ##zsk
    # bias = dt.biasshift(specfuncz,f_ex/1e9,delta_im/2/np.pi) ###yhs
    
    bias_offset = 0
    bias_offset_n = 0
    if dressenergy is not None:
        # if np.abs(dressenergy(0)-f_ex/1e9)>1e6:
        #     raise 'freq error'
        fnew = dressenergy(np.abs(imAmp)/2/np.pi)-dressenergy(0)
        bias_offset = dt.biasshift(specfuncz,f_ex/1e9,fnew,'lower')
        fnew_n = dressenergy(np.abs(imAmpn)/2/np.pi)-dressenergy(0)
        bias_offset_n = dt.biasshift(specfuncz,f_ex/1e9,fnew_n,'lower')

    # pulse = (wn.square(during,0e-9) << (during/2 + shift)) * (volt-bias_offset)
    zoffset_im = pulse_im*(volt-bias_offset)
    zoffset_imn = pulse_imn*(volt-bias_offset_n)
    pulse = zoffset_im + zoffset_imn
    pulse_read = (wn.square(3000e-9,0e-9) >> (3000e-9/2)) * readvolt 
    pulse += (pulse_read << shift)
    pulselist = (pulse,) 
    zwave, = pulseTowave(pulselist)

    pulse = bias+zwave
    # print(np.max(pulse))
    return (pulse,) if zCali is None else caliZ((pulse,),zCali), shift1

################################################################################
# 偏置sequence
################################################################################

async def Z_sequence_im(dcstate,measure,kindlist,v_or_t,arg,calimatrix=None,qnum=10,**paras):

    # awg_z = measure.awg[qubit.inst['z_awg']]
    for j,i in enumerate(tqdm(v_or_t,desc='Z_sequence_im')):
        paras[arg] = i
        flux_im = {}
        shiftlist = {}

        for k in dcstate:
            q = measure.qubits[k]
            paras['imAmp']=dcstate[k]
            # paras['volt'] = 0
            # paras['readvolt'] = 0

            pulse_im,shifttime = await funcarg(zWave_im,q,**paras)
            flux_im[k] = pulse_im[0] 
            shiftlist[k] = shifttime
            # print(pulse_im.max())
        # pulse_z, = await funcarg(zWave,qubit,**paras)
        # imbias, = await funcarg(specf2v,qubit,**paras)
        # pulse = pulse_z + imbias
        # pulse, = await funcarg(zWave_im,qubit,**paras)
        current_im = await mrw.zManage(measure,flux_im,calimatrix=calimatrix,qnum=qnum)
        # print(shiftlist)
        for c in current_im:
            l = len(measure.t_new)
            x = np.zeros(l)
            num = int(shiftlist[c]/0.4e-9)
            index = l-num
            x[:index] = current_im[c][num:]
            current_im[c] = x
        # flux_z = {i:measure.qubits[i].volt for i in dcstate}
        # volt_z = await mrw.zManage(measure,flux_z,calimatrix=calimatrix,qnum=qnum) 
        for k in dcstate:
            name_z = [measure.wave[kindlist[k]][0][j]]
            awg_z = measure.awg[measure.qubits[k].inst['z_awg']]
            # pulse_z, = await funcarg(zWave,measure.qubits[k],during=i,volt=volt_z[k][0],shift=paras["shift"])
            pulse = current_im[k] 
            # pulse = current_im[k] + pulse_z
            await writeWave(measure,awg_z,name_z,[pulse])
    gc.collect()


async def Z_sequence(qubit,measure,kind,v_or_t,arg,**paras):

    awg_z = measure.awg[qubit.inst['z_awg']]
    for j,i in enumerate(v_or_t):
        paras[arg] = i
        pulse_z = await funcarg(zWave,qubit,**paras)
        pulse = pulse_z 
        name_z = [measure.wave[kind][0][j]]
        await writeWave(measure,awg_z,name_z,pulse)


async def zWave_cos(volt=0,during=0e-9,shift=0e-9,timing={'z>xy':0,'read>xy':0},zCali=None,delta_im=0,imAmp=0,phaseim=0,readvolt=0):
    shift += timing['read>xy'] - timing['z>xy']
    pulse = (wn.square(during,0e-9) << (during/2 + shift)) * volt

    pulse_read = (wn.square(3000e-9,0e-9) >> (3000e-9/2)) * readvolt 
    pulse += (pulse_read << shift)

    pulselist = (pulse,) 
    zwave, = pulseTowave(pulselist) if zCali is None else caliZ(pulseTowave(pulselist),zCali)

    pulse_im = (wn.square(during,0e-9) << (during/2 + shift))
    if delta_im != 0:
        wav_I, wav_Q = wn.mixing(imAmp*pulse_im,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
    # fc,vtarget = dt.specshift(specfuncz,f_ex/1e9,volt)
    
    fshift, = pulseTowave((wav_I,))
    # vwindow, = pulseTowave((pulse_im,))
    bias = fshift
    pulse = bias+zwave
    # print(np.max(pulse))
    return (pulse,)

async def Z_sequence_cos(qubit,measure,kind,v_or_t,arg,**paras):

    awg_z = measure.awg[qubit.inst['z_awg']]
    for j,i in enumerate(tqdm(v_or_t,desc='Z_sequence_cos')):
        paras[arg] = i
        pulse_z = await funcarg(zWave_cos,qubit,**paras)
        pulse = pulse_z 
        name_z = [measure.wave[kind][0][j]]
        await writeWave(measure,awg_z,name_z,pulse)

################################################################################
# OTOC
################################################################################

async def zWave_otoc(specfuncz,dressenergy,f_ex,volt=0,voltn=0,during=0e-9,shift=0e-9,timing={'z>xy':0,'read>xy':0},\
    zCali=None,delta_im=0,imAmp=0,phaseim=0,readvolt=0,gateduring=0,imAmpn=0,volt_zgate=0,imAmp_off=0,rise=0):
    # print(phaseim)
    shift += timing['read>xy'] - timing['z>xy']
    shift1 = np.copy(shift) 
    shift = 0
    during1 = during
    during2 = during
    pulse_z0 = (wn.square(gateduring,0.8/1e9) << (gateduring/2 + shift + during2))
    pulse_im = (wn.square(during1,1.8/1e9) << (during1/2 + shift + during2 + gateduring))
    pulse_imn = (wn.square(during2,1.8/1e9) << (during2/2 + shift))
    if delta_im != 0:
        wav_I1, wav_Q1 = wn.mixing(imAmp*pulse_im,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
        wav_I2, wav_Q2 = wn.mixing(imAmpn*pulse_imn,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
        wav_I3, wav_Q3 = wn.mixing(imAmp_off*pulse_z0,phase=phaseim,freq=delta_im,ratioIQ=-1.0)
    # fc,vtarget = dt.specshift(specfuncz,f_ex/1e9,volt)
    wav_I = wav_I1 + wav_I2 + wav_I3
    fshift, = pulseTowave((wav_I,))
    # vwindow, = pulseTowave((pulse_im,))
    # bias = fshift
    bias = dt.biasshift(specfuncz,f_ex/1e9,fshift/2/np.pi) 
    
    bias_offset = 0
    bias_offset_n = 0
    bias_offset_off = 0
    if dressenergy is not None:
        # if np.abs(dressenergy(0)-f_ex/1e9)>1e6:
        #     raise 'freq error'
        fnew = dressenergy(np.abs(imAmp)/2/np.pi)-dressenergy(0)
        bias_offset = dt.biasshift(specfuncz,f_ex/1e9,fnew,'lower')
        fnew_n = dressenergy(np.abs(imAmpn)/2/np.pi)-dressenergy(0)
        bias_offset_n = dt.biasshift(specfuncz,f_ex/1e9,fnew_n,'lower')
        fnew_off = dressenergy(np.abs(imAmp_off)/2/np.pi)-dressenergy(0)
        # bias_offset_off = dt.biasshift(specfuncz,f_ex/1e9,fnew_off,'lower')

    # pulse = (wn.square(during,0e-9) << (during/2 + shift)) * (volt-bias_offset)
    zoffset_im = pulse_im*(volt-bias_offset)
    zoffset_imn = pulse_imn*(volt-bias_offset_n)
    zoffset_off = pulse_z0*(volt_zgate-bias_offset_off)
    # pulse_z = pulse_z0*(volt_zgate+volt)
    pulse = zoffset_im + zoffset_imn + zoffset_off
    pulse_read = (wn.square(3000e-9,0e-9) >> (3000e-9/2)) * readvolt 
    pulse += (pulse_read << shift)
    pulselist = (pulse,) 
    zwave, = pulseTowave(pulselist)

    pulse = bias+zwave
    # print(np.max(pulse))
    return (pulse,) if zCali is None else caliZ((pulse,),zCali), shift1

################################################################################
# 偏置sequence_otoc
################################################################################

async def Z_sequence_otoc(dcstate,measure,kindlist,v_or_t,arg,calimatrix=None,qnum=10,**paras):
    shift_copy = float(np.copy(paras['shift']))
    # awg_z = measure.awg[qubit.inst['z_awg']]
    for j,i in enumerate(tqdm(v_or_t,desc='Z_sequence_otoc')):
        paras[arg] = i
        flux_im = {}
        shiftlist = {}

        for k in dcstate:
            q = measure.qubits[k]
            paras['shift'] = q.shift + shift_copy
            # paras['imAmp']=dcstate[k]
            # paras['volt'] = 0
            # paras['readvolt'] = 0

            pulse_im,shifttime = await funcarg(zWave_otoc,q,**paras)
            flux_im[k] = pulse_im[0] 
            shiftlist[k] = shifttime
            # print(pulse_im.max())
        # pulse_z, = await funcarg(zWave,qubit,**paras)
        # imbias, = await funcarg(specf2v,qubit,**paras)
        # pulse = pulse_z + imbias
        # pulse, = await funcarg(zWave_im,qubit,**paras)
        current_im = await mrw.zManage(measure,flux_im,calimatrix=calimatrix,qnum=qnum)
        # print(shiftlist)
        for c in current_im:
            l = len(measure.t_new)
            x = np.zeros(l)
            num = int(shiftlist[c]/0.4e-9)
            index = l-num
            x[:index] = current_im[c][num:]
            current_im[c] = x
        # flux_z = {i:measure.qubits[i].volt for i in dcstate}
        # volt_z = await mrw.zManage(measure,flux_z,calimatrix=calimatrix,qnum=qnum) 
        for k in dcstate:
            name_z = [measure.wave[kindlist[k]][0][j]]
            awg_z = measure.awg[measure.qubits[k].inst['z_awg']]
            # pulse_z, = await funcarg(zWave,measure.qubits[k],during=i,volt=volt_z[k][0],shift=paras["shift"])
            pulse = current_im[k] 
            # pulse = current_im[k] + pulse_z
            await writeWave(measure,awg_z,name_z,[pulse])
    gc.collect()

################################################################################
# oToc_tomo
################################################################################

async def oToc_tomo_wave(envelopename=['square',1],axis='X',during=0,gateduring=0,amp=1,pi_len=75e-9,\
    shift=0e-9,delta_ex=110e6,phaseDiff=0.0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):
    shift += timing['read>xy']
    # trig_pi = wn.square(envelopename[1]*pi_len*2) << (trig_shift+shift+gateduring+pi_len+during+envelopename[1]*pi_len*0.5)
    trig_pi = whichEnvelope((envelopename[1]*pi_len+20e-9),'square',envelopename[1]) << (trig_shift+shift+gateduring+pi_len+during)
    pulse_pi = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,shift=(shift+gateduring+pi_len+during),axis='Xhalf',delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff)

    # trig_half = wn.square(envelopename[1]*pi_len) << (trig_shift+shift+envelopename[1]*pi_len*0.25)
    trig_half = whichEnvelope((pi_len+20e-9),'square',envelopename[1]) << (trig_shift+shift)
    pulse_half = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,axis=axis,shift=timing['read>xy'],delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff)
    trig_pi, trig_half = pulseTowave((trig_pi,)), pulseTowave((trig_half,))
    pulse = np.array(pulse_pi)+np.array(pulse_half)
    pulselist = pulse[0],pulse[1], np.array((None,)), [(trig_pi[0]+trig_half[0])/np.max(trig_pi+trig_half)]

    # print(np.array(trig_half)+np.array(trig_pi))
    return pulselist


async def tomoWave(qubit,axis='Xhalf',shift=0e-9):
    pulse_tomo = await funcarg(qgw.singleQgate,qubit,axis=axis)
    pulse = np.array(pulse_tomo)
    return pulse

async def qptomoWave(qubit,init='I',axis='Xhalf',shift=0e-9):
    pulse_init = await funcarg(qgw.singleQgate,qubit,axis=init,shift=(shift+qubit.pi_len))
    pulse_tomo = await funcarg(qgw.singleQgate,qubit,axis=axis,shift=0)
    pulse = np.array(pulse_tomo) + np.array(pulse_init)
    return pulse

async def qwalkWave(qubits,init=['I'],axis=['Xhalf'],shift=0e-9):
    pulses = [[]]*len(qubits)
    for i,qubit in qubits:
        pulse_init = await funcarg(qgw.singleQgate,qubits,axis=init[i],shift=(shift+qubits.pi_len))
        pulse_tomo = await funcarg(qgw.singleQgate,qubits,axis=axis[i],shift=0)
        pulse = np.array(pulse_tomo) + np.array(pulse_init)
        pulses[i] = pulse.copy()
    return pulses

################################################################################
# sqrtiswap_Tomography
################################################################################

async def sqrtiswaptomoWave(qubit,axis='Xhalf',shift=0e-9):
    
    pulse_rabi = await funcarg(qgw.singleQgate,qubit,axis='X',\
        shift=(shift+(qubit.during_swap+qubit.pi_len+qubit.gateduring*2)),virtualPhase=0)
    pulse_rabi1 = await funcarg(qgw.singleQgate,qubit,axis='Xhalf',shift=(qubit.pi_len+shift))
    pulse_tomo = await funcarg(qgw.singleQgate,qubit,axis=axis,virtualPhase=0)
    pulse = np.array(pulse_rabi) + np.array(pulse_tomo)
    # pulse = np.array(pulse_rabi) + np.array(pulse_tomo) + np.array(pulse_rabi1)
    return pulse
################################################################################
# phase_gate
################################################################################

async def Z_sequence_pg(qubit,measure,kind,v_or_t,arg,num_gate,**paras):
    shift_m = float(np.copy(paras['shift']))
    awg_z = measure.awg[qubit.inst['z_awg']]
    for j,i in enumerate(v_or_t):
        paras[arg] = i
        pulse_z = 0
        shift = shift_m
        for n in range(num_gate):
            paras['shift'] = shift
            pulse_gate = await funcarg(zWave,qubit,**paras)
            # shift += (paras['during'] + 5e-9)
            # paras['volt'], paras['shift'] = qubit.volt, shift
            # pulse_res = await funcarg(zWave,qubit,**paras)
            # pulse_z += (pulse_gate[0] + pulse_res[0])
            pulse_z += pulse_gate[0]
            shift += (paras['during'] + 5e-9)
        pulse = np.array([pulse_z]) 
        name_z = [measure.wave[kind][0][j]]
        await writeWave(measure,awg_z,name_z,pulse)

################################################################################
# phasegate_tomo
################################################################################

async def phasegate_tomo_wave(envelopename=['square',1],axis='X',during=0,gateduring=0,amp=1,pi_len=75e-9,\
    shift=0e-9,delta_ex=110e6,phaseDiff=0.0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):
    shift += timing['read>xy']

    # trig_pi = wn.square(envelopename[1]*pi_len*2) << (trig_shift+shift+gateduring+pi_len+during+envelopename[1]*pi_len*0.5)
    trig_pi = whichEnvelope((envelopename[1]*pi_len+20e-9),'square',envelopename[1]) << (trig_shift+shift+gateduring+pi_len+during)
    pulse_halfpi = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,shift=(shift+gateduring+pi_len+during),axis='Xhalf',delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff)

    # trig_half = wn.square(envelopename[1]*pi_len) << (trig_shift+shift+envelopename[1]*pi_len*0.25)
    trig_half = whichEnvelope((pi_len+20e-9),'square',envelopename[1]) << (trig_shift+shift)
    pulse_half = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,axis=axis,shift=timing['read>xy'],delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff)
    trig_pi, trig_half = pulseTowave((trig_pi,)), pulseTowave((trig_half,))
    pulse = np.array(pulse_halfpi)+np.array(pulse_half)
    pulselist = pulse[0],pulse[1], np.array((None,)), [(trig_pi[0]+trig_half[0])/np.max(trig_pi+trig_half)]

    # print(np.array(trig_half)+np.array(trig_pi))
    return pulselist

################################################################################
# 矫正相位
################################################################################

async def phaseCaliwave(envelopename=['square',1],t_run=0,amp=1,pi_len=75e-9,nwave=0,seqtype='CPMG',\
    detune=3e6,shift=0e-9,delta_ex=110e6,phaseDiff=0.0,DRAGScaling=None,timing={'z>xy':0,'read>xy':0}):
    shift += timing['read>xy']

    pulse1 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,shift=(shift+t_run+pi_len),axis='Xhalf',delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff,timing=timing)

    pulse2 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,amp=amp,axis='Xhalf',shift=shift,delta_ex=delta_ex,\
    DRAGScaling=DRAGScaling,phaseDiff=phaseDiff,timing=timing)
    pulse = np.array(pulse1)+np.array(pulse2)
    return pulse


# ################################################################################
# # 单比特tomo测试
# ################################################################################

# async def tomoTest(measure,awg,t,halfpi,axis,name,DRAGScaling):

#     gatepulse = await rabiWave(measure.envelopename,pi_len=t/1e9,shift=(halfpi)/1e9,\
#         delta_ex=110e6,DRAGScaling=DRAGScaling)
#     # gatepulse = await rabiWave(envelopename=measure.envelopename,pi_len=halfpi/1e9)
#     tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,\
#         delta_ex=110e6,axis=axis,DRAGScaling=DRAGScaling)
#     pulse = np.array(gatepulse) + np.array(tomopulse)
#     await writeWave(measure,awg,name,pulse)

# ################################################################################
# # ramseyZpulse波形
# ################################################################################

# async def ramseyZwave(measure,awg,halfpi,axis,name,DRAGScaling,shift):
#     t_int = 100e-9
#     gatepulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=(t_int+(shift+halfpi)/1e9),\
#         delta_ex=110e6,axis='Xhalf',DRAGScaling=DRAGScaling)
#     # pulsespinecho = await rabiWave(measure.envelopename,pi_len=halfpi/1e9,shift=(t_int+halfpi/1e9),\
#     #     phase=-np.pi/2,delta_ex=110e6,DRAGScaling=DRAGScaling)
#     tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=shift/1e9,\
#         delta_ex=110e6,axis=axis,DRAGScaling=DRAGScaling)
#     pulse = np.array(gatepulse) + np.array(tomopulse)
#     await writeWave(measure,awg,name,pulse)

# ################################################################################
# # ramseyZpulse_chen波形
# ################################################################################

# async def ramseyZwave_chen(measure,awg,halfpi,axis,shift,name,DRAGScaling):
#     t_int = 700e-9
#     gatepulse = await tomoWave(measure.envelopename,halfpi/1e9,shift=(t_int+shift+halfpi/1e9),\
#         delta_ex=110e6,axis='Xhalf',DRAGScaling=DRAGScaling)
#     # pulsespinecho = await rabiWave(measure.envelopename,pi_len=halfpi/1e9,shift=(t_int+halfpi/1e9),\
#     #     phase=-np.pi/2,delta_ex=110e6,DRAGScaling=DRAGScaling)
#     tomopulse = await tomoWave(measure.envelopename,halfpi/1e9,delta_ex=110e6,axis=axis,DRAGScaling=DRAGScaling)
#     pulse = np.array(gatepulse) + np.array(tomopulse)
#     await writeWave(measure,awg,name,pulse)

# ################################################################################
# # AllXY drag detune
# ################################################################################

# async def dragDetunewave(measure,awg,pilen,coef,axis,name_ch):
#     pulse1 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis=axis[0],DRAGScaling=coef)
#     pulse2 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis=axis[1],DRAGScaling=coef)
#     pulse = np.array(pulse1) + np.array(pulse2)
#     await writeWave(measure,awg,name_ch,pulse)
    
# ################################################################################
# # dragcoefHD
# ################################################################################

# async def HDWave(measure,awg,pilen,coef,axis,nwave,name_ch):
#     pulseequator2 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis='Xhalf',DRAGScaling=coef,phaseDiff=0)
#     pulsem, shift = np.array([wn.zero()]*4), 0
#     for i in range(nwave):
#         shift += pilen/1e9 +10e-9
#         pulse1 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,shift=shift,delta_ex=110e6,axis=axis[0],DRAGScaling=coef,phaseDiff=0)
#         shift += pilen/1e9 + 10e-9
#         pulse2 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,shift=shift,delta_ex=110e6,axis=axis[1],DRAGScaling=coef,phaseDiff=0)
#         pulsem += (np.array(pulse1)+np.array(pulse2))
#     pulseequator1 = await tomoWave(measure.envelopename,pi_len=pilen/1e9,shift=(shift+pilen/1e9+10e-9),\
#         delta_ex=110e6,axis='Xhalf',DRAGScaling=coef,phaseDiff=0)
#     pulse = np.array(pulseequator1) + np.array(pulseequator2) + np.array(pulsem)
#     await writeWave(measure,awg,name_ch,pulse)

# ################################################################################
# # RTO
# ################################################################################

# async def rtoWave(measure,awg,pilen,name_ch):
#     envelopename = measure.envelopename
#     pulsegate = await tomoWave(envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis='Xhalf',shift=(pilen+100)/1e9,DRAGScaling=None)
#     pulsetomoy = await tomoWave(envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis='Ynhalf',DRAGScaling=None)
#     pulsetomox = await tomoWave(envelopename,pi_len=pilen/1e9,delta_ex=110e6,axis='Xhalf',DRAGScaling=None)
#     pulse1 = np.array(pulsegate)+np.array(pulsetomoy)
#     pulse2 = np.array(pulsegate)+np.array(pulsetomox)
#     await writeWave(measure,awg,name_ch[0],pulse1)
#     await writeWave(measure,awg,name_ch[1],pulse2)

################################################################################
# RB波形
################################################################################

async def rbWave(m,gate,envelopename=['square',1],pi_len=30e-9,amp=1,delta_ex=110e6,shift=0,\
    DRAGScaling=None,phaseDiff=0,timing={'z>xy':0,'read>xy':0}):

    op = mx.op

    mseq = mx.cliffordGroup_single(m,gate)
    if mseq == []:
        return
    rotseq = []
    for i in mseq[::-1]:
        rotseq += op[i]
    waveseq_I, waveseq_Q, wav = np.zeros(len(t_new)), np.zeros(len(t_new)), np.zeros(len(t_new))
    # rotseq = ['Xhalf','Xnhalf','Yhalf','Ynhalf']*m
    # if rotseq == []:
    #     return
    # print(rotseq)
    for i in rotseq:
        # paras = genParas(i)
        if i == 'I':
            waveseq_I += 0
            waveseq_Q += 0
            # continue
        else:
            wav_I, wav_Q, mrk1, mrk2 = await qgw.singleQgate(envelopename=envelopename,pi_len=pi_len,\
                amp=amp,shift=shift,delta_ex=delta_ex,axis=i,\
                DRAGScaling=DRAGScaling,phaseDiff=phaseDiff,timing=timing)
            waveseq_I += wav_I
            waveseq_Q += wav_Q
        if envelopename[1] == 2:
            if i in ['X','Y','Z']:
                shift += envelopename[1]*pi_len
            else:
                shift += pi_len
        if envelopename[1] == 1:
            shift += pi_len
        # if paras[1] == 'pi':
        #     shift += envelopename[1]*pi_len
        # if paras[1] == 'halfpi':
        #     shift += pi_len

    return waveseq_I, waveseq_Q, wn.zero(), wn.zero()

################################################################################
# RB Sequence
################################################################################

async def rb_sequence(qubit,measure,kind,mlist,arg,**paras):
    awg= measure.awg[qubit.inst['ex_awg']]
    await awg.stop()
    for j,i in enumerate(tqdm(mlist,desc='RB_sequence')):
        paras[arg] = i
        name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
        pulse = await funcarg(rbWave,qubit,**paras)
        await writeWave(measure,awg,name_ch,pulse,mark=False)
        # pulse = await rbWave(measure,m,gate,pilen*1e-9)
        # await writeWave(measure,awg,name_ch,pulse)


# async def Rabi_sequence(qubit,measure,kind,v_or_t,arg,**paras):
#     # print(v_or_t,arg,paras)
#     awg= measure.awg[qubit.inst['ex_awg']]
#     await awg.stop()
#     for j,i in enumerate(tqdm(v_or_t,desc=''.join(('Rabi_',kind)))):
#     # for j,i in enumerate(v_or_t):
#         paras[arg] = i
#         name_ch = [measure.wave[kind][0][j],measure.wave[kind][1][j]]
#         pulse = await funcarg(rabiWave,qubit,**paras)
#         await writeWave(measure,awg,name_ch,pulse,mark=False)
    
# tomo
