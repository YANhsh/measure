
import sys, time, visa
from os.path import dirname, abspath
import matplotlib.pyplot as plt
from qulab.job import Job
from scipy import fftpack,signal


import numpy as np
import serial, time
from device_interface import DeviceInterface 
from qulab import computewave_ztw as cw
from qulab import ezQ as ezQ


class common():
    def __init__(self,freqall,ad,dc,psg,awg,jpa,qubits,att):
        self.freqall = freqall
        self.ad = ad
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = qubits
        self.wave = {}
        self.att = {}
        self.inststate = 0
        self.t_list = np.linspace(0,5000,10001)*1e-9
        self.t_range = (-45e-6,5e-6)
        self.depth = 2000
        self.delta = np.array([80e6])
        self.delta_ex = np.array([200e6])
        self.amp = np.array([0])

###################################################################################
#采集数据
####################################################################################
def getExpArray(f_list, depth, weight=None, sampleRate=1e9):
    e = []
    t = np.arange(0, depth, 1) / sampleRate
    if weight is None:
        weight = np.ones(depth)
    for f in f_list:
        e.append(weight * np.exp(-1j * 2 * np.pi * f * t))
    return np.asarray(e).T

def demodulation(qubit, measure, fft=False, avg=False, offset=True, is2ch=True):
    #待调整
    e = getExpArray(measure.f_list, measure.depth, weight=None, sampleRate=1e9)
    n = e.shape[0]
    measure.ad.ad_data_clear()
    ####注意顺序！！！！！！
    measure.ad.ad_enable_trig()
    measure.awg[qubit.inst['lo_awg']].da_master_trigg_enable()
    res = measure.ad.ad_getdata()
    A, B = res[1]/1e5, res[2]/1e5
    if fft:
        A = (A[:, :n]).dot(e)
        B = (B[:, :n]).dot(e)
    if avg:
        return A.mean(axis=0), B.mean(axis=0)
    else:
        return A, B

def getIQ(qubit, measure, fft=False, avg=False, offset=True, is2ch=False):
    return demodulation(qubit, measure, fft, avg, offset, is2ch)

def getTraces(qubit, measure, fft=False, avg=True, offset=True, is2ch=False):
    return demodulation(qubit, measure, fft, avg, offset, is2ch)


def resn(f_list):
    f_list = np.array(f_list)
    f_lo = f_list.max() - 80e6
    delta =  -(f_lo - f_list)
    n = len(f_list)
    return f_lo, delta, n

def awg_stop(measure):
    for i in (measure.awg):
        for j in [1,2,3,4]:
            measure.awg[i].da_stop_output_wave(j)

def dc_zero(measure):
    for k,v in enumerate(measure.dc):
        measure.dc[v].dc(0)

############################################################################
##利用crosstalk，设置多个比特的bias
############################################################################
def qnames(measure):
    qnames =[0]*len(measure.qubits)
    for k,v in enumerate (measure.qubits):
        qnames[k] = v.q_name
    return qnames

def Want_current(qubit,current, measure):
    biaslist = [0]*len(measure.qubits)
    if qubit.q_name in qnames(measure):
        index = int(qubit.q_name[1]) - int(((measure.qubits[0]).q_name)[1])
        biaslist[index]=current
        print('OK: %s in measure.qubits!'%qubit.q_name)
    else:
        print('Error: %s not in measure.qubits!'%qubit.q_name)
        biaslist = None
    print('Want_current: ')
    print(biaslist)
    return biaslist

def inst_current(measure,want_current,calimatrix):
    if len(want_current)>1:
        calimatrix = np.mat(calimatrix)
        inst_current = calimatrix.I*np.mat(want_current).T
        print('Inst_DC: ')
        for k,qubit in enumerate(measure.qubits):
            i = np.array(inst_current)[:,0][k]
            measure.dc[qubit.q_name].dc(i)
            print(i)
    else:
        i = want_current[0]
        measure.dc[qubit.q_name].dc(want_current[i])


def inst_Zpulse(measure,want_current,calimatrix,channel_output_delay=59e3):
    calimatrix = np.mat(calimatrix)
    inst_current = calimatrix.I*np.mat(want_current).T
    print('Inst_Zpulse: ')
    for k,qubit in enumerate(measure.qubits):
        i = np.array(inst_current)[:,0][k]
        cw.z_pulse(qubit,measure,width=30000,amp=i,channel_output_delay=channel_output_delay)
        # cw.z_envelope(qubit,measure,width=30000e-9,amp=i,channel_output_delay=channel_output_delay, Delta_lo=200e6,t=0)
        print(i)

def bias_zeros(measure):
    N = len(measure.qubits)
    inst_Zpulse(measure,[0]*N,np.eye(N))
    inst_current(measure,[0]*N,np.eye(N))

############################################################################
##S21   async def S21(qubit,flux,measure,freq,calimatrix):
############################################################################
async def S21(qubit,measure,freq):
    #波形write
    await measure.psg['psg_lo'].setValue('Output','ON')
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    time.sleep(0.01)
    for i in freq:
        await measure.psg['psg_lo'].setValue('Frequency', i)
        #采数
        res = getIQ(qubit, measure)
        if isinstance(res, int):
           break 
        a = np.mean(res[0], axis=0)  # 对各列求均值
        b = np.mean(res[1], axis=0)
        s = a + 1j*b
        yield i+measure.delta, s

    
############################################################################
#
##S21vsFlux
#
############################################################################
async def S21vsFlux(qubit,current,measure,freq,calimatrix):
    for i in current:
        # measure.dc[qubit.q_name].dc(i)
        print('current=%f'%i)
        # cw.z_pulse(qubit,measure,width=30000,amp=i,channel_output_delay=69e3)

        measure.dc[qubit.q_name].dc(qubit.bias)
        want_current =  Want_current(qubit,i, measure)
        inst_Zpulse(measure,want_current,calimatrix,channel_output_delay=69e3)

        job = Job(S21, (qubit,measure,freq),auto_save=True,max=len(freq),tags=[qubit.q_name])
        f_s21, s_s21 = await job.done()
        yield [i], f_s21, s_s21
    bias_zeros(measure)


############################################################################
##S21vsPower
############################################################################
async def S21vsPower(qubit,measure,freq,attv,attlo):
    for i in attv:
        # await measure.psg['psg_lo'].setValue('Power',i)
        await attlo.set_att(i)
        job = Job(S21, (qubit,measure,freq),auto_save=False,max=len(freq),tags=[qubit.q_name])
        print(i)
        f_s21, s_s21 = await job.done()
        yield [i], f_s21, s_s21
    
################################################################################
# 重新混频
################################################################################

async def again(qubit,measure,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,126)*1e6+f_lo
    # for i in measure.psg:
    #     if i != 'psg_lo' and i != 'psg_pump':
    #         await measure.psg[i].setValue('Output','OFF')
    length = len(freq) if freq is not None else 201
    job = Job(S21, (qubit,measure,freq),auto_save=True,max=len(freq),tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n =  resn(np.array(f_res))
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        #await cw.ats_setup(measure.ats,delta)
        cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
        base = 0
        for i in range(15):
            res = getIQ(qubit, measure)
            if isinstance(res, int):
                break 
            a = np.mean(res[0], axis=0)  # 对各列求均值
            b = np.mean(res[1], axis=0)
            s = a + 1j*b
            base += s
        base /= 15
    measure.base, measure.n, measure.delta, measure.f_lo = base, n, delta, np.array([f_lo])
    return f_lo, delta, n, f_res, base,f_s21, s_s21


############################################################################
##Singlespec
############################################################################
async def singlespec(qubit,flux, measure, readpoint, ex_freq,calimatrix):
    # cw.z_pulse(qubit,measure,width=3000,amp=10000)
    measure.dc[qubit.q_name].dc(qubit.bias)
    want_current =  Want_current(qubit,flux, measure)
    inst_Zpulse(measure,want_current,calimatrix,channel_output_delay=59e3)
    # inst_current(measure,want_current,calimatrix)
    
    readpoint = readpoint - 80e6
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    # if readpoint:
    #     f_lo, delta, n, f_res, base,f_s21, s_s21 = await again(qubit,measure,modulation=False,flo=None,freq=None)
    await measure.psg['psg_ex'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Output','ON')
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    cw.modulation_ex(qubit,measure)
    for i in ex_freq:
        await measure.psg['psg_ex'].setValue('Frequency',i)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i], s[0]
    await measure.psg['psg_ex'].setValue('Output','OFF')
    bias_zeros(measure)


################################################################################
# Spec2d
################################################################################
async def spec2d(qubit,measure,ex_freq,lo_freq,current,calimatrix):
    bias_zeros(measure)
    for i in current:
        # measure.dc[qubit.q_name].dc(flux)
        want_current =  Want_current(qubit,i, measure)
        inst_current(measure,want_current,calimatrix)

        job = Job(S21, (qubit,measure,lo_freq),auto_save=False,max=len(lo_freq),tags=[qubit.q_name])
        f_s21, s_s21 = await job.done()
        index = np.abs(s_s21).argmin(axis=0)
        # readpoint = f_s21[index]
        readpoint = lo_freq[index]+80e6
        print(index)
        job = Job(singlespec, (qubit,i, measure, readpoint, ex_freq,calimatrix),auto_save=True,max=len(ex_freq),tags=[qubit.q_name])    ###多个混频待解决
        f_ss, s_ss = await job.done()
        #n = np.shape(s_ss)[1]
        yield [i]*8, f_ss, s_ss

################################################################################
# Spec2d_zpulse
################################################################################
async def spec2d_zpulse(qubit,measure,ex_freq,lo_freq,current,calimatrix,othercurrents):
    bias_zeros(measure)
    for i in current:
        for k,v in enumerate(measure.qubits):
            measure.dc[v.q_name].dc(v.bias)
        measure.dc[qubit.q_name].dc(qubit.bias)
        # cw.z_pulse(qubit,measure,width=30000,amp=i)
        want_current =  Want_current(qubit,i, measure) + othercurrents
        inst_Zpulse(measure,want_current,calimatrix,channel_output_delay=59e3)

        job = Job(S21, (qubit,measure,lo_freq),auto_save=False,max=len(lo_freq),tags=[qubit.q_name])
        f_s21, s_s21 = await job.done()
        index = np.abs(s_s21).argmin(axis=0)
        # readpoint = f_s21[index]
        readpoint = lo_freq[index] + 80e6
        
        job = Job(singlespec, (qubit, i, measure, readpoint, ex_freq,calimatrix),auto_save=True,max=len(ex_freq),tags=[qubit.q_name])
        f_ss, s_ss = await job.done()
        #n = np.shape(s_ss)[1]
        yield [i]*1, f_ss, s_ss
    bias_zeros(measure)


async def spec2d_zt(qubit,measure,ex_freq,lo_freq,calimatrix,timing):
    bias_zeros(measure)
    readpoint = lo_freq + 80e6
    for i in timing:
        want_current =  Want_current(qubit,i, measure)
        inst_Zpulse(measure,want_current,calimatrix,channel_output_delay=59e3)

        job = Job(S21, (qubit,measure,lo_freq),auto_save=False,max=len(lo_freq),tags=[qubit.q_name])
        f_s21, s_s21 = await job.done()
        index = np.abs(s_s21).argmin(axis=0)
        # readpoint = f_s21[index]
        readpoint = lo_freq[index] + 80e6

        job = Job(singlespec, (qubit, qubit.bias, measure, readpoint, ex_freq,calimatrix),auto_save=True,max=len(ex_freq),tags=[qubit.q_name])
        f_ss, s_ss = await job.done()
        #n = np.shape(s_ss)[1]
        yield [i]*1, f_ss, s_ss
    bias_zeros(measure)


################################################################################
# Rabi
################################################################################
async def Rabi(qubit,measure,readpoint,f_ex,t_rabi,amp,calimatrix):
    bias_zeros(measure)
    want_current =  Want_current(qubit,qubit.bias, measure)
    inst_current(measure,want_current,calimatrix)

    readpoint=readpoint-80e6
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = f_ex-measure.delta_ex[0]
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_rabi:
        cw.rabiWave(qubit, measure,amp=amp, envelopename='cospulse',nwave=1,\
            during=i/1e9,Delta_lo=measure.delta_ex[0],phase=0,phaseDiff=0,DRAGScaling=None)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*1, s[0]
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')

async def RabiPower(qubit,measure,readpoint,f_ex, power, pi_len,calimatrix):
    bias_zeros(measure)
    want_current =  Want_current(qubit,qubit.bias, measure)
    inst_current(measure,want_current,calimatrix)

    readpoint=readpoint-80e6
    delta_ex = qubit.delta_ex[0]
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex[0]-delta_ex
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    print('freq=%s'%freq)
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in power:
        cw.rabiWave(qubit, measure, envelopename='square',nwave=1,\
            during=pi_len/1e9,Delta_lo=delta_ex,amp=i,phase=0,phaseDiff=0,DRAGScaling=None)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*1, s
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# T1
################################################################################
async def T1(qubit, measure,readpoint, f_ex, t_t1, pi_len,amp,calimatrix):
    bias_zeros(measure)
    flux = qubit.bias
    want_current =  Want_current(qubit,flux, measure)
    inst_current(measure,want_current,calimatrix)

    readpoint=readpoint-80e6
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = f_ex-measure.delta_ex[0]
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cw.rabiWave(qubit, measure, envelopename='cospulse',during=pi_len/1e9/2,shift=i/1e9,Delta_lo=measure.delta_ex[0],amp=amp)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*1, s[0]
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# Ramsey
################################################################################

async def Ramsey(qubit, measure,readpoint, f_ex, t_t1, pi_len,amp,calimatrix):
    bias_zeros(measure)
    want_current =  Want_current(qubit,qubit.bias, measure)
    inst_current(measure,want_current,calimatrix)

    readpoint=readpoint-80e6
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = f_ex-measure.delta_ex[0]
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cw.ramseyWave(qubit, measure,envelopename='cospulse', delay=i/1e9, halfpi=pi_len/1e9/2,fdetune=3e6,amp=amp)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s 
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# Spin echo
################################################################################

async def SpinEcho(qubit, measure,readpoint, f_ex, t_t1, pi_len,seqtype,amp,calimatrix):
    bias_zeros(measure)
    want_current =  Want_current(qubit,qubit.bias, measure)
    inst_current(measure,want_current,calimatrix)

    readpoint=readpoint-80e6
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Output','ON')
    cw.modulation_read(qubit, measure, measure.delta, readlen=measure.readlen) 
    freq = f_ex-measure.delta_ex[0]
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_t1:
        cw.coherenceWave(qubit, measure, envelopename='cospulse',t_run=i/1e9,during=pi_len/1e9/2,amp=amp,n_wave=1,seqtype=seqtype,detune=3e6,shift=0, channel_output_delay=89.9e3,Delta_lo=measure.delta_ex[0])
        # cw.ramseyWave(qubit, measure,envelopename='cospulse', delay=i/1e9, halfpi=pi_len/1e9/2,fdetune=3e6,amp=amp)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')

###############################################################################
# Crosstalk
################################################################################

async def CS21(target_qubit,bias_qubit,measure,readpoint,ex_freq,flux,clist):
    
    #波形write
    measure.dc[target_qubit.q_name].dc(flux)

    readpoint = readpoint - 80e6
    await measure.psg['psg_lo'].setValue('Frequency',readpoint)
    await measure.psg['psg_ex'].setValue('Frequency',ex_freq)
    
    await measure.psg['psg_ex'].setValue('Output','ON')
    await measure.psg['psg_lo'].setValue('Output','ON')
    cw.modulation_read(target_qubit, measure, measure.delta, readlen=measure.readlen) 
    cw.modulation_ex(target_qubit,measure)
    for i in clist:
        cw.z_pulse(bias_qubit,measure,width=3000,amp=i)
        #采数
        res = getIQ(target_qubit, measure)
        if isinstance(res, int):
           break 
        a = np.mean(res[0], axis=0)  # 对各列求均值
        b = np.mean(res[1], axis=0)
        s = a + 1j*b
        yield [i], s[0]
    await measure.psg['psg_ex'].setValue('Output','OFF')
    

async def Crosstalk(target_qubit,bias_qubit,measure,ex_freq,readpoint,compenlist,biaslist,flux):
    bias_zeros(measure)
    print('start')
    for i in compenlist:
        cw.z_pulse(target_qubit,measure,width=3000,amp=i)
        
        job = Job(CS21, (target_qubit,bias_qubit,measure,readpoint,ex_freq,flux,biaslist),auto_save=False,max=len(biaslist),tags=[bias_qubit.q_name])
        f_ss, s_ss = await job.done()
        yield [i], f_ss, s_ss
        
    bias_zeros(measure)
    await measure.psg['psg_ex'].setValue('Output','OFF')
    print('end')
    