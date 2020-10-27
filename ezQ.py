import sys, time, visa
from os.path import dirname, abspath
import matplotlib.pyplot as plt
from qulab.job import Job
from scipy import fftpack,signal
project_path = dirname(dirname(abspath(__file__)))
sys.path.append(project_path+r'\qulab\lib')



import numpy as np
import serial, time
from device_interface import DeviceInterface 
from qulab import computewave_ztw as cw

dev = DeviceInterface()




'''
da_id, da_ip = ('DACUSTCF215', '10.0.200.101') # 列出所有DA板ID
ad_id , ad_mac = ('ADCUSTCC013', '00-00-00-00-00-52') # 列出与da对应的ad
host_mac = "54-E1-AD-26-0F-FA" #本机MAC
channel_i = 3 #I通道
channel_q = 4 #Q通道
master_id, master_ip = ('DACUSTCF215', '10.0.200.100')
path = project_path+'/pic/'

dc_id = "DC1"
dc_ip = "10.0.200.110"
dc_port = 5000
'''





class common():
    def __init__(self,freqall,ad,dc,psg,awg,jpa,qubits):
        self.freqall = freqall
        self.ad = ad
        self.dc = dc
        self.psg = psg
        self.awg = awg
        self.jpa = jpa
        self.qubits = {i.q_name:i for i in qubits}
        self.wave = {}
        self.att = {}
        self.inststate = 0
        self.t_list = np.linspace(0,5000,10001)*1e-9
        self.t_range = (-45e-6,5e-6)
        self.depth = 1500


        

################################################################################
# 激励源设置
################################################################################

async def exMixing(f):
    if f == {}:
        return 
    qname = [i for i in f]
    f_ex = np.array([f[i] for i in f])
    ex_lo = f_ex.max() + 220e6
    delta =  ex_lo - f_ex
    delta_ex = {qname[i]:delta[i] for i in range(len(qname))}
    # n = len(f_ex)
    return ex_lo, delta_ex


async def exManage(measure,dcstate={},exstate={},calimatrix=None,qnum=10):
    qubits = measure.qubits
    matrix = np.mat(np.eye(qnum)) if calimatrix == None else np.mat(calimatrix)
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

################################################################
##DC
################################################################
class DC():
    def __init__(self,dc_id,dc_ip,dc_port,dc_channel):
        self.dc_id = dc_id
        self.dc_ip = dc_ip
        self.dc_port = dc_port
        self.dc_channel = dc_channel

    def dc(self,volt):
        ret = dev.dc_set(self.dc_id,self.dc_ip ,self.dc_port, self.dc_channel, ('VOLT', volt))
        
    def dc_query(self):
        ret,volt = dev.dc_query(self.dc_id,self.dc_ip ,self.dc_port, self.dc_channel,'VOLT')
        print (f'dc:[{self.dc_id}] query volt value:[{volt}].')
        return volt


##############################################################
#DA设置
##############################################################
class DA():
    def __init__(self, da_id, da_ip, master_id=None, master_ip=None):
        self.da_id = da_id
        self.da_ip = da_ip
        self.master_id = master_id
        self.master_ip = master_ip

    '''连接DA板，初始化DA板'''
    def da_connect_device(self):
        ret = 0
        channel_gain = [0,0,0,0]
        data_offset = [0,0,0,0]
        ret |= dev.da_connect_device(self.da_id, self.da_ip, channel_gain, data_offset)
        # if self.master_id is not None:
        #     ret |=dev.da_connect_device(self.master_id, self.master_ip, channel_gain, data_offset)
        #     ret |=dev.da_init_device(self.master_id)
        if ret != 0:
            print(f'ERROR:da board:[{self.da_id}] or [{self.master_id}]connect failure ,ret:[{ret}]')
            sys.exit(0)
        else:
            print(f'da board:[{self.da_id}] connect success .')
        return ret

    def da_init_device(self,trig_interval):
        ret = 0
        ret |= dev.da_init_device(self.da_id)
        # if self.master_id is not None:
        #     ret |=dev.da_connect_device(self.master_id, self.master_ip, channel_gain, data_offset)
        #     ret |=dev.da_init_device(self.master_id)
        ret |= dev.da_set_trigger_interval_l1(self.da_id, trig_interval)
        return ret
        
    '''
    设置触发次数触发间隔:
    如果是通过主板触发，为多板触发，则设置主板触发次数和间隔，模式默认为0，无需设置;
    如果是单板触发，则需要设置单板模式；
    mode为1为单板触发，mode为0为多板触发即需要主板触发。
    '''
    def da_trigg(self, trig_count):
        ret = 0
        ret |= dev.da_set_trigger_count_l1(self.master_id, trig_count)



    
    # def da_trigg(self, trig_count, trig_interval, mode=1):
    #     ret |= dev.da_set_trigger_interval_l1(self.da_id, trig_interval)
    #     if ret != 0:
    #        print(f'ERROR:da board:[{self.da_id}] set trig failure, ret:[{ret}].')
    #        sys.exit(0)
    #     else:
    #         print(f'da board:[{self.da_id}] set trig success .')
    #     return ret 
     

    '''DA板使能触发'''
    def da_master_trigg_enable(self):
        ret = 0
        # if self.master_id is not None:
        ret |= dev.da_trigger_enable(self.master_id)
        # else:
        #     ret |= dev.da_trigger_enable(self.da_id)
        return ret

    '''DA板断开连接'''
    def da_disconnect_device(self):
        dev.da_disconnect_device(self.da_id)


    '''设置通道默认偏置'''
    # def set_da_ch_offset(self, da_channel, data_offset):
    #     ret = 0
    #     ret |= dev.da_set_data_offset(self.da_id, da_channel, data_offset)
    #     #dev self.da_set_channel_default_voltage()
    #     return ret

    def da_set_channel_default_voltage(self, da_channel, channel_default_voltage):
        ret = 0
        ret |= dev.da_set_channel_default_voltage(self.da_id, da_channel, channel_default_voltage)
        return ret

    '''
    读取波形对象wave_obj的采样数据，
    根据参数component取实部或虚部
    (’i’为实部， ‘q’为虚部)
    mode=0,chufashuchu
    mode=1,单通道连续输出
    多通道连续输出模式 mode为1: 重复输出波形缓存区中的波形序列；在多通道连续输出模式下需要da_trigger_enable DA触发使能；
    单通道连续输出模式mode为2：重复输出波形缓存区中的波形序列；
 	触发输出模式 mode为0: 在收到触发信号后输出波形；

    '''
    def da_write_wave(self, da_channel, wave_obj, component, channel_output_delay, padding=0,  mode=0):
        # print(channel_output_delay)
        ret = 0 
        ret |= dev.da_stop_output_wave(self.da_id, da_channel)
        ret |= dev.da_write_wave(wave_obj, self.da_id, da_channel, component,  mode=mode, padding=padding, channel_output_delay=channel_output_delay/1e9)
        ret |= dev.da_start_output_wave(self.da_id, da_channel)
        return ret
    
    def da_stop_output_wave(self,da_channel):
        dev.da_stop_output_wave(self.da_id, da_channel)


    def da_set_trig_delay(self, ad_trig_delay):
        dev.da_set_trigger_delay(self.da_id, ad_trig_delay)


##############################################################
##AD设置
##############################################################

class AD():
    def __init__(self, ad_id, ad_mac, host_mac):
        self.ad_id = ad_id
        self.ad_mac = ad_mac
        self.host_mac = host_mac
        
    '''AD连接初始化'''
    def ad_connect_device(self):
        ret = 0
        print(self.host_mac,self.ad_mac)
        ret |= dev.ad_connect_device(self.ad_id, self.host_mac, self.ad_mac)
        # ret |= dev.ad_init_device(self.ad_id)
        # if ret != 0:
        #     print(f'Error:AD:[{self.ad_id}] connect fauilure, ret:[{ret}] .')
        #     sys.exit(0)
        # else:
        #     print(f'AD:[{self.ad_id}] connect success .')
        return ret

    def ad_init_device(self):
        ret = 0
        # ret |= dev.ad_connect_device(self.ad_id, self.host_mac, self.ad_mac)
        ret |= dev.ad_init_device(self.ad_id)
        # if ret != 0:
        #     print(f'Error:AD:[{self.ad_id}] connect fauilure, ret:[{ret}] .')
        #     sys.exit(0)
        # else:
        #     print(f'AD:[{self.ad_id}] connect success .')
        return ret
    '''
    设置AD参数
    1、设置AD板硬件解模模式
    2、设置AD板采样深度
    3、设置AD板触发次数
    '''
    def set_ad(self, depth, trig_count, mode=0):
        ret = 0
        ret |= dev.ad_set_mode(self.ad_id, mode)
        ret |= dev.ad_set_sample_depth(self.ad_id, depth)
        print(trig_count)
        ret |= dev.ad_set_trigger_count(self.ad_id, trig_count)
        if ret != 0:
            print(f'ERROR:ad board:[{self.ad_id}] set param failure, ret:[{ret}].')
            sys.exit(0)
        else:
            print(f'da board:[{self.ad_id}] set param success .')
        return ret


    '''设置n个频点窗口参数，包括解模窗口长度、解模起始位置、解模频率、提交'''
    def set_ad_freq(self, f_list, depth, window_start=0):
        k = 0
        ret = 0
        ret |= dev.ad_set_window_width(self.ad_id, depth)
        ret |= dev.ad_set_window_start(self.ad_id, window_start)
        for freq in f_list:
            # ret |= dev.ad_set_window_width(self.ad_id, depth)
            # ret |= dev.ad_set_window_start(self.ad_id, window_start)
            ret |= dev.ad_set_demod_freq(self.ad_id, freq)
            ret |= dev.ad_commit_demod_set(self.ad_id, k)
            k += 1
        return ret

    def ad_data_clear(self):
        ret = 0
        ret |= dev.ad_clear_wincap_data(self.ad_id)
        return ret

    ''''AD板触发使能输出'''
    def ad_enable_trig(self):
        ret = 0
        ret |=  dev.ad_enable_adc(self.ad_id)
        return ret
    
    def ad_getdata(self):
        return dev.ad_receive_data(self.ad_id)





# from configparser import ConfigParser

# def getconfig(key):
#     config = ConfigParser()
#     path = r'D:\QuLab\qulab\conf\config.ini'
#     config.read(path)
#     return config.items(key)



# def awg_connect(key):
#     c = getconfig(key)
#     m = getconfig('master_AWG')
#     DA(c[0][1], c[1][1], m[0][1], m[1][1])


# def ad_connect(key):
#     c = getconfig(key)
#     AD(c[0][1], c[1][1], c[2][1])


# def ad_connect(key):
#     c = getconfig(key)
#     AD(c[0][1], c[1][1], c[2][1])

# def dc_connect(key,dc_channel):
#     c = getconfig(key)
#     DC(c[0][1], c[1][1], 5000, dc_channel)



# def allset(measure,trig_count,depth):
#     measure.lo_awg.da_trigg_count(trig_count)
#     measure.lo_awg.set_ad(depth, trig_count, mode=0)
#     measure.ad.set_ad_freq(f_list, depth, window_start=0)


##############################################################
#比特标定方法
##############################################################

# 配置参数

depth = 800
trig_count = 1000
trig_interval = 260e-6

f_list = [80e6]
amp = 32767.5  # 避免越界，比32768略小
freq_str = ['{freq:6.2f} MHz'.format(freq=freq / 1e6) for freq in f_list]
ad_trig_delay = 90450#ns

dcch1 = DC('DC1','10.0.200.110',5000,1)
dcch2 = DC('DC1','10.0.200.110',5000,2)
dcch3 = DC('DC1','10.0.200.110',5000,3)
dcch4 = DC('DC1','10.0.200.110',5000,4)


da1 = DA('DACUSTCF215', '10.0.200.100','DACUSTCF215', '10.0.200.100')
da2 = DA('DACUSTCF215', '10.0.200.101','DACUSTCF215', '10.0.200.100')
da3 = DA('DACUSTCF215', '10.0.200.102','DACUSTCF215', '10.0.200.100')
da4 = DA('DACUSTCF215', '10.0.200.103','DACUSTCF215', '10.0.200.100')

lo_ch = [1, 2]

ad = AD('ADCUSTCC013', '00-00-00-00-00-52', "54-E1-AD-26-0F-FA")
#dev.da_connect_device('QF10K4N0050', '10.0.200.68', [0,0,0,0], [0,0,0,0])

def getExpArray(f_list, depth, weight=None, sampleRate=1e9):
    e = []
    t = np.arange(0, depth, 1) / sampleRate
    if weight is None:
        weight = np.ones(depth)
    for f in f_list:
        e.append(weight * np.exp(-1j * 2 * np.pi * f * t))
    return np.asarray(e).T

###################################################################################
#采集数据
####################################################################################
def demodulation(qubit, measure, fft=False, avg=False, hilbert=False, offset=True, is2ch=True):
    #待调整
    e = getExpArray(measure.f_list, measure.depth, weight=None, sampleRate=1e9)
    n = e.shape[0]
    measure.ad.ad_data_clear()
    ####注意顺序！！！！！！
    measure.ad.ad_enable_trig()
    measure.awg[qubit.inst['lo_awg']].da_master_trigg_enable()
    res = measure.ad.ad_getdata()
    # print(np.shape(res))
    A, B = res[1]/1e5, res[2]/1e5
    if hilbert:
        Analysis_cos = signal.hilbert(A,axis=1)
        Analysis_sin = signal.hilbert(B,axis=1)
        # theta = np.angle(Analysis_cos) - np.angle(Analysis_sin)
        # Analysis_sin *= np.exp(1j*(theta))
        # A, B = (np.real(Analysis_cos) + np.real(Analysis_sin)), (np.imag(Analysis_cos) + np.imag(Analysis_sin)) 
        if is2ch:
            A, B = (np.real(Analysis_cos) - np.imag(Analysis_sin)), (np.imag(Analysis_cos) + np.real(Analysis_sin))
        else: 
            A, B = np.real(Analysis_cos), np.imag(Analysis_cos)
    if fft:
        A = (A[:, :n]).dot(e)
        B = (B[:, :n]).dot(e)
        print(1)
    if avg:
        return A.mean(axis=0), B.mean(axis=0)
    else:
        return A, B

def getIQ(qubit, measure, fft=False, avg=False, hilbert=False, offset=True, is2ch=False):
    return demodulation(qubit, measure, fft, avg, hilbert, offset, is2ch)

def getTraces(qubit, measure, fft=False, avg=True, hilbert=False, offset=True, is2ch=False):
    return demodulation(qubit, measure, fft, avg, hilbert, offset, is2ch)


def resn(f_list):
    f_list = np.array(f_list)
    f_lo = f_list.max() - 80e6
    delta =  -(f_lo - f_list)
    n = len(f_list)
    return f_lo, delta, n


############################################################################
##S21
############################################################################
async def S21(qubit,measure,freq=None):
    #波形write
    await measure.psg['psg_lo'].setValue('Output','ON')
    if freq == None:
        f_lo, delta, n = resn(np.array(qubit.f_lo))
        freq = np.linspace(-2.5,2.5,126)*1e6 + f_lo
        cw.modulation_read(qubit, measure, delta, readlen=measure.readlen)

    else:
        delta = measure.delta
        freq = np.linspace(-2.5,2.5,126)*1e6 + freq   
        cw.modulation_read(qubit, measure, delta, readlen=measure.readlen)
        
    measure.n = len(delta)
    x = np.zeros(8)
    x[:len(delta)] = delta
    delta = x

    print(measure.depth)
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
        yield i+delta, s

###########################################################################
##again
###########################################################################
async def again(qubit,measure,freq=None):
    #f_lo, delta, n = qubit.f_lo, qubit.delta, len(qubit.delta)
    #freq = np.linspace(-2.5,2.5,126)*1e6+f_lo
    await measure.psg['psg_ex'].setValue('Output','OFF')
    job = Job(S21, (qubit,measure,freq),auto_save=True,max=126,tags=[qubit.q_name])
    f_s21, s_s21 = await job.done()
    index = np.abs(s_s21).argmin(axis=0)
    f_res = np.array([f_s21[:,i][j] for i, j in enumerate(index)])
    base = np.array([s_s21[:,i][j] for i, j in enumerate(index)])
    f_lo, delta, n = resn(f_res[:measure.n])
    print(measure.n)
    await measure.psg['psg_lo'].setValue('Frequency',f_lo)
    if n != 1:
        cw.modulation_read(qubit,measure,measure.delta)
        base = 0
        for i in range(25):
            res = getIQ(qubit, measure)
            Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            base += Am + 1j*Bm
        base /= 25
    measure.base, measure.delta, measure.f_lo = base, delta, np.array([f_lo])
    return f_lo, delta, f_res, base, f_s21, s_s21

############################################################################
#
##S21vsFlux
#
############################################################################
async def S21vsFlux(qubit, measure,current,f_read=None):
    for i in current:
        measure.dc[qubit.q_name].dc(i)
        job = Job(S21, (qubit,measure,f_read),auto_save=True,max=126,tags=[qubit.q_name])
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21
    measure.dc[qubit.q_name].dc(0)

############################################################################
##S21vsPower
############################################################################
async def S21vsPower(qubit,measure,att,com='com3',modulation=False):
    for i in att:
        await measure.att['attlo'].set_att(i)
        job = Job(S21, (qubit,measure,False),auto_save=False, no_bar=True)
        f_s21, s_s21 = await job.done()
        n = np.shape(s_s21)[1]
        yield [i]*n, f_s21, s_s21

############################################################################
##Singlespec
############################################################################
async def singlespec(qubit,measure,ex_freq,readponit=True,freq=None):
    if readponit:
        f_lo, delta, f_res, base, f_s21, s_s21 = await again(qubit,measure,freq)
    else:
        n, base = 8, measure.base
    await measure.psg['psg_ex'].setValue('Output','ON')
    cw.modulation_ex(qubit,measure)
    for i in ex_freq:
        await measure.psg['psg_ex'].setValue('Frequency',i)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s-base
    await measure.psg['psg_ex'].setValue('Output','OFF')

################################################################################
# Spec2d
################################################################################
async def spec2d(qubit,measure,ex_freq,current):
    for i in current:
        measure.dc[qubit.q_name].dc(i)
        job = Job(singlespec, (qubit,measure,ex_freq,True),auto_save=True,max=len(ex_freq),tags=[qubit.q_name])
        f_ss, s_ss = await job.done()
        #n = np.shape(s_ss)[1]
        yield [i]*8, f_ss, s_ss

###############################################################################
# Spec2d_zpulse
################################################################################
async def spec2d_zpulse(qubit,measure,ex_freq,current):
    await again(qubit,measure)
    for i in current:
        cw.z_pulse(qubit,measure,width=30000,amp=i)
        job = Job(singlespec, (qubit,measure,ex_freq,False),auto_save=True,max=len(ex_freq),tags=[qubit.q_name])
        f_ss, s_ss = await job.done()
        #n = np.shape(s_ss)[1]
        yield [i]*8, f_ss, s_ss

################################################################################
# Rabi
################################################################################
async def Rabi(qubit,measure,t_rabi):
    delta_ex = qubit.delta_ex[0]
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex[0]-delta_ex
    print('freq=%s'%freq)
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in t_rabi:
        pulse = cw.rabiWave(qubit, measure, envelopename='cospulse',nwave=1,\
            during=i/1e9,Delta_lo=delta_ex,amp=25e3,phase=0,phaseDiff=0,DRAGScaling=None)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s

################################################################################
# RabiPower
################################################################################
async def RabiPower(qubit,measure,power):
    delta_ex = qubit.delta_ex[0]
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    freq = qubit.f_ex[0]-delta_ex
    print('freq=%s'%freq)
    await measure.psg['psg_ex'].setValue('Frequency',freq)
    for i in power:
        pulse = cw.rabiWave(qubit, measure, envelopename='cospulse',nwave=1,\
            during=qubit.pi_len/1e9,Delta_lo=delta_ex,amp=i,phase=0,phaseDiff=0,DRAGScaling=None)
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s

################################################################################
# T1
################################################################################
async def T1(qubit, measure, t_t1):
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in t_t1:
        pulse = cw.rabiWave(qubit, measure, envelopename='cospulse',during=qubit.pi_len/1e9,channel_output_delay=89.9e3,shift=i/1e9,Delta_lo=qubit.delta_ex[0])
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s 

################################################################################
# Ramsey
################################################################################

async def Ramsey(qubit, measure, t_t1):
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in t_t1:
        pulse = cw.ramseyWave(qubit, measure, delay=i/1e9, halfpi=qubit.pi_len/1e9,envelopename='cospulse')
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s 

################################################################################
# Spin echo
################################################################################

async def SpinEcho(qubit, measure, t_t1, seqtype):
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in t_t1:
        pulse = cw.coherenceWave(qubit, measure, envelopename='cospulse',t_run=i/1e9,during=qubit.pi_len/1e9,n_wave=1,seqtype=seqtype,detune=3e6,shift=0, channel_output_delay=89.9e3,Delta_lo=qubit.delta_ex[0])
        res = getIQ(qubit, measure)
        Am, Bm = res[0].mean(axis=0),res[1].mean(axis=0)
        s = Am + 1j*Bm
        yield [i]*8, s 

###################################################################################
# threshHold
########################################################################################
async def threshHold(qubit,measure,amp,modulation=True):
    if modulation:
        pilen = qubit.pi_len
        pulse = cw.rabiWave(qubit,measure,envelopename='cospulse',during=pilen/1e9,amp=amp,DRAGScaling=None)
    for j, i in enumerate(['OFF','ON']):
        await measure.psg['psg_ex'].setValue('Output',i)
        time.sleep(2)
        ch_A, ch_B = getIQ(qubit,measure,fft=False,avg=False,hilbert=False,is2ch=False)
        s = ch_A + 1j*ch_B
        # theta0 = np.angle(Am) - np.angle(Bm)
        # Bm *= np.exp(1j*theta0)
        # s = Am + 1j*Bm
        yield j, s





###################################################################################
# readOp
########################################################################################
async def readOptest(qubit, measure):
    for j, i in enumerate(['OFF','ON']):
        await measure.psg['psg_ex'].setValue('Output',i)
        time.sleep(2)
        job = Job(S21, (qubit,measure,False),auto_save=False, max=126)
        f_s21, s_s21 = await job.done()
        yield [j]*8, f_s21, s_s21

###################################################################################
# readWavelen
########################################################################################

async def readWavelen(qubit,measure):
    pilen, t = qubit.pi_len, np.linspace(900,2000,21,dtype=np.int64)
    for k in t:
        cw.modulation_read(qubit, measure, delta, readlen=k)
        state = []
        for j, i in enumerate(['OFF','ON']):
            await measure.psg['psg_lo'].setValue('Output',i)
            ch_A, ch_B = getIQ()
            Am, Bm = ch_A[:,0],ch_B[:,0]
            # theta0 = np.angle(Am) - np.angle(Bm)
            # Bm *= np.exp(1j*theta0)
            s = Am + 1j*Bm
            state.append((j,np.mean(s),np.std(s)))
        yield k, state


################################################################################
# Tomo
################################################################################

async def tomo(qubit,measure,t_rabi,which,amp=25e3,DRAGScaling=None):
    pilen = qubit.pi_len
    await measure.psg['psg_lo'].setValue('Output','ON')
    await measure.psg['psg_ex'].setValue('Output','ON')
    for i in t_rabi:
        pop = []
        for axis in ['Ynhalf','Xhalf','Z']:
            cw.tomoTest(qubit,measure,i,halfpi=pilen,axis=axis,amp=amp,DRAGScaling=None)  
            ch_A, ch_B = getIQ(qubit,measure,fft=False,avg=False,hilbert=False,is2ch=False)
            Am, Bm = ch_A[:,0],ch_B[:,0]
            ss = Am + 1j*Bm
            d = list(zip(np.real(ss),np.imag(ss)))
            y = measure.predict[qubit.q_name](d)
            pop.append(list(y).count(which)/len(y))
        yield [i], pop