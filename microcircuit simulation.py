from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
'''statistics macro'''


def Mean_firing_Rate(state_monitor, population,running_time,):
    spike_count = len(state_monitor.t / ms)
    return spike_count / (population*running_time)


def spike_sorting(state_monitor):
    sorted_spike = [[] for _ in range(10000)]

    for number, spiking_times in enumerate(state_monitor.t):
        if((spiking_times/ms)<1000.0):
            continue
        sorted_spike[state_monitor.i[number]].extend([spiking_times / ms])
    return sorted_spike


def synchrony(state_monitor, population, running_time):
    N_p = 50  # partition size
    time_bin = 5  # time bin: unit ms

    Cov = np.zeros((N_p, N_p))  # covariance matrix
    Corr = np.zeros((N_p, N_p))
    Var = np.zeros(N_p)
    E = np.zeros(N_p)
    Spiking_number = np.zeros((N_p, (int)((running_time-1) / time_bin) + 1)) # -1 is caused by cutting

    sorted_spike = spike_sorting(state_monitor)

    for neuron_number, neuron_spike_train in enumerate(sorted_spike):
        for neuron_spiking_time in neuron_spike_train:
            if(neuron_spiking_time < 1000.0): # cutting trainsient effect
                continue
            Spiking_number[(int)(N_p * (neuron_number - 1) / population)][(int)((neuron_spiking_time-1) / time_bin)] += 1.0

    E = Spiking_number.sum(axis=1) / ((int)(running_time / time_bin) + 1)
    Var = (np.multiply(Spiking_number,Spiking_number).sum(axis=1)) / ((int)(running_time / time_bin) + 1) - np.multiply(E,E)

    for i in range(N_p):
        for j in range(N_p):
            E_x = np.multiply(Spiking_number[i][:], Spiking_number[j][:]) / ((int)(running_time / time_bin) + 1)
            Cov[i][j] = E_x.sum(axis=0) - E[i] * E[j]

    for i in range(N_p):
        for j in range(N_p):
            if abs(Var[i] * Var[j]) < 1e-10:
                Corr[i][j]=0
                continue
            Corr[i][j] = Cov[i][j] / sqrt(Var[i] * Var[j])

    return Corr.sum(axis=1).sum(axis=0)/(N_p * N_p)

def irregularity(state_monitor, population):
    mean = [0 for _ in range(10000)]
    s_o_s = [0 for _ in range(10000)]
    Ir = [0 for _ in range(10000)]
    Irrg = 0

    sorted_spike = spike_sorting(state_monitor)

    for neuron_number, neuron_spike in enumerate(sorted_spike):
        ISI = []
        for index in range(len(neuron_spike) - 1):
            ISI.append(neuron_spike[index + 1] - neuron_spike[index])

        if (len(ISI) == 0):
            continue
        for index in range(len(ISI)):
            mean[neuron_number] += ISI[index]
            s_o_s[neuron_number] += ISI[index] * ISI[index]

        mean[neuron_number] /= len(ISI)
        s_o_s[neuron_number] /= len(ISI)

        if (mean[neuron_number] == 0):
            continue

        Ir[neuron_number] = (s_o_s[neuron_number] - mean[neuron_number] * mean[neuron_number]) / (
                    mean[neuron_number] * mean[neuron_number])

    for index, element in enumerate(Ir):
        Irrg += element

    return Irrg / population

    summation = 0
    for index in range(cnt):
        summation += Ir[index]

    return summation / population




'''
Neuron들 setting

편의상 reset membrane potential을 resting membrane potential로 일단 설정해
두었습니다
'''
defaultclock.dt = 0.05 * ms

g_list = [2.5 * cnt for cnt in range(2)]
freq_list = [5 * cnt for cnt in range(2)]

Mean_rate = np.zeros((2,2))
Irregular = np.zeros((2,2))
Syn = np.zeros((2,2))


for i, g in enumerate(g_list):
    for j, freq in enumerate(freq_list):
        ''' Inhibitory neuron '''

        sigma = 2 * mV

        eqs_1 = """
        dv_1/dt  = (-(v_1+71.2*mV + sigma*sqrt(2*ms/16.7)*xi)/(16.7*ms)) : volt
        dv_1_t/dt = (-48.1*mV-v_1_t)/(1000*ms): volt
        """

        reset_1 = '''
        v_1=-75.4*mV
        v_1_t+=1*mV
        '''

        ''' external input의 decaying time constant를 L5e끼리에서의 decaying time constant와 동일하다고 가정함.'''
        L_5_i = NeuronGroup(1065, eqs_1, threshold='v_1>v_1_t', reset=reset_1, refractory=2 * ms, method='euler')
        L_5_i.v_1 = -75.4 * mV
        L_5_i.v_1_t = -48.1 * mV

        ''' excitatory neuron '''

        eqs_2 = """
        dv_2/dt  = (-(v_2+71.2*mV+sigma*sqrt(2*ms/27.1)*xi)/(27.1*ms)) : volt
        dv_2_t/dt = (-43.1*mV-v_2_t)/(1000*ms): volt
        """
        reset_2 = '''
        v_2=-71.2*mV
        v_2_t+=1*mV

        '''

        L_5_e = NeuronGroup(4850, eqs_2, threshold='v_2>v_2_t', reset=reset_2, refractory=2 * ms, method='euler')
        L_5_e.v_2 = -71.2 * mV
        L_5_e.v_2_t = -43.1 * mV

        C_i_1 = Synapses(L_5_e, L_5_e, on_pre='v_2+=1.3*mV', delay=2.1 * ms)
        C_i_1.connect(p=0.083)

        C_i_2 = Synapses(L_5_e, L_5_i, on_pre='v_1+=1*mV', delay=1 * ms)
        C_i_2.connect(p=0.060)

        C_i_3 = Synapses(L_5_i, L_5_e, on_pre='v_2-=g*1*mV', delay=2.1 * ms)
        C_i_3.connect(p=0.373)

        C_i_4 = Synapses(L_5_i, L_5_i, on_pre='v_1-=1.3*mV', delay=1.7 * ms)
        C_i_4.connect(p=0.316)

        ''' external input '''

        P_1 = PoissonGroup(2000, freq * Hz)

        S_1 = Synapses(P_1, L_5_e, on_pre='v_2+=0.15*mV')
        S_1.connect()

        P_2 = PoissonGroup(2000, freq * Hz)

        S_2 = Synapses(P_2, L_5_i, on_pre='v_1+=0.15*mV')
        S_2.connect()

        print('variable completed')

        '''run simulation'''

        A = SpikeMonitor(L_5_e)
        B = PopulationRateMonitor(L_5_e, name='L5_Excitatory_neurons')

        C = SpikeMonitor(L_5_i)
        D = PopulationRateMonitor(L_5_i, name='L5_inhibitatory_neurons')

        run(5 * second)
        ''' spike statistics '''

        print('g', g, 'freq', freq, 'mean firing rate', Mean_firing_Rate(A,4850,5), 'Irr', irregularity(A, 4850),'Syn',synchrony(A,4850,5000))

        Mean_rate[i][j]= Mean_firing_Rate(A, 5,4850)
        Irregular[i][j]= irregularity(A, 4850)
        Syn[i][j]=synchrony(A,4850,5000)


#########plotting#########

subplot(311)
plt.imshow(Mean_rate, cmap='hot',interpolation='nearest')
plt.colorbar()

subplot(312)
plt.imshow(Irregular, cmap='hot',interpolation='nearest')
plt.colorbar()

subplot(313)
plt.imshow(Syn, cmap='hot',interpolation='nearest')
plt.colorbar()


plt.show()
