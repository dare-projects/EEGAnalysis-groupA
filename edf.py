import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from pyentrp import entropy as ent

f_12 = pyedflib.EdfReader("InputFiles/chb05_12.edf")
n_12 = f_12.signals_in_file
signal_labels = f_12.getSignalLabels()
sigbufs_12 = np.zeros((n_12, f_12.getNSamples()[0]))
for i in np.arange(n_12):
     sigbufs_12[i, :] = f_12.readSignal(i)
     
f_13 = pyedflib.EdfReader("InputFiles/chb05_13s.edf")
n_13 = f_13.signals_in_file
sigbufs_13 = np.zeros((n_13, f_13.getNSamples()[0]))
for i in np.arange(n_13):
     sigbufs_13[i, :] = f_13.readSignal(i)
     
f_14 = pyedflib.EdfReader("InputFiles/chb05_14.edf")
n_14 = f_14.signals_in_file
sigbufs_14 = np.zeros((n_14, f_14.getNSamples()[0]))
for i in np.arange(n_14):
     sigbufs_14[i, :] = f_14.readSignal(i)

#%%
# FREQUENCY AT WHICH THE SAMPLES WERE TAKEN
sample_frequency = 256
# WINDOW AND PLOT VALUES
window_size = 256 * 6
plotscale = window_size  # used to determine the plot scale

# PLOT A CERTAIN ELECTROD SIGNAL (BEFORE DATA FILTERING)
electrod_signal = sigbufs_12[0,:]  
fig = plt.figure(figsize=(20,10))
plt.plot(electrod_signal[:plotscale])

################################ CLEAN ELECTRODS SIGNALS ##################################
#%%#
clean_signals_12 = []

for i in range(0,22):
    
    electrod_signal = sigbufs_12[i,:]
    nyq = 0.5 * sample_frequency
    radiant_const = 1/(2*np.pi)     # used to round the wave
    
    # LOWPASS FILTRING
    order = 4
    normal_cutoff = (125/nyq) * radiant_const
    a,b = signal.butter(order, normal_cutoff, btype='low', analog=False)
    lowFiltered = signal.filtfilt(a,b,electrod_signal)
    '''
    fig = plt.figure(figsize=(20,10))
    plt.title(signal_labels[i])
    plt.plot(lowFiltered[:plotscale])
    '''
    # BANDSTOP FILTRING
    order = 4
    start_normal_cutoff = (58/nyq) * radiant_const
    stop_normal_cutoff = (62/nyq) * radiant_const
    a,b = signal.butter(order, [start_normal_cutoff,stop_normal_cutoff], btype='bandstop', analog=False)
    bandstopFiltered = signal.filtfilt(a,b,lowFiltered)

    # APPEND DATA LIST
    clean_signals_12.append(bandstopFiltered)
    # PLOTTING
    '''
    fig = plt.figure(figsize=(20,10))
    plt.title(signal_labels[i])
    plt.plot(bandstopFiltered[:plotscale])
    '''
#%%
clean_signals_13 = []
for i in range(0,22):
    
    electrod_signal = sigbufs_13[i,:]
    
    # LOWPASS FILTRING
    lowFiltered = signal.filtfilt(a,b,electrod_signal)
    
    # BANDSTOP FILTRING
    bandstopFiltered = signal.filtfilt(a,b,lowFiltered)

    # APPEND DATA LIST
    clean_signals_13.append(bandstopFiltered)
    # PLOTTING
    '''
    fig = plt.figure(figsize=(20,10))
    plt.title(signal_labels[i])
    plt.plot(bandstopFiltered[:plotscale])
    '''
    
clean_signals_14 = []
for i in range(0,22):
    
    electrod_signal = sigbufs_14[i,:]
    
    # LOWPASS FILTRING
    lowFiltered = signal.filtfilt(a,b,electrod_signal)
    
    # BANDSTOP FILTRING
    bandstopFiltered = signal.filtfilt(a,b,lowFiltered)

    # APPEND DATA LIST
    clean_signals_14.append(bandstopFiltered)
    # PLOTTING
    '''
    fig = plt.figure(figsize=(20,10))
    plt.title(signal_labels[i])
    plt.plot(bandstopFiltered[:plotscale])
    '''

############################### HURST ##################################################
def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk, vice verse. """
    
    X = np.array(X)
    N = X.size
    T = np.arange(1, N + 1)
    Y = np.cumsum(X)
    Ave_T = Y / T

    S_T = np.zeros(N)
    R_T = np.zeros(N)

    for i in range(N):
        S_T[i] = np.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = np.ptp(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = np.log(R_S)[1:]
    n = np.log(T)[1:]
    A = np.column_stack((n, np.ones(n.size)))
    [m, c] = np.linalg.lstsq(A, R_S)[0]
    H = m
    return H

################################ POWER ##################################################
def bin_power(X, Band, Fs):
    """Compute power in each frequency bin specified by Band from FFT result of
    X. By default, X is a real signal.

    Note
    -----
    A real signal can be synthesized, thus not real.

    Parameters
    -----------

    Band
        list

        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
        You can also use range() function of Python to generate equal bins and
        pass the generated list to this function.

        Each element of Band is a physical frequency and shall not exceed the
        Nyquist frequency, i.e., half of sampling frequency.

     X
        list

        a 1-D real time series.

    Fs
        integer

        the sampling rate in physical frequency

    Returns
    -------

    Power
        list

        spectral power in each frequency bin.

    Power_ratio
        list

        spectral power in each frequency bin normalized by total power in ALL
        frequency bins.

    """

    C = np.fft.fft(X)
    C = abs(C)
    Power = np.zeros(len(Band) - 1)
    for Freq_Index in range(0, len(Band) - 1):
        Freq = float(Band[Freq_Index])
        Next_Freq = float(Band[Freq_Index + 1])
        Power[Freq_Index] = sum(
            C[int(np.floor(Freq / Fs * len(X))): 
                int(np.floor(Next_Freq / Fs * len(X)))]
        )
    Power_Ratio = Power / sum(Power)
    return Power, Power_Ratio
########################### CALCULATE SIGNALS WINDOWS VALUES ############################
#%%#
first_electrode_signal = sigbufs_12[0,:]
window_signal = []
n_windows = int(len(first_electrode_signal)/window_size)

frequencies = [0.5,4,7,12,30,100]


# first file dataframe and csv
Measure_12_df = pd.DataFrame(columns=['Electrode','Window','Mean','Std','Max','Min','Power','Hurst','Entropy'])
for x in range(0,n_12-1):
    for y in range(0,n_windows):
        window_signal = clean_signals_12[x][y*window_size:(y+1)*window_size]
        std_window = np.std(window_signal)
        sample_entropy = ent.sample_entropy(window_signal, 4, 0.2 * std_window)
        Measure_12_df = Measure_12_df.append({'Electrode': signal_labels[x], 'Window': y, 'Mean':  np.mean(window_signal),
                              'Std' : np.std(window_signal),'Max' : np.max(window_signal),'Min' : np.min(window_signal),
                              'Power' : max(bin_power(window_signal, frequencies, 256)[0]),
                              'Hurst' : hurst(window_signal),
                              'Entropy' : sample_entropy},
                                ignore_index=True)
        print('                           Measure_12')
        print(Measure_12_df.iloc[x*n_windows+y])
Measure_12_df.to_csv('OutputFiles/Measure_12_df.csv',index=False)

# second file dataframe and csv
Measure_13_df = pd.DataFrame(columns=['Electrode','Window','Mean','Std','Max','Min','Power','Hurst','Entropy'])
for x in range(0,n_13-1):
    for y in range(0,n_windows):
        window_signal = clean_signals_13[x][y*window_size:(y+1)*window_size]
        std_window = np.std(window_signal)
        sample_entropy = ent.sample_entropy(window_signal, 4, 0.2 * std_window)
        Measure_13_df = Measure_13_df.append({'Electrode': signal_labels[x], 'Window': y, 'Mean':  np.mean(window_signal),
                              'Std' : np.std(window_signal),'Max' : np.max(window_signal),'Min' : np.min(window_signal),
                              'Power' : max(bin_power(window_signal, frequencies, 256)[0]),
                              'Hurst' : hurst(window_signal),
                              'Entropy' : sample_entropy},
                                ignore_index=True)
        print('                           Measure_13')
        print(Measure_13_df.iloc[x*n_windows+y])
Measure_13_df.to_csv('OutputFiles/Measure_13_df.csv',index=False)

# third file dataframe and csv 
Measure_14_df = pd.DataFrame(columns=['Electrode','Window','Mean','Std','Max','Min','Power','Hurst','Entropy'])
for x in range(0,n_14-1):
    for y in range(0,n_windows):
        window_signal = clean_signals_14[x][y*window_size:(y+1)*window_size]
        std_window = np.std(window_signal)
        sample_entropy = ent.sample_entropy(window_signal, 4, 0.2 * std_window)
        Measure_14_df = Measure_14_df.append({'Electrode': signal_labels[x], 'Window': y, 'Mean':  np.mean(window_signal),
                              'Std' : np.std(window_signal),'Max' : np.max(window_signal),'Min' : np.min(window_signal),
                              'Power' : max(bin_power(window_signal, frequencies, 256)[0]),
                              'Hurst' : hurst(window_signal),
                              'Entropy' : sample_entropy},
                                ignore_index=True)
        print('                           Measure_14')
        print(Measure_14_df.iloc[x*n_windows+y])
Measure_14_df.to_csv('OutputFiles/Measure_14_df.csv',index=False)

