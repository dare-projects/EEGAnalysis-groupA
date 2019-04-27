import pandas as pd
import numpy as np
import pyedflib
import matplotlib.pyplot as plt

#####################################################################
################### Read Cluster Data
cluster_kmeans = pd.read_csv('kmeans3.csv', delimiter=",")
cluster_kmeans = cluster_kmeans[['Electrode', 'Window', 'cluster']]

#####################################################################
#################### Split Data frame

len_file = int(len(cluster_kmeans)/3)
df_file_12 = cluster_kmeans.iloc[:len_file]
df_file_13 = cluster_kmeans.iloc[len_file:(len_file*2)]
df_file_14 = cluster_kmeans.iloc[(len_file*2):(len_file*3)]

###############################################################
#################### Read Raw Data
f_12 = pyedflib.EdfReader("InputFiles/chb05_12.edf")
n_12 = f_12.signals_in_file
signal_labels_12 = f_12.getSignalLabels()
sigbufs_12 = np.zeros((n_12, f_12.getNSamples()[0]))
for i in np.arange(n_12):
     sigbufs_12[i, :] = f_12.readSignal(i)
     
f_13 = pyedflib.EdfReader("InputFiles/chb05_13s.edf")
n_13 = f_13.signals_in_file
signal_labels_13 = f_13.getSignalLabels()
sigbufs_13 = np.zeros((n_13, f_13.getNSamples()[0]))
for i in np.arange(n_13):
     sigbufs_13[i, :] = f_13.readSignal(i)
     
f_14 = pyedflib.EdfReader("InputFiles/chb05_14.edf")
n_14 = f_14.signals_in_file
signal_labels_14 = f_14.getSignalLabels()
sigbufs_14 = np.zeros((n_14, f_14.getNSamples()[0]))
for i in np.arange(n_14):
     sigbufs_14[i, :] = f_14.readSignal(i)
     
#####################################################################
################## Plot signal with selected cluster
windows_size = 256*6
files_dict = {}
files_dict[12] = [signal_labels_12, df_file_12, sigbufs_12]
files_dict[13] = [signal_labels_13, df_file_13, sigbufs_13]
files_dict[14] = [signal_labels_14, df_file_14, sigbufs_14]
#%%
# just for one measurement (change 12, 13, 14 to get oter measurements)
for key, value in files_dict.items():
    fig = plt.figure(figsize=(15,15))
    for x in range(n_12-1):
        electrode = value[0][x]
        anomalies = value[1].loc[value[1]['Electrode'] == electrode]
        anomalies = anomalies.loc[np.repeat(anomalies.index.values, windows_size)]
        anomalies['signal'] = value[2][x, :]
        anomalies['id'] = anomalies.index
        cluster0 = anomalies[anomalies['cluster'] == 0]
        cluster1 = anomalies[anomalies['cluster'] == 1]
        cluster2 = anomalies[anomalies['cluster'] == 2]
        #cluster3 = anomalies[anomalies['cluster'] == 3]
        #cluster4 = anomalies[anomalies['cluster'] == 4]
        ax = plt.subplot(n_12-1, 1, x+1)
        #plt.plot(anomalies.index, anomalies['signal'], c=anomalies['cluster'])
        ax.plot(cluster0['id'], cluster0['signal'],linewidth=1, color='r')
        ax.plot(cluster1['id'], cluster1['signal'],linewidth=1, color='b')
        ax.plot(cluster2['id'], cluster2['signal'],linewidth=1, color='g')
        
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #plt.plot(cluster3['id'], cluster3['cluster'], 'y.')
        #plt.plot(cluster4['id'], cluster4['cluster'], 'm.')
    plt.show()
    plt.close("all")
    