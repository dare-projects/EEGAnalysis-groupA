import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


measure12 = pd.read_csv('OutputFiles/Measure_12_df.csv')
measure13 = pd.read_csv('OutputFiles/Measure_13_df.csv')
measure14 = pd.read_csv('OutputFiles/Measure_14_df.csv')

# the last value of entropy is most accurate
measure12['Entropy'] = measure12['Entropy'].map(lambda x: float(x.replace(']', '').strip(' ').split(' ')[-1]))
measure13['Entropy'] = measure13['Entropy'].map(lambda x: float(x.replace(']', '').strip(' ').split(' ')[-1]))
measure14['Entropy'] = measure14['Entropy'].map(lambda x: float(x.replace(']', '').strip(' ').split(' ')[-1]))

#add a column to identify measurement
measure12['Measurement'] = 12
measure13['Measurement'] = 13
measure14['Measurement'] = 14

measures = pd.concat([measure12, measure13, measure14], ignore_index=True)
data = measures.drop(['Electrode', 'Window', 'Measurement'], axis = 1)
corr = data.corr()
print(corr)
'''
             Mean       Std       Max       Min     Power     Hurst   Entropy
Mean     1.000000  0.033990  0.093157  0.036611  0.036565  0.020701 -0.023950
Std      0.033990  1.000000  0.853947 -0.877207  0.974375 -0.019238 -0.615864
Max      0.093157  0.853947  1.000000 -0.706555  0.829550  0.033347 -0.578820
Min      0.036611 -0.877207 -0.706555  1.000000 -0.853358  0.007475  0.568593
Power    0.036565  0.974375  0.829550 -0.853358  1.000000 -0.010718 -0.676153
Hurst    0.020701 -0.019238  0.033347  0.007475 -0.010718  1.000000 -0.238968
Entropy -0.023950 -0.615864 -0.578820  0.568593 -0.676153 -0.238968  1.000000
'''

# calculate with different number of centroids to see the loss plot (ELBOW METHOD)
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters=i).fit(data) for i in n_cluster]
scores = [kmeans[i].score(data) for i in range(len(kmeans))]

fig, ax = plt.subplots()
ax.plot(n_cluster, scores)
plt.show()


# from elbow we select k = 3; eeg have three stages (normal, pre-epileptic, epileptic) which confirms number of clusters

# try with k=3 (best one)
measures['cluster'] = kmeans[2].predict(data)
#measures['principal_feature1'] = data['Power']
#measures['principal_feature2'] = data['Std']
#measures['cluster'].value_counts()

measures.to_csv('kmeans3.csv')