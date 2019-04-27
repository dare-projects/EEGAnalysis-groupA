import pyedflib
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.adfvalues import mackinnonp

################################################ READ FILE AND STORE VALUES ###############################################

f = pyedflib.EdfReader("chb05_12.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
     sigbufs[i, :] = f.readSignal(i)

#%%
########################################## AUGMENTED DICKEY-FULLER UNIT ROOT TEST #########################################
window_size = 256 * 6

electrod_signal = sigbufs[0,:]
scores = []
accepted_values = 0
i = 0

while (i <= len(electrod_signal)-1):
    fract_signal = electrod_signal[i:(i+window_size)]

    # make sure we are working with an array, convert if necessary
    fract_signal = np.asarray(fract_signal)
    
    # Get the dimension of the array
    nobs = fract_signal.shape[0]
    
    # We use 1 as maximum lag in our calculations
    maxlag = 1
    
    # Calculate the discrete difference
    tsdiff = np.diff(fract_signal)
    
    # Create a 2d array of lags, trim invalid observations on both sides
    tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
    
    # Get dimension of the array
    nobs = tsdall.shape[0] 
        
    # replace 0 xdiff with level of x
    tsdall[:, 0] = fract_signal[-nobs - 1:-1]  
    tsdshort = tsdiff[-nobs:]
    
    # Calculate the linear regression using an ordinary least squares model    
    results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
    adfstat = results.tvalues[0]
    
    # Get approx p-value from a precomputed table (from stattools)
    pvalue = mackinnonp(adfstat, 'c', N=1)

    scores.append(pvalue)
    
    # Increase counter to change window
    i += window_size
    
    # Check value
    if(pvalue < 0.05):
        accepted_values += 1
    
# Calculate percentage of accepted values over all values
percentage = accepted_values/((len(electrod_signal)-1)/window_size)
# Change "scores" from list to array of values
scores = scores[1:]
# Print out results
print('Accepted values percentage: ', percentage)
print('scores mean: ', np.mean(scores))
print('scores variance: ',np.var(scores))