import numpy as np
import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
from sklearn import preprocessing

def generateSeries(sd, numDataPts, maxLag, maxSummands, maxFactors, noiseStDev):
    # Generate data for a target time series that's driven by randomly
    # selected stocks. 

    # The way that the target depends on the selected stocks is also
    # generated randomly; its form is that of a sum of products of
    # transformations of lagged variables of the selected stocks.

    # sd = DataFrame containing the stock data
    # numDataPts = desired number of data points
    # maxLag = maximum lag considered
    # maxJ = maximum number of time series to use
    # maxSummands = maximum number of summands to use
    # maxFactors = for each summand, the maximum number of factors
    # noiseStDev = standard deviation of the Gaussian noise with mean zero

    ## Normalize the stock data
    sdNorm = np.zeros([numDataPts+maxLag, len(sd.columns)])
    sdNorm[:,:] = sd.iloc[-(numDataPts+maxLag):,:]
    sdNorm = preprocessing.scale(sdNorm,axis=0,with_mean=True,with_std=True)

    ## Select the number of stocks to use
    J = np.random.choice(range(1, len(sd.columns)+1))
    ## Select the stocks
    selectedStocks = np.sort(np.random.choice(range(len(sd.columns)), size=J, replace=False))
    print "Selected Stocks: ", selectedStocks
    ## Get the stock data and put it into the data array dd; the last column
    ## is reserved for the generated time series
    dd = np.zeros([numDataPts+maxLag, J+1])
    dd[:,:-1] = sdNorm[:,selectedStocks]
    
    ## Select the number of summands to use
    M = np.random.choice(range(1, maxSummands+1))
    ## Generate the data for the target series
    outString = "y(t) = "
    for m in range(M):
        # Select the number of factors to use for the current summand
        N = np.random.choice(range(1, maxFactors+1))
        ss = np.ones(numDataPts)
        # Select a coefficient for the summand
        coef = np.random.choice(range(1,5))
        ss *= coef
        outString += str(coef)
        # Form the rest of the summand
        for n in range(N):
            # Select the stock to use
            j = np.random.choice(range(J))
            # Select the lag
            l = np.random.choice(range(1, maxLag+1))
            # Select the transformation
            f = np.random.choice(range(5))
            if f==0:
                ss *= dd[(maxLag-l):-l,j]
                outString += "x_" + str(selectedStocks[j]) + "(t-" + str(l) + ")"
            elif f==1:
                ss *= dd[(maxLag-l):-l,j]**2
                outString += "x_" + str(selectedStocks[j]) + "(t-" + str(l) + ")^2"
            elif f==2:
                ss *= np.sin(dd[(maxLag-l):-l,j])
                outString += "sin(x_" + str(selectedStocks[j]) + "(t-" + str(l) + "))"
            elif f==3:
                ss *= np.cos(dd[(maxLag-l):-l,j])
                outString += "cos(x_" + str(selectedStocks[j]) + "(t-" + str(l) + "))"
            elif f==4:
                ss *= np.log(np.absolute(dd[(maxLag-l):-l,j]))
                outString += "log(|x_" + str(selectedStocks[j]) + "(t-" + str(l) + ")|)"
        dd[maxLag:,-1] += ss
        if m < M-1:
            outString += " + "
    dd[maxLag:,-1] += noiseStDev * np.random.randn(numDataPts)
    print outString + " + error(t)"
    return np.column_stack((sdNorm, dd[:,-1]))
