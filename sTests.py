import numpy as np
from sklearn import preprocessing, linear_model, ensemble
# import GPy as gp
import sgenRandomized as sgr

def genRegInput(inSeries, maxLag):
    # Create design matrix for regression from time series data

    # Inputs:
    # inSeries = array where each column is a time series that may drive the 
    # target time series
    # tgtSeries = target time series
    # maxLag = maximum lag considered

    # Outputs:
    # X = design matrix

    numOutData = len(inSeries) - maxLag
    numInSeries = len(inSeries[0])
    X = np.empty([numOutData, numInSeries*maxLag])
    for i in range(numOutData):
        for j in range(numInSeries):
            for l in range(maxLag):
                X[i, j*maxLag+l] = inSeries[maxLag+i-l-1, j]
    return X

def fitModel(inSeries, outSeries, maxLag, method, threshold):
    # Use the specified method to fit a model to each variable represented in 
    # outSeries, using the data in inSeries with a maximum lag of maxLag.

    # Inputs:
    # inSeries = array where each column is a time series
    # outSeries = array where each column is a time series assumed to be driven by 
    # some of the series in inSeries.
    # maxLag = maximum lag considered
    # method = string indicating the fitting method to use:
    #     'lassocv' = LASSO with cross-validation
    # threshold = threshold to use for identifying the relevancy of a time series
    
    # Outputs:
    # depFitted = array whose ij-th element is a tuple (k, l) where k is the index 
    # of a column of inSeries that was identified as relevant to the i-th column 
    # of outSeries, and l is the relevant lag

    depFitted = []
    coefList = []
    numInSeries = len(inSeries[0])
    if len(outSeries.shape)==1:
        numOutSeries = 1
    else:
        numOutSeries = len(outSeries[0])
    for series in range(numOutSeries):
        # Format the data for the methods
        if numOutSeries==1:
            y = outSeries[maxLag:]
        else:
            y = outSeries[maxLag:,series]
        X = genRegInput(inSeries, maxLag)
        X_scaled = preprocessing.scale(X)
        coefs = np.empty([numInSeries, maxLag])
        # Fit a model with one of the methods
        if method=='lassocv':
            lasso = linear_model.LassoCV(max_iter=10000)
            lasso.fit(X_scaled, y)
            for i in range(numInSeries):
                coefs[i,:] = lasso.coef_[i*maxLag:(i+1)*maxLag]
            coefList.append(coefs)
        # Get the maximal coefficient, and the corresponding lag, for each input 
        # series
        maxCoefs = np.zeros(numInSeries)
        maxLags = np.zeros(numInSeries)
        for j in range(numInSeries):
            maxCoefs[j] = np.amax(np.absolute(coefs[j,:]))
            maxLags[j] = np.argsort(np.absolute(coefs[j,:]))[-1] + 1
        # Compute the "importance score" for each input series
        scores = np.zeros(numInSeries)
        mc = np.amax(maxCoefs)
        for j in range(numInSeries):
            scores[j] = maxCoefs[j]/mc
        # Use the scores to identify relevant input series
        dfit = []
        for j in range(numInSeries):
            if scores[j] >= threshold:
                dfit.append((j, int(maxLags[j])))
        depFitted.append(dfit)
    return (depFitted, coefList)
