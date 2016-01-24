from __future__ import division
import numpy as np
from sklearn import preprocessing, linear_model, ensemble
import GPy as gp
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
    # depFitted = array whose ij-th element is a tuple (k, l) where k is the 
    # index of a column of inSeries that was identified as relevant to the i-th 
    # column of outSeries, and l is the relevant lag

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
        elif method=='elasticnetcv':
            elasticnet = linear_model.ElasticNetCV(max_iter=10000)
            elasticnet.fit(X_scaled, y)
            for i in range(numInSeries):
                coefs[i,:] = elasticnet.coef_[i*maxLag:(i+1)*maxLag]
            coefList.append(coefs)            
        elif method=='gbm':
            gbm = ensemble.GradientBoostingRegressor()
            gbm.fit(X_scaled, y)
            for i in range(numInSeries):
                coefs[i,:] = gbm.feature_importances_[i*maxLag:(i+1)*maxLag]
            coefList.append(coefs)
        elif method=='gbmhuber':
            gbmhuber = ensemble.GradientBoostingRegressor(loss='huber')
            gbmhuber.fit(X_scaled, y)
            for i in range(numInSeries):
                coefs[i,:] = gbmhuber.feature_importances_[i*maxLag:(i+1)*maxLag]
            coefList.append(coefs)
        elif method=='randomforest':
            randomforest = ensemble.RandomForestRegressor()
            randomforest.fit(X_scaled, y)
            for i in range(numInSeries):
                coefs[i,:] = randomforest.feature_importances_[i*maxLag:(i+1)*maxLag]
            coefList.append(coefs)
        elif method=='gaussianprocessARD':
            ARDkernel = gp.kern.RBF(input_dim=X.shape[1], variance=1., lengthscale=1., ARD=True)
            y = np.array(np.matrix(y).T)
            m = gp.models.GPRegression(X_scaled, y, ARDkernel)
            m.optimize()
            for i in range(numInSeries):
                coefs[i,:] = 1./m.parameters[1]['lengthscale'][i*maxLag:(i+1)*maxLag]
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
    return depFitted

def computeF1Score(dep, depFitted):
    # Compute the F1 score for the fitted dependencies.

    # Here we're only interested in which time series were identified by the 
    # fitted model (i.e. we're ignoring the "important" lags indicated by the 
    # model).

    # Inputs:
    # dep = array whose ij-th element is a tuple (k, l) where k is the index of a 
    # stock that was used to generate the i-th series, and l is the lag 
    # that was used
    # depFitted = array whose ij-th element is a tuple (k, l) where k is the 
    # index of a column of inSeries that was identified as relevant to the i-th 
    # column of outSeries, and l is the relevant lag

    # Outputs:
    # relSeries = list of sets, where the i-th set contains the indices of the 
    # actually relevant time series
    # idSeries = list of sets, where the i-th set contains the indices of the 
    # time series identified as relevant
    # F1_scores = array containing the F1 scores associated with each generated 
    # series
    # For each series, the F1 score is 2*(precision*recall)/(precision + recall), 
    # where:
    #    precision = (number of actually relevant series identified)/(total 
    #                number of series identified)
    #    recall = (number of actually relevant series identified)/(total number 
    #             of actually relevant series) 

    # Pre-process the inputs
    relSeries = []
    idSeries = []
    for series in range(len(dep)):
        rs = []
        for i in dep[series]:
            rs.append(i[0])
        relSeries.append(set(rs))
    for series in range(len(depFitted)):
        ids = []
        if len(depFitted[series])>0:            
            for i in depFitted[series]:
                ids.append(i[0])
        idSeries.append(set(ids))
    # Compute the F1 scores
    F1_scores = []
    for series in range(len(idSeries)):
        numRelId = 0
        for i in idSeries[series]:
            if i in relSeries[series]:
                numRelId += 1
        if len(idSeries[series])==0:
            P = np.nan # no relevant series identified
        else:
            P = numRelId / len(idSeries[series])
        R = numRelId / len(relSeries[series])
        if P==0 and R==0:
            F1 = 0
        else:
            F1 = (2*P*R) / (P + R)
        F1_scores.append(F1)
    return (relSeries, idSeries, F1_scores)
