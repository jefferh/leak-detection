import time
import numpy as np
import pandas as pd
import pandas.io.data
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
import GPy
import sgenRandomized as sgr
import helpers as hf

def printEdges(coefArray, tgt, numTargets, numFeatures):
    maxCoefs = np.zeros(numFeatures)  # magnitude of largest coefficient for each feature
    maxLags = np.zeros(numFeatures)   # lag of largest coefficient for each feature
    for f in range(numFeatures):
        maxCoefs[f] = np.amax(coefArray[f,:])
        maxLags[f] = np.argsort(coefArray[f,:])[-1] + 1
    colorVals = np.zeros(numFeatures) # Percentage of red to use for the feature's edge in the graph drawing
    mcf = np.amax(maxCoefs) # Largest coefficient for the current target
    for f in range(numFeatures):
        colorVals[f] = int((maxCoefs[f] / mcf) * 100)
    minColorVal = 20
    for f in range(numFeatures):
        if colorVals[f]>=minColorVal:
            print "(%d) edge [red!%d] node [below right] {%d} (%d)" % (f+numTargets, colorVals[f], maxLags[f], tgt)



