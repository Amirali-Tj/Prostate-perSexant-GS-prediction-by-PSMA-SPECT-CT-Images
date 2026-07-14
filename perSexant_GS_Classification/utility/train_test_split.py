import numpy as np
from pprint import pprint


def dataSplitPerSample(sets , testRatio , seed=None) :
    sets = list(sets)
    if seed != None :
        np.random.seed(seed)

    fullTrainSet = []
    fullTestSet  = []

    for set in sets :
        np.random.shuffle(set)
        setLen = len(set)
        testLen  = np.round(testRatio*setLen)
        trainLen = int(setLen - testLen)
        
        trainSet = set[0:trainLen]
        testSet  = set[trainLen:]

        fullTrainSet.extend(trainSet)
        fullTestSet.extend(testSet)
    
    return fullTrainSet , fullTestSet

def stratified_CV_split(sets , n_fold , seed=None) :
    sets = list(sets)
    if seed != None :
        np.random.seed(seed)

    foldsPerClass = []

    for set in sets :
        np.random.shuffle(set)
        setLen   = len(set)
        foldLen  = int(np.ceil(setLen/n_fold)) # to have better division floor used
        classFolds    = []

        ix_s = 0
        ix_e = foldLen 
        for n in range(n_fold) :
            if n == n_fold - 1 :
                fold = set[ix_s:]
            
            fold = set[ix_s:ix_e]
            print(fold)
            ix_s = ix_s + foldLen
            ix_e = ix_e + foldLen

            classFolds.append(fold)
        foldsPerClass.append(classFolds)
    folds = {}
    for n in range(n_fold) :
        fold_n_train = []
        fold_n_test  = []

        for c in foldsPerClass :
            fold_n_test.extend(c[n])
            val = c.pop(n)
            for splt in c :
                fold_n_train.extend(splt)
            c.insert(n , val)
        else :
            if len(fold_n_test) != 0 :
                folds[n] = {
                    "train" : fold_n_train ,
                    "test"  : fold_n_test 
                }

    return folds
