import numpy as np


def adj_list2mat(adjList, *values):
    # adjacency list to adjacency matrix
    # note: adjacency list is shifted by 1
    for val in values:
        assert val.shape[:2] == adjList.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency list"

    adjMat = np.zeros((len(adjList), len(adjList)), dtype=int)
    for i in range(len(adjList)):
        adjMat[i, adjList[i, adjList[i] > 0] - 1] = 1

    if len(values) == 0:
        return adjMat
    else:
        idx1, idx3 = np.where(adjMat)
        idx2 = adjList[adjList > 0] - 1

        outVal = [np.zeros(list(adjMat.shape) + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idx1, idx2] = val[adjList > 0]

        return tuple([adjMat] + outVal)


def adj_mat2list(adjMat, *values):
    # adjacency matrix to adjacency list
    # note: the indices are shifted by 1 in the adjacency list

    adjMat = np.array(adjMat, dtype=int)
    for val in values:
        assert val.shape[:2] == adjMat.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency matrix"

    idx1, idx2 = np.where(adjMat)
    numNb = adjMat.sum(-1)
    maxNb = numNb.max()
    idxNb = np.zeros((len(adjMat), maxNb), dtype=int)
    for i in range(len(adjMat)): idxNb[i, :numNb[i]] = 1
    idxNb[idxNb == 1] = idx2 + 1

    if len(values) == 0:
        return maxNb, idxNb
    else:
        outVal = [np.zeros([len(idxNb), maxNb] + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idxNb > 0] = val[adjMat > 0]

        return tuple([maxNb, idxNb] + outVal)



def adjList2adjMat_old(adjList, *values):
    # adjacency list to adjacency matrix
    # note: adjacency list is shifted by 1
    for val in values:
        assert val.shape[:2] == adjList.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency list"

    adjMat = np.array([[1 if k + 1 in item[item > 0] else 0 for k in range(len(adjList))] for item in adjList])

    if len(values) == 0:
        return adjMat
    else:
        idx1, idx3 = np.where(adjMat)
        idx2 = adjList[adjList > 0] - 1

        outVal = [np.zeros(list(adjMat.shape) + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idx1, idx2] = val[adjList > 0]

        return tuple([adjMat] + outVal)


def adjMat2adjList_old(adjMat, *values):
    # adjacency matrix to adjacency list
    # note: the indices are shifted by 1 in the adjacency lsit
    for val in values:
        assert val.shape[:2] == adjMat.shape, \
            "The first 2 dimensions of the input values must be the same as the adjacency matrix"


    idx1, idx2 = np.where(adjMat)
    maxNb = np.array([list(idx1).count(item) for item in list(idx1)]).max()
    idxNb = np.array(
        [np.concatenate([idx2[idx1 == item] + 1, np.zeros(maxNb - list(idx1).count(item), dtype=int)]) for item in
         list(set(idx1))])

    if len(values) == 0:
        return maxNb, idxNb
    else:
        outVal = [np.zeros([len(idxNb), maxNb] + list(val.shape)[2:], dtype=val.dtype) for val in values]

        for iVal, val in enumerate(values):
            (outVal[iVal])[idxNb > 0] = val[adjMat > 0]

        return tuple([maxNb, idxNb] + outVal)
