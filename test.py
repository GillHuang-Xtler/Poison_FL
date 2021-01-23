import random
import math
import heapq
import numpy as np
from scipy.stats import wasserstein_distance
from statistics import mean
client_distribution = [[138, 0, 95, 129, 121, 122, 112, 121, 118, 118],
                       [129, 0, 120, 107, 134, 121, 116, 123, 125, 113],
                       [131, 0, 128, 112, 109, 128, 129, 112, 122, 121],
                       [137, 0, 129, 121, 110, 139, 110, 110, 130, 109],
                       [110, 0, 108, 128, 136, 120, 119, 111, 116, 135],
                       [108, 0, 118, 106, 133, 104, 125, 136, 130, 119],
                       [131, 0, 135, 122, 129, 123, 116, 106, 113, 113],
                       [122, 0, 121, 120, 130, 110, 112, 112, 128, 125],
                       [121, 0, 99, 115, 116, 103, 129, 138, 130, 123],
                       [116, 0, 125, 125, 120, 121, 144, 103, 99, 122],
                       [111, 0, 123, 118, 117, 129, 99, 129, 123, 120],
                       [133, 0, 122, 121, 116, 121, 119, 106, 111, 109],
                       [112, 0, 111, 138, 116, 121, 130, 129, 133, 104],
                       [118, 0, 112, 112, 123, 104, 125, 141, 127, 116],
                       [117, 0, 127, 123, 138, 130, 119, 102, 122, 112],
                       [126, 0, 115, 105, 127, 119, 132, 106, 126, 124],
                       [123, 0, 116, 122, 116, 109, 117, 122, 128, 131],
                       [116, 0, 124, 122, 110, 109, 125, 112, 119, 140],
                       [121, 0, 130, 129, 120, 112, 129, 110, 116, 116],
                       [108, 0, 135, 133, 122, 113, 113, 119, 110, 124],
                       [103, 0, 129, 135, 119, 128, 124, 119, 122, 113],
                       [130, 0, 115, 132, 115, 100, 119, 132, 123, 130],
                       [134, 0, 119, 120, 101, 120, 113, 120, 117, 120],
                       [122, 0, 120, 125, 121, 139, 129, 124, 97, 107],
                       [120, 0, 103, 124, 126, 129, 127, 123, 122, 107],
                       [129, 0, 127, 104, 110, 126, 123, 122, 115, 126],
                       [123, 0, 113, 117, 132, 121, 112, 114, 150, 98],
                       [103, 0, 123, 139, 121, 124, 120, 126, 114, 119],
                       [118, 0, 100, 108, 126, 120, 125, 147, 123, 128],
                       [129, 0, 115, 94, 100, 133, 140, 132, 123, 115],
                       [103, 0, 127, 147, 111, 95, 123, 121, 118, 121],
                       [127, 0, 120, 124, 113, 117, 115, 99, 139, 134],
                       [116, 0, 139, 118, 104, 115, 129, 115, 121, 129],
                       [124, 0, 112, 115, 118, 133, 125, 121, 110, 124],
                       [125, 0, 122, 137, 140, 111, 101, 123, 109, 120],
                       [108, 0, 129, 115, 116, 131, 108, 113, 129, 121],
                       [120, 0, 123, 124, 118, 108, 106, 128, 132, 118],
                       [112, 0, 128, 118, 137, 98, 116, 121, 130, 115],
                       [144, 0, 116, 94, 116, 128, 103, 136, 121, 122],
                       [108, 0, 140, 118, 100, 120, 123, 120, 122, 127],
                       [126, 0, 118, 119, 133, 134, 104, 113, 112, 117],
                       [119, 0, 119, 113, 116, 130, 122, 127, 87, 128],
                       [115, 0, 142, 103, 123, 111, 126, 123, 119, 121],
                       [119, 0, 136, 129, 98, 105, 120, 118, 123, 143],
                       [125, 0, 119, 101, 132, 123, 119, 119, 118, 119],
                       [128, 0, 112, 100, 111, 122, 147, 105, 112, 122],
                       [109, 0, 109, 118, 130, 126, 129, 131, 118, 103],
                       [121, 0, 108, 121, 119, 133, 119, 117, 110, 125],
                       [101, 0, 123, 130, 127, 132, 110, 110, 127, 111],
                       [111, 6000, 101, 150, 124, 130, 103, 133, 111, 123]]

global_distribution = [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]
# current_distribution = [300,1800,300,300,300,300,300,300,300,300]
# current_distribution = [100,0,100,100,100,100,100,100,100,100]
current_distribution = [1000,6000,1000,1000,1000,1000,1000,1000,1000,1000]
# current_distribution = [15000, 60000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000]
# current_distribution = [5000,30000,5000,5000,5000,5000,5000,5000,5000,5000]
# current_distribution = [8000,40000,8000,8000,8000,8000,8000,8000,8000,8000]
def compute_wasserstein_distance(distribution1, distribution2):
    return wasserstein_distance(distribution1, distribution2)

def softmax(x):
     """Compute softmax values for each sets of scores in x."""
     e_x = np.exp(x - np.max(x))
     return e_x / e_x.sum(axis=0)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def sigmoid(X):
	return 1.0 / (1 + np.exp(-float(X)));

def compute_probability(global_distribution,current_distribution, client_distribution ):
    alpha = 0.1
    EMDG = []
    for i in client_distribution:
        EMDG.append(compute_wasserstein_distance(global_distribution, i))
    EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]

    #
    EMDC = []
    for i in client_distribution:
        EMDC.append(compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
    EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
    EMDC = [i / 1 for i in EMDC]

    _emd = []

    for i in range(len(client_distribution)):
        _emd.append((alpha * EMDG[i] - EMDC[i]))

    print(softmax(_emd))
    return softmax(_emd)

compute_probability(global_distribution,current_distribution, client_distribution)

def a_res(samples, m):
    """
    :samples: [(item, weight), ...]
    :k: number of selected items
    :returns: [(item, weight), ...]
    """

    heap = []
    for sample in samples:
        wi = sample
        ui = random.uniform(0, 1)
        ki = ui ** (1/wi)

        if len(heap) < m:
            heapq.heappush(heap, (ki, sample))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, sample))

            if len(heap) > m:
                heapq.heappop(heap)

    return [samples.index(item[1]) for item in heap]
#
# print(a_res([(1,0,1),(2,0.2),(3,0.3),(4,0.4)],2))