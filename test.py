import numpy as np
from scipy.stats import wasserstein_distance
import random
import heapq
def norm1(dis):
    min_dis = 6000
    for i in dis:
        if i < min_dis and i !=0:
            min_dis = i
    return [i/min_dis for i in dis]
#
def norm(dis):
    a = dis[0]/100
    return [i/(100*a) for i in dis ]

def a_Reservoir(samples, m):
    """
    :samples: [(item, weight), ...]
    :k: number of selected items
    :returns: [(item, weight), ...]
    """

    heap = []
    for sample in samples:
        wi = sample
        ui = random.uniform(0, 1)
        ki = ui ** (1 / wi)

        if len(heap) < m:
            heapq.heappush(heap, (ki, sample))
        elif ki > heap[0][0]:
            heapq.heappush(heap, (ki, sample))

            if len(heap) > m:
                heapq.heappop(heap)

    return [samples.index(item[1]) for item in heap]
client_distribution = [[97, 0, 97, 82, 110, 107, 97, 105, 94, 112]
,[100, 0, 102, 111, 96, 91, 102, 88, 99, 108]
,[105, 0, 89, 108, 100, 104, 93, 121, 99, 86]
,[105, 0, 91, 106, 93, 101, 108, 94, 102, 92]
,[89, 0, 91, 106, 92, 115, 124, 88, 88, 104]
,[95, 0, 103, 98, 107, 89, 102, 102, 99, 122]
,[92, 0, 90, 99, 113, 100, 106, 105, 106, 95]
,[96, 0, 113, 108, 111, 77, 80, 103, 116, 86]
,[96, 0, 93, 86, 108, 97, 94, 109, 99, 107]
,[107, 0, 84, 103, 105, 112, 90, 104, 90, 92]
,[101, 0, 123, 102, 105, 109, 101, 101, 80, 94]
,[97, 0, 118, 104, 79, 104, 108, 110, 85, 92]
,[96, 0, 101, 100, 95, 97, 87, 107, 112, 105]
,[117, 0, 99, 86, 90, 106, 100, 97, 104, 90]
,[94, 0, 96, 114, 116, 93, 85, 111, 89, 101]
,[98, 0, 107, 99, 90, 99, 110, 98, 85, 111]
,[91, 0, 87, 107, 94, 101, 121, 111, 108, 96]
,[85, 0, 105, 94, 111, 117, 95, 92, 104, 103]
,[93, 0, 107, 106, 99, 83, 108, 96, 110, 107]
,[130, 0, 92, 100, 89, 80, 118, 91, 108, 87]
,[113, 0, 102, 100, 104, 90, 105, 104, 92, 92]
,[96, 0, 91, 102, 97, 111, 88, 111, 103, 97]
,[92, 0, 115, 102, 96, 88, 98, 90, 108, 103]
,[100, 0, 99, 99, 107, 98, 107, 90, 100, 108]
,[105, 0, 108, 97, 101, 84, 103, 96, 80, 125]
,[112, 0, 83, 103, 97, 103, 77, 117, 96, 109]
,[112, 0, 100, 97, 102, 102, 99, 97, 104, 97]
,[108, 0, 92, 94, 108, 82, 97, 105, 96, 113]
,[96, 0, 113, 104, 96, 95, 106, 89, 91, 120]
,[94, 0, 103, 99, 104, 103, 94, 93, 117, 83]
,[91, 0, 106, 104, 109, 106, 98, 94, 105, 98]
,[114, 0, 111, 119, 96, 96, 85, 93, 96, 95]
,[102, 0, 99, 86, 81, 113, 107, 111, 94, 104]
,[119, 0, 96, 84, 90, 116, 110, 101, 108, 90]
,[98, 0, 94, 109, 103, 93, 90, 96, 101, 106]
,[95, 0, 100, 116, 106, 98, 86, 104, 99, 91]
,[105, 0, 97, 83, 103, 100, 85, 103, 97, 104]
,[94, 0, 88, 90, 128, 99, 100, 106, 84, 97]
,[101, 0, 112, 103, 88, 101, 119, 90, 97, 83]
,[90, 0, 110, 85, 106, 116, 97, 97, 111, 92]
,[94, 0, 90, 111, 87, 115, 109, 99, 111, 94]
,[99, 0, 97, 96, 94, 87, 111, 97, 104, 88]
,[89, 0, 116, 104, 95, 97, 108, 98, 108, 88]
,[103, 0, 90, 103, 98, 102, 98, 102, 96, 101]
,[105, 0, 84, 93, 109, 105, 81, 101, 121, 118]
,[98, 0, 106, 94, 108, 96, 102, 101, 97, 102]
,[113, 0, 108, 103, 95, 112, 91, 86, 108, 93]
,[97, 0, 102, 103, 104, 101, 104, 87, 108, 100]
,[100, 0, 108, 96, 92, 105, 118, 95, 102, 91]
,[81, 5000, 92, 102, 93, 104, 98, 114, 89, 128]]
# client_distribution = [[111, 0, 104, 109, 137, 96, 127, 130, 134, 121],
#                         [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                         [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                         [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                         [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                         [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                         [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                         [111, 0, 104, 109, 137, 96, 127, 130, 134, 121],
#                         [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                         [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                         [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                         [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                         [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                         [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                        [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                        [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                        [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                        [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [111, 0, 104, 109, 137, 96, 127, 130, 134, 121],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                        [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                        [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                        [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                        [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                        [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                        [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                        [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                        [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [111, 0, 104, 109, 137, 96, 127, 130, 134, 121],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                        [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                        [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                        [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                        [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                        [134, 0, 129, 136, 129, 109, 105, 101, 106, 123],
#                        [113, 0, 117, 133, 124, 113, 107, 135, 122, 108],
#                        [122, 0, 128, 116, 120, 121, 103, 141, 117, 112],
#                        [128, 0, 111, 137, 116, 102, 119, 108, 117, 116],
#                        [111, 0, 104, 109, 137, 96, 127, 130, 134, 121],
#                        [106, 0, 139, 129, 127, 131, 102, 116, 108, 118],
#                        [108, 0, 137, 118, 113, 111, 121, 117, 128, 128],
#                         [132, 6000, 127, 123, 108, 136, 104, 114, 125, 114]
#                         ]
# global_distribution = [i/6000 for i in [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]]
#
#
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
#
def compute_wasserstein_distance(distribution1, distribution2):
    return wasserstein_distance(distribution1, distribution2)
def compute_probability(global_distribution, current_distribution, client_distribution, epoch):
    alpha = 1
    EMDG = []
    for i in client_distribution:
        EMDG.append(compute_wasserstein_distance(global_distribution, i))
    # EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]

    #
    EMDC = []
    for i in client_distribution:
        EMDC.append(compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
    # EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
    # EMDC = [i / 1 for i in EMDC]

    # print(EMDG)
    _emd = []
    # print(EMDC)
    for i in range(len(client_distribution)):
        _emd.append((0.105 * EMDG[i] - 0.0015*epoch*EMDC[i]))
        # _emd.append(EMDC[i]/10)

    return softmax(_emd)
#
#
def select(client_distribution):
    selected_workers = []
    global_distribution=np.ones(10)
    current_distribution = np.ones(10)
    current_distribution[1] = 50
    _tmp1 = [1,50,1,1,1,1,1,1,1,1]
    _tmp2 = [1,0,1,1,1,1,1,1,1,1]
    for epoch in range(200):
        current_probability = compute_probability(global_distribution, current_distribution, client_distribution, epoch)
        selected = a_Reservoir(current_probability.tolist(), 5)
        selected_workers.append(selected)
        if 49 in selected:
            _current_distribution = [i*(epoch+1)*100 for i in current_distribution]
            _current_distribution = [_current_distribution[i]+_tmp1[i]*100 for i in range(10)]
            current_distribution = norm(_current_distribution)
        else:
            _current_distribution = [i*(epoch+1)*100 for i in current_distribution]
            _current_distribution = [_current_distribution[i]+_tmp2[i]*100 for i in range(10)]
            current_distribution = norm(_current_distribution)
        print(current_probability[49])
    return selected_workers

#
# client_distribution = [norm(i) for i in client_distribution]

# print(select(client_distribution))
# # _tmp1 = [1,50,1,1,1,1,1,1,1,1]
# # _tmp2 = [1,3,1,1,1,1,1,1,1,1]
# # print(compute_wasserstein_distance(_tmp1,_tmp2))
# select = select(client_distribution)
# res = []
# cnt = 0
# for i in range(len(select)):
#     if 49 in select[i]:
#         res.append(i)
#         cnt+=1
# print(cnt,res)
path1 = './res/9112_results.csv'
filename1 = path1
X1 = []
with open(filename1, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(',')]
        X1.append(value[0])

max_acc = 90.2
print(max_acc)
max60 = max_acc * 0.87
max80 = max_acc * 0.99

res4 = []
res6 = []
res8 = []
for i in range(len(X1)):
    if X1[i] > max60:
        res6.append(i)
    if X1[i] > max80:
        res8.append(i)
if len(res8) == 0:
    res8.append(0)
print( [res6[0], res8[0]])