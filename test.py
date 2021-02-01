# import numpy as np
# from scipy.stats import wasserstein_distance
# import random
# import heapq
# def norm1(dis):
#     min_dis = 6000
#     for i in dis:
#         if i < min_dis and i !=0:
#             min_dis = i
#     return [i/min_dis for i in dis]
#
# def norm(dis):
#     a = dis[0]/120
#     return [i/(120*a) for i in dis ]
#
# def a_Reservoir(samples, m):
#     """
#     :samples: [(item, weight), ...]
#     :k: number of selected items
#     :returns: [(item, weight), ...]
#     """
#
#     heap = []
#     for sample in samples:
#         wi = sample
#         ui = random.uniform(0, 1)
#         ki = ui ** (1 / wi)
#
#         if len(heap) < m:
#             heapq.heappush(heap, (ki, sample))
#         elif ki > heap[0][0]:
#             heapq.heappush(heap, (ki, sample))
#
#             if len(heap) > m:
#                 heapq.heappop(heap)
#
#     return [samples.index(item[1]) for item in heap]
#
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
# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
# def compute_wasserstein_distance(distribution1, distribution2):
#     return wasserstein_distance(distribution1, distribution2)
# def compute_probability(global_distribution, current_distribution, client_distribution, epoch):
#     alpha = 1
#     EMDG = []
#     for i in client_distribution:
#         EMDG.append(compute_wasserstein_distance(global_distribution, i))
#     # EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]
#
#     #
#     EMDC = []
#     for i in client_distribution:
#         EMDC.append(compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
#     # EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
#     # EMDC = [i / 1 for i in EMDC]
#
#     # print(EMDG)
#     _emd = []
#     # print(EMDC)
#     for i in range(len(client_distribution)):
#         _emd.append((0.15 * EMDG[i] - 0.0015*epoch*EMDC[i]))
#         # _emd.append(EMDC[i]/10)
#
#     return softmax(_emd)
#
#
# def select(client_distribution):
#     selected_workers = []
#     global_distribution=np.ones(10)
#     current_distribution = np.ones(10)
#     current_distribution[1] = 50
#     _tmp1 = [1,50,1,1,1,1,1,1,1,1]
#     _tmp2 = [1,0,1,1,1,1,1,1,1,1]
#     for epoch in range(200):
#         current_probability = compute_probability(global_distribution, current_distribution, client_distribution, epoch)
#         selected = a_Reservoir(current_probability.tolist(), 5)
#         selected_workers.append(selected)
#         if 49 in selected:
#             _current_distribution = [i*(epoch+1)*120 for i in current_distribution]
#             _current_distribution = [_current_distribution[i]+_tmp1[i]*120 for i in range(10)]
#             current_distribution = norm(_current_distribution)
#         else:
#             _current_distribution = [i*(epoch+1)*120 for i in current_distribution]
#             _current_distribution = [_current_distribution[i]+_tmp2[i]*120 for i in range(10)]
#             current_distribution = norm(_current_distribution)
#         print(current_distribution)
#     return selected_workers
#
# #
# client_distribution = [norm(i) for i in client_distribution]
#
# # print(select(client_distribution))
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
path1 = './res/2283_results.csv'
filename1 = path1
X1 = []
with open(filename1, 'r') as f:
    lines = f.readlines()
    for line in lines:
        value = [float(s) for s in line.split(',')]
        X1.append(value[0])

max_acc = 90.37
print(max_acc)
max60 = max_acc * 0.97
max80 = max_acc * 0.99

res4 = []
res6 = []
res8 = []
for i in range(len(X1)):
    if X1[i] > max60:
        res6.append(i)
    if X1[i] > max80:
        res8.append(i)
print( [res6[0], res8[0]])