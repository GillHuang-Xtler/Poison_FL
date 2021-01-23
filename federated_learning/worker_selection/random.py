from .selection_strategy import SelectionStrategy
import random
import math
from scipy.stats import wasserstein_distance
import numpy as np
import heapq
from federated_learning.utils.tensor_converter import convert_distributed_data_into_numpy

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, workers, poisoned_workers, kwargs):
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])


    def select_round_workers_minus_1(self, workers, poisoned_workers, kwargs):
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"]-1)


    def select_round_workers_except_49(self, workers, poisoned_workers, kwargs):
        workers.remove(49)
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"]-1)

    def compute_wasserstein_distance(self, distribution1, distribution2):
        return wasserstein_distance(distribution1,distribution2)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def S(self, weight):
        R = random.random()
        return math.pow(R, 1 / weight)

    def a_Reservoir(self, samples, m):
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

    def compute_probability(self, global_distribution, current_distribution, client_distribution):
        alpha = 0.1
        EMDG = []
        for i in client_distribution:
            EMDG.append(self.compute_wasserstein_distance(global_distribution, i))
        EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]

        #
        EMDC = []
        for i in client_distribution:
            EMDC.append(self.compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
        EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
        EMDC = [i / 1 for i in EMDC]

        _emd = []

        for i in range(len(client_distribution)):
            _emd.append((alpha * EMDG[i] - EMDC[i]))

        return self.softmax(_emd)

    def select_round_workers_distribution(self, workers, poisoned_workers,clients, current_distribution, kwargs):
        client_distribution = []
        global_distribution = [6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]
        for client_idx in range(len(clients)):
            _client_distribution = clients[client_idx].get_client_distribution()
            global_distribution = [global_distribution[i]+_client_distribution[i] for i in range(len(_client_distribution))]
            client_distribution.append(_client_distribution)

        print('current_distribution:'+ str(current_distribution))
        probability = self.compute_probability(global_distribution, current_distribution, client_distribution)
        print("probability"+ str(probability))

        num_round_workers  = kwargs["NUM_WORKERS_PER_ROUND"]
        choosed_workers = self.a_Reservoir(probability.tolist(), num_round_workers)
        return choosed_workers

