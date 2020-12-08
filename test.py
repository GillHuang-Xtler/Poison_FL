from client import Client
from loguru import logger
import torch
import time
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters

from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
import copy
from federated_learning.worker_selection import RandomSelectionStrategy
from federated_learning.utils import replace_1_with_9
import os
import numpy as np
import itertools
import math
from client import Client


#
# KWARGS = {
#         "NUM_WORKERS_PER_ROUND" : 5
#     }
#
#
# args = Arguments(logger)
#
#
def load_model_from_file(args, clients, client, model_file_path):
    """
    Load a model from a file.

    :param model_file_path: string
    """
    model_class = args.get_net()
    model = model_class()
    test_data_loader = load_test_data_loader(logger, args)

    if os.path.exists(model_file_path):
        try:
            model.load_state_dict(torch.load(model_file_path))
        except:
            print("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

            model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    else:
        print("Could not find model: {}".format(model_file_path))

    clients[client].set_net(model)
    result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = clients[client].test()
    return result_deletion_accuracy.test()

def save_temporary_model(args, epoch, subset_idx):
    """
    Saves the model if necessary.
    """
    args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

    if not os.path.exists(args.get_save_model_folder_path()):
        os.mkdir(args.get_save_model_folder_path())

    full_save_path = os.path.join(args.get_save_model_folder_path(),
                                  "model_" + str(subset_idx) + "_" + str(epoch) + ".model")
    torch.save(Client.get_nn_parameters(), full_save_path)


def make_all_subsets(n_client, random_workers):
    """
    make all subsets from a list.
    :param n_client: number of clients this round
    :type n_client: int
    :param random_workers: selected workers
    :type random_workers: list[int]
    :return: set subsets
    """
    client_list = list(np.arange(n_client))
    set_of_all_subsets = set([])
    for i in range(len(client_list), -1, -1):
        for element in itertools.combinations(random_workers, i):
            set_of_all_subsets.add(frozenset(element))
    return sorted(set_of_all_subsets)


def get_subset_index(subset):
    """
    get index from a subset
    :param subset: subset
    :type set(int)
    :return: index of subset
    :type str joined by '_'
    """
    subset_idx = '_'.join(sorted(set(str(i) for i in subset)))
    return subset_idx

def calculate_shapley_values(args, clients, random_workers, epoch):
    result_deletion = []
    args.get_logger().info("Selected workers #{}", str(random_workers))
    args.get_logger().info("Start calculating Shapley result on epoch #{}", str(epoch))
    client_list = list(np.arange(len(random_workers)))
    shapley = []
    clientShapley = 0
    total = 0
    factorialTotal = math.factorial(len(random_workers))
    set_of_all_subsets = make_all_subsets(n_client=len(random_workers), random_workers=random_workers)
    temp_save_dir = './temp_' + str(epoch)
    if not os.path.exists(temp_save_dir):
        os.mkdir(temp_save_dir)
    for client in random_workers:
        for subset in set_of_all_subsets:
            if client in subset:
                remainderSet = subset.difference(set([client]))
                b = len(remainderSet)
                factValue = (len(client_list) - b - 1)
                temp_save_path = os.path.join(temp_save_dir, get_subset_index(subset = subset))
                if not os.path.exists(temp_save_path):
                    other_parameters = [clients[client].get_nn_parameters() for client in subset]
                    new_other_params = average_nn_parameters(other_parameters)
                    clients[client].update_nn_parameters(new_other_params)
                    result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = clients[client].test()
                    result_deletion.append([result_deletion_accuracy, result_deletion_loss])
                    save_temporary_model(args = args, epoch = epoch, subset_idx= get_subset_index(subset))
                # else:
                #     load_model_from_file(args, clients= clients, client=clients[client], model_file_path = temp_save_path)

                if len(remainderSet) > 0:
                    remainder_parameters = [clients[client].get_nn_parameters() for client in remainderSet]
                    new_remainder_params = average_nn_parameters(remainder_parameters)
                    clients[client].update_nn_parameters(new_remainder_params)
                    remainder_accuracy, remainder_loss, remainder_precision, remainder_class_recall = clients[
                        client].test()
                else:
                    remainder_accuracy, remainder_loss = 0, 0
                difference = result_deletion_accuracy - remainder_accuracy
                divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                weightValue = divisor * difference
                clientShapley += weightValue
        shapley.append(clientShapley)
        # total = total + clientShapley
        args.get_logger().info("Finished calculating Shapley Value #{} on client #{}", str(clientShapley), str(client))
        clientShapley = 0

    return shapley


if __name__ == "__main__":
    print(calculate_shapley_values())
