from federated_learning.utils import average_nn_parameters
import numpy as np
import math
import itertools

def calculate_influence(args, clients, random_workers, epoch):
    """
    get influence of each client.
    :param clients: clients
    :type clients: list(Clients)
    """
    workers = []
    for client in clients:
        if client.get_client_index() in random_workers:
            workers.append(client)

    result_deletion = []

    args.get_logger().info("test result on epoch #{}", str(epoch))

    for client in workers:
        other_clients_idx =[worker_id for worker_id in random_workers if worker_id != client.get_client_index()]
        args.get_logger().info("Removing parameters on client #{}", str(client.get_client_index()))
        other_parameters = [clients[client_idx].get_nn_parameters() for client_idx in other_clients_idx]
        new_other_params = average_nn_parameters(other_parameters)
        client.update_nn_parameters(new_other_params)
        args.get_logger().info("Finished calculating Influence on client #{}", str(client.get_client_index()))
        result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = client.test()
        result_deletion.append([result_deletion_accuracy, result_deletion_loss])

    return result_deletion

def make_all_subsets(n_client):
    client_list = list(np.arange(n_client))
    set_of_all_subsets = set([])
    for i in range(len(client_list),-1,-1):
        for element in itertools.combinations(client_list,i):
            set_of_all_subsets.add(frozenset(element))
    return sorted(set_of_all_subsets)




