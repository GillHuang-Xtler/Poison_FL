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
from client import Client
import contribution_evaluation
import copy
import test


def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                               str(clients[client_idx].get_client_index()))
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    new_nn_params = average_nn_parameters(parameters)

    if args.contribution_measurement_metric == 'None':
        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(new_nn_params)

    elif args.contribution_measurement_metric == 'Influence' and (args.contribution_measurement_round == epoch or args.contribution_measurement_round == epoch+1 or args.contribution_measurement_round == epoch+2 or args.contribution_measurement_round == epoch+3 or args.contribution_measurement_round == epoch+4):
        result_deletion = contribution_evaluation.calculate_influence(args, clients, random_workers, epoch)
        result_deletion_acc = [i[0] for i in result_deletion]
        result_deletion_loss = [i[1] for i in result_deletion]
        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(new_nn_params)

        accuracy, loss, class_precision, class_recall = clients[0].test()
        Influence_acc = result_deletion_acc[:] = [accuracy - x[0] for x in result_deletion]
        Influence_loss = result_deletion_loss[:] = [loss - x[1] for x in result_deletion]
        args.get_logger().info("Influence on clients: by acc: #{}, by loss: #{} on selected #{}", str(Influence_acc), str(Influence_loss), str(random_workers))

    elif args.contribution_measurement_metric == 'Shapley' and args.contribution_measurement_round == epoch:
        shapley_acc, shapley_loss = contribution_evaluation.calculate_shapley_values(args, clients, random_workers, epoch)
        args.get_logger().info("Shapley on clients: by acc: #{}, by loss: #{} on selected #{}", str(shapley_acc), str(shapley_loss), str(random_workers))

        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(new_nn_params)

    else:
        for client in clients:
            args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
            client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers


def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients


def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    for epoch in range(1, args.get_num_epochs() + 1):
        # torch.cuda.synchronize()
        start = time.time()
        results, workers_selected = train_subset_of_clients(epoch, args, clients, poisoned_workers)
        # torch.cuda.synchronize()
        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        end = time.time()
        args.get_logger().debug(
            'Time for training ' + str(args.get_net()) + ' for a round without contribution evaluation is: ' + str(
                (end - start)) + ' seconds')

    return convert_results_to_csv(epoch_test_set_results), worker_selection


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,
                                            replacement_method, args.get_poison_effort)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset,
                                                                        args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    save_results(results, results_files[0])
    save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
