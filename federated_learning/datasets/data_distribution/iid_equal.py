import torch

def distribute_batches_equally(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers

        distributed_dataset[worker_idx].append((data, target))

    return distributed_dataset

def distribute_batches_reduce_1(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        try:
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            distributed_dataset[worker_idx].append((data_r, target_r))
        except:
            distributed_dataset[worker_idx].append((data, target))


    return distributed_dataset

def distribute_batches_reduce_1_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        distributed_dataset[num_workers-1].append((data_p, target_p))


    return distributed_dataset

def distribute_batches_reduce_1_only(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % (num_workers-1)
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        if len(plus)>0:
            distributed_dataset[num_workers-1].append((data_p, target_p))

    return distributed_dataset

def distribute_batches_reduce_2_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce0 = (target !=0 ).nonzero()
        plus0 = (target == 0).nonzero()
        plus1 = (target == 1).nonzero()
        reduce1 = (target !=1 ).nonzero()
        target_r0 = torch.index_select(target, 0, reduce0.view(-1))
        data_r0 = torch.index_select(data, 0, reduce0.view(-1))
        target_r = torch.index_select(target_r0, 0, reduce1.view(-1))
        data_r = torch.index_select(data_r0, 0, reduce1.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p0 = torch.index_select(target, 0, plus0.view(-1))
        data_p0 = torch.index_select(data, 0, plus0.view(-1))
        target_p1 = torch.index_select(target, 0, plus1.view(-1))
        data_p1 = torch.index_select(data, 0, plus1.view(-1))
        distributed_dataset[num_workers-1].append((data_p0, target_p0))
        distributed_dataset[num_workers-2].append((data_p1, target_p1))

    return distributed_dataset

def distribute_batches_reduce_3_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce0 = (target !=0 ).nonzero()
        plus0 = (target == 0).nonzero()
        plus1 = (target == 1).nonzero()
        reduce1 = (target !=1 ).nonzero()
        reduce2 = (target !=2 ).nonzero()
        plus2 = (target == 2).nonzero()
        target_r0 = torch.index_select(target, 0, reduce0.view(-1))
        data_r0 = torch.index_select(data, 0, reduce0.view(-1))
        target_r1 = torch.index_select(target_r0, 0, reduce1.view(-1))
        data_r1 = torch.index_select(data_r0, 0, reduce1.view(-1))
        target_r = torch.index_select(target_r1, 0, reduce2.view(-1))
        data_r = torch.index_select(data_r1, 0, reduce2.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p0 = torch.index_select(target, 0, plus0.view(-1))
        data_p0 = torch.index_select(data, 0, plus0.view(-1))
        target_p1 = torch.index_select(target, 0, plus1.view(-1))
        data_p1 = torch.index_select(data, 0, plus1.view(-1))
        target_p2 = torch.index_select(target, 0, plus2.view(-1))
        data_p2 = torch.index_select(data, 0, plus2.view(-1))
        distributed_dataset[num_workers-1].append((data_p0, target_p0))
        distributed_dataset[num_workers-2].append((data_p1, target_p1))
        distributed_dataset[num_workers-3].append((data_p2, target_p2))

    return distributed_dataset