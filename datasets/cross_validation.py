import os
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler

def kfold_cv(targets, kfold=5, random_seed=0, verbose=True):
    """
        K-Fold Cross Validation
        Effort to divide labels equally.
        Args:
            targets - label data. eg. [0, 1, 1, 2, ..., 0]
        Return:
            (splited train index, splited test index)

    """
    assert isinstance(targets, np.ndarray)
    np.random.seed(random_seed)

    classes = np.unique(targets)
    Cs = {i:np.arange(len(targets))[targets == i] for i in classes}
    train_data_index = [list() for i in range(kfold)]
    test_data_index = [list() for i in range(kfold)]

    if verbose:
        print('---- k-fold info -----')
    for key in Cs:
        C = Cs[key]
        perm = np.random.permutation(len(C))
        b_size = len(C) // kfold
        if verbose:
            print('class: {}, N/{}: {}, residue: {}'.format(key, kfold, b_size, len(C) % kfold))

        for cnt, i in enumerate(range(0, len(C), b_size)):
            if cnt == kfold:
                break
            ind = np.zeros(len(perm), dtype=bool)
            ind[i:i + b_size] = True
            train_data_index[cnt].append(C[~ind])
            test_data_index[cnt].append(C[ind])

    train_data_index = [ np.concatenate(index, 0) for index in train_data_index]
    test_data_index = [ np.concatenate(index, 0) for index in test_data_index]
    
    train_data_index = list(map(np.random.permutation, train_data_index))
    test_data_index = list(map(np.random.permutation, test_data_index))

    return train_data_index, test_data_index
    
def get_splited_dataloader(args, dataset, train_data_index, test_data_index, shuffle=False):
    train_loaders = []
    valid_loaders = []
    for index1, index2 in zip(train_data_index, test_data_index):
        if shuffle:
            train_sampler = SubsetRandomSampler(index1)
            test_sampler = SubsetRandomSampler(index2)
        else:
            train_sampler = SequentialSampler(index1)
            test_sampler = SequentialSampler(index2)
            
        train_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=False, sampler=test_sampler,
            num_workers=args.num_workers, pin_memory=True)
        
        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
    return train_loaders, valid_loaders    