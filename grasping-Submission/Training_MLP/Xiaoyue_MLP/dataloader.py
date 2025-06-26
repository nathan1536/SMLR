
import numpy as np
import torch

class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        def build_batch_iterator(dataset, batch_size, shuffle):
            shuffle = self.shuffle
            batch_size = self.batch_size
            dataset = self.dataset
            if shuffle:
                index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
            else:
                index_iterator = iter(range(len(dataset)))

            batch = []
            for index in index_iterator:  # iterate over indices using the iterator
                batch.append(dataset[index])
                if self.drop_last:
                    if len(batch) == batch_size:
                        yield batch  # use yield keyword to define a iterable generator
                        batch = []
                else:
                    if len(batch) == batch_size:
                        yield batch  # use yield keyword to define a iterable generator
                        batch = []
                    if ((len(dataset) - index) <= len(dataset) % batch_size) and (len(batch) == len(dataset) % batch_size):
                        yield batch
                        batch = []


        def combine_batch_dicts(batch):
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            return batch_dict

        def batch_to_numpy(batch):
            numpy_batch = {}
            # for key, value in batch.items():
            #     numpy_batch[key] = np.array(value)
            # return numpy_batch
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    # Add debugging statements
                    print(f"Key: {key}, Shape: {value.shape}")

                    # Convert tensor to numpy array
                    numpy_batch[key] = np.array(value)
                else:
                    numpy_batch[key] = value
            return numpy_batch

        batch_iterator = build_batch_iterator(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        batches = []
        for batch in batch_iterator:
            batches.append(batch)

        for batch in batches:
            yield batch_to_numpy(combine_batch_dicts(batch))


    def __len__(self):
        if self.drop_last:
            length = len(self.dataset) // self.batch_size
        elif len(self.dataset) % self.batch_size:
            length = len(self.dataset) // self.batch_size + 1
        else:
            length = len(self.dataset) // self.batch_size

        return length
