import torch
from torch.utils.data import DataLoader, Dataset

from kaggle_google_qa_labeling.length_sort_sampler import LengthSortSampler
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import pad_sequences


class CrossDataset(Dataset):
    def __init__(self, X, F, T, Y, indexes=None):
        self.X = list(X)
        self.T = list(T)
        self.F = list(F) if F is not None else F
        self.Y = list(Y) if Y is not None else Y

        self.indexes = list(indexes) if indexes is not None else [-1] * len(X)

    def __getitem__(self, ind):
        item = [self.T[ind], self.X[ind]]
        if self.F is not None:
            item.append(self.F[ind].tolist())
        if self.Y is not None:
            item = [item, self.Y[ind]]
        else:
            item = [item, ]
        return item

    def __len__(self):
        return len(self.X)

    def add_extra(self, other: 'CrossDataset'):
        self.X.extend(other.X)
        self.T.extend(other.T)
        self.Y.extend(other.Y)
        self.indexes.extend(other.indexes)

    @staticmethod
    def get_collate_fn(max_len, pad_id):
        def collate_fn(data):
            data = list(zip(*data))
            Y = data[1] if len(data) == 2 else None
            d = data[0]
            d = list(zip(*d))

            T, X = d[0], d[1]
            F = d[2] if len(d) == 3 else None

            T = list(zip(*T))
            X = list(zip(*X))

            seq_len = min(max_len, max([len(x) for x in T[0]]))

            T = [torch.LongTensor(pad_sequences(T_, seq_len, 'post', 1)) for T_ in T]
            X = [torch.LongTensor(pad_sequences(X_, seq_len, 'post', pad_id)) for X_ in X]

            if Y is not None:
                Y = torch.FloatTensor(Y)
                res = ([X, T], [Y])
            else:
                res = ([X, T],)

            if F is not None:
                F = torch.LongTensor(F)
                res[0].append(F)

            return res

        return collate_fn

    def get_data_loader(self, bs, max_len, pad_id, drop_last, use_length_sampler=True):
        lengths = [len(x[0]) for x in self.X]

        if use_length_sampler:
            sampler_ = LengthSortSampler(lengths, bs=bs)
        else:
            sampler_ = None

        dl = DataLoader(self, batch_size=bs, collate_fn=CrossDataset.get_collate_fn(max_len, pad_id),
                        sampler=sampler_, drop_last=drop_last)
        return dl
