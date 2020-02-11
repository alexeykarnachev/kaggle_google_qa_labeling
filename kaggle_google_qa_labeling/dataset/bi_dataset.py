import torch
from torch.utils.data import DataLoader, Dataset

from kaggle_google_qa_labeling.length_sort_sampler import LengthSortSampler
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import pad_sequences


class BiDataset(Dataset):
    def __init__(self, X_tq, X_a, F, T_tq, T_a, Y, indexes=None):

        self.X_tq = list(X_tq)
        self.X_a = list(X_a)

        self.F = list(F) if F is not None else F

        self.T_tq = list(T_tq)
        self.T_a = list(T_a)

        self.Y = list(Y) if Y is not None else Y
        self.indexes = list(indexes) if indexes is not None else [-1] * len(X_tq)

    def __getitem__(self, ind):
        item = [self.X_tq[ind], self.X_a[ind], self.T_tq[ind], self.T_a[ind]]
        if self.F is not None:
            item.append(self.F[ind].tolist())

        if self.Y is not None:
            item = [item, self.Y[ind]]
        else:
            item = [item, ]

        return item

    def __len__(self):
        return len(self.X_tq)

    def add_extra(self, other: 'BiDataset'):
        self.X_tq.extend(other.T_tq)
        self.X_a.extend(other.X_a)
        self.T_tq.extend(other.T_tq)
        self.T_a.extend(other.T_a)
        self.Y.extend(other.Y)
        self.indexes.extend(other.indexes)

    @staticmethod
    def get_collate_fn(max_len, pad_id):
        def collate_fn(data):
            data = list(zip(*data))
            Y = data[1] if len(data) == 2 else None
            d = data[0]
            d = list(zip(*d))

            X_tq, X_a, T_tq, T_a = d[:4]
            F = d[4] if len(d) == 5 else None

            X_tq = list(zip(*X_tq))
            X_a = list(zip(*X_a))
            T_tq = list(zip(*T_tq))
            T_a = list(zip(*T_a))

            seq_len_a = min(max_len, max([len(x) for x in T_a[0]]))
            seq_len_tq = min(max_len, max([len(x) for x in T_tq[0]]))

            T_tq = [torch.LongTensor(pad_sequences(T_, seq_len_tq, 'post', 1)) for T_ in T_tq]
            T_a = [torch.LongTensor(pad_sequences(T_, seq_len_a, 'post', 1)) for T_ in T_a]
            X_tq = [torch.LongTensor(pad_sequences(X_, seq_len_tq, 'post', pad_id)) for X_ in X_tq]
            X_a = [torch.LongTensor(pad_sequences(X_, seq_len_a, 'post', pad_id)) for X_ in X_a]

            if Y is not None:
                Y = torch.FloatTensor(Y)
                res = ([X_tq, X_a, T_tq, T_a], [Y])
            else:
                res = ([X_tq, X_a, T_tq, T_a],)

            if F is not None:
                F = torch.LongTensor(F)
                res[0].append(F)

            return res

        return collate_fn

    def get_data_loader(self, bs, max_len, pad_id, drop_last, use_length_sampler=True):
        lengths = [len(x_tq[0]) for x_tq in self.X_tq]

        if use_length_sampler:
            sampler_ = LengthSortSampler(lengths, bs=bs)
        else:
            sampler_ = None

        dl = DataLoader(
            self,
            batch_size=bs,
            collate_fn=BiDataset.get_collate_fn(max_len, pad_id),
            sampler=sampler_,
            drop_last=drop_last)

        return dl
