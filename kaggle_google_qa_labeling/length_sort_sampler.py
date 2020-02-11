import numpy as np
from torch.utils.data import Sampler
from kaggle_google_qa_labeling.dataset.cross_dataset_utilities import get_chunks, flatten


class LengthSortSampler(Sampler):
    def __init__(self, data_source, bs):
        super().__init__(data_source)
        self.data_source = data_source
        self.bs = bs

        # TODO: dirty shit
        try:
            int(self.data_source[0])
            lengths = self.data_source
        except TypeError:
            lengths = [len(x) for x in self.data_source]

        inds = np.argsort(lengths)[::-1]
        chunks = list(get_chunks(inds, bs))
        chunk_inds = list(range(len(chunks) - 1))
        np.random.shuffle(chunk_inds)
        chunk_inds = list(chunk_inds) + [len(chunk_inds)]
        self.inds = flatten([chunks[i] for i in chunk_inds])

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(self.inds)
