import numpy as np
from kaggle_google_qa_labeling.dataset.cross_dataset import CrossDataset


class MulticlassCrossDataset(CrossDataset):
    VALUES_MAP = {
        0: 0,
        200: 1,
        267: 2,
        300: 3,
        333: 4,
        400: 5,
        444: 6,
        467: 7,
        500: 8,
        533: 9,
        556: 10,
        600: 11,
        667: 12,
        700: 13,
        733: 14,
        778: 15,
        800: 16,
        833: 17,
        867: 18,
        889: 19,
        900: 20,
        933: 21,
        1000: 22
    }

    def __init__(self, X, T, Y, indexes):
        super().__init__(X, T, Y, indexes)
        self.X = X
        self.T = T

        self.Y = np.apply_along_axis(lambda x: [self.VALUES_MAP[y] for y in x], 0, (np.round(Y * 1000)).astype(int)) \
            if Y is not None else None

        self.indexes = indexes
