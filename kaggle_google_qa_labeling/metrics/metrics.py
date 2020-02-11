import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr

from kaggle_google_qa_labeling.blend_utils import blend_sigmoids
from kaggle_google_qa_labeling.metrics.abstract_metric import AbstractMetric


class SpearmanMetric(AbstractMetric):

    def __init__(self):
        super().__init__()

    def calculate(self, logits, labels):
        if len(logits[0].shape) == 3:
            logits = logits[0]
            logits = softmax(logits, axis=1)
            logits = np.argmax(logits, axis=1)
        else:
            logits = blend_sigmoids(logits)
            
        labels = labels[0]

        res = 0

        for i in range(labels.shape[1]):
            corr = spearmanr(labels[:, i], logits[:, i]).correlation

            if np.isnan(corr) or not corr:
                corr = 0

            res += corr / labels.shape[1]

        return res
