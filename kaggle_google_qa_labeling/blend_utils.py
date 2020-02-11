from typing import List

import numpy as np
from scipy.stats import rankdata
from kaggle_google_qa_labeling.utilities import sigmoid


def blend_arrays(arrays: List[np.ndarray], apply_each, apply_res) -> np.ndarray:
    res = None

    for arr in arrays:
        arr_ = apply_each(arr) if apply_each is not None else arr
        res = arr_ if res is None else res + arr_

    res = apply_res(res) if apply_res is not None else res

    return res


def blend_ranks(arrays: List[np.ndarray]):
    return blend_arrays(
        arrays=arrays,
        apply_each=lambda arr: np.apply_along_axis(lambda x: rankdata(x, method='dense'), axis=0, arr=arr),
        apply_res=lambda res: np.apply_along_axis(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0, arr=res)
    )


def blend_sigmoids(arrays: List[np.ndarray]):
    return blend_arrays(
        arrays=arrays,
        apply_each=lambda arr: sigmoid(arr) / len(arrays),
        apply_res=None
    )


def blend_mean(arrays: List[np.ndarray]):
    return blend_arrays(
        arrays=arrays,
        apply_each=lambda arr: arr / len(arrays),
        apply_res=None
    )
