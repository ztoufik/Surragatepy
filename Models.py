from Utils import FModel
import numpy as np
from numpy.typing import ArrayLike


class No_DelayLinear_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[0]


class NoDelay_Tanh_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[-1])

class Q2mulQ1_FModel(FModel):

    def __call__(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        return np.tanh(arr[2]*arr[-1])


class ExpQ2AddQ1_FModel(FModel):
    def __call__(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        return np.exp(arr[2]+arr[-1])
