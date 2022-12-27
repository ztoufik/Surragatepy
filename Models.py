from Utils import FModel
import numpy as np
from numpy.typing import ArrayLike


class No_DelayLinear_FModel(FModel):

    def evaluate(self,arr:ArrayLike)->ArrayLike:
        return arr[0]

    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        return arr_of_arrs[-1]

class NoDelay_Tanh_FModel(FModel):

    def evaluate(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        return np.tanh(arr_of_arrs[-1])

class Q2mulQ1_FModel(FModel):

    def evaluate(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[2]*arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        return np.tanh(arr_of_arrs[-2]*arr_of_arrs[-1])

class ExpQ2AddQ1_FModel(FModel):
    def evaluate(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[2]+arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        return np.exp(arr_of_arrs[2]+arr_of_arrs[-1])
