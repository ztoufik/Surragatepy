import os
from Utils import FModel,hash_numpy_arr
import numpy as np
from numpy.typing import ArrayLike


class No_DelayLinear_FModel(FModel):

    def evaluate(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        return arr[0]

    def batch_evaluate(self,arr_of_arrs:ArrayLike,cache_enable=True)->ArrayLike:
        return arr_of_arrs[-1]

class NoDelay_Tanh_FModel(FModel):

    def evaluate(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        if cache_enable:
            file_cache=f"{type(self).__name__}{hash_numpy_arr(arr)}.npy"
            if os.path.exists(file_cache):
                return self.load_data(file_cache)
            out=np.tanh(arr[-1])
            self.save_data(file_cache,out)
            return out
        return np.tanh(arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike,cache_enable=True)->ArrayLike:
        if cache_enable:
            file_cache=f"{type(self).__name__}{hash_numpy_arr(arr_of_arrs)}.npy"
            if os.path.exists(file_cache):
                return self.load_data(file_cache)
            out=np.tanh(arr_of_arrs[-1])
            self.save_data(file_cache,out)
            return out
        return np.tanh(arr_of_arrs[-1])

class Q2mulQ1_FModel(FModel):

    def evaluate(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        return np.tanh(arr[2]*arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike,cache_enable=True)->ArrayLike:
        return np.tanh(arr_of_arrs[-2]*arr_of_arrs[-1])

class ExpQ2AddQ1_FModel(FModel):
    def evaluate(self,arr:ArrayLike,cache_enable=True)->ArrayLike:
        return np.exp(arr[2]+arr[-1])

    def batch_evaluate(self,arr_of_arrs:ArrayLike,cache_enable=True)->ArrayLike:
        return np.exp(arr_of_arrs[2]+arr_of_arrs[-1])
