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

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[2]*arr[-1])

class Q1mulQ0_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[0]*arr[1])

class ExpQ2AddQ1_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[2]+arr[-1])

class PolarCoord_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        #if len(arr.shape)==1:#scalar model:
            #arr=np.array([arr])#vector model with one entry
        print("arr shape:",arr.shape)
        norms_list=[np.linalg.norm(row_arr) for row_arr in arr.T]
        scalar_list=[row_arr[-1] for row_arr in arr.T]
        out=np.array([[np.linalg.norm(row_arr),row_arr[-1]] for row_arr in arr.T]).T
        print("*"*20)

        print("out shape:",out.shape)
        return out
