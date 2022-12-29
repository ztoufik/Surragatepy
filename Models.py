from Utils import FModel
import numpy as np
from numpy.typing import ArrayLike

class Q0_Linear_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[0]

class Q0_Tanh_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[0])

class Q1_Linear_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[1]

class Q1_Tanh_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.tanh(arr[1])

class Q2mulQ1_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[2]*arr[1]

class ExpQ2AddQ1_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[2]+arr[1])

class Q0mulQ0_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[0]*arr[0]

class ExpQ0AddQ0_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[0]+arr[0])

class Q1mulQ1_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[1]*arr[1]

class ExpQ1AddQ1_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[1]+arr[1])

class Q2mulQ2_FModel(FModel):

    def __call__(self,arr:ArrayLike)->ArrayLike:
        return arr[2]*arr[2]

class ExpQ2AddQ2_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.exp(arr[2]+arr[2])

class PolarCoord_FModel(FModel):
    def __call__(self,arr:ArrayLike)->ArrayLike:
        return np.array([[np.linalg.norm(row_arr),row_arr[-1]] for row_arr in arr.T]).T
