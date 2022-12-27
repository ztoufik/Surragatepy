import hashlib
import numpy as np
import pickle as pk
from abc import ABC,abstractmethod
from numpy.typing import ArrayLike

def load_data(file_name:str)->tuple:
    with open(file_name,"rb") as f:
        out=pk.load(f)   
        print(f"loading {f.name} success")
        return out

def save_data(file_name:str,inputs:object)->None:
    with open(file_name,"wb") as f:
        pk.dump(inputs,f)
        print(f"dumping {f.name} success")

def hash_numpy_arr(arr):
     return hashlib.blake2b(arr.tobytes(), digest_size=20).hexdigest()


class FModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self,arr:ArrayLike)->ArrayLike:
        ...

    @abstractmethod
    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        ...

    def save_data(self,file_cache:str,arr:ArrayLike)->None:
        print("saving",file_cache)
        np.save(file_cache,arr)

    def load_data(self,file_cache:str)->ArrayLike:
        print("loading",file_cache)
        return np.load(file_cache)
