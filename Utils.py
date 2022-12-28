import hashlib,os
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
    '''
    __call__: to be defined by custom child class
    '''

    @abstractmethod
    def __call__(self,arr:ArrayLike)->ArrayLike:
        ...

    def evaluate(self,arr:ArrayLike)->ArrayLike:
        self.cache_dir=os.path.join('.','cache_dir')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        file_cache=os.path.join(self.cache_dir,f"{type(self).__name__}{hash_numpy_arr(arr)}.npy")
        if os.path.exists(file_cache):
            return self.load_data(file_cache)
        out=self(arr)
        self.save_data(file_cache,out)
        return out


    def save_data(self,file_cache:str,arr:ArrayLike)->None:
        print(f"saving {file_cache} success")
        np.save(file_cache,arr)

    def load_data(self,file_cache:str)->ArrayLike:
        print(f"loading {file_cache} success")
        return np.load(file_cache)
