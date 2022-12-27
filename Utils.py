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

class FModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self,arr:ArrayLike)->ArrayLike:
        ...

    @abstractmethod
    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        ...

