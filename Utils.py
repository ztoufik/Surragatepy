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

class FModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self,arr:ArrayLike)->ArrayLike:
        ...

    @abstractmethod
    def batch_evaluate(self,arr_of_arrs:ArrayLike)->ArrayLike:
        ...

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

if __name__=="__main__":
    from main import Scalar_Poly_Expanser

    number_joint_RV=3
    poly_order=3
    quadrature_intg_order=5

    models={}
    models["linear"]=No_DelayLinear_FModel()
    models["tanh"]=NoDelay_Tanh_FModel()
    models["Q2*Q1"]=Q2mulQ1_FModel()

    poly_expanser=Scalar_Poly_Expanser(number_joint_RV,poly_order,quadrature_intg_order)
    print("generate nodes & weights")
    poly_expanser.generate_quad_nodes_weights()
    print("generate poly expansion")
    poly_expanser.generate_poly_expansion()

    evaluation_nodes=np.random.rand(number_joint_RV,10)
    error={}
    for model in models.keys():
        model_evals=models[model].batch_evaluate(poly_expanser.nodes)
        poly_expanser.estimate_fourier_coefs(model_evals)

        poly_model_evals=poly_expanser.poly_expansion(*evaluation_nodes)
        model_evals=models[model].batch_evaluate(evaluation_nodes)
        print("model:",model_evals)

        error_vector=(model_evals-poly_model_evals)/model_evals
        error[model]=np.mean(np.abs(error_vector)).round(3)

    for model in error:
        print(model,error[model])
