import numpy as np
import chaospy as cp
from Expanser import Expanser
from Utils import FModel

class Surrogate_Model(Expanser):
    def __init__(self,Model:FModel,num_RV:int,poly_ord:int,quad_int_ord:int):
        super(Surrogate_Model,self).__init__(num_RV,poly_ord,quad_int_ord)
        self.Model=Model

    def estimate_fourier_coefs(self):
        '''
        estimate polynomial expansion on self.nodes
        model_evals:numpy.array of shape mXlen(weights)=the output of Model applied to self.nodes
        '''
        self.generate_quad_nodes_weights()

        model_evals=self.Model.evaluate(self.nodes).round(Expanser.ROUND_ERROR)
        if len(model_evals.shape)>2:
            raise ValueError(f"expected shape of arg: (m,n). got {model_evals.shape}")

        if len(model_evals.shape)==1:#scalar model:
            model_evals=np.array([model_evals])#vector model with one entry

        self.generate_polynomials()

        fourier_coefs_arrs=[0]*model_evals.shape[0]
        expansions_arrs=[0]*model_evals.shape[0]
        poly_evals_arrs=[0]*model_evals.shape[0]

        for index,model in enumerate(model_evals):
            expansions_arrs[index],fourier_coefs_arrs[index],poly_evals_arrs[index]=cp.fit_quadrature(
                    self.polynomials, self.nodes, self.weights, model,retall=2)
        self.fourier_coefs_arrs=np.array(fourier_coefs_arrs).round(Expanser.ROUND_ERROR)
        self.expansions_arrs=cp.polynomial(expansions_arrs).round(Expanser.ROUND_ERROR)
        self.poly_evals_arrs=np.array(poly_evals_arrs).round(Expanser.ROUND_ERROR)

    def evaluate(self,evaluation_nodes):
        return np.array(
                [poly(*evaluation_nodes)for poly in self.expansions_arrs]
                ).round(Expanser.ROUND_ERROR)

    def get_error(self,evaluation_nodes):
        poly_model_evals=self.evaluate(evaluation_nodes)
        model_evals=self.Model.evaluate(evaluation_nodes)
        error_vector=(model_evals-poly_model_evals)/model_evals
        return np.array([np.mean(np.abs(err_row)) for err_row in error_vector]).round(3)

    def calculate_IPC(self):
        self.ipc={}
        for poly,arr in zip(self.polynomials,self.fourier_coefs_arrs.T):
            self.ipc[str(poly)]=(arr*arr).round(Expanser.ROUND_ERROR)
        return self.ipc
