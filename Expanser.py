import os
import chaospy as cp
import numpy as np
import Utils as utils

class Expanser:
    '''
    Esitmate the polynomial expansion & the fourier coefficient of model using quadrature integration technique
    The input is assumed to be follow IID joint uniform [0,1] pdf
    nodes & weights used in estimating the fourier coefficient are generated based on quadrature integration
    file based caching is used where file are named using hashed dependent variables
    '''
    def __init__(self, num_RV,poly_ord,quad_int_ord):
        '''
        num_RV:number of iid RV
        poly_ord:maximal order of polynomial expansion
        quad_int_ord:order of quadrature integration
        '''
        self.num_RV=num_RV
        self.poly_ord=poly_ord
        self.quad_int_ord=quad_int_ord
        self.joint_dist=cp.Iid(cp.Uniform(0,1),self.num_RV)# (Remove the serialized cache if you changed dist (Hash don't change if you change the distribution)

        self.nodes_hash=hash((self.quad_int_ord,self.num_RV))
        self.expansions_hash=hash((self.poly_ord,self.num_RV))

        self.cache_dir=os.path.join('.','cache_dir')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.nodes_file_name=os.path.join(self.cache_dir,f"nodes{self.nodes_hash}.pi")
        self.expansions_file_name=os.path.join(self.cache_dir,f"expansions{self.expansions_hash}.pi")

    def generate_quad_nodes_weights(self):
        if os.path.exists(self.nodes_file_name):
            self.nodes,self.weights=utils.load_data(self.nodes_file_name)
        else:
             self.nodes,self.weights=cp.generate_quadrature(self.quad_int_ord, self.joint_dist, rule="gaussian",
                                                            recurrence_algorithm="stieltjes" ,tolerance=1e10-5,scaling=5)
             utils.save_data(self.nodes_file_name,(self.nodes,self.weights))

    def generate_polynomials(self):
        if os.path.exists(self.expansions_file_name):
            self.polynomials=utils.load_data(self.expansions_file_name)
        else:
             self.polynomials = cp.generate_expansion(self.poly_ord, self.joint_dist,normed=True,reverse=True)
             utils.save_data(self.expansions_file_name,self.polynomials)

    def estimate_fourier_coefs(self,model_evals):
        '''
        also estimate polynomial expansion on self.nodes
        model_evals:numpy.array of shape 1Xlen(weights)=the output of Model applied to self.nodes
        '''
        if len(model_evals.shape)>2:
            raise ValueError(f"expected shape of arg: (m,n). got {model_evals.shape}")

        if len(model_evals.shape)==1:#scalar model:
            model_evals=np.array([model_evals])#vector model with one entry


        fourier_coefs_arrs=[0]*model_evals.shape[0]
        expansions_arrs=[0]*model_evals.shape[0]
        poly_evals_arrs=[0]*model_evals.shape[0]

        for index,model in enumerate(model_evals):
            expansions_arrs[index],fourier_coefs_arrs[index],poly_evals_arrs[index]=cp.fit_quadrature(
                    self.polynomials, self.nodes, self.weights, model,retall=2)
        self.fourier_coefs_arrs=np.array(fourier_coefs_arrs)
        self.expansions_arrs=cp.polynomial(expansions_arrs)
        self.poly_evals_arrs=np.array(poly_evals_arrs)

    def evaluate(self,evaluation_nodes):
        raise NotImplementedError()

    def calculate_IPC(self):
        ipc={}
        for poly,arr in zip(self.polynomials,self.fourier_coefs_arrs.T):
            ipc[str(poly)]=arr*arr
        return ipc
