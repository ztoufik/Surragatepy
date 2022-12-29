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
    ROUND_ERROR=4
    def __init__(self, num_RV:int,poly_ord:int,quad_int_ord:int):
        '''
        num_RV:number of iid RV
        poly_ord:maximal order of polynomial expansion
        quad_int_ord:order of quadrature integration
        '''
        self.num_RV=num_RV
        self.poly_ord=poly_ord
        self.quad_int_ord=quad_int_ord
        self.joint_dist=cp.Iid(cp.Uniform(-2,2),self.num_RV)# (Remove the serialized cache if you changed dist (Hash don't change if you change the distribution)

        self.nodes_hash=hash((self.quad_int_ord,self.num_RV))
        self.expansions_hash=hash((self.poly_ord,self.num_RV))

        self.cache_dir=os.path.join('.','cache_dir')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.nodes_file_name=os.path.join(self.cache_dir,f"nodes{self.nodes_hash}.pi")
        self.expansions_file_name=os.path.join(self.cache_dir,f"expansions{self.expansions_hash}.pi")

    def generate_quad_nodes_weights(self):
        '''
        generate num_RVxlen(weights) matrix of nodes to be evaluated by target model
        '''
        if os.path.exists(self.nodes_file_name):
            self.nodes,self.weights=utils.load_data(self.nodes_file_name)
        else:
             self.nodes,self.weights=cp.generate_quadrature(self.quad_int_ord, self.joint_dist, rule="gaussian",
                                                            recurrence_algorithm="stieltjes" ,tolerance=1e10-5,scaling=5)
             utils.save_data(self.nodes_file_name,(self.nodes,self.weights))
        self.nodes=self.nodes.round(Expanser.ROUND_ERROR)
        self.weights=self.weights.round(Expanser.ROUND_ERROR)

    def generate_polynomials(self):
        if os.path.exists(self.expansions_file_name):
            self.polynomials=utils.load_data(self.expansions_file_name).round(Expanser.ROUND_ERROR)
        else:
             self.polynomials = cp.generate_expansion(self.poly_ord, self.joint_dist,normed=True,reverse=False)
             utils.save_data(self.expansions_file_name,self.polynomials)

