import os
import chaospy as cp
import Utils as utils

class Scalar_Poly_Expanser:
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
        self.nodes_file_name=f"nodes{self.nodes_hash}.pi"
        self.expansions_file_name=f"expansions{self.expansions_hash}.pi"

    def generate_quad_nodes_weights(self):
        if os.path.exists(self.nodes_file_name):
            self.nodes,self.weights=utils.load_data(self.nodes_file_name)
        else:
             self.nodes,self.weights=cp.generate_quadrature(self.quad_int_ord, self.joint_dist, rule="gaussian",
                                                            recurrence_algorithm="stieltjes" ,tolerance=1e10-5,scaling=5)
             utils.save_data(self.nodes_file_name,(self.nodes,self.weights))

    def generate_poly_expansion(self):
        if os.path.exists(self.expansions_file_name):
            self.expansions=utils.load_data(self.expansions_file_name)
        else:
             self.expansions = cp.generate_expansion(self.poly_ord, self.joint_dist,normed=True,reverse=True)
             utils.save_data(self.expansions_file_name,self.expansions)

    def estimate_fourier_coefs(self,model_evals):
        '''
        also estimate polynomial expansion on self.nodes
        model_evals:numpy.array of shape 1Xlen(weights)=the output of Model applied to self.nodes
        '''
        self.poly_expansion,self.fourier_coefs,self.poly_evals=cp.fit_quadrature(
                self.expansions, self.nodes, self.weights, model_evals,retall=2)

    def evaluate(self,evaluation_nodes):
        '''
        evaluation_nodes:numpy.array of shape number_RVxL
        return:nump.array of shape 1xL the value of polynomial estimation on each colon of RVs
        '''
        return self.poly_expansion(*evaluation_nodes)
