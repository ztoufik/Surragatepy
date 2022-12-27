import os
import chaospy as cp
import Utils as utils

class Scalar_Poly_Expanser:
    def __init__(self, number_joint_RV,poly_order,quadrature_intg_order):
        self.number_joint_RV=number_joint_RV#(iid RV) number of parameters model to evaluate accepts 
        self.poly_order=poly_order
        self.quadrature_intg_order=quadrature_intg_order
        self.iid_joint_dist=cp.Iid(cp.Uniform(0,1),self.number_joint_RV)# (Remove the serialized cache if you changed dist (Hash don't change if you change the distribution)
        self.nodes_hash=hash((self.quadrature_intg_order,self.number_joint_RV))
        self.expansions_hash=hash((self.poly_order,self.number_joint_RV))
        self.nodes_file_name=f"nodes{self.nodes_hash}.pi"
        self.expansions_file_name=f"expansions{self.expansions_hash}.pi"

    def generate_quad_nodes_weights(self):
        if os.path.exists(self.nodes_file_name):
            self.nodes,self.weights=utils.load_data(self.nodes_file_name)
        else:
             self.nodes,self.weights=cp.generate_quadrature(self.quadrature_intg_order, self.iid_joint_dist, rule="gaussian",
                                                            recurrence_algorithm="stieltjes" ,tolerance=1e10-5,scaling=5)
             utils.save_data(self.nodes_file_name,(self.nodes,self.weights))

    def generate_poly_expansion(self):
        if os.path.exists(self.expansions_file_name):
            self.expansions=utils.load_data(self.expansions_file_name)
        else:
             self.expansions = cp.generate_expansion(self.poly_order, self.iid_joint_dist,normed=True,reverse=True)
             utils.save_data(self.expansions_file_name,self.expansions)

    def estimate_fourier_coefs(self,model_evals):
        self.poly_expansion,self.fourier_coefs,self.poly_evals=cp.fit_quadrature(
                self.expansions, self.nodes, self.weights, model_evals,retall=2)
    def evaluate(self,evaluation_nodes):
        return self.poly_expansion(*evaluation_nodes)