import os
import chaospy as cp
import numpy as np
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

if __name__=="__main__":
    number_joint_RV=3
    poly_order=5
    quadrature_intg_order=5

    models={}
    models["tanh"]=lambda arr:np.tanh(arr[0])
    models["linear"]=lambda arr:0.34*arr[0]
    models["delay1"]=lambda arr:arr[1]
    models["delay2"]=lambda arr:arr[2]
    models["delay2*delay1"]=lambda arr:arr[2]*arr[1]
    models["exp(delay2+delay1)"]=lambda arr:np.exp(arr[2]+arr[1])

    poly_expanser=Scalar_Poly_Expanser(number_joint_RV,poly_order,quadrature_intg_order)
    print("generate nodes & weights")
    poly_expanser.generate_quad_nodes_weights()
    print("generate poly expansion")
    poly_expanser.generate_poly_expansion()

    evaluation_nodes=np.random.rand(10,number_joint_RV)
    error={}
    for model in models.keys():
        model_evals=[models[model](node) for node in poly_expanser.nodes.T]
        poly_expanser.estimate_fourier_coefs(np.array(model_evals))
        poly_model_evals=poly_expanser.poly_expansion(*evaluation_nodes.T)
        #print("poly model:",poly_model_evals)
        model_evals=np.array([models[model](node) for node in evaluation_nodes])
        #print("model_evals:",model_evals)
        error_vector=(model_evals-poly_model_evals)/model_evals
        error[model]=np.mean(np.abs(error_vector)).round(3)

    for model in error:
        print(model,error[model])
