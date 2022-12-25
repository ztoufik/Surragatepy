import chaospy as cp
import numpy as np

class Scalar_Poly_Expanser:
    def __init__(self, number_joint_RV,poly_order,quadrature_intg_order):
        self.number_joint_RV=number_joint_RV#(iid RV) number of parameters model to evaluate accepts 
        self.poly_order=poly_order
        self.quadrature_intg_order=quadrature_intg_order
        self.normal_dist=cp.Uniform(0,1)
        self.iid_normal_dist=cp.Iid(self.normal_dist,self.number_joint_RV)

    def generate_quad_nodes_weights(self,tolerance=1e-5,recurrence_algorithm="stieltjes",scaling=5):
        self.nodes,self.weights=cp.generate_quadrature(self.quadrature_intg_order, self.iid_normal_dist, rule="gaussian",
                                         recurrence_algorithm=recurrence_algorithm ,tolerance=tolerance,scaling=scaling)

    def generate_poly_expansion(self):
        self.expansions = cp.generate_expansion(self.poly_order, self.iid_normal_dist,normed=True,reverse=True)

    def estimate_fourier_coefs(self,model):
        self.model_evals=np.array([model(node) for node in self.nodes.T])
        self.poly_expansion,self.fourier_coefs,self.poly_evals=cp.fit_quadrature(self.expansions, self.nodes, self.weights, self.model_evals,retall=2)

if __name__=="__main__":
    number_joint_RV=3
    poly_order=5
    quadrature_intg_order=5
    tolerance=1e10-5
    quadrature_algo="stieltjes"
    scaling=5

    models={}
    models["tanh"]=lambda arr:np.tanh(arr[0])
    models["linear"]=lambda arr:0.34*arr[0]
    models["delay1"]=lambda arr:arr[1]
    models["delay2"]=lambda arr:arr[2]
    models["delay2*delay1"]=lambda arr:arr[2]*arr[1]
    models["exp(delay2+delay1)"]=lambda arr:np.exp(arr[2]+arr[1])

    poly_expanser=Scalar_Poly_Expanser(number_joint_RV,poly_order,quadrature_intg_order)
    print("generate nodes & weights")
    poly_expanser.generate_quad_nodes_weights(tolerance,quadrature_algo,scaling)
    print("generate poly expansion")
    poly_expanser.generate_poly_expansion()

    evaluation_nodes=np.random.rand(10,number_joint_RV)
    error={}
    for model in models.keys():
        poly_expanser.estimate_fourier_coefs(models[model])
        poly_model_evals=poly_expanser.poly_expansion(*evaluation_nodes.T)
        #print("poly model:",poly_model_evals)
        model_evals=np.array([models[model](node) for node in evaluation_nodes])
        #print("model_evals:",model_evals)
        error_vector=(model_evals-poly_model_evals)/model_evals
        error[model]=np.mean(np.abs(error_vector)).round(3)

    for model in error:
        print(model,error[model])
