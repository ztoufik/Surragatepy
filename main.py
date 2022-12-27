

if __name__=="__main__":
    from Expanser import Scalar_Poly_Expanser
    import numpy as np
    from Models import *

    number_joint_RV=3
    poly_order=3
    quadrature_intg_order=5

    models={}
    models["linear"]=No_DelayLinear_FModel()
    models["tanh"]=NoDelay_Tanh_FModel()
    models["Q2*Q1"]=Q2mulQ1_FModel()
    models["exp(Q1+Q2)"]=ExpQ2AddQ1_FModel()

    poly_expanser=Scalar_Poly_Expanser(number_joint_RV,poly_order,quadrature_intg_order)
    poly_expanser.generate_quad_nodes_weights()
    poly_expanser.generate_poly_expansion()

    evaluation_nodes=np.random.rand(number_joint_RV,100)
    error={}
    for model in models.keys():
        model_evals=models[model].batch_evaluate(poly_expanser.nodes)
        poly_expanser.estimate_fourier_coefs(model_evals)

        poly_model_evals=poly_expanser.poly_expansion(*evaluation_nodes)
        model_evals=models[model].batch_evaluate(evaluation_nodes)

        error_vector=(model_evals-poly_model_evals)/model_evals
        error[model]=np.mean(np.abs(error_vector)).round(3)

    for model in error:
        print(model,error[model])
