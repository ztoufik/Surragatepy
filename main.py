from Expanser import Expanser
from Surrogate_Model import Surrogate_Model
import numpy as np
from Models import *

number_joint_RV=3
poly_order=4
quadrature_intg_order=10

models={}
models["linear"]=Surrogate_Model(No_DelayLinear_FModel(),number_joint_RV,poly_order,quadrature_intg_order)
models["tanh"]=Surrogate_Model(NoDelay_Tanh_FModel(),number_joint_RV,poly_order,quadrature_intg_order)
models["Q2*Q1"]=Surrogate_Model(Q2mulQ1_FModel(),number_joint_RV,poly_order,quadrature_intg_order)
models["Q1*Q0"]=Surrogate_Model(Q1mulQ0_FModel(),number_joint_RV,poly_order,quadrature_intg_order)
models["exp(Q1+Q2)"]=Surrogate_Model(ExpQ2AddQ1_FModel(),number_joint_RV,poly_order,quadrature_intg_order)
models["Polar_Coord"]=Surrogate_Model(PolarCoord_FModel(),number_joint_RV,poly_order,quadrature_intg_order)

evaluation_nodes=np.random.rand(number_joint_RV,100)
error={}

for model in models.keys():
    models[model].estimate_fourier_coefs()
    
    error[model]=models[model].get_error(evaluation_nodes)

for model in error:
    print(model,error[model])
