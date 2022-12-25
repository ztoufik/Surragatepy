import chaospy as cp
import numpy as np
#model
'''
def model(x,u0,c0,c1,c2):
    def c(x):
        if x<0.5: return c0
        elif 0.5<=x<0.7: return c1
        else:  return c2

    N=len(x)
    u=np.zeros(N)

    u[0]=u0
    for n in range(N-1):
        dx=x[n+1]-x[n]
        K1=-dx*u[n]*c(x[n])
        K2=-dx*u[n]+K1/2*c(x[n]+dx/2)
        u[n+1]=u[n]+K1+K2
    return u
'''
model=lambda arr:arr[1]

#parameters
normal_dist=cp.Normal(0.0,1.0)
number_joint_RV=3#memory delay parameters
quadrature_intg_order=10
hermite_order=5#polynomial orders
joint_normal_dist=cp.Iid(normal_dist,number_joint_RV)

#generate qudrature nodes
print("generate qudrature nodes")
nodes,weights=cp.generate_quadrature(quadrature_intg_order, joint_normal_dist, rule="gaussian")
nodes=nodes.round(3)
weights=weights.round(3)
print("nodes",nodes)
print("weights",weights)
#print("len(nodes[0])",len(nodes[0]))
#Evaluating model 
print("Evaluating model")
evals=[model(node) for node in nodes.T]

#polynomial expansion
print("polynomial expansion")
expansions = cp.generate_expansion(hermite_order, joint_normal_dist).round(3)
print("expansions:",expansions)

#Fourier coefficients
print("Fourier coefficients")
model_approx,coefs=cp.fit_quadrature(expansions, nodes, weights, evals,retall=True)
print("coefs:",coefs)
approx_evals = model_approx(*nodes).round(3)

#evaluations
print("appxorimation",approx_evals)
print("model",evals)

'''




hermite_polys=cp.generate_expansion(hermite_order, joint_normal_dist).round(5)
print(hermite_polys)
nodes,weights=cp.generate_quadrature(order=quadrature_intg_order,dist=joint_normal_dist,rule="Gaussian")
model_evals=np.array([model(node,0.3,0.5,0.5,0.03) for node in nodes.T])
fourier_coef=cp.fit_quadrature(hermite_polys,nodes,weights,model_evals)
print(fourier_coef)
x=np.linspace(0,1,100)
true_evals=model(x,0.3,0.5,0.5,0.03) 
approx_evals=fourier_coef(*x)
print("true evals:",true_evals)
print("approx evals:",approx_evals)
#print(np.linalg.norm(approx_evals-true_evals))
'''
