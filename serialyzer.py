import pickle as pk
import numpy as np
import chaospy as cp


if __name__=="__main__":

    file_name="cache.pi"
    #dumping

    with open(file_name,"wb") as f:
        q0, q1 = cp.variable(2)
        poly = cp.polynomial([1, q0**2, q0*q1])
        print("poly=",poly)
        pk.dump(poly,f)
        print("dumping success")

    #loading
    with open(file_name,"rb") as f:
        loaded_poly=pk.load(f)
        print("loading success")
        print("poly=",loaded_poly)
        print(loaded_poly(2,1))
