import os
import pickle as pk

def load_data(file_name):
    with open(file_name,"rb") as f:
        out=pk.load(f)   
        print(f"loading {f.name} success")
        return out

def save_data(file_name,inputs):
    with open(file_name,"wb") as f:
        pk.dump(inputs,f)
        print(f"dumping {f.name} success")
