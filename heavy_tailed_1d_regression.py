import numpy as np
from scipy.stats import levy_stable,norm
import math
from scipy.optimize import minimize
from scipy.optimize import brentq
from scipy.optimize import newton
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.optimize import show_options
import json
import time
import concurrent.futures as futures
file_path = "lookup_table_entropy_g1(1.1).json"

# Initialize an empty dictionary
lookup_table_entropy = {}
try:
    with open(file_path, "r") as json_file:
        lookup_table_entropy = json.load(json_file)
        # print("Dictionary loaded from JSON file:")
        # print(lookup_table_ro[0])
except FileNotFoundError:
    print(f"The file {file_path} does not exist.")

lookup_table_entropy = {float(key): value for key, value in lookup_table_entropy.items()}

def entropy(alpha,gamma):
    try:
        
        entro=lookup_table_entropy[alpha]
        return entro + np.log(gamma)
    except ValueError:
        print("key not found in the dictionary.")
        return 0
    
def get_strength_new(W0,x, y,alpha):
    return W0[-1]

def get_params_g(W0,x, y,alpha):
    bnds=((None, None), (None, None),(0,None))

    opts={"verbose":2,"gtol":1e-12,"xtol":1e-12,"maxiter":3000}
    result=minimize(get_strength_new,W0,options=opts, args=(x, y,alpha),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:fa1(W0,x, y,alpha)}) ,method='trust-constr')
    # print(f"constraint value {-fa2(result.x,x, y,alpha)}")
    print(result)
    return result.x

def fa1(W0,x, y,alpha):
    n_points = len(x)
    sum=0
    
    A=W0[0]
    B=W0[1]
    S=W0[2]
    for i in range(n_points):
        temp=levy_stable.pdf((y[i] - (A * x[i] + B))/S, alpha, 0, loc=0, scale=(1/alpha)**(1/alpha))
        if (temp>10**(-64)):
        # sum-=levy_stable.logpdf((x-nu)/gamma, alpha, 0, loc=0, scale=(1))
            sum-=np.log(temp) 
    gamma=(1/alpha)**(1/alpha)
    return sum/n_points -entropy(alpha,gamma) 


def get_params_c(W0,x, y,alpha):
    x_dim=1
    y_dim=1
    bn=((None,None),)
    bnds=bn*(x_dim*y_dim + y_dim)+((0,None),)
    opts={"verbose":-1,"gtol":1e-11,"xtol":1e-11,"maxiter":3000}
    result=minimize(get_strength_new,W0,options=opts, args=(x, y,alpha),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:cost_f1(W0,x, y)}) ,method='trust-constr')
    print(result)
    return result.x
def cost_f1(W,x, y):
    n_points = len(x)
    A=W[0]
    print(A)
    B=W[1]    
    S=W[-1]   
    predicted_y=x*A+B
    sum=0
    # for i in range(n_points):
        # sum +=np.log(1+((y[i] - (A * x[i] + B)) ** 2)/S**2 ) 
    sum=np.sum(np.log(1+((y -  predicted_y) ** 2/S**2) ))
    return sum/n_points -math.log(4)

def train(x, y,alpha,method="cauchy"):
    W0=np.array([1,1,1])    
    if method=="cauchy":
        W0=get_params_c(W0,x, y,alpha)
    elif method=="alpha":
        W0=get_params_g(W0,x, y,alpha)
    print(f" weight={W0}")        
    return W0
