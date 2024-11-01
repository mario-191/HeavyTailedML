import numpy as np
import random as rand
from scipy import special
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize



def alpha_rotational_data_nd(n,alpha=1,beta=0,loc=0, scale=1,num=100000,seed=None):
#     loc = delta ; gamma = scale
    rng = np.random.RandomState(seed)
    mean = np.zeros(n)
    cov =np.diag(np.ones(n))*(2*scale**2)
    # np.random.RandomState.seed(seed)
    points = rng.multivariate_normal(mean, cov, num)
    # print(points)
    A = levy_stable.rvs(alpha/2, 1,0,math.cos(math.pi*alpha/4)**(2/alpha),size=num,random_state=seed) 
    A_half = np.sqrt(A)  
    # print(A_half)
    points= np.multiply(points, A_half[:, np.newaxis])
    mu = np.array(loc)
    points=points+ mu
    # print(points)
    return points

def alphaIID_data_nd(n,alpha=1,beta=0,loc=0, scale=1,num=100000,seed=None):
#     loc = delta ; gamma = scale

      
    points = levy_stable.rvs(alpha, beta,0,scale, size=n*num,random_state=seed)
   
    points = points.reshape(-1, n)
    mu = np.array(loc)
    points=points + mu
   
    
     
    # points = np.hstack((x_r_n.reshape(-1, 1), y_r_n.reshape(-1, 1)))
    return points

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def get_strength_new(W0,x, y,x_dim,y_dim):
    return W0[-1]

def get_params_st(W0,x, y,x_dim,y_dim):
    # bnds=((None, None),(None,None),(None, None),(None,None),(None, None),(None,None),(None, None),(None,None), (None, None),(None,None),(None, None),(None,None),(None, None),(None, None),(None,None),(None, None),(None,None), (None, None),(None,None),(None, None),(None,None),(None, None),(None, None),(None,None),(None, None),(None,None), (None, None),(None,None),(None, None),(None,None),(None, None),(None, None),(None,None),(None, None),(None,None), (None, None),(0,None))
    bn=((None,None),)
    bnds=bn*(x_dim*y_dim+y_dim)+((0,None),)
    #constraints=({'type': 'ineq', 'fun': -fa2(W0,x, y,alpha)}),
    # show_options(solver="minimize", method="trust-constr", disp=True)
    opts={"verbose":2,"gtol":1e-5,"xtol":1e-5,"maxiter":9500}
    result=minimize(get_strength_new,W0,options=opts, args=(x, y,x_dim,y_dim),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:cost_f1(W0,x, y,x_dim,y_dim) }) ,method='trust-constr')
    # print(f"constraint value {-fa2(result.x,x, y,alpha)}")
    print(result)
    return result.x

def cost_f1(W,x, y,x_dim,y_dim):
    eulergamma=0.577215664901533
    n_points = len(x)
    A=W[:-y_dim-1]
    print(A)
    A=A.reshape(y_dim,x_dim)
    B=W[-y_dim-1:-1]
    S=W[-1]

    # print(f"A:{A}")
    # print(f"B:{B}")
    predicted_y=x @ A.T+B
    total_error = 0.0
    for i in range(n_points):
        total_error +=np.log(1+(euclidean_distance(y[i],predicted_y[i]) ** 2)/S**2 )
        # total_error+= (euclidean_distance(y[i],predicted_y[i]) ** 2)
    

    return (total_error/n_points - special.digamma((y_dim+1)/2)-np.log(4)-eulergamma)



def train(x, y):
    if(np.array(x).ndim!=1):
        x_dim=np.shape(x)[-1]
    else:
        x_dim=1
    if(np.array(y).ndim!=1):
        y_dim=np.shape(y)[-1]
    else:
        y_dim=1
    W=np.ones(x_dim*y_dim+y_dim+1)
    
   
    
    W=get_params_st(W,x, y,x_dim,y_dim)
 
    # p=gamma_op(p,W,x, y,x_dim,y_dim)
 
    a=W[:-y_dim-1].reshape(y_dim,x_dim)
    b=W[-y_dim-1:-1]
    S=W[-1]
    
    print(f"weight={a} bias={b}  power={S}")
         
    return W
