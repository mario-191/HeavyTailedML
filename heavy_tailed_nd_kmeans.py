import numpy as np
import random as rand
from scipy import special
from scipy.stats import levy_stable,ortho_group
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize,fsolve
import sympy
import json
import concurrent.futures as futures

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
  """
  Calculates the Euclidean distance between two points in n dimensions.

  Args:
    p1: A list of n numbers representing the coordinates of the first point.
    p2: A list of n numbers representing the coordinates of the second point.

  Returns:
    The Euclidean distance between the two points.
  """



  return np.linalg.norm(a - b)

def quartile_centroids(n,all_vals, K):
    centroids = []
    for j in range(n):
        for i in range(K):
            centroids.append(np.percentile(all_vals[:, j], 100*(2*i+1)/(K*2)))
    # print(f"inital centroids picked: {centroids}")   
    centroids=np.array(centroids)
    centroids = (centroids.reshape(n, K)).T
    print(centroids)  
    return centroids

def get_strength_new(W0,x):
    return W0[-1]

def get_mu_s(x,W0):
    print(f"pppppppppppppppppp{np.shape(x)}")

    x_dim=np.shape(x)[1]
    bn=((None,None),)
    bnds=bn*(x_dim)+((0,None),)
    #constraints=({'type': 'ineq', 'fun': -fa2(W0,x, y,alpha)}),
    # show_options(solver="minimize", method="trust-constr", disp=True)
    opts={"verbose":2,"gtol":1e-5,"xtol":1e-5,"maxiter":9500}
    result=minimize(get_strength_new,W0,options=opts, args=(x),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:cost_fun_cauchy(W0,x,x_dim) }) ,method='trust-constr')
    # print(f"constraint value {-fa2(result.x,x, y,alpha)}")
    print(result)
    return result.x
def cost_fun_cauchy(W,x,x_dim):
    
    n_points = len(x)
    mu=W[:-1]
    print(mu)

    S=W[-1] 
    
    total_error = 0.0
    for i in range(n_points):
        total_error +=np.log(1+(euclidean_distance(x[i],mu) ** 2)/S**2 ) 
       
    

    return (total_error/n_points - special.digamma((x_dim+1)/2)-np.log(4)+special.digamma(1))
def new_centroids(all_vals, centroids, assignments,strengths, K):
    new_centroids = []
    new_strengths=[]
    for i in range(K):
        pt_cluster = []
        
        for x in range(len(all_vals)):
                if (assignments[x] == i):
                    pt_cluster.append(all_vals[x])
        print("###########################xxxxxxxxx")
        print(centroids[i])
        W0=np.array(centroids[i])
        
        W0=np.append(W0,strengths[i])
        mean_c=get_mu_s(pt_cluster,W0)
        # mean_c=np.mean(pt_cluster)
        print(f"meanC={mean_c[:-1]}")
        
        new_centroids.append(mean_c[:-1])
        # gam=gamma_op(2,mean_c,pt_cluster,gammas[i])
        
        # print(f"gamma_op gave {gam} as output")
        new_strengths.append(mean_c[-1])
        
    
    # print(f"the centroids are: {new_centroids}")
    # print(f"the gammas are: {new_gammas}")
    return (new_centroids,new_strengths)

def get_inertia_cauchy(assignments, points, centroids):
    def f_inertia(S):
        total_sum = 0
        x_dim=np.shape(points)[-1]
        nb_points = len(points)
        # Iterate over each point
        for i, point in enumerate(points):
            # Get the corresponding centroid index from the assignments
            centroid_index = assignments[i]

            # Calculate the difference between the point and its assigned centroid
            difference = euclidean_distance(point , centroids[centroid_index])

            # Apply the function f to the difference and accumulate the result
            total_sum += np.log(1+(difference ** 2)/S**2 ) 

        return (total_sum/nb_points - special.digamma((x_dim+1)/2)-np.log(4)+special.digamma(1)) 
    root_S = fsolve(f_inertia, x0=1)  # x0 is the initial guess for 'a'
    return root_S[0] 

def assign_cluster(all_vals, centroids ):
    assignments = []
    for data_point in all_vals:
        dist_point_clust = []

        for centroid in centroids:
            d_clust =  euclidean_distance(np.array(data_point),np.array(centroid))
        
            dist_point_clust.append(d_clust)
        
        assignment = np.argmin(dist_point_clust)
        assignments.append(assignment)
    
    return assignments
  
    

def kmeans_clustering(all_vals,K,max_iter = 100, tol = pow(10,-6),type=None ):
    it = -1
    n=np.shape(all_vals)[1]
    all_se = []
    assignments = []
    strengths=np.ones(K)
    print(np.size(all_vals))
    #Place K centroids at random locations
    
    centroids = quartile_centroids(n,all_vals, K)
    print(f"initial{centroids}")
    print(np.shape(centroids))
    # centroids=random_k_points(all_vals, K)
    # gammas=np.ones(K)
    #Until algorithm converges (needs two iterations before comparing the errors)
    while (len(all_se)<=1 or (it < max_iter and np.absolute((all_se[it] - all_se[it-1])/all_se[it-1]) >= tol)):
        it += 1
        #Assign all data points to the closest center
        assignments = assign_cluster(all_vals, centroids)
        
        #Compute the new centroids
        V = new_centroids(all_vals, centroids, assignments,strengths, K)
        centroids =np.array(V[0])
        strengths =np.array(V[1])
         
        inertia=get_inertia_cauchy(assignments, all_vals, centroids)
       
        all_se=np.append(all_se,inertia)
        
        #Compute SE
        # se_kmeans = sum_errors(all_vals, assignments, centroids)
        # all_se.append(se_kmeans)
    # centroids=np.array(centroids)
    sort=centroids[:, 0].argsort()
    centroids=centroids[sort]        
    strengths=strengths[sort]
     
    return (assignments,centroids,strengths,inertia,all_se)
