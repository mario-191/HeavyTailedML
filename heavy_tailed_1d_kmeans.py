import numpy as np
import random 
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize,fsolve
import json
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




def quartile_centroids(all_vals, K):
    centroids = []
    for i in range(K):
        centroids.append(np.percentile(all_vals, 100*(2*i+1)/(K*2)))
    # print(f"inital centroids picked: {centroids}")    
    return centroids
def assign_cluster(all_vals, centroids,gamma ):
    assignments = []
    for data_point in all_vals:
        dist_point_clust = []
        # print(centroids)
        for centroid in centroids:

            d_clust =  euclidean_distance(np.array(data_point),np.array(centroid))
        
            dist_point_clust.append(d_clust)
        
        assignment = np.argmin(dist_point_clust)
        assignments.append(assignment)
    
    return assignments
def euclidean_distance(point1, point2):
    
    square_dif = (point1 - point2) ** 2
    return np.sqrt(square_dif)
def new_centroids(all_vals, centroids, assignments, K,alpha,gammas,strengths,method="cauchy"):
    new_centroids = []
    new_strengths=[]
    for i in range(K):
        pt_cluster = []
        
        
        for x in range(len(all_vals)):
                if (assignments[x] == i):
                    pt_cluster.append(all_vals[x])
        print("###########################xxxxxxxxx")
        if (method=="alpha"):
            W0=np.array([centroids[i],strengths[i]])
            mean_c= get_mu_s(pt_cluster,W0,alpha)
            new_centroids.append(mean_c[0])
            new_strengths.append(mean_c[-1])
            # mean_c=ln_mean_opalpha(pt_cluster,centroids[i],alpha,gammas[i])
            # new_centroids.append(mean_c[0][0])
            # new_inertias.append(mean_c[1])
        elif(method=="cauchy"):
            W0=np.array([centroids[i],strengths[i]])
            mean_c= get_mu_sc(pt_cluster,W0)
            new_centroids.append(mean_c[0])
            new_strengths.append(mean_c[-1])
        # elif(method=="standard"):
        #     mean_c=sum(pt_cluster)/len(pt_cluster)
        #     new_centroids.append(mean_c)
        #     inertia = sum_errors(pt_cluster,mean_c)
        #     new_inertia.append(inertia)
        # elif(method=="kmedian"):
        #      median_c=kmedian(pt_cluster,centroids[i])
        #      new_centroids.append(median_c[0])
        #      inertia = sum_errors(pt_cluster,median_c)
        #      new_inertia.append(inertia)
            
        # mean_c=ln_mean_opa(pt_cluster,centroids[i],gammas[i])
        # mean_c=sum(pt_cluster)/len(pt_cluster)
        # new_centroids.append(mean_c) 
        # new_centroids.append(mean_c[0][0])
        # new_inertias.append(mean_c[1])
        # gam=gamma_opalpha(mean_c,pt_cluster,alpha,gammas[i])
        # gam=gamma_op(mean_c,pt_cluster,gammas[i])

        # gam=1
        # print(f"gamma_op gave {gam} as output")
        # new_gammas.append(gam)
        
    
    print(f"the centroids are: {new_centroids}")
    
    return (new_centroids,new_strengths)
def get_mu_sc(x,W0):
    bn=((None,None),)
    bnds=bn+((0,None),)
    #constraints=({'type': 'ineq', 'fun': -fa2(W0,x, y,alpha)}),
    # show_options(solver="minimize", method="trust-constr", disp=True)
    opts={"verbose":0,"gtol":1e-5,"xtol":1e-5,"maxiter":9500}
    result=minimize(get_strength_new,W0,options=opts, args=(x),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:cost_fun_genc(W0,x) }) ,method='trust-constr')
    # print(f"constraint value {-fa2(result.x,x, y,alpha)}")
    # print(result)
    return result.x
def cost_fun_genc(W,x):
    n_points = len(x)
    mu=W[0]
    print(mu)

    S=W[-1]
    sum = 0.0
    for i in range(n_points):
        sum +=np.log(1+(euclidean_distance(x[i],mu) ** 2)/S**2 ) 
        # total_error+= (euclidean_distance(y[i],predicted_y[i]) ** 2)
    
    # sum=np.sum(np.log(1+((x-mu)/S)**2))    
    return sum/n_points -np.log(4)   


def get_mu_s(x,W0,alpha):
    bn=((None,None),)
    bnds=bn+((0,None),)
    #constraints=({'type': 'ineq', 'fun': -fa2(W0,x, y,alpha)}),
    # show_options(solver="minimize", method="trust-constr", disp=True)
    opts={"verbose":0,"gtol":1e-5,"xtol":1e-5,"maxiter":9500}
    result=minimize(get_strength_new,W0,options=opts, args=(x),bounds=bnds,
                    constraints=({'type': 'eq', 'fun':lambda W0:cost_fun_gen(W0,x,alpha) }) ,method='trust-constr')
    # print(f"constraint value {-fa2(result.x,x, y,alpha)}")
    # print(result)
    return result.x
def get_strength_new(W0,x):
    return W0[-1]

def cost_fun_gen(W,x,alpha):
    n_points = len(x)
    mu=W[:-1]
    print(mu)

    S=W[-1]
    sum = 0.0
    # for i in range(n_points):
        # total_error +=np.log(1+(euclidean_distance(x[i],mu) ** 2)/S**2 ) 
        # total_error+= (euclidean_distance(y[i],predicted_y[i]) ** 2)
    # for i in range(n_points):
    pdf_x=levy_stable.pdf((x-mu)/S, alpha, 0, loc=0, scale=(1/alpha)**(1/alpha))
        # if (pdf_x>10**(-9)):
        
            # sum-=levy_stable.logpdf((x-nu)/gamma, alpha, 0, loc=0, scale=(1))
    sum=-np.sum(np.log(pdf_x))
        
    return sum/n_points -entropy(alpha,gamma=(1/alpha)**(1/alpha))  

def entropy(alpha,gamma=1):
    try:
        
        entro=lookup_table_entropy[alpha]
        return entro+np.log(gamma)
    except ValueError:
        print("key not found in the dictionary.")
        return 0
def get_inertia_gen(alpha,assignments, points, centroids):
    def f_inertia(S):
        total_sum = 0
        nb_points = len(points)
        # Iterate over each point
        for i, point in enumerate(points):
            # Get the corresponding centroid index from the assignments
            centroid_index = assignments[i]

            # Calculate the difference between the point and its assigned centroid
            difference = point - centroids[centroid_index]

            # Apply the function f to the difference and accumulate the result
            total_sum -= np.log(levy_stable.pdf((difference)/S, alpha, 0, loc=0, scale=(1/alpha)**(1/alpha)))

        return total_sum/nb_points - entropy(alpha,gamma=(1/alpha)**(1/alpha))  
    root_S = fsolve(f_inertia, x0=1)  # x0 is the initial guess for 'a'
    return root_S[0] 

def get_inertia_cauchy(assignments, points, centroids):
    def f_inertia(S):
        total_sum = 0
        nb_points = len(points)
        # Iterate over each point
        for i, point in enumerate(points):
            # Get the corresponding centroid index from the assignments
            centroid_index = assignments[i]

            # Calculate the difference between the point and its assigned centroid
            difference = point - centroids[centroid_index]

            # Apply the function f to the difference and accumulate the result
            total_sum += np.log(1+(difference ** 2)/S**2 ) 

        return total_sum/nb_points - np.log(4)  
    root_S = fsolve(f_inertia, x0=1)  # x0 is the initial guess for 'a'
    return root_S[0] 

def kmeans_clustering(all_vals,K,alpha=1,max_iter = 100, tol = pow(10,-6),method="cauchy" ):
    it = -1
    all_se = np.array([])
    assignments = []
    strengths=np.ones(K)

    print(np.size(all_vals))
    
    centroids = quartile_centroids(all_vals, K)
        
    #Place K centroids at random locations
    # centroids = median_centroids(all_vals, K)
    # centroids = quartile_centroids(all_vals, K)
    # centroids = all_vals[np.random.choice(range(len(all_vals)),K, replace=False)]
    print("the initial centroids are: ",centroids)
    gammas=np.ones(K)
    #Until algorithm converges (needs two iterations before comparing the errors)
    while (len(all_se)<=1 or (it < max_iter and np.absolute((all_se[it] - all_se[it-1])/all_se[it-1]) >= tol)):
        it += 1
        #Assign all data points to the closest center
        assignments = assign_cluster(all_vals, centroids,gammas)
        
        #Compute the new centroids
        V = new_centroids(all_vals, centroids, assignments, K,alpha,gammas,strengths,method=method)
        centroids =np.array(V[0])
        strengths=V[1]
        
        if (method=="alpha"):    
            inertia=get_inertia_gen(alpha,assignments, all_vals, centroids)

        elif(method=="cauchy"):
            inertia=get_inertia_cauchy(assignments, all_vals, centroids)
        #Compute SE
        # se_kmeans=np.sum(inertias)
        # se_kmeans = sum_errors(all_vals, assignments, centroids,alpha=1)
        print((inertia))
        all_se=np.append(all_se, inertia)   
        
     
    return (assignments,centroids,strengths,inertia, all_se)
