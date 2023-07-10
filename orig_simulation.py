from gurobipy import *
import numpy as np
import time
import optimals as op
import utils
import sys
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
import pandas as pd
import pickle
from optimals import sec_stg, ato_opt, nn_opt
import os

cwd = os.getcwd()

Ns = [50, 100]

CVs = [.5, 1, 2]

our_policy = False

dist_list = ['indep', 'pos', 'neg']


indep_rand = np.random.RandomState(0)
pos_cor_rand = np.random.RandomState(1)
neg_cor_rand = np.random.RandomState(2)


def construct_AB(n, dem_mean=10, cov = {}, samples=1000, c_cost=1, p_cost=8, cv=1, dist = '', our_policy = False):
    out_dir = cwd + "/newpickle"
    
    A = np.empty((n, n**2), int)
    B = np.array([])
    np.array([]).reshape((0,3))

    mean_vec = np.array([dem_mean] * n)


    d = []
    zero_vec = np.array([0])
    one_vec = np.array([1])
    
    if our_policy:
        for i in range(n):
            hold = np.repeat(zero_vec, [n*i], axis=0)
            hold2 = np.concatenate((hold,np.repeat(one_vec, [n])))
            hold3 = np.concatenate((hold2, np.repeat(zero_vec, [n*(n-i-1)])))
            A[i,] = hold3

            #A = np.concatenate((A, hold3))

        for k in range(n):
            if len(B) == 0:
                B = np.identity(n)
            else:
                B=np.concatenate((B,np.identity(n)))
    else:
        A = np.identity(n)

        B =  np.identity(n)

    stddev = (dem_mean*cv)

    variance = stddev**2

#np.maximum(0, np.around(pos_cor_rand.multivariate_normal(scale*product_dem_rate, var, W_ecom))) n sample = 1000

#ordering cost = 1, holding cost = .05*ordering cost, bl cost = 5,10


    if dist == 'indep':
        d = np.maximum(indep_rand.multivariate_normal(mean=mean_vec, cov=np.identity(n)*variance,size=samples),0)
    if dist == 'pos':
        d =    np.maximum(pos_cor_rand.multivariate_normal(mean=mean_vec, cov=variance*(np.identity(n) + 0.5*(np.ones([n,n]) - np.identity(n))),size=samples),0)
    if dist == 'neg': 
        d = np.maximum(neg_cor_rand.multivariate_normal(mean=mean_vec, cov=variance*(np.identity(n) + (-1/(n-1))*(np.ones([n,n]) - np.identity(n))),size=samples),0)

    mu = (1/samples) * np.ones(samples)

    c = np.ones(n) * c_cost

    p = np.ones(n) * p_cost
    

    radius  = 1  #initialize circle radius, unit circle is fine for now, we can change it later if needed

    coord = {}  #initialize coordinates dictionary

    rad_unit = 2*np.pi/n  # radians between each warehouse on circle

    for i in range(n):

        coord[i] = np.array([radius*np.cos(i*rad_unit), radius*np.sin(i*rad_unit)])

    

    fix_cst = 0.5  # add some fixed cost to each distance b/c otherwise a warehouse to its own region has zero cost

    var_cst = 1  # multiplier times the distance to determine the distance based cost

    fulf_costs = np.zeros((n,n)) # this code was for a setting where the fulfillment costs could be stored in a matrix like this, but we’ll need to flatten it in the same way as the length n^2 activity vector

    for j in range(n):
        for i in range(n):
            fulf_costs[i,j] = np.linalg.norm(coord[j]-coord[i]) + fix_cst
    
    if our_policy:
        q  = fulf_costs.flatten()

    else:
        q = np.ones(n) * fix_cst

    instance = {'c': c, 'q': q, 'p': p, 'd': d, 'mu': mu, 'A': A, 'B':B}

    [cost, r, x, y, time] = nn_opt(instance, var_type='C', return_xy=True)

    LP_solution = {'cost':cost, 'r':r, 'x':x, 'y':y, 'time':time} 

    instance_pkl = {'instance': instance, 'LP_solution':LP_solution}

    #with open("simpleinstances/ecom" + str(row) + ".pkl", 'wb') as fpick:
    #    pickle.dump({'instance': ecom_inst, 'LP_solution': ecom_lp_sol}, fpick)

    with open( out_dir + '/simple_network' + str(n) + 'cv' + str(int(cv*10)) + dist + '.pkl', 'wb') as handle:
        pickle.dump(instance_pkl, handle)

    
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for dist in dist_list:
        for cv in CVs:
            for n in Ns:
                construct_AB(n=n, cv=cv, dist = dist, our_policy=our_policy)
    