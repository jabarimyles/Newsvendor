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
import random

seed_rand = np.random.RandomState(923)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
lead_time = 100
lead_times = [20,30,40]
# ATO set Cover
A_sc = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [1, 1, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1],
                  [1, 0, 1, 1, 0, 1, 0],
                  [0, 1, 1, 1, 1, 0, 0],
                  [1, 1, 0, 1, 0, 0, 1]])
[M, N] = A_sc.shape
c_sc = np.ones(M)
q_sc = 0.5*np.ones(N)
p_sc = 10*np.ones(N)

W = 1000
mu = 1/W*np.ones(W)
np.random.seed(2)
mean = 0.1739
d = np.random.binomial(1, mean, size=(W, N))

# NN set cover
A_nn = np.array([[1, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 1]])
B_nn = np.array([[1, 0, 0],
                 [1, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1],
                 [0, 0, 1]])
[M_nn, L_nn] = A_nn.shape
N_nn = B_nn.shape[1]
c_nn = np.ones(M_nn)
q_nn = 0.5*np.ones(L_nn)
p_nn = 10*np.ones(N_nn)
W_nn = 100
mu_nn = 1/W*np.ones(W_nn)
np.random.seed(2)
mean = 0.1
d_nn = np.random.binomial(1, mean, size=(W_nn, N_nn))

##### NN auto manufacturing with long chain #####
fac_counts = [3, 5, 7] # number of facilities and regions
flex_deg = 2 # flexibility degree, 2 for long chain, need to update code for anything else
# A_base = A_sc
A_auto = np.array([[1, 0, 0],
                   [0, 1, 1],
                   [1, 1, 0],
                   [0, 0, 1]])
A_ls = np.array([[1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 0, 0, 0, 1],
                  [0, 0, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 1],
                  [1, 1, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0, 1]])
np.random.seed(2)
rnd_M = 50
rnd_N = 20
A_rnd = 1*(np.random.random_sample(size=[rnd_M, rnd_N]) > .7)
systems = {'auto': A_auto, 'ls': A_ls, 'rnd': A_rnd}

c_list = {'auto': [2.5, 3, 1, 1.5],
          'ls': [1, 1.5, 0.5, 1, 2, 0.5, 0.5, 1.5, 1, 1, 1, 2, 1, 0.5],
          'rnd': np.random.random_sample(rnd_M) + 0.5}

q_list = {'auto': [1, 1.1, 1.3],
          'ls': [1, 1, 1.1, 1.1, 1.2, 1.2],
          'rnd': 1.5*np.random.random_sample(rnd_N) + 0.25}
q_nhb_mult = 1.7

markup_list = {'auto': [3, 5, 6],
               'ls': [2, 2, 3, 3.5, 4, 4],
               'rnd': 4*np.random.random_sample(rnd_N) + 3}

prod_dem_rates = {'auto': [.6, .3, .5],
                  'ls': [.4, .6, .6, .25, .25, .5],
                  'rnd': np.random.random_sample(rnd_N)}

dem_scale_manuf = [1, 5]



def manuf_system_gen(A_base, fac_n, flex_deg, c_base, q_base, q_nhb_mult, markup_base):
    [M_base, N_base] = A_base.shape
    # M_auto = M_base*fac_n
    L = N_base*flex_deg*fac_n
    # N_auto = N_base*fac_n
    A = np.kron(np.eye(fac_n, dtype=int), np.tile(A_base, flex_deg)) # block diagonal with repeating A_base matrix
    B_init = np.kron(np.eye(fac_n, dtype=int), np.tile(np.eye(N_base, dtype=int), (flex_deg, 1))) # initial diagonal matrix
    [B_split1, B_split2] = np.vsplit(B_init, [-N_base]) # split out last N_base rows
    B = np.block([[B_split2], [B_split1]]) # concatenate those rows to beginning to model long chain

    # c_base = np.array([2.5, 3, 1, 1.5])
    np.random.seed(0)
    c_mult = np.random.uniform(0, .2, size=fac_n) + .9 # scale costs to be slightly different at different facilities
    c = np.repeat(c_mult, M_base)*np.tile(c_base, fac_n)

    q = np.tile(np.concatenate((q_nhb_mult*np.array(q_base), q_base)), fac_n)

    act_cost = np.dot(c, A) + q
    prod_cost = np.reshape(act_cost, [L, 1])*B
    prod_cost_min = np.where(prod_cost>0, prod_cost, np.iinfo(int).max).min(axis=0)
    prod_cost_max = prod_cost.max(axis=0)
    prod_cost_marg = prod_cost_max/prod_cost_min
    prod_markup = np.repeat(c_mult, N_base)*np.tile(markup_base, fac_n)
    p = np.maximum(prod_markup, prod_cost_marg)*prod_cost_min
    return [A, q, B, p, c]

# q_auto = 0.5*np.ones(L_auto)
# p_auto = 10*np.ones(N_auto)

W_manuf = 1000
mu_manuf = 1/W_manuf*np.ones(W_manuf)
# np.random.seed(2)
# mean = 0.7
# d_auto = np.random.binomial(5, mean, size=(W_auto, N_auto))


#### Flexible Production ####

np.random.seed(2)
rnd_M_N = [[10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [20, 20], [50, 20]]


systems_prod = {'auto': A_auto, 'ls': A_ls}

c_list_prod = {'auto': [2.5, 3, 1, 1.5],
          'ls': [1, 1.5, 0.5, 1, 2, 0.5, 0.5, 1.5, 1, 1, 1, 2, 1, 0.5]}

q_list_prod = {'auto': [1, 1.1, 1.3],
          'ls': [1, 1, 1.1, 1.1, 1.2, 1.2]}

q_nhb_mult_prod = 1.7

markup_list_prod = {'auto': [3, 5, 6],
               'ls': [2, 2, 3, 3.5, 4, 4]}

dem_rates_prod = {'auto': [.6, .3, .5],
                  'ls': [.4, .6, .6, .25, .25, .5]}

dem_scale_prod = [1, 5]

for [rnd_M, rnd_N] in rnd_M_N:
    systems_prod['rnd_' + str(rnd_M) + '_' + str(rnd_N)] = 1*(np.random.random_sample(size=[rnd_M, rnd_N]) > .7)
    c_list_prod['rnd_' + str(rnd_M) + '_' + str(rnd_N)] = np.random.random_sample(rnd_M) + 0.5
    q_list_prod['rnd_' + str(rnd_M) + '_' + str(rnd_N)] = 1.5*np.random.random_sample(rnd_N) + 0.25
    markup_list_prod['rnd_' + str(rnd_M) + '_' + str(rnd_N)] = 4*np.random.random_sample(rnd_N) + 3
    dem_rates_prod['rnd_' + str(rnd_M) + '_' + str(rnd_N)] = np.random.random_sample(rnd_N)

W_prod = 1000
mu_prod = 1/W_prod*np.ones(W_prod)


def prod_system_gen(A_base, flex_deg, c_base, q_base, q_nhb_mult, markup_base):
    [M_base, N_base] = A_base.shape
    L = N_base*flex_deg
    A = np.empty([0, 0])
    c = np.empty(0)

    np.random.seed(0)
    plant_mult = np.random.uniform(0, .2, size=N_base) + .9  # scale costs to be slightly different at different facilities

    for j in range(N_base):
        comp_jm1_j = A_base[:, [(j-1) % N_base, j]]
        comp_used = np.sum(comp_jm1_j, axis=1) > 0
        comp_mat = comp_jm1_j[comp_used, :]
        [A_row, A_col] = A.shape
        [comp_row, comp_col] = comp_mat.shape
        A = np.block([[A, np.zeros([A_row, comp_col])],
                      [np.zeros([comp_row, A_col]), comp_mat]])
        c = np.concatenate((c, plant_mult[j]*np.array(c_base)[comp_used]))
    B_init = np.kron(np.eye(N_base, dtype=int), np.tile(np.eye(1, dtype=int), (flex_deg, 1)))  # initial diagonal matrix
    [B_split1, B_split2] = np.vsplit(B_init, [-1])  # split out last row
    B = np.block([[B_split2], [B_split1]])  # concatenate row to beginning to model long chain

    q_init = np.empty(flex_deg*len(q_base))
    q_init[0::2] = q_base
    q_init[1::2] = q_nhb_mult*np.array(q_base)
    [q_split1, q_split2] = np.split(q_init, [-1])  # split out last row
    q = np.concatenate((q_split2, q_split1))

    act_cost = np.dot(c, A) + q
    prod_cost = np.reshape(act_cost, [L, 1])*B
    prod_cost_min = np.where(prod_cost>0, prod_cost, np.iinfo(int).max).min(axis=0)
    prod_cost_max = prod_cost.max(axis=0)
    prod_cost_marg = prod_cost_max/prod_cost_min
    # prod_markup = np.repeat(c_mult, N_base)*np.tile(markup_base, fac_n)
    p = np.maximum(markup_base, prod_cost_marg)*prod_cost_min
    return [A, q, B, p, c]


def prod_system_gen_share_inv(A_base, flex_deg, c_base, q_base, q_nhb_mult, markup_base):
    [M_base, N_base] = A_base.shape
    L = N_base*flex_deg
    A = np.zeros([M_base, L])
    c = np.empty(0)

    np.random.seed(0)
    plant_mult = np.random.uniform(0, .2, size=N_base) + .9  # scale costs to be slightly different at different facilities

    for j in range(N_base):
        A[:, [flex_deg*j, flex_deg*j + 1]] = A_base[:, [(j-1) % N_base, j]]

    A = np.block([[A],[np.kron(np.eye(N_base, dtype=int), np.tile(np.eye(1, dtype=int), (1, flex_deg)))]] )
    c = np.concatenate((c_base, np.random.uniform(0, 2, size=N_base)+2))

    B_init = np.kron(np.eye(N_base, dtype=int), np.tile(np.eye(1, dtype=int), (flex_deg, 1)))  # initial diagonal matrix
    [B_split1, B_split2] = np.vsplit(B_init, [-1])  # split out last row
    B = np.block([[B_split2], [B_split1]])  # concatenate row to beginning to model long chain

    q_init = np.empty(flex_deg*len(q_base))
    q_init[0::2] = q_base
    q_init[1::2] = q_nhb_mult*np.array(q_base)
    [q_split1, q_split2] = np.split(q_init, [-1])  # split out last row
    q = np.concatenate((q_split2, q_split1))

    act_cost = np.dot(c, A) + q
    prod_cost = np.reshape(act_cost, [L, 1])*B
    prod_cost_min = np.where(prod_cost>0, prod_cost, np.iinfo(int).max).min(axis=0)
    prod_cost_max = prod_cost.max(axis=0)
    prod_cost_marg = prod_cost_max/prod_cost_min
    # prod_markup = np.repeat(c_mult, N_base)*np.tile(markup_base, fac_n)
    p = np.maximum(markup_base, prod_cost_marg)*prod_cost_min
    return [A, q, B, p, c]




#### E-commerce ####
item_counts = [2, 5, 10]
item_bundles = {2: [[0], [1], [0, 1]],
                5: [[0], [1], [2], [3], [4], [0, 1], [0, 2], [3, 4], [2, 3, 4]],
                10: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [0, 1], [0, 2], [3, 6], [5, 8], [4, 9], [7, 8], [2, 3, 4], [5, 7, 9]]}
reg_counts = [3, 5, 7]

ship_loc_list = {2: [1, 1.5],
                 5: [1, 1.5, 1.5, 1.6, 1.6],
                 10: [1, 1.5, 1.5, 1.6, 1.6, 1.7, 1.7, 1.8, 1.8, 2]} # per item local shipping cost
short_cst_list = {2: [5, 7],
                  5: [5, 7, 7, 9, 9],
                  10: [5, 7, 7, 9, 9, 11, 11, 13, 13, 15]} # per item shortage cost

item_cst_list = {2: [0.5, 0.75],
                 5: [0.5, 0.75, 0.75, 1, 1],
                 10: [0.5, 0.75, 0.75, 1, 1, 1.1, 1.1, 1.2, 1.2, 1.3]}

item_dem_rates = {2: [.4, .6],
                  5: [.4, .6, .6, .25, .25],
                  10: [.4, .6, .6, .25, .25, .5, .5, .75, .75, .1]}
dem_scale = [1, 5]

# item_n = 2
# reg_n = 3


# c_ecom = np.tile(item_cst, reg_n)
ship_fix_cst = 1 # fixed shipping cost per each warehouse shipped from
ship_nhb_mult = 1.5 # multiply by local for cost to ship from neighboring warehouse
# ship_loc_cst = [1, 1.5] # per item local shipping cost
# short_cst_item = [5, 7] # per item shortage cost
# bundles = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
# bundles = [[0], [1], [0, 1]]


def ecom_system_gen(reg_n, item_n, bundles, short_cst_item, ship_loc_cst, ship_nhb_mult, ship_fix_cst):
    resource_n = item_n * reg_n
    bundles_n = len(bundles)
    product_n = reg_n * bundles_n
    A = []
    B = []
    q = []
    p = []
    for reg in range(reg_n):
        warehouses = [-1, reg, (reg+1)%reg_n]  # -1 means not filled, modulo is to connect long chain back to beginning
        for bundle_count, bundle in enumerate(bundles):
            p.append(np.sum(np.array(short_cst_item)[bundle])) #shortage cost for entire bundle
            bundle_size = len(bundle)
            for comb in it.product(warehouses, repeat=bundle_size):
                resource_required = [0]*resource_n
                activity_cost = 0
                warehouse_counter = np.zeros(reg_n)
                for item_count, warehouse in enumerate(comb):
                    item = bundle[item_count]
                    if warehouse >= 0:
                        resource_required[warehouse*item_n + item] = 1
                        warehouse_counter[warehouse] += 1
                        activity_cost += ship_loc_cst[item]*(1 + (ship_nhb_mult - 1)*(warehouse != reg)) #local or neighbor shipping cost
                    else:
                        activity_cost += short_cst_item[item] #shortage cost
                activity_cost += ship_fix_cst*np.sum(warehouse_counter > 0)
                if np.sum(resource_required) > 0: # exclude activities corresponding to lost sale of entire bundle
                    A.append(resource_required)
                    product_filled = [0]*product_n
                    product_filled[reg*bundles_n + bundle_count] = 1
                    B.append(product_filled)
                    q.append(activity_cost)
    return [np.array(A).T, np.array(q), np.array(B), np.array(p)]

W_ecom = 1000
mu_ecom = 1/W_ecom*np.ones(W_ecom)
# np.random.seed(2)
dist_list = ['indep', 'pos_cor', 'neg_cor'] #same for both ecom and manuf
# mean = 0.3
# cor_p = np.random.dirichlet(np.ones(product_n), size=W_ecom)
# d_ecom = np.random.binomial(1, cor_p) #, size=(W_ecom, product_n))
#
# ecom_inst = {'c': c_ecom, 'q': q_ecom, 'p': p_ecom, 'd': d_ecom, 'mu': mu_ecom, 'A': A_ecom, 'B': B_ecom}
indep_rand = np.random.RandomState(0)
pos_cor_rand = np.random.RandomState(1)
neg_cor_rand = np.random.RandomState(2)



def prod_sim(load_prod):
    ### Production simulation ###
        output_prod = []
        row = 0
        for syst in systems_prod:
            if not load_prod:
                A_base = systems_prod[syst]
                c_base = c_list_prod[syst]
                q_base = q_list_prod[syst]
                markup_base = markup_list_prod[syst]
                prod_dem_rate = np.array(dem_rates_prod[syst])
                [A, q, B, p, c] = prod_system_gen_share_inv(A_base, flex_deg, c_base, q_base, q_nhb_mult, markup_base)
                product_n = len(prod_dem_rate)
            r_L = np.zeros(len(c))
            for i in range(len(c)):
                r_L[i] = seed_rand.choice(lead_times)
            for scale in dem_scale_prod:
                for dist in dist_list:
                    d_sol = np.zeros((W_prod, product_n))
                    if not load_prod:
                        prod_lead_times = np.matmul((A.T*r_L).T, B)
                        prod_avg_lead_times = np.round(prod_lead_times.sum(0)/(prod_lead_times != 0).sum(0))
                        max_lead_time = int(max(prod_avg_lead_times))
                        d_hold = np.zeros((max_lead_time, product_n))
                        for i in range(max_lead_time+1):
                            if dist == 'indep':
                                d = np.random.binomial(scale, prod_dem_rate, size=(W_prod, product_n))
                                for j in range(product_n):
                                    if i <= prod_avg_lead_times[j]+1:
                                        d_sol[:,j] += d[:,j] 
                            if dist == 'pos_cor':
                                var = scale * (np.identity(product_n) + 0.5 * (
                                            np.ones([product_n, product_n]) - np.identity(product_n)))
                                d = np.maximum(0,
                                            np.around(np.random.multivariate_normal(scale * prod_dem_rate, var, W_prod)))
                                for j in range(product_n):
                                    if i <= prod_avg_lead_times[j]+1:
                                        d_sol[:,j] += d[:,j] 
                            if dist == 'neg_cor':
                                cor_p = np.minimum(1, np.sum(prod_dem_rate) * np.random.dirichlet(prod_dem_rate, size=W_prod))
                                d = np.random.binomial(scale, cor_p)
                                for j in range(product_n):
                                    if i <= prod_avg_lead_times[j]+1:
                                        d_sol[:,j] += d[:,j] 
                        
                        prod_inst = {'c': c, 'q': q, 'p': p, 'd': d, 'mu': mu_prod, 'A': A, 'B': B, 'd_sol': d_sol, 'r_L':r_L}
                        [prod_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(prod_inst, var_type='C', return_xy=True)
                    # print("LP r:", r_opt)
                    # print("time:", opt_time)
                    # print("LP cost:", manuf_opt)
                        prod_lp_sol = {'r': r_opt, 'x': x_opt, 'y': y_opt, 'cost': prod_opt, 'time': opt_time}
                        with open("instances/prod" + str(row) + ".pkl", 'wb') as fpick:
                            pickle.dump({'instance': prod_inst, 'LP_solution': prod_lp_sol}, fpick)
                    else:
                        with open("instances/prod" + str(row) + ".pkl", 'rb') as fpick:
                            inst_sol = pickle.load(fpick)
                        prod_inst = inst_sol['instance']
                        prod_lp_sol = inst_sol['LP_solution']
                        [prod_opt, r_opt, x_opt, y_opt, opt_time] = [prod_lp_sol['cost'], prod_lp_sol['r'],
                                                                     prod_lp_sol['x'], prod_lp_sol['y'],
                                                                     prod_lp_sol['time']]

                    start = time.time()
                    [min_b, beta_cst, r_rnd] = utils.min_beta_grid(r_opt, x_opt, y_opt, prod_inst, max_iter=10) #, plot_on=True)
                    # print("Rounded r:", r_rnd)
                    # print("Approximation Factor:", beta_cst/manuf_opt)
                    beta_tm = time.time() - start

                    start = time.time()
                    [min_a, cor_cst, r_rnd] = utils.min_cor_grid(r_opt, x_opt, y_opt, prod_inst, max_iter=10)
                    cor_tm = time.time() - start

                    start = time.time()
                    [min_a, min_b, mrkv_cst, r_rnd] = utils.min_mrkv_grid(r_opt, x_opt, y_opt, prod_inst, max_iter=5)
                    mrkv_tm = time.time() - start

                    start = time.time()
                    [near_int_cst, r_rnd] = utils.rnd_near_int(r_opt, x_opt, y_opt, prod_inst)
                    near_int_tm = time.time() - start

                    output_prod.append([syst, 0, scale, dist, prod_opt, opt_time, beta_cst, cor_cst, mrkv_cst,
                                        near_int_cst, beta_tm, cor_tm, mrkv_tm, near_int_tm])
                    print(row)
                    row += 1

        labels = ['System', 'placeholder', 'demand_scale', 'distribution',
                  'LP_cost', 'LP_time', 'beta_cost', 'cor_cost', 'markov_cost', 'near_int_cost', 'beta_time',
                  'cor_time', 'markov_time', 'near_int_time']
        output_prod = pd.DataFrame(output_prod, columns=labels)
        flnm = 'prod_simulation_near_int.csv'
        output_prod.to_csv(flnm)

    
### Ecommerce simulation ###
def ecom_sim(load_ecom):
        output = []
        row = 0
        # if load_ecom:
        #     with open("instances/ecom" + str(row) + ".pkl", 'rb') as fpick:
        #         inst_sol = pickle.load(fpick)
        #     ecom_inst = inst_sol['instance']
        #     ecom_lp_sol = inst_sol['LP_solution']
        #     ecom_opt = inst_sol['cost']
        #     opt_time = inst_sol['time']

        for item_n in item_counts:
            if not load_ecom:
                bundles = item_bundles[item_n]
                short_cst_item = short_cst_list[item_n]
                ship_loc_cst = ship_loc_list[item_n]
                dem_rate = np.array(item_dem_rates[item_n])
                bundle_dem_rate = np.empty(len(bundles))
                for bundle_count, bundle in enumerate(bundles):
                    bundle_dem_rate[bundle_count] = np.prod(dem_rate[bundle])**(1.0/len(bundle))
            for reg_n in reg_counts:
                if not load_ecom:
                    product_dem_rate = np.tile(bundle_dem_rate, reg_n)
                    product_n = len(product_dem_rate)
                    c = np.tile(item_cst_list[item_n], reg_n)
                    [A, q, B, p] = ecom_system_gen(reg_n, item_n, bundles, short_cst_item, ship_loc_cst, ship_nhb_mult, ship_fix_cst)
                r_L = np.zeros(len(c))
                for i in range(len(c)):
                    r_L[i] = seed_rand.choice(lead_times)
                for scale in dem_scale:
                    for dist in dist_list:
                        d_sol = np.zeros((W_ecom, product_n))
                        if not load_ecom:
                            prod_lead_times = np.matmul((A.T*r_L).T, B)
                            prod_avg_lead_times = np.round(prod_lead_times.sum(0)/(prod_lead_times != 0).sum(0))
                            max_lead_time = int(max(prod_avg_lead_times))
                            d_hold = np.zeros((max_lead_time, product_n))
                            for i in range(max_lead_time+1):
                                if dist == 'indep':
                                    d = indep_rand.binomial(scale, product_dem_rate, size=(W_ecom, product_n))
                                    for j in range(product_n):
                                        if i <= prod_avg_lead_times[j]+1:
                                            d_sol[:,j] += d[:,j] 
                                if dist == 'pos_cor':
                                    var = scale*(np.identity(product_n) + 0.5*(np.ones([product_n, product_n]) - np.identity(product_n)))
                                    d = np.maximum(0, np.around(pos_cor_rand.multivariate_normal(scale*product_dem_rate, var, W_ecom)))
                                    for j in range(product_n):
                                        if i <= prod_avg_lead_times[j]+1:
                                            d_sol[:,j] += d[:,j] 
                                if dist == 'neg_cor':
                                    cor_p = np.minimum(1, np.sum(product_dem_rate)*neg_cor_rand.dirichlet(product_dem_rate, size=W_ecom))
                                    d = neg_cor_rand.binomial(scale, cor_p)
                                    for j in range(product_n):
                                        if i <= prod_avg_lead_times[j]+1:
                                            d_sol[:,j] += d[:,j] 
                            ecom_inst = {'c': c, 'q': q, 'p': p, 'd': d, 'mu': mu_ecom, 'A': A, 'B': B, 'd_sol': d_sol, 'r_L':r_L}
                            [ecom_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(ecom_inst, var_type='C', return_xy=True)
                            ecom_lp_sol = {'r': r_opt, 'x': x_opt, 'y': y_opt, 'cost': ecom_opt, 'time': opt_time}
                            with open("instances/ecom" + str(row) + ".pkl", 'wb') as fpick:
                                pickle.dump({'instance': ecom_inst, 'LP_solution': ecom_lp_sol}, fpick)
                        else:
                            with open("instances/ecom" + str(row) + ".pkl", 'rb') as fpick:
                                inst_sol = pickle.load(fpick)
                            ecom_inst = inst_sol['instance']
                            ecom_lp_sol = inst_sol['LP_solution']
                            [ecom_opt, r_opt, x_opt, y_opt, opt_time] = [ecom_lp_sol['cost'], ecom_lp_sol['r'],
                                                                         ecom_lp_sol['x'], ecom_lp_sol['y'],
                                                                         ecom_lp_sol['time']]

                        # print("LP r:", r_opt)
                        # print("time:", opt_time)
                        # print("LP cost:", ecom_opt)

                        start = time.time()
                        [min_b, beta_cst, r_rnd] = utils.min_beta(r_opt, x_opt, y_opt, ecom_inst, max_iter=10)
                        beta_tm = time.time() - start

                        start = time.time()
                        [min_a, cor_cst, r_rnd] = utils.min_cor_grid(r_opt, x_opt, y_opt, ecom_inst, max_iter=10)
                        cor_tm = time.time() - start

                        start = time.time()
                        [min_a, min_b, mrkv_cst, r_rnd] = utils.min_mrkv_grid(r_opt, x_opt, y_opt, ecom_inst, max_iter=5)
                        mrkv_tm = time.time() - start

                        start = time.time()
                        [near_int_cst, r_rnd] = utils.rnd_near_int(r_opt, x_opt, y_opt, ecom_inst)
                        near_int_tm = time.time() - start

                        output.append([item_n, reg_n, scale, dist, ecom_opt, opt_time, beta_cst, cor_cst, mrkv_cst,
                                       near_int_cst, beta_tm, cor_tm, mrkv_tm, near_int_tm])
                        print(row)
                        row += 1

        labels = ['items', 'regions', 'demand_scale', 'distribution',
                  'LP_cost', 'LP_time', 'beta_cost', 'cor_cost', 'markov_cost', 'near_int_cost', 'beta_time',
                  'cor_time', 'markov_time', 'near_int_time']
        output = pd.DataFrame(output, columns=labels)
        flnm = 'ecom_simulation_near_int.csv'
        output.to_csv(flnm)

### Manufacturing simulation ###
def manuf_sim(load_manuf):
        output_manuf = []
        row = 0
        # for syst in {'auto'}:
        for syst in systems:
            if not load_manuf:
                A_base = systems[syst]
                c_base = c_list[syst]
                q_base = q_list[syst]
                markup_base = markup_list[syst]
                prod_dem_rate = prod_dem_rates[syst]
            # for fac_n in [3, 5]:
            for fac_n in fac_counts:
                if not load_manuf:
                    [A, q, B, p, c] = manuf_system_gen(A_base, fac_n, flex_deg, c_base, q_base, q_nhb_mult, markup_base)
                    dem_rate_manuf = np.tile(prod_dem_rate, fac_n)
                    product_n = len(dem_rate_manuf)
                # for scale in [1, 5]:
                for scale in dem_scale_manuf:
                    # for dist in ['indep', 'pos_cor', 'neg_cor']:
                    for dist in dist_list:
                        d_sol = np.zeros((W_manuf, product_n))
                        if not load_manuf:
                            for i in range(lead_time+1):
                                if dist == 'indep':
                                    d = np.random.binomial(scale, dem_rate_manuf, size=(W_manuf, product_n))
                                    d_sol += d
                                if dist == 'pos_cor':
                                    var = scale * (np.identity(product_n) + 0.5 * (
                                                np.ones([product_n, product_n]) - np.identity(product_n)))
                                    d = np.maximum(0,
                                                np.around(np.random.multivariate_normal(scale * dem_rate_manuf, var, W_manuf)))
                                    d_sol += d
                                if dist == 'neg_cor':
                                    cor_p = np.minimum(1, np.sum(dem_rate_manuf) * np.random.dirichlet(dem_rate_manuf, size=W_manuf))
                                    d = np.random.binomial(scale, cor_p)
                                    d_sol += d
                            manuf_inst = {'c': c, 'q': q, 'p': p, 'd': d, 'mu': mu_manuf, 'A': A, 'B': B, 'd_sol': d_sol}
                            [manuf_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(manuf_inst, var_type='C', return_xy=True)
                        # print("LP r:", r_opt)
                        # print("time:", opt_time)
                        # print("LP cost:", manuf_opt)
                            manuf_lp_sol = {'r': r_opt, 'x': x_opt, 'y': y_opt, 'cost': manuf_opt, 'time': opt_time}
                            with open("instances/manuf" + str(row) + ".pkl", 'wb') as fpick:
                                pickle.dump({'instance': manuf_inst, 'LP_solution': manuf_lp_sol}, fpick)
                        else:
                            with open("instances/manuf" + str(row) + ".pkl", 'rb') as fpick:
                                inst_sol = pickle.load(fpick)
                            manuf_inst = inst_sol['instance']
                            manuf_lp_sol = inst_sol['LP_solution']
                            [manuf_opt, r_opt, x_opt, y_opt, opt_time] = [manuf_lp_sol['cost'], manuf_lp_sol['r'],
                                                                          manuf_lp_sol['x'], manuf_lp_sol['y'],
                                                                          manuf_lp_sol['time']]

                        start = time.time()
                        [min_b, beta_cst, r_rnd] = utils.min_beta_grid(r_opt, x_opt, y_opt, manuf_inst, max_iter=10) #, plot_on=True)
                        # print("Rounded r:", r_rnd)
                        # print("Approximation Factor:", beta_cst/manuf_opt)
                        beta_tm = time.time() - start

                        start = time.time()
                        [min_a, cor_cst, r_rnd] = utils.min_cor_grid(r_opt, x_opt, y_opt, manuf_inst, max_iter=10)
                        cor_tm = time.time() - start

                        start = time.time()
                        [min_a, min_b, mrkv_cst, r_rnd] = utils.min_mrkv_grid(r_opt, x_opt, y_opt, manuf_inst, max_iter=5)
                        mrkv_tm = time.time() - start

                        start = time.time()
                        [near_int_cst, r_rnd] = utils.rnd_near_int(r_opt, x_opt, y_opt, manuf_inst)
                        near_int_tm = time.time() - start

                        output_manuf.append([syst, fac_n, scale, dist, manuf_opt, opt_time, beta_cst, cor_cst, mrkv_cst,
                                             near_int_cst, beta_tm, cor_tm, mrkv_tm, near_int_tm])
                        print(row)
                        row += 1

        labels = ['System', 'facilities', 'demand_scale', 'distribution',
                  'LP_cost', 'LP_time', 'beta_cost', 'cor_cost', 'markov_cost', 'near_int_cost', 'beta_time',
                  'cor_time', 'markov_time', 'near_int_time']
        output_manuf = pd.DataFrame(output_manuf, columns=labels)
        flnm = 'manuf_simulation_near_int.csv'
        output_manuf.to_csv(flnm)

if __name__ == '__main__':
    prod_sim(False)
    #manuf_sim(False)
    ecom_sim(False)

    # G = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(sp.csr_matrix(A))
    # top = nx.bipartite.sets(G)[0]
    # pos = nx.bipartite_layout(G, top)
    # nx.draw(G, pos=pos)
    # plt.show()
    #
    # G2 = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(sp.csr_matrix(B))
    # top = nx.bipartite.sets(G2)[0]
    # pos = nx.bipartite_layout(G2, top)
    # nx.draw(G2, pos=pos)
    # plt.show()

    # sys.exit()

    # [ato_opt, r_opt, opt_time] = op.ato_opt(c_sc, q_sc, p_sc, d, mu, A_sc, var_type='C')
    # print(r_opt)
    # print("time:", opt_time)
    # print("cost:", ato_opt)

    # print(A_auto)
    # print(B_auto)

    # [nn_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(c_nn, q_nn, p_nn, d_nn, mu_nn, A_nn, B_nn, var_type='C', return_xy=True)
    # print(r_opt)
    # print("time:", opt_time)
    # print("cost:", nn_opt)
    #
    # [min_b, min_cst, r_rnd] = utils.min_beta(r_opt, x_opt, y_opt, c_nn, q_nn, p_nn, d_nn, A_nn, B_nn, mu_nn, max_iter=10)
    # print("rounded:", r_rnd)
    # print("rounded cots:", min_cst)
    # print("Beta:", min_b)

    # [auto_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(c_auto, q_auto, p_auto, d_auto, mu_auto, A_auto, B_auto, var_type='C', return_xy=True)
    # print("LP r:", r_opt)
    # print("time:", opt_time)
    # print("LP cost:", auto_opt)
    #
    # [min_b, min_cst, r_rnd] = utils.min_beta(r_opt, x_opt, y_opt, c_auto, q_auto, p_auto, d_auto, A_auto, B_auto, mu_auto, max_iter=10, plot_on=True)
    # print("beta rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("Beta:", min_b)
    #
    # # [min_b, min_cst, r_rnd] = utils.min_beta_grid(r_opt, x_opt, y_opt, c_auto, q_auto, p_auto, d_auto, A_auto, B_auto,
    # #                                          mu_auto, max_iter=10, plot_on=True)
    # # print("rounded:", r_rnd)
    # # print("rounded cost:", min_cst)
    # # print("Beta:", min_b)
    #
    # # [auto_opt, r_opt, opt_time] = op.nn_opt(c_auto, q_auto, p_auto, d_auto, mu_auto, A_auto, B_auto,
    # #                                                       var_type='I')
    # # print("opt r:", r_opt)
    # # print("opt time:", opt_time)
    # # print("opt cost:", auto_opt)
    #
    # [min_a, min_cst, r_rnd] = utils.min_cor_grid(r_opt, x_opt, y_opt, c_auto, q_auto, p_auto, d_auto, A_auto, B_auto,
    #                                          mu_auto, max_iter=10, plot_on=True)
    # print("correlated rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("alpha:", min_a)
    #
    # [min_a, min_b, min_cst, r_rnd] = utils.min_mrkv_grid(r_opt, x_opt, y_opt, c_auto, q_auto, p_auto, d_auto, A_auto, B_auto,
    #                                              mu_auto, max_iter=10)
    # print("markov rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("alpha:", min_a)
    # print("beta:", min_b)

    # [ecom_opt, r_opt, x_opt, y_opt, opt_time] = op.nn_opt(ecom_inst, var_type='C', return_xy=True)
    # print("LP r:", r_opt)
    # print("time:", opt_time)
    # print("LP cost:", ecom_opt)
    #
    # [min_b, min_cst, r_rnd] = utils.min_beta(r_opt, x_opt, y_opt, ecom_inst, max_iter=10, plot_on=True)
    # print("beta rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("Beta:", min_b)

    # [min_b, min_cst, r_rnd] = utils.min_beta_grid(r_opt, x_opt, y_opt, c_auto, q_auto, p_auto, d_auto, A_auto, B_auto,
    #                                          mu_auto, max_iter=10, plot_on=True)
    # print("rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("Beta:", min_b)

    # [auto_opt, r_opt, opt_time] = op.nn_opt(c_auto, q_auto, p_auto, d_auto, mu_auto, A_auto, B_auto,
    #                                                       var_type='I')
    # print("opt r:", r_opt)
    # print("opt time:", opt_time)
    # print("opt cost:", auto_opt)

    # [min_a, min_cst, r_rnd] = utils.min_cor_grid(r_opt, x_opt, y_opt, ecom_inst, max_iter=10, plot_on=True)
    # print("correlated rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("alpha:", min_a)
    #
    # [min_a, min_b, min_cst, r_rnd] = utils.min_mrkv_grid(r_opt, x_opt, y_opt, ecom_inst, max_iter=10)
    # print("markov rounded:", r_rnd)
    # print("rounded cost:", min_cst)
    # print("alpha:", min_a)
    # print("beta:", min_b)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
        
    
    
        

    
