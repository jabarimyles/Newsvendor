import time
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import copy


def obj_cost(r, x, y, c, q, p, mu):
    return np.dot(c, r) + np.dot(mu,np.dot(x, q)) + np.dot(mu,np.dot(y, p))


def beta_rnd_cost(beta, r, x_max_ind, y, c, q, p, d, A, B, mu, return_r = False):
    prod_thresh = y > (1 - beta) * d
    x_rnd = x_max_ind*np.dot(np.logical_not(prod_thresh)*d, B.T)
    y_rnd = d*prod_thresh
    r_rnd = np.amax(np.dot(x_rnd, A.T), axis=0) # use maximum usage in x rounding as inventory
    # delta = np.amax(np.sum(B,axis=0))
    # r_rnd = np.floor(delta*r / beta) # use scaling and rounding as inventory
    if return_r:
        return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]
    else:
        return obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu)


def beta_rnd_cost_frac(beta, r, x_floor, x_max_ind, y_floor, y_frac, c, q, p, d_frac, A, B, mu, return_r=False):
    prod_thresh = y_frac > (1 - beta) * d_frac
    x_rnd = x_max_ind*np.dot(np.logical_not(prod_thresh)*d_frac, B.T) + x_floor
    y_rnd = d_frac*prod_thresh + y_floor
    r_rnd = np.amax(np.dot(x_rnd, A.T), axis=0) # use maximum usage in x rounding as inventory
    # delta = np.amax(np.sum(B,axis=0))
    # r_rnd = np.floor(delta*r / beta) # use scaling and rounding as inventory
    if return_r:
        return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]
    else:
        return obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu)


def min_beta(r, x, y, nn_inst, max_iter, plot_on=None):
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']
    x_max_ind = np.zeros_like(x)
    for j in range(len(p)):
        max_ind = np.argmax(B[:, j] * x - 1 + B[:, j], axis=1) # choose maximum x_k among k serving j, set remaining k to -1 so they aren't chosen
        x_max_ind[np.arange(len(x)), max_ind] = 1

    tol = 0.0001
    lft_b = 0.01
    rgt_b = 0.99
    gr_splt = 0.3819660112501051 # equals (3-np.sqrt(5))/2, to split interval from left endpoint into golden ratio
    lft_cst = beta_rnd_cost(lft_b, r, x_max_ind, y, c, q, p, d, A, B, mu)
    rgt_cst = beta_rnd_cost(rgt_b, r, x_max_ind, y, c, q, p, d, A, B, mu)
    if lft_cst <= rgt_cst:
        min_b = lft_b
        min_cst = lft_cst
        max_b = rgt_b
        max_cst = rgt_cst
    else:
        min_b = rgt_b
        min_cst = rgt_cst
        max_b = lft_b
        max_cst = lft_cst

    if plot_on:
        plot_mat = [[lft_b, lft_cst], [rgt_b, rgt_cst]]

    # print(lft_cst)
    # print(rgt_cst)

    k = 1
    while (k <= max_iter) and (max_cst - min_cst >= tol*abs(max_b - min_b)):
        # print(k)
        if (min_b - lft_b >= rgt_b - min_b): #split to left of min
            splt_b = lft_b + gr_splt*(min_b - lft_b)
            splt_cst = beta_rnd_cost(splt_b, r, x_max_ind, y, c, q, p, d, A, B, mu)
            if plot_on:
                plot_mat.append([splt_b, splt_cst])
            if splt_cst < min_cst: #split becomes new min
                rgt_b = min_b
                rgt_cst = min_cst
                min_b = splt_b
                min_cst = splt_cst
                max_b = lft_b
                max_cst = lft_cst
            else: #split becomes new left
                lft_b = splt_b
                lft_cst = splt_cst
                if lft_cst <= rgt_cst:
                    max_b = rgt_b
                    max_cst = rgt_cst
                else:
                    max_b = lft_b
                    max_cst = lft_cst
        else: #split to right of min
            splt_b = min_b + gr_splt*(rgt_b - min_b)
            splt_cst = beta_rnd_cost(splt_b, r, x_max_ind, y, c, q, p, d, A, B, mu)
            if plot_on:
                plot_mat.append([splt_b, splt_cst])
            if splt_cst < min_cst: #split becomes new min
                lft_b = min_b
                lft_cst = min_cst
                min_b = splt_b
                min_cst = splt_cst
                max_b = rgt_b
                max_cst = rgt_cst
            else: #split becomes new right
                rgt_b = splt_b
                rgt_cst = splt_cst
                if lft_cst <= rgt_cst:
                    max_b = rgt_b
                    max_cst = rgt_cst
                else:
                    max_b = lft_b
                    max_cst = lft_cst
        k += 1
        # print(min_cst)
        # print(max_cst)

    if plot_on:
        plot_np = np.array(plot_mat)
        plot_np = plot_np[plot_np[:,0].argsort()]
        plt.plot(plot_np[:,0], plot_np[:,1])
        plt.show()#block=False)
        
    [min_cst, r_rnd] = beta_rnd_cost(min_b, r, x_max_ind, y, c, q, p, d, A, B, mu, return_r = True)
    return [min_b, min_cst, r_rnd]


def min_beta_grid(r, x, y, nn_inst, max_iter, plot_on=None):
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    [x_frac, x_floor] = np.modf(x)
    [y_frac, y_floor] = np.modf(y)
    d_frac = d - y_floor - np.dot(x_floor, B)
    x_max_ind = np.zeros_like(x_frac)
    for j in range(len(p)):
        max_ind = np.argmax(B[:, j] * x_frac - 1 + B[:, j],
                            axis=1)  # choose maximum x_k among k serving j, set remaining k to -1 so they aren't chosen
        x_max_ind[np.arange(len(x_frac)), max_ind] = 1

    if plot_on:
        plot_mat = []
    lft_b = 0.01
    rgt_b = 0.99
    beta_list = np.linspace(lft_b, rgt_b, max_iter)
    min_cst = np.inf
    for beta in beta_list:
        cst = beta_rnd_cost_frac(beta, r, x_floor, x_max_ind, y_floor, y_frac, c, q, p, d_frac, A, B, mu)
        if plot_on:
            plot_mat.append([beta, cst])
        if cst < min_cst:
            min_cst = cst
            min_b = beta

    if plot_on:
        plot_np = np.array(plot_mat)
        plot_np = plot_np[plot_np[:, 0].argsort()]
        plt.plot(plot_np[:, 0], plot_np[:, 1])
        plt.show()  # block=False)

    [min_cst, r_rnd] = beta_rnd_cost_frac(min_b, r, x_floor, x_max_ind, y_floor, y_frac, c, q, p, d_frac, A, B, mu, return_r=True)

    # x_max_ind = np.zeros_like(x)
    # for j in range(len(p)):
    #     max_ind = np.argmax(B[:, j] * x - 1 + B[:, j],
    #                         axis=1)  # choose maximum x_k among k serving j, set remaining k to -1 so they aren't chosen
    #     x_max_ind[np.arange(len(x)), max_ind] = 1
    #
    # if plot_on:
    #     plot_mat = []
    # lft_b = 0.01
    # rgt_b = 0.99
    # beta_list = np.linspace(lft_b, rgt_b, max_iter)
    # min_cst = np.inf
    # for beta in beta_list:
    #     cst = beta_rnd_cost(beta, r, x_max_ind, y, c, q, p, d, A, B, mu)
    #     if plot_on:
    #         plot_mat.append([beta, cst])
    #     if cst < min_cst:
    #         min_cst = cst
    #         min_b = beta
    #
    # if plot_on:
    #     plot_np = np.array(plot_mat)
    #     plot_np = plot_np[plot_np[:,0].argsort()]
    #     plt.plot(plot_np[:,0], plot_np[:,1])
    #     plt.show()#block=False)
    #
    # [min_cst, r_rnd] = beta_rnd_cost(min_b, r, x_max_ind, y, c, q, p, d, A, B, mu, return_r=True)
    return [min_b, min_cst, r_rnd]


def cap_x(x, q, d, B):
    # cap off x's that sum to over d
    x_cap = np.zeros_like(x)
    for j in range(d.shape[1]):
        k_ind_j, = np.where(B[:, j] > 0)
        k_ind_j_sort = k_ind_j[np.argsort(q[k_ind_j], kind='stable')]
        x_j_sort = x[:, k_ind_j_sort]
        x_j_sum = np.cumsum(x_j_sort, axis=1)
        x_j_cap = np.minimum(x_j_sort, np.maximum(0, d[:, [j]] + x_j_sort - x_j_sum))
        for count, k in enumerate(k_ind_j_sort):
            x_cap[:, k] = x_j_cap[:, count]

    return x_cap


def cor_rnd_cost(alpha, r, x, y, c, q, p, d, A, B, mu, chi, return_r = False):
    [x_frac, x_floor] = np.modf(x)
    x_rnd = np.where(chi <= alpha*x_frac, np.ceil(x), x_floor)
    x_rnd = cap_x(x_rnd, q, d, B)
    y_rnd = np.maximum(0, d - np.dot(x_rnd, B))
    r_rnd = np.amax(np.dot(x_rnd, A.T), axis=0) # use maximum usage in x rounding as inventory
    if return_r:
        return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]
    else:
        return obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu)


def best_of_n_cor_rnd(alpha, r, x, y, c, q, p, d, A, B, mu, n=5, return_r = False):
    np.random.seed(0)
    min_cst = np.inf
    if return_r:
        r_rnd_store = np.zeros_like(r)
    for i in range(n):
        chi = np.random.uniform(size=len(q))
        if return_r:
            [cst, r_rnd] = cor_rnd_cost(alpha, r, x, y, c, q, p, d, A, B, mu, chi, return_r)
        else:
            cst = cor_rnd_cost(alpha, r, x, y, c, q, p, d, A, B, mu, chi, return_r)
        if cst < min_cst:
            min_cst = copy.deepcopy(cst)
            if return_r:
                r_rnd_store = copy.deepcopy(r_rnd)

    if return_r:
        return [min_cst, r_rnd_store]
    else:
        return min_cst


def min_cor_grid(r, x, y, nn_inst, max_iter, plot_on=None):
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    if plot_on:
        plot_mat = []

    act_cost = np.dot(c, A) + q
    prod_cost = np.reshape(act_cost, [len(q), 1]) * B
    prod_cost_min = np.where(prod_cost > 0, prod_cost, np.iinfo(int).max).min(axis=0)
    prod_markup = p/prod_cost_min
    lft_a = 1
    rgt_a = 1 + np.log(np.mean(prod_markup))
    alpha_list = np.linspace(lft_a, rgt_a, max_iter)
    min_cst = np.inf
    for alpha in alpha_list:
        cst = best_of_n_cor_rnd(alpha, r, x, y, c, q, p, d, A, B, mu)
        if plot_on:
            plot_mat.append([alpha, cst])
        if cst < min_cst:
            min_cst = copy.deepcopy(cst)
            min_a = copy.deepcopy(alpha)
    # print("COST!:", min_cst)

    if plot_on:
        plot_np = np.array(plot_mat)
        plot_np = plot_np[plot_np[:,0].argsort()]
        plt.plot(plot_np[:,0], plot_np[:,1])
        plt.show()#block=False)

    [min_cst, r_rnd] = best_of_n_cor_rnd(min_a, r, x, y, c, q, p, d, A, B, mu, return_r=True)
    return [min_a, min_cst, r_rnd]


def mrkv_rnd_cost(alpha, beta, r, x, y, c, q, p, d, A, B, mu, chi, return_r = False):
    [x_frac, x_floor] = np.modf(x)
    [y_frac, y_floor] = np.modf(y)
    d_frac = d - y_floor - np.dot(x_floor, B)

    d_act = np.dot(d_frac, B.T)
    x_rnd = np.where(chi * d_act <= alpha * x_frac, d_act, 0) + x_floor
    x_rnd = cap_x(x_rnd, q, d, B)
    r_rnd = np.floor(r / beta)  # initially use this to to test for feasibility
    r_over = (np.dot(x_rnd, A.T) > r_rnd)  # find i, omega where usage is over inventory
    x_rnd = x_rnd * (np.dot(r_over, A) == 0)  # 0 out x if it uses any i that was over inventory
    y_rnd = np.maximum(0, d - np.dot(x_rnd, B))
    r_rnd = np.amax(np.dot(x_rnd, A.T), axis=0)  # use maximum usage in x rounding as final inventory
    if return_r:
        return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]
    else:
        return obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu)

    # d_act = np.dot(d, B.T)
    # x_rnd = np.where(chi*d_act <= alpha*x, d_act, 0)
    # x_rnd = cap_x(x_rnd, q, d, B)
    # r_rnd = np.floor(r/beta) # initially use this to to test for feasibility
    # r_over = (np.dot(x_rnd, A.T) > r_rnd) # find i, omega where usage is over inventory
    # x_rnd = x_rnd*(np.dot(r_over, A) == 0) # 0 out x if it uses any i that was over inventory
    # y_rnd = np.maximum(0, d - np.dot(x_rnd, B))
    # r_rnd = np.amax(np.dot(x_rnd, A.T), axis=0) # use maximum usage in x rounding as final inventory
    # if return_r:
    #     return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]
    # else:
    #     return obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu)


def best_of_n_mrkv_rnd(alpha, beta, r, x, y, c, q, p, d, A, B, mu, n=3, return_r = False):
    np.random.seed(0)
    min_cst = np.inf
    if return_r:
        r_rnd_store = np.zeros_like(r)
    for i in range(n):
        chi = np.random.uniform(size=len(q))
        if return_r:
            [cst, r_rnd] = mrkv_rnd_cost(alpha, beta, r, x, y, c, q, p, d, A, B, mu, chi, return_r)
        else:
            cst = mrkv_rnd_cost(alpha, beta, r, x, y, c, q, p, d, A, B, mu, chi, return_r)
        if cst < min_cst:
            min_cst = copy.deepcopy(cst)
            if return_r:
                r_rnd_store = copy.deepcopy(r_rnd)

    if return_r:
        return [min_cst, r_rnd_store]
    else:
        return min_cst


def min_mrkv_grid(r, x, y, nn_inst, max_iter, plot_on=None):
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    if plot_on:
        plot_mat = []

    act_cost = np.dot(c, A) + q
    prod_cost = np.reshape(act_cost, [len(q), 1]) * B
    prod_cost_min = np.where(prod_cost > 0, prod_cost, np.iinfo(int).max).min(axis=0)
    prod_markup = p/prod_cost_min
    lft_a = 0.01
    rgt_a = np.maximum(2, np.log(np.mean(prod_markup)))
    alpha_list = np.linspace(lft_a, rgt_a, max_iter)
    # print(alpha_list)
    lft_b = 0.01
    rgt_b = 0.99
    min_cst = np.inf
    beta_list = np.linspace(lft_b, rgt_b, max_iter)
    # print(beta_list)
    for alpha in alpha_list:
        for beta in beta_list:
            cst = best_of_n_mrkv_rnd(alpha, beta, r, x, y, c, q, p, d, A, B, mu)
            if plot_on:
                plot_mat.append([alpha, cst])
            if cst < min_cst:
                min_cst = copy.deepcopy(cst)
                min_a = copy.deepcopy(alpha)
                min_b = copy.deepcopy(beta)
    # print("COST!:", min_cst)

    if plot_on:
        plot_np = np.array(plot_mat)
        plot_np = plot_np[plot_np[:,0].argsort()]
        plt.plot(plot_np[:,0], plot_np[:,1])
        plt.show()#block=False)

    [min_cst, r_rnd] = best_of_n_mrkv_rnd(min_a, min_b, r, x, y, c, q, p, d, A, B, mu, return_r=True)
    return [min_a, min_b, min_cst, r_rnd]


def rnd_near_int(r, x, y, nn_inst):
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    r_rnd = np.rint(r)
    x_rnd = np.floor(x)
    y_rnd = np.maximum(0, d - np.dot(x_rnd, B))
    return [obj_cost(r_rnd, x_rnd, y_rnd, c, q, p, mu), r_rnd]