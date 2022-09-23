from gurobipy import *
import numpy as np
import time


def sec_stg(r_in, p, d, mu, A, var_type, ret_y=None):
    """ compute optimal second stage cost given inventory vector """
    # var_type will be the gurobi vtype variable, 'C'->continuous, 'I'->integer
    M = A.shape[0]
    [W, N] = d.shape
    components = range(M)
    products = range(N)
    scenarios = range(W)

    sec_stg_output = []

    # Optimize Second Stage
    ss = Model('Scnd_Stg')
    # ss.Params.Method = 1 #set dual simplex method to save memory (default concurrent method uses too much memory)
    y_ss = {}
    for j in products:
        y_ss[j] = ss.addVar(vtype=var_type, name='y_%s' % j, obj=p[j], lb=0)  # will set upper bound in each scenario
    ss.update()

    cinv_ss = {}
    for i in components:
        yvar_ss = [y_ss[j] for j in products]
        cinv_ss[i] = ss.addConstr(LinExpr(A[i, :], yvar_ss) >= 0, name='cinv_%s' % i)  # will set RHS for each scenario
    ss.update()
    ss.setAttr("ModelSense", GRB.MINIMIZE)
    ss.setParam('OutputFlag', False)

    shrt_cst = 0
    if ret_y:
        y_out = np.empty([W, N])
    # sec_stg_strt = time.time()
    for w in scenarios:
        for j in products:
            y_ss[j].setAttr(GRB.attr.UB, d[w, j])
        for i in components:
            cinv_ss[i].setAttr(GRB.attr.RHS, np.dot(A[i, :], d[w, :]) - r_in[i])  # setting RHS for each scenario
        ss.update()
        ss.optimize()
        shrt_cst = shrt_cst + ss.ObjVal * mu[w]
        if ret_y:
            for j in products:
                y_out[w, j] = y_ss[j].x

    # sec_stg_end = time.time()
    # print("Avg Sec Stg Time: ", (sec_stg_end-sec_stg_strt)/W)

    del ss

    sec_stg_output.append(shrt_cst)
    if ret_y:
        sec_stg_output.append(y_out)
    return sec_stg_output


def ato_opt(c, q, p, d, mu, A, var_type):
    """ solve ATO problem """
    # var_type will be the gurobi vtype variable, 'C'->continuous, 'I'->integer
    M = A.shape[0]
    [W, N] = d.shape

    components = range(M)
    products = range(N)
    scenarios = range(W)

    start = time.time()
    ato = Model('ATO')
    # ato.Params.Method = 1
    r_o = {}
    x_o = {}
    y_o = {}
    for i in components:
        r_o[i] = ato.addVar(vtype=var_type, name='r_%s' % i, obj=c[i], lb=0)
    for j in products:
        for w in scenarios:
            x_o[w, j] = ato.addVar(vtype=var_type, name='x_%s' % j + '%s' % w, obj=q[j] * mu[w], lb=0)  # , ub=d[w, j])
            y_o[w, j] = ato.addVar(vtype=var_type, name='y_%s' % j + '%s' % w, obj=p[j] * mu[w], lb=0)  # , ub=d[w,j])
    ato.update()

    cinv_o = {}  # inventory
    cdem_o = {}  # inventory
    for w in scenarios:
        for i in components:
            xvar = [x_o[w, j] for j in products]
            cinv_o[i, w] = ato.addConstr(LinExpr(A[i, :], xvar) <= r_o[i], name='cinv_o_%s' % i + '%s' % w)
        for j in products:
            cdem_o[w, j] = ato.addConstr(x_o[w, j] + y_o[w, j] == d[w,j], name='cdem_o_%s' % j + '%s' % w)
    ato.update()

    ato.setAttr("ModelSense", GRB.MINIMIZE)
    ato.setParam('OutputFlag', False)
    ato.optimize()

    opt_time = time.time() - start

    ato_opt = ato.ObjVal

    r_opt = np.empty(M)
    for i in components:
        r_opt[i] = r_o[i].x

    ##    # remove variables and constraints
    ##    for v in range(ato.NumVars):
    ##        ato.remove(ato.getVars()[v])
    ##    for c in range(ato.NumConstrs):
    ##        ato.remove(ato.getConstrs()[c])
    ##    ato.update()

    del ato

    return [ato_opt, r_opt, opt_time]


def nn_opt(nn_inst, var_type, return_xy=False):
    """ solve ATO problem """
    # var_type will be the gurobi vtype variable, 'C'->continuous, 'I'->integer
    # nn_inst is: {'c': , 'q': , 'p': , 'd': , 'mu': , 'A': , 'B': }
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    [M, L] = A.shape
    [W, N] = d.shape

    components = range(M)
    activities = range(L)
    products = range(N)
    scenarios = range(W)

    start = time.time()
    nn = Model('NN')
    # nn.Params.Method = 1
    r_o = {}
    x_o = {}
    y_o = {}
    for i in components:
        r_o[i] = nn.addVar(vtype=var_type, name='r_%s' % i, obj=c[i], lb=0)
    for w in scenarios:
        for k in activities:
            x_o[w, k] = nn.addVar(vtype=var_type, name='x_%s' % k + '%s' % w, obj=q[k] * mu[w], lb=0)  # , ub=d[w, j])
        for j in products:
            y_o[w, j] = nn.addVar(vtype=var_type, name='y_%s' % j + '%s' % w, obj=p[j] * mu[w], lb=0)  # , ub=d[w,j])
    nn.update()

    cinv_o = {}  # inventory
    cdem_o = {}  # inventory
    for w in scenarios:
        xvar = [x_o[w, k] for k in activities]
        for i in components:
            cinv_o[i, w] = nn.addConstr(LinExpr(A[i, :], xvar) <= r_o[i], name='cinv_o_%s' % i + '%s' % w)
        for j in products:
            cdem_o[w, j] = nn.addConstr(LinExpr(B[:, j], xvar) + y_o[w, j] == d[w,j], name='cdem_o_%s' % j + '%s' % w)
    nn.update()

    nn.setAttr("ModelSense", GRB.MINIMIZE)
    nn.setParam('OutputFlag', False)
    nn.optimize()

    opt_time = time.time() - start

    nn_opt = nn.ObjVal

    r_opt = np.empty(M)
    for i in components:
        r_opt[i] = r_o[i].x

    ##    # remove variables and constraints
    ##    for v in range(nn.NumVars):
    ##        nn.remove(nn.getVars()[v])
    ##    for c in range(nn.NumConstrs):
    ##        nn.remove(nn.getConstrs()[c])
    ##    nn.update()

    if return_xy:
        x_opt = np.empty([W, L])
        y_opt = np.empty([W, N])
        for w in scenarios:
            for k in activities:
                x_opt[w, k] = x_o[w, k].x
            for j in products:
                y_opt[w, j] = y_o[w, j].x

        del nn
        return [nn_opt, r_opt, x_opt, y_opt, opt_time]
    else:
        del nn
        return [nn_opt, r_opt, opt_time]

def nn_opt_high(nn_inst, var_type, return_xy=False, delta = 0.95):
    """ solve ATO problem """
    # var_type will be the gurobi vtype variable, 'C'->continuous, 'I'->integer
    # nn_inst is: {'c': , 'q': , 'p': , 'd': , 'mu': , 'A': , 'B': }
    c = nn_inst['c']
    q = nn_inst['q']
    p = nn_inst['p']
    d = nn_inst['d']
    mu = nn_inst['mu']
    A = nn_inst['A']
    B = nn_inst['B']

    [M, L] = A.shape
    [W, N] = d.shape

    #check if b is ever positive


    components = range(M)
    activities = range(L)
    products = range(N)
    scenarios = range(W)

    start = time.time()
    nn = Model('NN')
    # nn.Params.Method = 1
    r_o = {}
    x_o = {}
    y_o = {}
    b_o = {}

    for l in products:
        b_o[l] = nn.addVar(vtype=var_type, name='b_%s' % l, obj=p[l]*(1-delta), lb=0)
    for i in components:
        r_o[i] = nn.addVar(vtype=var_type, name='r_%s' % i, obj=c[i], lb=0)
    for w in scenarios:
        for k in activities:
            x_o[w, k] = nn.addVar(vtype=var_type, name='x_%s' % k + '%s' % w, obj=q[k] * mu[w], lb=0)  # , ub=d[w, j])
        for j in products:
            y_o[w, j] = nn.addVar(vtype=var_type, name='y_%s' % j + '%s' % w, obj=p[j] * mu[w] * delta, lb=0)  # , ub=d[w,j])
    nn.update()

    cinv_o = {}  # inventory
    cdem_o = {}  # inventory
    for w in scenarios:
        xvar = [x_o[w, k] for k in activities]
        for i in components:
            cinv_o[i, w] = nn.addConstr(LinExpr(A[i, :], xvar) <= r_o[i], name='cinv_o_%s' % i + '%s' % w)
        for j in products:
            cdem_o[w, j] = nn.addConstr(LinExpr(B[:, j], xvar) + y_o[w, j] == d[w,j]+b_o[j], name='cdem_o_%s' % j + '%s' % w)
    nn.update()

    nn.setAttr("ModelSense", GRB.MINIMIZE)
    nn.setParam('OutputFlag', False)
    nn.optimize()

    opt_time = time.time() - start

    nn_opt = nn.ObjVal

    r_opt = np.empty(M)
    for i in components:
        r_opt[i] = r_o[i].x

    ##    # remove variables and constraints
    ##    for v in range(nn.NumVars):
    ##        nn.remove(nn.getVars()[v])
    ##    for c in range(nn.NumConstrs):
    ##        nn.remove(nn.getConstrs()[c])
    ##    nn.update()

    if return_xy:
        x_opt = np.empty([W, L])
        y_opt = np.empty([W, N])
        b_opt = np.empty([W, N])
        for w in scenarios:
            for k in activities:
                x_opt[w, k] = x_o[w, k].x
            for j in products:
                y_opt[w, j] = y_o[w, j].x
            for j in products:
                b_opt[w,j] = b_o[j].x
                

        del nn
        return [nn_opt, r_opt, x_opt, y_opt, opt_time]
    else:
        del nn
        return [nn_opt, r_opt, opt_time]

    