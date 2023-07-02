from pyparsing import alphanums
from dynamic_simulation import *
from optimals import *
from simulation import *
cwd = os.getcwd()

solve_LP = True 

novel_lower = False

# Controls whether we just run the simulation or modpickle files and run with those
just_sim = True

# Whether we solved for the optimal policy, this is almost never True
optimal_policy=False

simple_network = False

#### Run Controls ####
run_ecom = False
load_ecom = False
run_manuf = False
load_manuf = False
run_prod = True
load_prod = False
zero_order = True

# Parameters the simulation runs over
alphas = [.01]
betas = [.5, .8, .9, 1] 
deltas = [1, .99, .97, .95, .9, .85, .8]

# Specify output path, if blank it goes to cwd
path = ''
cwd = os.getcwd()
new_path = cwd + '/instances'

#initialize simulation dataframe and columns of interest to summarize
column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'upper cost' ,'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost']
sim_df = pd.DataFrame(columns = column_names)


def min_act_cost(pkl, novel=False, this_beta=1):
        pkl['instance']['min_actv'] = np.zeros((pkl['instance']['q'].shape[0],pkl['instance']['p'].shape[0]))
        min_cost = []
        for i in range(pkl['instance']['p'].shape[0]):
            pos_actvs = np.where(pkl['instance']['B'][:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                if novel == True:
                    actv_cost = np.matmul(pkl['instance']['A'][:,j], pkl['instance']['c'] - this_beta*(pkl['instance']['c']*alpha))
                else:
                    actv_cost = np.matmul(pkl['instance']['c'] ,pkl['instance']['A'][:,j])

                cost_dict[j] = pkl['instance']['q'][j] + actv_cost
            
            if novel == False:
                min_cost.append(min(cost_dict.values()))
            elif novel == True:
                min_cost.append(min(min(cost_dict.values()), pkl['instance']['p'][i])) 
        
        return min_cost

def modpickles(path, alpha=1, novel=True, lead_time=0, simple_network=False):
    
    if lead_time == 0:
        new_path = cwd + '/instances'+str(int(alpha*100))
    elif lead_time > 0 and simple_network: 
        new_path =  cwd + '/pos_leads/simple_network/instances'+str(int(alpha*100))
    elif lead_time > 0 and not simple_network:
        new_path = cwd + '/pos_leads/real_network/lead_out'+str(lead_time)

    if file.endswith(".pkl"):
                
        pklfile = file
        openpkl = open(orig_path + '/' + pklfile , 'rb')
        loadedpkl = pickle.load(openpkl)
        newpkl = copy.deepcopy(loadedpkl)
        lower_list = []
        lower_dict = {}
        run='instances'

        if lead_time == 0:
            # Prepare to solve upper bound
            loadedpkl['instance']['p'] = np.add(np.float_(loadedpkl['instance']['p']), np.float_(min_act_cost(loadedpkl, novel=False)))
            loadedpkl['instance']['q'] = (1-alpha)*np.matmul(loadedpkl['instance']['c'],loadedpkl['instance']['A'])+loadedpkl['instance']['q']
            loadedpkl['instance']['c'] = alpha * loadedpkl['instance']['c']
            
            #upper bound
            [upper_cost_opt, upper_r_opt, upper_x_opt, upper_y_opt, upper_opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True) #not using scaled, but use true min cost
            loadedpkl['instance']['p'] = newpkl['instance']['p']

            #original lower bound for alpha > 1
            if alpha > 1 or zero_order:
                if zero_order:
                    newpkl['instance']['c'] = np.zeros(len(loadedpkl['instance']['c']))
                    loadedpkl['instance']['c'] = np.zeros(len(loadedpkl['instance']['c']))
                    q = (loadedpkl['instance']['B'].T * loadedpkl['instance']['q']).T
                    new_q = (q - np.where(q>0, q, np.inf).min(axis=0)[None,:])
                    new_q[new_q<0] = 0
                    newpkl['instance']['q'] = new_q.sum(axis=1)
                    loadedpkl['instance']['q'] = new_q.sum(axis=1)
                for d in deltas:
                    [lower_cost_opt, lower_r_opt, lower_x_opt, lower_y_opt, lower_opt_time] = nn_opt_high(loadedpkl['instance'], var_type='C', return_xy=True, delta=d) #never uses scaled
                    lower_dict[run+'_original' + '_alpha' + str(alpha)  + '_delta' + str(d)] = lower_cost_opt

            elif alpha <= 1:
                #original lower for alpha <= 1
                # q -> formula of u
                [lower_cost_opt, lower_r_opt, lower_x_opt, lower_y_opt, lower_opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True) #uses new scaled cost
                lower_dict[run+'_original' + '_alpha' + str(alpha) ] = lower_cost_opt

                for beta in betas:
                    #first novel lower alpha <= 1
                    hold_p = copy.deepcopy(loadedpkl['instance']['p'])
                    hold_c = copy.deepcopy(loadedpkl['instance']['c'])
                    loadedpkl['instance']['p'] = np.add(np.float_(loadedpkl['instance']['p']), np.float_(min_act_cost(loadedpkl, novel=True, this_beta=beta)))
                    loadedpkl['instance']['c'] = beta * loadedpkl['instance']['c'] 
                    [lower_cost_opt, lower_r_opt, lower_x_opt, lower_y_opt, lower_opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True) #uses new scaled cost
                    lower_dict[run+ '_novel' + '_alpha' + str(alpha) + '_beta' + str(beta)] = lower_cost_opt
                    loadedpkl['instance']['p'] = hold_p
                    loadedpkl['instance']['c'] = hold_c


            #return p back for sim runs
            loadedpkl['instance'] = newpkl['instance']
            newpkl['LP_solution'] = {'r': upper_r_opt, 'x': upper_x_opt, 'y': upper_y_opt, 'cost': lower_dict, 'time': upper_opt_time, 'upper_cost':upper_cost_opt}
            with open(new_path + '/' + file, 'wb') as f:
                pickle.dump(newpkl, f)

        elif lead_time > 0:
            mean_vec = loadedpkl['instance']['d'].mean(axis=0)
            variance = np.var(loadedpkl['instance']['d'], axis=0)
            n = loadedpkl['instance']['d'].shape[1]
            samples = loadedpkl['instance']['d'].shape[0]
            d = loadedpkl['instance']['d']
            d_sol = loadedpkl['instance']['d_sol']
            """
            if dist == 'indep':
                d_sol = np.maximum(indep_rand.multivariate_normal(mean=mean_vec*(lead_time+1), cov=(np.identity(n)*variance)*(lead_time+1),size=samples),0)
            if dist == 'pos':
                d_sol = np.maximum(pos_cor_rand.multivariate_normal(mean=mean_vec*(lead_time+1), cov=(variance*(np.identity(n) + 0.5*(np.ones([n,n]) - np.identity(n))))*(lead_time+1),size=samples),0)
            if dist == 'neg': 
                d_sol = np.maximum(neg_cor_rand.multivariate_normal(mean=mean_vec*(lead_time+1), cov=(variance*(np.identity(n) + (-1/(n-1))*(np.ones([n,n]) - np.identity(n))))*(lead_time+1),size=samples),0)
            """

            # Create cost vars for solving SP
            h = (alpha * loadedpkl['instance']['c']) #holding cost for SP
            u_k = ((1-alpha*(lead_time+1))*np.matmul(loadedpkl['instance']['c'],loadedpkl['instance']['A'])+loadedpkl['instance']['q'])/(lead_time+1) #long-run activity cost
            instance = {'c': h, 'q': u_k, 'p': loadedpkl['instance']['p'], 'd': d_sol, 'mu': mu, 'A': loadedpkl['instance']['A'], 'B':loadedpkl['instance']['B'], 'lead_time':lead_time}
            [cost, r, x, y, time] = nn_opt(instance, var_type='C', return_xy=True)


            # Reset activity cost, demand, and ordering cost for simulation
            instance['q'] = loadedpkl['instance']['q']
            instance['d'] = loadedpkl['instance']['d']
            instance['c'] = loadedpkl['instance']['c']
            

            """
            holding_cost = np.mean(np.subtract(np.matmul(r,h), np.matmul(h.T,np.matmul(A,x.T))))
            ordering_cost = 1/(lead_time+1) * np.mean(np.matmul(c.T, np.matmul(A, x.T)))
            activity_cost = 1/(lead_time+1) * np.mean(np.matmul(u_k,x.T))
            shortage_cost = np.mean(np.matmul(y,p))

            instance['holding_cost_SP'] = holding_cost
            instance['ordering_cost_SP'] = ordering_cost
            instance['activity_cost_SP'] = activity_cost
            instance['shortage_cost_SP'] = shortage_cost
            """

            # Calculate mean of x and y
            x_bar = x.mean(axis=0)
            y_bar = y.mean(axis=0)

            # Calculate mean demands 
            exp_d = d_sol.mean(axis=0)
            exp_d_jk = np.matmul(loadedpkl['instance']['B'],d_sol.mean(axis=0))

            # Calculate phi and psi proportions
            phi = x_bar/exp_d_jk
            psi = y_bar/exp_d

            # Add phi, psi and lead time to the pickle file
            pos_leads = {'phi':phi, 'psi':psi, 'lead_time':lead_time}
            LP_solution = {'cost':cost, 'r':r, 'x':x, 'y':y, 'time':time} 
            instance_pkl = {'instance': instance, 'LP_solution': LP_solution, 'pos_leads': pos_leads}


            with open(new_path + '/' + file, 'wb') as f:
                pickle.dump(instance_pkl, f)


    else:
        print("Not a pickle file: "  + file)

    
def simple_mod(path, alpha=1, novel=True, n=1):
    
    if path == '':
        new_path = cwd + '/instances'+str(int(alpha*100))
    else: 
        new_path = path + '/instances'+str(int(alpha*100))
    if file.endswith(".pkl"):
                
        pklfile = file
        openpkl = open(orig_path + '/' + pklfile , 'rb')
        loadedpkl = pickle.load(openpkl)
        newpkl = copy.deepcopy(loadedpkl)
        lower_list = []

        lower_dict = {}
    
        run='instances'
        
        loadedpkl['instance']['A']

        loadedpkl['instance']['B']

#B is a qxp matrix linking activities to products
#A is a cxq matrix mapping resources to activities

#new B (n^2x n) mapping  to demand in each region

#new A (nxn^2) mapping inventory of each item at each warehouse to activities (warehouse fulfilling region demand)

#4x16, 16x4
# 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 
# 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1

# 1 0 0 0
# 0 1 0 0
# 0 0 1 0 
# 0 0 0 1
# 1 0 0 0
# 0 1 0 0
# 0 0 1 0 
# 0 0 0 1
# 1 0 0 0
# 0 1 0 0
# 0 0 1 0 
# 0 0 0 1
# 1 0 0 0
# 0 1 0 0
# 0 0 1 0 
# 0 0 0 1




if __name__ == '__main__':
    
    #print(begin_time)
#pickle file instance you load has parameters: c0, q0, p0
#then create new instance with c1 = alpha*c0 w/ alpha < 1
#q1 = (1-alpha)*A*c0 + q0, p1 = p0
#then run nn_opt on instance with c1, q1, p1
#then run simulation with: h = alpha*c0
#c = c0, q=q0, p=p0
    orig_path = cwd + '/instances'
    for root, dir, files, in os.walk(orig_path):
        for alpha in alphas:
            for file in files:
                if file.endswith(".pkl"):
                    print(file)
                    begin_time = datetime.datetime.now()
                    if path == '':
                        new_path = cwd + '/instances'+str(int(alpha*100))
                    else: 
                        new_path = path + '/instances'+str(int(alpha*100))
    
                    
                    if not os.path.isdir(new_path):
                        os.makedirs(new_path)
                    if not just_sim:
                        modpickles(path=path,alpha=alpha, novel=novel_lower)
                    
                    print(datetime.datetime.now() - begin_time)

            print('loadpickles started')
            ####### dynamic simulation stuff ######
            simlist = loadpickles(path = new_path, alpha=alpha)
            print('run_sim started')
            run_sim(sim_list=simlist,alpha=alpha, novel=novel_lower, optimal_policy=optimal_policy)

    

