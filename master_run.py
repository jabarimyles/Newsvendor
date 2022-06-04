from pyparsing import alphanums
from dynamic_simulation import *
from optimals import *
from simulation import *

cwd = os.getcwd()

solve_LP = True 

#### Run Controls ####
run_ecom = False
load_ecom = False
run_manuf = False
load_manuf = False
run_prod = True
load_prod = False

alphas = [1.05]

path = ''

cwd = os.getcwd()

new_path = cwd + '/instances'

column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'upper cost' 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost']
sim_df = pd.DataFrame(columns = column_names)

def min_act_cost(pkl):
        pkl['instance']['min_actv'] = np.zeros((pkl['instance']['q'].shape[0],pkl['instance']['p'].shape[0]))
        min_cost = []
        for i in range(pkl['instance']['p'].shape[0]):
            pos_actvs = np.where(pkl['instance']['B'][:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(pkl['instance']['c'] ,pkl['instance']['A'][:,j])
                cost_dict[j] = pkl['instance']['q'][j] + actv_cost
            
            min_cost.append(min(cost_dict.values()))
        
        return min_cost
        

def modpickles(path, alpha=1):
    if path == '':
        new_path = cwd + '/instances'+str(int(alpha*100))
    else: 
        new_path = path
    
    os.makedirs(new_path)

    orig_path = cwd + '/instances'

    for root, dir, files, in os.walk(orig_path):
        for file in files:
            if file.endswith(".pkl"):
                pklfile = file
                print(pklfile)
                openpkl = open(orig_path + '/' + pklfile , 'rb')
                loadedpkl = pickle.load(openpkl)
                newpkl = copy.deepcopy(loadedpkl)

                #loadedpkl['instance']['p'] += min_act_cost(loadedpkl)
                loadedpkl['instance']['p'] = np.add(np.float_(loadedpkl['instance']['p']), np.float_(min_act_cost(loadedpkl)))
                loadedpkl['instance']['q'] = (1-alpha)*np.matmul(loadedpkl['instance']['c'],loadedpkl['instance']['A'])+loadedpkl['instance']['q']
                loadedpkl['instance']['c'] = alpha * loadedpkl['instance']['c']
                if alpha > 1:
                    [upper_cost_opt, upper_r_opt, upper_x_opt, upper_y_opt, upper_opt_time] = nn_opt_high(loadedpkl['instance'], var_type='C', return_xy=True)
                else:
                    [upper_cost_opt, upper_r_opt, upper_x_opt, upper_y_opt, upper_opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True)
                loadedpkl['instance']['p'] = newpkl['instance']['p']
                [lower_cost_opt, lower_r_opt, lower_x_opt, lower_y_opt, lower_opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True)
                newpkl['LP_solution'] = {'r': upper_r_opt, 'x': upper_x_opt, 'y': upper_y_opt, 'cost': lower_cost_opt, 'time': upper_opt_time, 'upper_cost':upper_cost_opt}
                with open(new_path + '/' + file, 'wb') as f:
                    pickle.dump(newpkl, f)
            else:
                print("No files found!")
    
    return new_path



if __name__ == '__main__':

#pickle file instance you load has parameters: c0, q0, p0
#then create new instance with c1 = alpha*c0 w/ alpha < 1
#q1 = (1-alpha)*A*c0 + q0, p1 = p0
#then run nn_opt on instance with c1, q1, p1
#then run simulation with: h = alpha*c0
#c = c0, q=q0, p=p0
    for alpha in alphas:
        begin_time = datetime.datetime.now()
        print(begin_time)

        if not os.path.isdir(cwd + '/instances'+str(int(alpha*100))):
            new_path = modpickles(path=path,alpha=alpha)

        print('loadpickles started')
        ####### dynamic simulation stuff ######
        simlist = loadpickles(path = new_path, alpha=alpha)
        print('run_sim started')
        run_sim(sim_list=simlist, alpha=alpha)

        print(datetime.datetime.now() - begin_time)