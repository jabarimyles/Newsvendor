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

alpha = .10

path = ''

cwd = os.getcwd()

new_path = cwd + '/instances'

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
                loadedpkl['instance']['q'] = (1-alpha)*np.matmul(loadedpkl['instance']['c'],loadedpkl['instance']['A'])+loadedpkl['instance']['q']
                loadedpkl['instance']['c'] = alpha * loadedpkl['instance']['c']
                [cost_opt, r_opt, x_opt, y_opt, opt_time] = nn_opt(loadedpkl['instance'], var_type='C', return_xy=True)
                newpkl['LP_solution'] = {'r': r_opt, 'x': x_opt, 'y': y_opt, 'cost': cost_opt, 'time': opt_time}
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

    begin_time = datetime.datetime.now()
    print(begin_time)

    if not os.path.isdir(cwd + '/instances'+str(int(alpha*100))):
        new_path = modpickles(path=path,alpha=alpha)

    ####### dynamic simulation stuff ######
    simlist = loadpickles(path = new_path, alpha=alpha)
    run_sim(sim_list=simlist, alpha=alpha)

    print(datetime.datetime.now() - begin_time)