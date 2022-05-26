import argparse
import pickle
import os
import numpy as np
#import matplotlib.pyplot as plt

cwd = os.getcwd()


n_sims = 1000
days = 100

#track costs over sample paths, 

#m num of resources - len of c
#c_i is ordering cost (c in paper)
#p_i is backlog cost (b in paper)
#h_i is holding cost (h in paper)
#q_k is processing activity cost (p_k in paper)
#n num of products - len(p)
#l num of activities - len(q)
#B is a qxp matrix linking activities to products
#A is a cxq matrix mapping resources to activities

#M resources (indexed by i) to fill demand for N products (indexed by j) 
#by means of L processing activities (indexed by k)

#r (B in pickle) is S_i in paper, S-i is justt how much we want on hand at the end of the day
#page 6 equation k(j) argmin , for product j what is its lowest cost activity to fill it. q_k + \sum_i a_ik c_i

#x in pickle is z in paper, 

#inventory should stay constant
#backlog shoudl grow


class Simulation(object):

    def __init__(self, dictionary):

        # Set attributes from dictionary
        for key in dictionary:
            setattr(self, key, dictionary[key])
        self.I = np.zeros(dictionary['A'].shape[0])  #Inventory
        self.I_hist = np.zeros(dictionary['A'].shape[0])
        self.BL = np.zeros(dictionary['p'].shape[0])  #Backlog
        self.BL_hist = np.zeros(dictionary['p'].shape[0])
        self.x = np.zeros(len(self.c)) 
        self.z = np.zeros(len(self.q)) #processing activities
        self.h = self.c
        self.cost = 0
        self.demand = np.zeros(dictionary['p'].shape[0])


    # Getter/setter stuff
    #@property
    def smart_order(self):
        #res_to_prod = np.matmul(self.A, self.B)
#processing + sum(c*a) c is per unit odering cost, a is the resource requirements for each processing activity
        #self.r - self.I + 
        #for k in range(self.r.shape[0]):
        min_actv = np.zeros((self.q.shape[0],self.p.shape[0]))

        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.c ,self.A[:,j])
                cost_dict[j] = self.q[j] + actv_cost
            min_actv[j,i] += 1 #lowest activity for given product
            #FIX BEFORE LINE 70

        self.z = self.r[i] - self.I[i] + np.matmul(self.A,np.matmul(min_actv, self.BL))

        self.cost += np.matmul(self.z, self.c) #updates cost


               # np.where(self.A[:,i] != 0) #resource for each pos_actvs
            #cheapest_activity = low_actvs[0][np.argmin(self.q[low_actvs])]
            #cheapest_act_cost = self.q[cheapest_activity]

        #    for j in prods:
        #        for k in 
        #        actv = np.where(self.A[])
    
    def update_cost(self):
        d=c+c

    def smart_fulfillment(self):
        min_actv = np.zeros((self.q.shape[0],self.p.shape[0]))

        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.c ,self.A[:,j])
                cost_dict[j] = self.q[j] + actv_cost
            min_actv[j,i] += 1 #lowest activity for given product
        
        t + np.matmul(min_actv, self.BL) #Figure out what the activity processing matrix is

        #self.y[self.demand_index] 


    def demand_draw(self):
        index = np.random.choice(self.mu.shape[0])
        self.demand_index = index
        self.demand = self.d[index]
        
    
    def get_cheapest(self):
        self.A = self.A[np.array(np.argsort(self.p))]



    def update_inventory(self):
        self.I += self.x - np.matmul(self.A, self.z)
        self.I_hist = np.vstack([self.I_hist, self.I])


    def update_backlog(self):
        self.BL += self.demand - np.matmul(self.z, self.B)
        self.BL_hist = np.vstack([self.BL_hist, self.BL])

    def plot_sim_backlog(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.BL_hist)
        plt.show()

    def plot_sim_inventory(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.I_hist)
        plt.show()

''' 
    def get_c(self):
        return self.c
  
    def set_c(self, x):
        self.c = x

    def get_q(self):
        return self.q
    
    def set_q(self, x):
        self.q = x
    
    def get_p(self):
        return self.p
    
    def set_p(self, x):
        self.p = x

    def get_d(self):
        return self.d
    
    def set_d(self, x):
        self.d = x
    
    def get_mu(self):
        return self.mu
    
    def set_mu(self, x):
        self.mu = x
    
    def get_A(self):
        return self.A
    
    def set_A(self, x):
        self.A = x
    
    def get_B(self):
        return self.B
    
    def set_B(self, x):
        self.B = x
   ''' 

def run():
    '''run simulation'''
    #global n_sims
    #global n_paths
    n_params = len(sim_list)

    i,j,k = 1,1,1

    try:
        while i < n_params:
            while j < n_sims:
                while k < days:
                    sim = sim_list[i]
                    sim.smart_order()
                    sim.demand_draw()
                    sim.smart_fulfillment()
                    sim.update_backlog()
                    sim.update_inventory()

                    k+=1
                    
                sim.plot_sim_backlog()
                sim.plot_sim_inventory()
                j+=1
            i+=1

    except:
        print("Simulation failed")
    




def loadpickles(path):
    simlist = []
    if path == '':
        file_path = cwd + '/instances'
    else: 
        file_path = path

    for root, dir, files, in os.walk(file_path):
        for file in files:
            if file.endswith(".pkl"):
                pklfile = file
                print(pklfile)
                openpkl = open(file_path + '/' + pklfile , 'rb')
                loadedpkl = pickle.load(openpkl)
                #print(loadedpkl)
                simlist.append(Simulation({**loadedpkl['LP_solution'], **loadedpkl['instance']}))
            else:
                print("No files found!")


    return simlist




if __name__ == '__main__':

    sim_list = loadpickles(path = '')

    run()

    
#THIS IS USED TO RUN FROM CMD LINE
#    parser = argparse.ArgumentParser(description='Get path to simulation information.')

#    parser.add_argument(
#        '--path',
#        required=True,
#        type=str,
#        help='Path to folder with pickle files'
#    )

#   args = parser.parse_args()

#    loadpickles(args.path)
    
    

    

