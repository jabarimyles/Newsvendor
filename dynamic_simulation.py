import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

cwd = os.getcwd()


n_sims = 1000
days = 100


#m num of resources - len of c
#n num of products - len(p)
#l num of activities - len(q)

#r (b in pickle) is S_i in paper, S-i is justt how much we want on hand at the end of the day
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
        self.cost = 0
        self.demand = np.zeros(dictionary['p'].shape[0])


    # Getter/setter stuff
    #@property
    def constant_order(self):
        self.x.fill(0)
    
    def demand_draw(self):
        index = np.random.choice(self.mu.shape[0])
        self.demand = self.d[index]
        
    
    def get_cheapest(self):
        self.A = self.A[np.array(np.argsort(self.p))]

    def constant_fulfillment(self):
        self.x.fill(0)


    def update_inventory(self):
        self.I += self.x - np.matmul(self.A, self.z)
        self.I_hist = np.vstack([self.I_hist, self.I])


    def update_backlog(self):
        self.BL += self.demand - np.matmul(self.z, self.B)
        self.BL_hist = np.vstack([self.BL_hist, self.BL])

    def plot_sim_backlog(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = list(range(100))
        y1 = self.BL_hist[:,1]
        y2 = self.BL_hist[:,2]
        y3 = self.BL_hist[:,3]
        y4 = self.BL_hist[:,4]
        y5 = self.BL_hist[:,5]
        y6 = self.BL_hist[:,6]
        plt.plot(x,y1,x,y2,x,y3,x,y4,x,y5,x,y6)
        plt.show()

    def plot_sim_inventory(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = list(range(100))
        y1 = self.I_hist[:,1]
        y2 = self.I_hist[:,2]
        y3 = self.I_hist[:,3]
        y4 = self.I_hist[:,4]
        y5 = self.I_hist[:,5]
        y6 = self.I_hist[:,6]
        plt.plot(x,y1,x,y2,x,y3,x,y4,x,y5,x,y6)
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
                    sim.constant_order()
                    sim.demand_draw()
                    sim.constant_fulfillment()
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
                simlist.append(Simulation(loadedpkl['instance']))
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
    
    

    

