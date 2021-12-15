import argparse
import pickle
import os
import numpy as np

cwd = os.getcwd()

n_sims = 0



class Simulation(object):

    def __init__(self, dictionary):

        # Set attributes from dictionary
        for key in dictionary:
            setattr(self, key, dictionary[key])
            self.I = [0] * dictionary['A'].shape[0] #inventory #np.zeros((0,)*dictionary['A'].shape[0])  #
            self.L = [0] * dictionary['q'].shape[0] #backlog #np.zeros((0,)*dictionary['q'].shape[0])  # [0] * dictionary['q'].shape[0] #backlog


    # Getter/setter stuff
    @property
    def constant_order(self, n):
        self.A += n
    
    def demand_draw(self):
        index = np.random.choice(self.mu.shape[0])
        demand = self.d[index]
        return demand
    
    def get_cheapest(self):
        self.A = self.A[np.array(np.argsort(self.p))]

    def update_inventory(self):
        self.I = 

    def update_backlog(self):
        self.L = 

    
    def get_c(self):
        return self.c
  
    def set_c(self, x):
        self.c = x
'''
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
    global n_sims
    i = 0

    try:
        while i < n_sims:
            sim = sim_list[i]
            #sim.constant_order(n=0)
            sim.demand_draw()

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

    global n_sims 
    n_sims = len(simlist)
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
    
    

    

