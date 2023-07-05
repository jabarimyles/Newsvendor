import argparse
import pickle
import os
import sys
import numpy as np
import csv
import copy
import pdb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import sem
import cProfile
import time
from scipy.stats import variation 
#from optimals import sec_stg, ato_opt, nn_opt
pd.options.mode.chained_assignment = None  # default='warn'

seed_rand = np.random.RandomState(0)

cwd = os.getcwd()

n_sims = 1000
days = 100


csv_name = 'newsvendoroutput.csv'

#alphas = [ 2, 3, 4, 5]
#
#betas = [.5, .8, .9, 1] 


#product shortage cost as backlog cost
#percentage of holding cost is ordering cost
#exclude manuf
#ecom app and produc application


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





class Simulation(object):

    def __init__(self, dictionary,alpha):
        
        self.theta=1
        # Set attributes from dictionary
        for key in dictionary:
            setattr(self, key, dictionary[key])
        self.I = np.zeros(dictionary['A'].shape[0])  #Inventory
        self.I_hist = np.zeros(dictionary['A'].shape[0])
        self.BL = np.zeros(dictionary['p'].shape[0])  #Backlog
        self.BL_hist = np.zeros(dictionary['p'].shape[0])
        self.z = np.zeros(len(self.c)) #processing activities
        self.h = self.c * alpha
        self.sim_cost = 0
        self.ordering_cost = 0
        self.backlog_cost = 0
        self.holding_cost = 0
        self.fulfillment_cost = 0
        self.sim_cost_hist = []
        self.X_tilde = np.zeros(len(self.q))
        self.B_hat = np.zeros(len(self.q))
        self.B_bar = np.zeros(len(self.q))
        self.B_tilde = np.zeros(len(self.q))
        self.D_hat = np.zeros(len(self.q))
        if self.lead_time > 0:
            self.I = copy.deepcopy(self.r)
            self.u_k = (1/(self.lead_time+1))(self.q + np.matmul(self.A,(self.c - (self.lead_time-1)*self.h)))
            self.z_hist = np.zeros((len(self.c), self.lead_time+1))
            self.B_bar_hist = np.zeros((len(self.q), self.lead_time))
            self.B_tilde_hist = np.zeros((len(self.q), self.lead_time))
            self.D_hat_hist = np.zeros((len(self.q), self.lead_time))
            self.L_k = (self.theta+1) * self.x.mean(axis=0)
            self.r = (self.theta+1) * (self.r**self.lead_time)
            mu_j = self.d.mean(axis=0)
            numer = np.matmul((self.p + self.h_bar), (mu_j * variation(self.d, axis=0)**2))
            denom = np.matmul(np.matmul(mu_j,self.B), (self.min_actv * self.q))
            self.max_theta = np.sqrt(numer/denom)

        self.demand = np.zeros(dictionary['p'].shape[0])
        self.min_actv = np.zeros((self.q.shape[0],self.p.shape[0]))
        self.h_bar = np.zeros(self.h.shape[0])
        self.x_k = np.zeros(len(self.q)) 
        self.num_fill = np.zeros(len(self.q)) 

    def get_min_actv(self):
        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.c ,self.A[:,j])
                cost_dict[j] = self.q[j] + actv_cost
            self.min_actv[min(cost_dict,key=cost_dict.get),i] += 1 #lowest activity for given product
    
    def get_max_hold(self):
        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.h ,self.A[:,j])
                cost_dict[j] = self.h[j] + actv_cost
            self.h_bar[max(cost_dict,key=cost_dict.get),i] += 1 #lowest activity for given product

    def smart_order(self):
        self.z = self.r - self.I + np.matmul(self.A,np.matmul(self.min_actv, self.BL)) #base stock policy
        self.sim_cost += np.matmul(self.z, self.c) #ordering cost
        self.ordering_cost += np.matmul(self.z, self.c)
    
    def simple_smart_order(self):
        self.z[0] = self.BL[0] + min(self.BL[1],1)
        self.z[1] = max(0, self.BL[1]-1)
        self.sim_cost += np.matmul(self.z, self.c) #ordering cost
        self.ordering_cost += np.matmul(self.z, self.c)

    def smart_fulfillment(self):   
        num_fill =  np.matmul(self.min_actv, self.BL)
        self.x_k += num_fill  #Update fulfilled
        self.sim_cost += np.sum(self.q * self.x_k) #fulfillment cost
        self.fulfillment_cost += np.sum(self.q * self.x_k)


    def simple_smart_fulfillment(self):   
        self.x_k[0] = min(self.I[0]+self.z[0], self.BL[0]+self.demand[0])
        self.x_k[1] = min(max(self.I[0]+self.z[0]-self.BL[0]-self.demand[0],0), max(self.BL[1]+self.demand[1]-self.I[1]-self.z[1],0))
        self.x_k[2] = min(self.I[1]+self.z[1], self.BL[1]+self.demand[1])
        #self.x_k += self.num_fill  #Update fulfilled
        self.sim_cost += np.sum(self.q * self.x_k) #fulfillment cost
        self.fulfillment_cost += np.sum(self.q * self.x_k)


    def demand_draw(self):
        index = seed_rand.choice(self.mu.shape[0], p=self.mu)
        self.demand_index = index
        self.demand = self.d[index]
        self.x_k = copy.deepcopy(self.x[index])
        

    def get_cheapest(self):
        self.A = self.A[np.array(np.argsort(self.p))]



    def update_inventory(self):
        if self.lead_time == 0:
            self.I += self.z - np.matmul(self.A, self.x_k)
        else:
            self.I + self.z_hist[:,self.lead_time] - np.matmul(self.B,self.x_k)
        #if np.any(self.I < 0):
        #    pdb.set_trace()
        self.sim_cost += np.matmul(self.I, self.h)
        self.holding_cost += np.matmul(self.I, self.h) #holding cost
        self.I_hist = np.vstack([self.I_hist, self.I])


    def update_backlog(self):
        self.BL += self.demand - np.matmul(self.x_k, self.B)
        self.sim_cost += np.matmul(self.BL, self.p) #np.any(self.BL < 0)
        self.backlog_cost += np.matmul(self.BL, self.p) #backlog cost
        self.BL_hist = np.vstack([self.BL_hist, self.BL])
    

    def update_D_hat(self):
        self.D_hat = np.matmul(self.phi, np.matmul(self.demand, self.B))
    
    def update_B_tilde(self):
        self.B_tilde = np.matmul(self.min_actv, self.psi * self.demand)
    
    def update_X_tilde(self):
        self.X_tilde = np.minimum(self.D_hat + self.B_hat, )
    
    def update_B_bar(self):
        self.B_bar += self.D_hat + self.B_tilde - self.x_k

    def deterministic_order(self):
        self.z = self.r - self.I - np.sum(self.z_hist)

    def deterministic_fulfillment(self):
        self.x_k = self.X_tilde + self.B_tilde_hist[:,self.lead_time+1]

    def plot_sim_backlog(self):
        #fig = plt.figure()
        #fig.add_subplot(111)
        plt.plot(self.BL_hist)
        plt.title(self.file_name)
        plt.show(block=False)




    def plot_sim_inventory(self):
        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        plt.plot(self.I_hist)
        plt.show()
        #plt.savefig("temp.png")

    def output_to_csv(self, sim_df):
            csv_file = open(csv_name, "w")
            writer = csv.writer(csv_file) 

            csv_file.close()

    
    
def summarize_sims(df):
    df = df.drop(['sim number', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'], ax_ks=1)
    df['simulation cost'] = pd.to_numeric(df['simulation cost'])
    df['cost'] = pd.to_numeric(df['cost'])
    df['upper cost'] = pd.to_numeric(df['upper cost'])
    #df['largest lower'] = pd.to_numeric(df['largest lower'])
    df['sim_cost_per_day'] = df['simulation cost']/days
    #df = df.groupby(['file name'], as_index=False).sem()
    #df = df.groupby(['file name'], as_index=False).mean()
    cols = df.columns.drop(['file name', 'largest lower'])
    df[cols] = df[cols].apply(pd.to_numeric)
    df = df.groupby(['file name', 'largest lower']).agg([np.mean, sem])
    df.columns = df.columns.map('_'.join)
    df = df.reset_index()
    df['lower cost ratio'] = df['sim_cost_per_day_mean']/df['cost_mean']
    df['upper cost ratio'] = df['sim_cost_per_day_mean']/df['upper cost_mean']

    df_out = df.drop(['upper cost_sem', 'upper cost_mean', 'upper cost ratio'], ax_ks=1)

    return df_out

def plot_sim_cost_hist(sim_df, days,j):
        #fig = plt.figure()
        #fig.add_subplot(111)
    time.sleep(1.5)
    print(str(j) + ':' + str(np.mean(sim_df['simulation cost']/days)/np.mean(sim_df['cost'])))
    #plt.plot(np.mean(sim_df['simulation cost']/days)/np.mean(sim_df['cost']))
    #plt.title(sim_df['file name'].iloc[0])
    #plt.show(block=False)
 

def run_sim(sim_list,alpha=1, novel=False, optimal_policy=False):
    '''run simulation'''
    #global n_sims
    #global n_paths
    n_params = len(sim_list)
    column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'largest lower', 'upper cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] + list(sim_list[0].cost.keys())
    sim_df = pd.DataFrame(columns = column_names)

    i=0

    try:
        while i < n_params:
            
            j=0
            while j < n_sims:    
                sim = copy.deepcopy(sim_list[i])
                column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'largest lower', 'upper cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] + list(sim.cost.keys())
                #del sim_list[i]
                sim.get_min_actv()
                sim.get_max_hold()
                sim.k=0       
                while sim.k < days:
                        if optimal_policy:
                            sim.simple_smart_order()
                        else:
                            sim.smart_order()

                        sim.demand_draw()
                        
                        if optimal_policy:
                            sim.simple_smart_fulfillment()
                        else:
                            sim.smart_fulfillment()

                        sim.update_backlog()
                        sim.update_inventory()

                        sim.k+=1
                        
                        
                    
                #sim.plot_sim_backlog()
                #sim.plot_sim_inventory()
                
                j+=1
                

                sim.cost_dict = copy.deepcopy(sim.cost)
                sim.largest_lower = max(sim.cost, key=sim.cost.get)
                sim.cost = max(sim.cost.values())
                new_row = pd.DataFrame(data=np.array([[sim.file_name, j, sim.sim_cost, sim.cost, sim.largest_lower, sim.upper_cost, sim.holding_cost, sim.backlog_cost, sim.fulfillment_cost, sim.ordering_cost]+ list(sim.cost_dict.values())]), columns=sim_df.columns)
                sim_df = pd.concat([sim_df,new_row], ignore_index=True)

                #plot_sim_cost_hist(sim_df, days,j)
                
            i+=1
            print("i:" + str(i) + '/' + str(n_params), end="")
            print("\r", end="")
        sum_sim_df = summarize_sims(sim_df)
        sum_sim_df.to_csv('newsvendoroutput_summary' + str(int(alpha*100)) + '.csv', sep='\t')
        sim_df.to_csv('newsvendoroutput' + str(int(alpha*100)) + '.csv', sep='\t')

        print("Simulation passed")
    

    except:
        print("Simulation failed")
    


def run_pos_sim(sim_list,alpha=1, novel=False, optimal_policy=False, lead_time=1):
    '''run simulation'''
    #global n_sims
    #global n_paths
    n_params = len(sim_list)
    column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'largest lower', 'upper cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] + list(sim_list[0].cost.keys())
    sim_df = pd.DataFrame(columns = column_names)

    i=0

    try:
        while i < n_params:
            
            j=0
            while j < n_sims:    
                sim = copy.deepcopy(sim_list[i])
                column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'largest lower', 'upper cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] + list(sim.cost.keys())
                #del sim_list[i]
                sim.get_min_actv()
                k=0       
                while k < days:
                        if optimal_policy:
                            sim.simple_smart_order()
                        else:
                            sim.smart_order()

                        sim.demand_draw()
                        
                        if optimal_policy:
                            sim.simple_smart_fulfillment()
                        else:
                            sim.smart_fulfillment()

                        sim.update_backlog()
                        sim.update_inventory()

                        k+=1
                        
                        
                    
                sim.plot_sim_backlog()
                sim.plot_sim_inventory()
                
                j+=1
                

                sim.cost_dict = copy.deepcopy(sim.cost)
                sim.largest_lower = max(sim.cost, key=sim.cost.get)
                sim.cost = max(sim.cost.values())
                new_row = pd.DataFrame(data=np.array([[sim.file_name, j, sim.sim_cost, sim.cost, sim.largest_lower, sim.upper_cost, sim.holding_cost, sim.backlog_cost, sim.fulfillment_cost, sim.ordering_cost]+ list(sim.cost_dict.values())]), columns=sim_df.columns)
                sim_df = pd.concat([sim_df,new_row], ignore_index=True)

                #plot_sim_cost_hist(sim_df, days,j)
                
            i+=1
            print("i:" + str(i) + '/' + str(n_params), end="")
            print("\r", end="")
        sum_sim_df = summarize_sims(sim_df)
        sum_sim_df.to_csv('newsvendoroutput_summary' + str(int(alpha*100)) + '.csv', sep='\t')
        sim_df.to_csv('newsvendoroutput' + str(int(alpha*100)) + '.csv', sep='\t')

        print("Simulation passed")
    

    except:
        print("Simulation failed")




def loadpickles(path,alpha=1):
    simlist = []
    if path == '':
        file_path = cwd + '/instances'+str(int(alpha*100))
    else: 
        file_path = path

    for root, dir, files, in os.walk(file_path):
        for file in files:
            if file.endswith(".pkl"):
                pklfile = file
                #print(pklfile)
                openpkl = open(file_path + '/' + pklfile , 'rb')
                loadedpkl = pickle.load(openpkl)
                #print(loadedpkl)
                simlist.append(Simulation({**loadedpkl['LP_solution'], **loadedpkl['instance'], **{'file_name':pklfile}},alpha=alpha))



    return simlist

def get_novels(path = '', novel=False):

    for alpha in alphas:
        
        for beta in betas:
            costs_df = pd.DataFrame(columns = ['lower cost mean'+str(beta), 'file name'])
            if path == '':
                file_path = cwd + '/instances'+str(int(alpha*100)) 
            else: 
                file_path = path
            
            for root, dir, files, in os.walk(file_path + '_novel' + str(int(beta*100)) ):
                for file in files:
                    if file.endswith(".pkl"):
                        pklfile = file
                        print(pklfile)
                        openpkl = open(file_path + '/' + pklfile , 'rb')
                        opennovelpkl = open(file_path + '_novel' + str(int(beta*100)) + '/' + pklfile , 'rb')
                        novelpkl = pickle.load(opennovelpkl)
                        costs_df.loc[len(costs_df)] = [novelpkl['LP_solution']['cost'], pklfile]
            costs_df.to_csv('novels' + str(int(alpha*100)) + '_novel' + str(int(beta*100)) + '.csv', sep='\t')


def comparepickles(path,alpha=1, beta=1, novel=False):
    novel_greater = 0
    if path == '':
        file_path = cwd + '/instances'+str(int(alpha*100))
    else: 
        file_path = path

    for root, dir, files, in os.walk(file_path + '_novel' + str(int(beta*100)) ):
        for file in files:
            if file.endswith(".pkl"):
                pklfile = file
                print(pklfile)
                openpkl = open(file_path + '/' + pklfile , 'rb')
                opennovelpkl = open(file_path + '_novel' + str(int(beta*100)) + '/' + pklfile , 'rb')
                oldpkl = pickle.load(openpkl)
                novelpkl = pickle.load(opennovelpkl)
                print(str(novelpkl['LP_solution']['cost']) + ' : ' + str(oldpkl['LP_solution']['cost']))
                if novelpkl['LP_solution']['cost'] > oldpkl['LP_solution']['cost']:
                    novel_greater +=1
        
                

                #print(loadedpkl)
                #simlist.append(Simulation({**loadedpkl['LP_solution'], **loadedpkl['instance'], **{'file_name':pklfile}},alpha=alpha))
            else:
                print("No files found!")
            print(str(novel_greater) + ' greater!')


    return None
###cost is a lower bound on average of sample paths
###ratio should be greater than 1, must be less than 2
###


"""
if __name__ == '__main__':

    begin_time = datetime.datetime.now()
    print(begin_time)

    #sim_list = loadpickles(path = '')

    get_novels(path='')

    run_sim()

    print(datetime.datetime.now() - begin_time)




"""


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
    
    

    
