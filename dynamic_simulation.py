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

n_sims = 50
days = 100

burn_in = 0

days_mins_burn = days - burn_in

csv_name = 'newsvendoroutput.csv'

#alphas = [ 2, 3, 4, 5]
#
#betas = [.5, .8, .9, 1] 


#product shortage cost as backlog cost
#percentage of holding cost is ordering cost
#exclude manuf
#ecom app and produc application


#m num of resources - len of c
#resource i, product j, activity k.
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
        # Set attributes from dictionary
        for key in dictionary:
            setattr(self, key, dictionary[key])
        self.I = np.zeros(dictionary['A'].shape[0])  #Inventory
        self.I_hist = np.zeros(dictionary['A'].shape[0])
        self.BL = np.zeros(dictionary['p'].shape[0])  #Backlog
        self.BL_hist = np.zeros(dictionary['p'].shape[0])
        self.z = np.zeros(len(self.c)) #processing activities
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
        self.demand = np.zeros(dictionary['p'].shape[0])
        self.min_actv = np.zeros((self.q.shape[0],self.p.shape[0]))
        self.h_bar = np.zeros(self.p.shape[0])
        self.x_k = np.zeros(len(self.q)) 
        self.num_fill = np.zeros(len(self.q))

        if hasattr(self, 'lead_time'):
            #self.u_k = (1/(self.lead_time+1))*(self.q + np.matmul(self.A.T,(self.c - (self.lead_time+1)*self.h)))
            #((1-alpha*(self.lead_time+1))*np.matmul(loadedpkl['instance']['c'],loadedpkl['instance']['A'])+loadedpkl['instance']['q'])/(lead_time+1)
            if self.lead_time > 0:
                self.theta = 0
                self.z_hist = np.zeros((len(self.c), self.lead_time+1))
                self.B_bar_hist = np.zeros((len(self.q), self.lead_time))
                self.B_tilde_hist = np.zeros((len(self.q), self.lead_time+2))
                self.D_hat_hist = np.zeros((len(self.q), self.lead_time+1))
                self.L_k = (self.theta+1) * self.x.mean(axis=0)
                self.r = (self.theta+1) * (self.r)
                self.I = copy.deepcopy(self.r)
                self.h = self.c * alpha
        else:
            self.lead_time = 0
            self.k = 0

            

        

    def get_min_actv(self):
        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.c ,self.A[:,j])
                cost_dict[j] = self.q[j] + actv_cost
            self.min_actv[min(cost_dict,key=cost_dict.get),i] += 1 #lowest activity for given product
    
    def get_max_hold(self):
        h_bar = []
        for i in range(self.p.shape[0]):
            pos_actvs = np.where(self.B[:,i] == 1)[0] #activities that can fulfill each product
            cost_dict = {}
            for j in pos_actvs:
                actv_cost = np.matmul(self.h ,self.A[:,j])
                cost_dict[j] = self.h[j] + actv_cost
            h_bar.append(max(cost_dict.values()) )
            #self.h_bar[max(cost_dict,key=cost_dict.get)] +=  #lowest activity for given product
        self.h_bar = np.array(h_bar)

    def smart_order(self):
        self.z = self.r - self.I + np.matmul(self.A,np.matmul(self.min_actv, self.BL)) #base stock policy
        self.z[self.z < 0] = 0
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
        if self.lead_time == 0:
            self.x_k = copy.deepcopy(self.x[index])
        

    def get_cheapest(self):
        self.A = self.A[np.array(np.argsort(self.p))]



    def update_inventory(self):
        if self.lead_time == 0:
            self.I += self.z - np.matmul(self.A, self.x_k)
        else:
            self.I += self.z_hist[:,self.lead_time] - np.matmul(self.A,self.x_k)
        #if np.any(self.I < 0):
        #    pdb.set_trace()
        if self.k >= burn_in:
            self.sim_cost += np.matmul(self.I, self.h)
            self.holding_cost += np.matmul(self.I, self.h) #holding cost
        self.I_hist = np.vstack([self.I_hist, self.I])


    def update_backlog(self):
        self.BL += self.demand - np.matmul(self.x_k, self.B)
         #np.any(self.BL < 0)
        if self.k >= burn_in:
            self.sim_cost += np.matmul(self.BL, self.p)
            self.backlog_cost += np.matmul(self.BL, self.p) #backlog cost
        self.BL_hist = np.vstack([self.BL_hist, self.BL])
    

    def update_D_hat(self):
        self.D_hat = self.phi * np.matmul(self.demand, self.B.T)
        self.D_hat_hist = np.roll(self.D_hat_hist, 1, axis=1)
        self.D_hat_hist[:,0] = self.D_hat
    
    def update_B_tilde(self):
        self.B_tilde = np.matmul(self.min_actv, self.psi * self.demand)
        self.B_tilde_hist = np.roll(self.B_tilde_hist, 1, axis=1)
        self.B_tilde_hist[:,0] = self.B_tilde

    def update_X_tilde(self):
        self.X_tilde = np.minimum(self.D_hat + self.B_hat, self.L_k + self.B_hat - np.sum(self.D_hat_hist[:,1:self.lead_time+1], axis=1))
    
    def update_B_bar(self):
        self.B_bar += self.D_hat + self.B_tilde - self.x_k
        self.B_bar_hist = np.roll(self.B_bar_hist, 1, axis=1)
        self.B_bar_hist[:,0] = self.B_bar
    
    def update_B_hat(self):
        self.B_hat += self.D_hat - self.X_tilde
    
    def update_D_hat_rand(self):
        self.D_hat = self.X_tilde
        self.D_hat_hist = np.roll(self.D_hat_hist, 1, axis=1)
        self.D_hat_hist[:,0] = self.D_hat
    
    def update_B_tilde_rand(self):
        self.B_tilde = np.matmul(self.min_actv, (self.demand - np.matmul(self.B.T, self.X_tilde)))
        self.B_tilde_hist = np.roll(self.B_tilde_hist, 1, axis=1)
        self.B_tilde_hist[:,0] = self.B_tilde

    def update_X_tilde_rand(self):
        mask = np.where((self.d_L >= self.demand).all(axis=1))
        if len(mask[0]) == 0:
            lst_gt = []
            for i in range(self.d_L.shape[0]):
                num_gt = np.sum(self.d_L[i] > self.demand)
                lst_gt.append(num_gt)
            index = lst_gt.index(max(lst_gt))
        else:
            index = seed_rand.choice(list(mask[0]))
        d_L = np.array([self.d_L[index]])
        x_cut = np.array([self.x[index]])
        self.D_jt = np.array([np.matmul(self.demand, self.B.T)])
        numerator = x_cut * self.D_jt
        denom = np.matmul(d_L, self.B.T)
        W_kt = numerator/denom
        W_kt[0][np.isnan(W_kt[0])] = 0
        #fulfill_propor = (self.D_jt/denom).T
        #W_kt = (np.matmul(x_cut.T, d_L) * fulfill_propor).max(axis=0)
        num = (self.I + self.z_hist[:,self.lead_time] - np.matmul(self.A, self.X_tilde))
        num[num<0] = 0
        X_alt = np.empty((0,))
        for i in range(self.A.shape[1]):
            X_alt = np.append(X_alt, np.nanmin(num/self.A[:,i])) 
            #X_alt = np.nanmin(((self.I + self.z_hist[:,self.lead_time] - np.matmul(self.A, self.X_tilde))/self.A.T), axis=1)
        self.X_tilde = np.minimum(W_kt[0], X_alt)
        a=1

    def update_determ_vars(self):
        self.update_D_hat()
        self.update_B_tilde()
        self.update_X_tilde()
        self.update_B_bar() # after dhat, b_tilde
        self.update_B_hat() #after dhat and x_tilde


    def deterministic_order(self):
        self.z = self.r - self.I - np.sum(self.z_hist[:,0:self.lead_time], axis=1) + np.matmul(self.A, self.B_bar)
        self.z[self.z < 0] = 0
        self.z_hist = np.roll(self.z_hist, 1, axis=1)
        self.z_hist[:,0] = self.z
        if self.k >= burn_in:
            self.sim_cost += np.matmul(self.z, self.c) #ordering cost
            self.ordering_cost += np.matmul(self.z, self.c)


    def deterministic_fulfillment(self):
        self.x_k = self.X_tilde + self.B_tilde_hist[:,self.lead_time+1]
        if self.k >= burn_in:
            self.fulfillment_cost += np.sum(self.q * self.x_k)
            self.sim_cost += np.sum(self.q * self.x_k) #fulfillment cost

    def check_equalities(self):
        if (np.round(self.I,2) != np.round(self.r - np.matmul(self.A, np.sum(self.D_hat_hist[:,0:self.lead_time+1], axis=1) - self.B_hat),2)).any():
            print('Equality 1 doesnt hold')
        if (np.round(self.z,3) != np.round(np.matmul(self.A, (self.D_hat_hist[:,1] + self.B_tilde_hist[:,1])),3)).any():
            print('Equality 2 doesnt hold')
        if (np.round(self.B_bar,2) != np.round(self.B_hat + np.sum(self.B_tilde_hist[:,0:self.lead_time+1], axis=1),2)).any():
            print('Equality 3 doesnt hold!')
        if (np.round(self.demand,2) != np.round(np.matmul(self.B.T, (self.D_hat + self.B_tilde)))).all():
            print("Demand equality doesnt hold")

        

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

    
    
def summarize_sims(df, lead_time):
    df = df.drop(['sim number', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'], axis=1)
    df['simulation cost'] = pd.to_numeric(df['simulation cost'])
    df['cost'] = pd.to_numeric(df['cost'])
    if lead_time == 0:
        df['upper cost'] = pd.to_numeric(df['upper cost'])
    #df['largest lower'] = pd.to_numeric(df['largest lower'])
    df['sim_cost_per_day'] = df['simulation cost']/days_mins_burn
    #df = df.groupby(['file name'], as_index=False).sem()
    #df = df.groupby(['file name'], as_index=False).mean()
    if lead_time == 0:
        cols = df.columns.drop(['file name', 'largest lower'])
    else:
        cols = df.columns.drop(['file name'])    
    df[cols] = df[cols].apply(pd.to_numeric)
    if lead_time == 0:
        df = df.groupby(['file name', 'largest lower']).agg([np.mean, sem])
    elif lead_time > 0:
        df = df.groupby(['file name']).agg([np.mean, sem])
    df.columns = df.columns.map('_'.join)
    df = df.reset_index()
    df['lower cost ratio'] = df['sim_cost_per_day_mean']/df['cost_mean']
    if lead_time == 0:
        df['upper cost ratio'] = df['sim_cost_per_day_mean']/df['upper cost_mean']
    if lead_time == 0:
        df_out = df.drop(['upper cost_sem', 'upper cost_mean', 'upper cost ratio'], axis=1)
        return df_out
    return df

def plot_sim_cost_hist(sim_df, days,j):
        #fig = plt.figure()
        #fig.add_subplot(111)
    time.sleep(1.5)
    print(str(j) + ':' + str(np.mean(sim_df['simulation cost']/days)/np.mean(sim_df['cost'])))
    #plt.plot(np.mean(sim_df['simulation cost']/days)/np.mean(sim_df['cost']))
    #plt.title(sim_df['file name'].iloc[0])
    #plt.show(block=False)
 

def run_pos_sim(sim_list,alpha=1, novel=False, optimal_policy=False,lead_time=0, lead_policy = 'proportional'):
    '''run simulation'''
    #global n_sims
    #global n_paths
    n_params = len(sim_list)
    column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] #+ list(sim_list[0].cost.keys())
    sim_df = pd.DataFrame(columns = column_names)
    max_theta_df = pd.DataFrame(columns = column_names)

    i=0

    try:
        while i < n_params:
            sim = copy.deepcopy(sim_list[i])
            sim.get_min_actv()
            #sim.get_max_hold()
            #mu_j = sim.d.mean(axis=0)
            #numer = np.matmul((sim.p + sim.h_bar), (mu_j * variation(sim.d, axis=0)**2))
            #denom = np.matmul(mu_j, np.matmul(sim.min_actv.T, sim.q))
            #max_theta = np.sqrt(numer/denom)
            max_theta = 3
            max_theta_cols = ['file name', 'simulation cost_mean', 'simulation cost_sem', 'cost_mean', 'cost_sem', 'sim_cost_per_day_mean', 'sim_cost_per_day_sem', 'lower cost ratio']
            thetas_dict = {}
            chosen_theta = max_theta/10
            ratios_lst = []
            sim.L_k = (chosen_theta+1) * sim.x.mean(axis=0)
            sim.r = (chosen_theta+1) * (sim.r)
            sim.I = copy.deepcopy(sim.r)
            while chosen_theta <= max_theta: 
                j=0
                column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] #+ list(sim_list[0].cost.keys())
                sim_df = pd.DataFrame(columns = column_names)
                while j < n_sims:
                    
                    sim = copy.deepcopy(sim_list[i])
                    sim.get_min_actv()
                    sim.L_k = (chosen_theta+1) * sim.x.mean(axis=0)
                    sim.r = (chosen_theta+1) * (sim.r)
                    sim.I = copy.deepcopy(sim.r)
                    #sim.get_max_hold()
                    #mu_j = sim.d.mean(axis=0)
                    #numer = np.matmul((sim.p + sim.h_bar), (mu_j * variation(sim.d, axis=0)**2))
                    #denom = np.matmul(mu_j, np.matmul(sim.min_actv.T, sim.q))
                    #max_theta = np.sqrt(numer/denom) 
                    max_theta = 3
                    column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] 
                    # if novel_lower == True   
                    #column_names = ['file name', 'sim number', 'simulation cost', 'cost', 'largest lower', 'upper cost', 'holding cost', 'backlog cost', 'fulfillment cost', 'ordering cost'] #+ list(sim.cost.keys())
                    #del sim_list[i]
                    #sim.get_min_actv()
                    #sim.get_max_hold()
                    sim.k=0
                    while sim.k < days:
                        if lead_time == 0:
                            # Ordering
                            if optimal_policy:
                                sim.simple_smart_order()
                            else:
                                sim.smart_order()
                            # Demand realized
                            sim.demand_draw()
                            # Fulfillment
                            if optimal_policy:
                                sim.simple_smart_fulfillment()
                            else:
                                sim.smart_fulfillment()
                        elif lead_time > 0 and lead_policy == 'proportional':
                            sim.deterministic_order()
                            sim.demand_draw()
                            sim.update_D_hat()
                            sim.update_B_tilde()
                            sim.update_X_tilde()
                            sim.deterministic_fulfillment()
                            sim.update_B_bar() # after dhat, b_tilde
                            sim.update_B_hat() #after dhat and x_tilde
                        elif lead_time > 0 and lead_policy == 'randomized':
                            sim.deterministic_order()
                            sim.demand_draw()
                            sim.update_X_tilde_rand()
                            sim.update_D_hat_rand()
                            sim.update_B_tilde_rand()
                            sim.deterministic_fulfillment()
                            sim.update_B_bar() # after dhat, b_tilde
                            sim.update_B_hat() #after dhat and x_tilde
                        
                        # ordering is done at beginning of period
                        #t is at the end of the period for I and BL
                        #fulfillment is decided after demand is drawn


                        # Update state variables
                        sim.update_backlog()
                        sim.update_inventory()
                        sim.check_equalities()
                        # End of day
                        sim.k+=1
                            
                            
                        
                    #sim.plot_sim_backlog()
                    #sim.plot_sim_inventory()
                    
                    j+=1
                    sim.cost_dict = copy.deepcopy(sim.cost)
                    #if novel_lower == True sim.largest_lower = max(sim.cost, key=sim.cost.get)
                    #sim.cost = max(sim.cost.values())
                    #new_row = pd.DataFrame(data=np.array([[sim.file_name, j, sim.sim_cost, sim.cost, sim.largest_lower, sim.upper_cost, sim.holding_cost, sim.backlog_cost, sim.fulfillment_cost, sim.ordering_cost]+ list(sim.cost_dict.values())]), columns=sim_df.columns)

                    new_row = pd.DataFrame(data=np.array([[sim.file_name, j, sim.sim_cost, sim.cost, sim.holding_cost, sim.backlog_cost, sim.fulfillment_cost, sim.ordering_cost]]), columns=sim_df.columns)
                    sim_df = pd.concat([sim_df,new_row], ignore_index=True)

                    #plot_sim_cost_hist(sim_df, days,j)
                hold_df = summarize_sims(sim_df, sim.lead_time)
                hold_df['chosen_theta'] = chosen_theta
                thetas_dict[hold_df['lower cost ratio'][0]] = hold_df
                ratios_lst.append(hold_df['lower cost ratio'][0])
                chosen_theta += max_theta/10
                if (len(ratios_lst) >=2) and  (ratios_lst[-2] < ratios_lst[-1]):
                    break
            max_theta_df = pd.concat([max_theta_df, thetas_dict[min(thetas_dict.keys())]], ignore_index=True)        
            i+=1
            print("i:" + str(i) + '/' + str(n_params), end="")
            print("\r", end="")
        max_theta_df.to_csv(cwd + '/pos_leads/' + 'newsvendoroutput_summary' + str(int(alpha*100)) + '.csv', sep='\t')
        #sum_sim_df = summarize_sims(sim_df, lead_time)
        #sum_sim_df.to_csv('newsvendoroutput_summary' + str(int(alpha*100)) + '.csv', sep='\t')
        #sim_df.to_csv('newsvendoroutput' + str(int(alpha*100)) + '.csv', sep='\t')

        print("Simulation passed")
    

    except:
        print("Simulation failed")
    


def run_sim(sim_list,alpha=1, novel=False, optimal_policy=False, lead_time=1):
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
        sum_sim_df = summarize_sims(sim_df, lead_time)
        sum_sim_df.to_csv('newsvendoroutput_summary' + str(int(alpha*100)) + '.csv', sep='\t')
        sim_df.to_csv('newsvendoroutput' + str(int(alpha*100)) + '.csv', sep='\t')

        print("Simulation passed")
    

    except:
        print("Simulation failed")




def loadpickles(path,alpha=1, simple_network=False, lead_time=0, zero_order=False):
    simlist = []
    if lead_time == 0:
        file_path = cwd + '/instances'+str(int(alpha*100))
    elif lead_time > 0 and not simple_network:
        file_path = cwd + '/pos_leads/real_network/lead_out'+str(int(lead_time))
    elif lead_time > 0 and simple_network:
        file_path = cwd + '/pos_leads/simple_network/instances'+str(int(alpha*100))
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
                if lead_time == 0:
                    simlist.append(Simulation({**loadedpkl['LP_solution'], **loadedpkl['instance'], **{'file_name':pklfile}, **{'zero_order': zero_order}},alpha=alpha))
                elif lead_time > 0:
                    simlist.append(Simulation({**loadedpkl['LP_solution'], **loadedpkl['instance'], **loadedpkl['pos_leads'], **{'file_name':pklfile}},alpha=alpha))



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
    
    

    
