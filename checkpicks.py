from pyparsing import alphanums
from dynamic_simulation import *
from optimals import *
from simulation import *
import csv

cwd = os.getcwd()

solve_LP = True 

#### Run Controls ####
run_ecom = False
load_ecom = False
run_manuf = False
load_manuf = False
run_prod = True
load_prod = False

alphas = .1

path = ''

cwd = os.getcwd()

new_path = cwd + '/instances'

def checkpickles(path='', alpha=.1):

    new_path = cwd + '/instances'+str(int(alpha*100))

    Qs = []

    for root, dir, files, in os.walk(new_path):
        for file in files:
            if file.endswith(".pkl"):
                pklfile = file
                print(pklfile)
                openpkl = open(new_path + '/' + pklfile , 'rb')
                loadedpkl = pickle.load(openpkl)
                Qs.append(loadedpkl['instance']['q'])
    
    file = open('Qs' + str(alpha) + '.csv', 'w+', newline ='') 
    with file:     
        write = csv.writer(file) 
        write.writerows(Qs) 
               
if __name__ == '__main__':

    checkpickles(path=cwd, alpha = alphas)