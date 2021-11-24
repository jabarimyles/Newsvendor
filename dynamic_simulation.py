import argparse
import pandas as pd
import pickle
import os

cwd = os.getcwd()



class Simu(object):

    def __init__(self, dictionary):

        # Set attributes from dictionary
        for key in dictionary:
            setattr(self, '_'+key, dictionary[key])

    # Getter/setter stuff
    @property
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
                print(loadedpkl)
                simlist.append(Simu(loadedpkl['instance']))
            else:
                lipgui.msgbox("No files found!")
    
    return simlist



if __name__ == '__main__':

    sim_list = loadpickles(path = '')

    
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
    
    

    

