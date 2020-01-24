from qutip import *
import numpy as np
from QuantumSimulator import *

class Experiment:
    class Data:
        def __init__(self,
                     number_of_cases,
                     n_points,
                     name_cases,
                     fock,
                     n_oscillators,
                     purity,
                     expect_a,
                     var,
                     rhos,
                     tasks,
                     a,
                     b,
                     r):
            self.number_of_cases = number_of_cases
            self.n_points = n_points
            self.name_cases = name_cases
            self.fock = fock
            self.n_oscillators = n_oscillators
            self.purity = purity
            self.expect_a = expect_a
            self.var = var
            self.rhos = rhos
            self.tasks = tasks
            self.a = a
            self.b = b
            self.r = r
        
    def __init__(self,
                 dirname,
                 number_of_cases,
                 n_points,
                 name_cases,
                 N,
                 n_oscillators):
        self.dirname=dirname
        self.number_of_cases = number_of_cases
        self.n_points = n_points
        self.name_cases = name_cases
        self.fock = N
        self.n_oscillators = n_oscillators
        
    def process(self):
        self.purity = {}
        self.expect_a = {}
        self.var = {}
        self.rhos = {}
        self.tasks = {}
        self.rhos = {}
        for name_case in self.name_cases:
            pp = []
            ea = []
            v = []
            for i in range(0,self.number_of_cases):
                n_points,task,rhos,na,p,a,b,r = get_graphs_case(self.dirname,name_case+str(i),self.fock,n_points=self.n_points)
                self.tasks[name_case+str(i)] = task
                self.rhos[name_case+str(i)] = np.reshape(rhos,(self.n_points,
                                                               self.fock**self.n_oscillators,
                                                               self.fock**self.n_oscillators))
                pp=np.append(pp,p)
                ea = np.append(ea,na)
                v = np.append(v,task[name_case])
            self.purity[name_case]=np.reshape(pp,(self.number_of_cases,self.n_points))
            self.expect_a[name_case]=np.reshape(ea,(self.number_of_cases,self.n_points))
            self.var[name_case]=v
        self.a = a
        self.b = b
        self.r = r
        
    def rho(self,name_case,idx):
        dims = self.n_oscillators*[self.fock]
        return Qobj(self.rhos[name_case][idx],dims=[dims,dims])

    def load(self,filename):
        
        file = open(filename,"rb")
        data = pickle.load(file)
        file.close()
        
        self.number_of_cases = data.number_of_cases
        self.n_points = data.n_points
        self.name_cases = data.name_cases
        self.fock = data.fock
        self.n_oscillators = data.n_oscillators
        self.purity = data.purity
        self.expect_a = data.expect_a
        self.var = data.var
        self.rhos = data.rhos
        self.tasks = data.tasks
        self.a = data.a
        self.b = data.b
        self.r = data.r
        
        del data
    
    def save(self,filename):
        data  = Experiment.Data(self.number_of_cases,
                               self.n_points,
                               self.name_cases,
                               self.fock,
                               self.n_oscillators,
                               self.purity,
                               self.expect_a,
                               self.var,
                               self.rhos,
                               self.tasks,
                               self.a,
                               self.b,
                               self.r)
        file = open(filename,"wb")
        pickle.dump(data,file)
        file.close()
        
        
        del data
    

if __name__ == "__main__":

    
    tasks = []
    
    factor = 2.0*np.pi*1e9
    
    N = 4
    wa = 5.1
    wb = 5.7
    wr = 1.0
    ka = 1.0e-4
    kb = 1.0e-4
    kr = 1.0e-2
    gb = 0.05
    wd_begin = wa - 50*1e-6
    wd_end = wa + 50*1e-6
    T = 10e-3
    n_points = 50
    n_case = 0
    
    gas = np.linspace(0.0,2,5) *1e-3
    A = 5e-6
    
    for ga in gas:
        tasks.append(create_task(N,
                                 wa,
                                 wb,
                                 wr,
                                 ga,
                                 gb,
                                 ka,
                                 kb,
                                 kr,
                                 T,
                                 A,
                                 wd_begin, # begin_w
                                 wd_end, # end_w
                                 n_points,
                                 0, #idx
                                 "case_ga{}".format(n_case))) #name
        
        n_case = n_case + 1
        
    n_case=0
        
    ga = 2 *1e-3
    As = np.linspace(0.0,5,5) *1e-6
    
    for A in As:
        tasks.append(create_task(N,
                                 wa,
                                 wb,
                                 wr,
                                 ga,
                                 gb,
                                 ka,
                                 kb,
                                 kr,
                                 T,
                                 A,
                                 wd_begin, # begin_w
                                 wd_end, # end_w
                                 n_points,
                                 0, #idx
                                 "case_A{}".format(n_case))) #name
        
        n_case = n_case + 1
        
        
    

    simulate("2 Cavities 1 Resonator Drive Simulation",tasks)
    
