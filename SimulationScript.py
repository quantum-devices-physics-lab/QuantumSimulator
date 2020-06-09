from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import time
import multiprocessing as mp
import scipy.constants as sc
import pickle
import os
import re
import sys

# Returns the average photon number for a given temperature and frequency.
def calculate_n_th(T,w):
    return 1/(np.exp(sc.h*w*1e9/(sc.k*T))-1)

# Return the system hamiltonian.
def drive_Hamiltonian(a,wa,ga,wd_a,Aa,
                      b,wb,gb,wd_b,Ab,
                      r,wr):
    H= (wa-wd_a)*a.dag()*a + (wb-wd_b)*b.dag()*b + wr*r.dag()*r + Aa*(a.dag()+a) + Ab*(b.dag()+b) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
    return H

def create_parameters(N,T,c_ops,
                      a,n_th_a,rate_relaxation_a,rate_excitation_a,wa,ka,wd_a,Aa,ga,
                      b,n_th_b,rate_relaxation_b,rate_excitation_b,wb,kb,wd_b,Ab,gb,
                      r,n_th_r,rate_relaxation_r,rate_excitation_r,wr,kr):
    return {"N":N,
            "c_ops":c_ops,
            "a":a,
            "n_th_a":n_th_a,
            "rate_relaxation_a":rate_relaxation_a,
            "rate_excitation_a":rate_excitation_a,
            "b":b,
            "n_th_b":n_th_b,
            "rate_relaxation_b":rate_relaxation_b,
            "rate_excitation_b":rate_excitation_b,
            "r":r,
            "n_th_r":n_th_r,
            "rate_relaxation_r":rate_relaxation_r,
            "rate_excitation_r":rate_excitation_r,
            "wa":wa,
            "wb":wb,
            "wr":wr,
            "ga":ga,
            "gb":gb,
            "ka":ka,
            "kb":kb,
            "kr":kr,
            "T":T,
            "Aa":Aa,
            "Ab":Ab,
            "wd_a":wd_a,
            "wd_b": wd_b}

# Call this function to create a new task.        
def create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_a,wd_b,n_points,initial_parameter,final_parameter,idx,name):
    n_th_a = calculate_n_th(T,wa)
    n_th_b = calculate_n_th(T,wb)
    n_th_r = calculate_n_th(T,wr)

    rate_relaxation_a = ka*(1+n_th_a)
    rate_relaxation_b = kb*(1+n_th_b)
    rate_relaxation_r = kr*(1+n_th_r)

    rate_excitation_a = ka*(n_th_a)
    rate_excitation_b = kb*(n_th_b)
    rate_excitation_r = kr*(n_th_r)
    
    # The destruction operator
    a = tensor(destroy(N),qeye(N),qeye(N))
    b = tensor(qeye(N),destroy(N),qeye(N))
    r = tensor(qeye(N),qeye(N),destroy(N))
    
    c_ops = []

    if rate_excitation_a > 0.0:
        c_ops.append(np.sqrt(rate_excitation_a)*a.dag())

    if rate_relaxation_a > 0.0:
        c_ops.append(np.sqrt(rate_relaxation_a)*a)
        
    if rate_excitation_b > 0.0:
        c_ops.append(np.sqrt(rate_excitation_b)*b.dag())

    if rate_relaxation_b > 0.0:
        c_ops.append(np.sqrt(rate_relaxation_b)*b)
        
    if rate_excitation_r > 0.0:
        c_ops.append(np.sqrt(rate_excitation_r)*r.dag())

    if rate_relaxation_r > 0.0:
        c_ops.append(np.sqrt(rate_relaxation_r)*r)
    
    task = create_parameters(N,T,c_ops,
                             a,n_th_a,rate_relaxation_a,rate_excitation_a,wa,ka,wd_a,Aa,ga,
                             b,n_th_b,rate_relaxation_b,rate_excitation_b,wb,kb,wd_b,Ab,gb,
                             r,n_th_r,rate_relaxation_r,rate_excitation_r,wr,kr)
    task["n_points"] = n_points;
    task["initial_parameter"] = initial_parameter;
    task["final_parameter"] = final_parameter;
    task["idx"] = idx;
    task["name"] = name;
    task["H"] = drive_Hamiltonian(task['a'],task['wa'],task['ga'],task['wd_a'],task['Aa'],
                                  task['b'],task['wb'],task['gb'],task['wd_b'],task['Ab'],
                                  task['r'],task['wr'])
    return task;

def create_wd_a_sweep(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_a_initial,wd_a_final,wd_b,n_points,name):
    sweep = np.array([])
    wd_as = np.linspace(wd_a_initial,wd_a_final,n_points)
    for (idx,wd_a) in enumerate(wd_as):
        task = create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_a,wd_b,n_points,wd_a_initial,wd_a_final,idx,name)
        sweep = np.append(sweep,task)
    return sweep


class Experiment:
    def __init__(self,
                 tagname="",
                 sweeps=[],
                 n_points=0,
                 name_cases=[],
                 number_of_cases=[],
                 fock=0,
                 n_oscillators=0):
        
        self.tagname = tagname
        self.sweeps = np.array(sweeps)
        self.name_cases = np.array(name_cases)
        self.number_of_cases = {}
        for (name_case,number) in zip(self.name_cases,number_of_cases):
            self.number_of_cases[name_case] = number
        self.n_points = n_points
        self.fock = fock
        self.n_oscillators = n_oscillators
                 
    class Data:
        def __init__(self,
                     tagname,
                     name_cases,
                     number_of_cases,
                     n_points,
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
            self.tagname = tagname
            self.number_of_cases = number_of_cases
            self.name_cases = name_cases
            self.n_points = n_points
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

    def load(self,filename):
        
        file = open(filename,"rb")
        data = pickle.load(file)
        file.close()
        
        self.tagname = data.tagname
        self.number_of_cases = data.number_of_cases
        self.name_cases = data.name_cases
        self.n_points = data.n_points
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
        data  = Experiment.Data(self.tagname,
                                self.name_cases,
                                self.number_of_cases,
                                self.n_points,
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
            
    def process(self,l):

        self.purity = {}
        self.expect_a = {}
        
        purity_case = {}
        expect_a_case = {}
        
        self.tasks = {}
        self.var = {}
        self.rhos = {}

        for name_case in self.name_cases:
            self.var[name_case] = np.zeros(self.n_points,dtype=float)
            for i in range(0,self.number_of_cases[name_case]):
                purity_case[name_case+str(i)] = np.zeros(self.n_points,dtype=complex)
                expect_a_case[name_case+str(i)] = np.zeros(self.n_points,dtype=complex)
                self.rhos[name_case+str(i)] = np.zeros(self.n_points,dtype=Qobj)

        for result in l:
            self.tasks[result.task['name']] = result.task
            purity_case[result.task['name']][result.task['idx']] = result.purity
            expect_a_case[result.task['name']][result.task['idx']] = result.expect_a
            self.rhos[result.task['name']][result.task['idx']] = result.rho

        self.a = result.a
        self.b = result.b
        self.r = result.r
        
        for name_case in self.name_cases:
            pp = []
            ea = []
            v = []
            for i in range(0,self.number_of_cases[name_case]):
                p = purity_case[name_case+str(i)]
                na = expect_a_case[name_case+str(i)]
                task = self.tasks[name_case+str(i)]

                pp=np.append(pp,p)
                ea = np.append(ea,na)
                v = np.append(v,task[name_case])
    
            self.purity[name_case]=np.reshape(pp,(self.number_of_cases[name_case],self.n_points))
            self.expect_a[name_case]=np.reshape(ea,(self.number_of_cases[name_case],self.n_points))
            self.var[name_case]=v
            
# A class that is actually a struct of the data produced by the simulation.
# The really important variable is rho. The rest is for management simplicity.
class SimulationData():
    def __init__(self,name,task,expect_a,purity,rho):
        self.name = name
        self.task = task
        self.rho = rho
        self.purity  = purity
        self.expect_a = expect_a
        self.a = task['a']
        self.b = task['b']
        self.r = task['r']
  
# The parallel function simulate.
def simulate(logger,experiment,f,datetime_snapshot=""):

    # Obtain the number of cpus to be used
    task_count = len(experiment.sweeps)
    cpu_count = mp.cpu_count()
    
    manager = mp.Manager()
    
    l = manager.list()

    dirName = ""
    if datetime_snapshot != "":
        dirName = "{}_data_backup".format(datetime_snapshot)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
    
    logger.info("Starting Simulation")

    logger.info("#CPU {}".format(cpu_count))

    # Parellel code
    try:
        t_start = time.time()
        time_1 = []

        # get a pool object and apply the function execute for each task.
        pool = mp.Pool(processes = cpu_count)

        results = [pool.apply_async(f,args=(sweep,l,dirName)) for sweep in experiment.sweeps]

        # For each hour passed, the progress is logged. The loops continue until all tasks have been finished.
        passedAnHour = 0
        while True:
            incomplete_count = sum(1 for x in results if not x.ready())

            if incomplete_count == 0:
                logger.info("[100.0%] of the simulation calculated")
                logger.info("All done! Total time: %s"%datetime.timedelta(seconds=int(dif_time)))
                break
            else:
                p = float(task_count - incomplete_count) / task_count * 100
                dif_time = (time.time()-t_start)
          

            if(int(dif_time/3600)-passedAnHour > 0):
                passedAnHour = int(dif_time/3600)
                logger.info("[%4.1f%%] of the simulations calculated, progress time: %s "%(p,datetime.timedelta(seconds=int(dif_time))) )
            
            
            time.sleep(1)

        
        # When it is finished, get all the data.
        while not all([ar.ready() for ar in results]):
            for ar in results:
                ar.wait(timeout=0.1)


        pool.close()
        pool.join
        
        for ar in results:
            data = ar.get()

        logger.info("Formatting data")
        experiment.process(l)

    except Exception as e:
        logger.exception(e)
        pool.terminate()
        pool.join()
        raise e

    
    return experiment;

# Main processing function execute.
def execute(sweep,l,dirName=""):

    for task in sweep:

        # The steadysate function from QuTiP
        rho_ss = steadystate(task['H'], task['c_ops'])
        
        purity = (rho_ss*rho_ss).tr()
        expect_a = (rho_ss*task['a']).tr()

        data = SimulationData(task['name'],
                              task,
                              expect_a,
                              purity,
                              rho_ss)
        print("Acquired data {} point {}".format(data.task["name"],data.task["idx"]))
        l.append(data)
        
        if dirName != "":
            filename = dirName+"/{}_{}".format(task['name'],task['idx'])
            file = open(filename,"wb")
            pickle.dump(data,file)
            file.close()
        
        
    return data


def run_experiment1(name):
    
    # This is a setup for the logging system
    filename = name.replace(" ","_")    
    logging.basicConfig(level=logging.DEBUG) 
    logger = logging.getLogger(name)
    
    datetime_snapshot = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    handler = logging.FileHandler('{}_{}.log'.format(datetime_snapshot,filename))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    sweeps = []
    N = 3
    number_of_oscillators = 3
    wa = 5.1
    wb = 5.7
    wr = 1.0
    ka = 1.0e-4
    kb = 1.0e-4
    kr = 10.0e-3
    gb = 5e-3
    wd_begin = wa - 500*1e-6
    wd_end = wa + 500*1e-6
    T = 10e-3
    n_points = 1
    
    number_of_cases_ga = 10
    number_of_cases_Ab = 5
    
    gas = np.linspace(0,10e-3,number_of_cases_ga)
    Aa = 5.0e-6
    wd_b = wb - 0.0001
    Abs = np.linspace(0,40e-6,number_of_cases_Ab)
    Ab = 5.0e-6

 
    n_case = 0
    for ga in gas:
        logger.info("Creating sweep {}".format("ga{}".format(n_case)))
        sweeps.append(create_wd_a_sweep(N,
                                        wa,
                                        wb,
                                        wr,
                                        ga,
                                        gb,
                                        ka,
                                        kb,
                                        kr,
                                        T,
                                        Aa,
                                        Ab,
                                        wd_begin,
                                        wd_end,
                                        wd_b,
                                        n_points,
                                        "ga{}".format(n_case)))
        n_case = n_case + 1
        

    
    ga = 5e-3
    n_case = 0
    for Ab in Abs:
        logger.info("Creating sweep {}".format("Ab{}".format(n_case)))
        sweeps.append(create_wd_a_sweep(N,
                                        wa,
                                        wb,
                                        wr,
                                        ga,
                                        gb,
                                        ka,
                                        kb,
                                        kr,
                                        T,
                                        Aa,
                                        Ab,
                                        wd_begin,
                                        wd_end,
                                        wd_b,
                                        n_points,
                                        "Ab{}".format(n_case)))
        n_case = n_case + 1
    
    logger.info("Registering experiment")
    experiment = Experiment(name,
                            sweeps,
                            n_points,
                            ['ga','Ab'],
                            [number_of_cases_ga,number_of_cases_Ab],
                            fock,
                            number_of_oscillators)

    
    
    experiment = simulate(logger,experiment,execute)
    
    logger.info("Saving full experiment data")
    filename = "{}_{}.data".format(datetime_snapshot,name.replace(" ","_"))
    experiment.save(filename)
    
    logger.info("End of simulation")
    handler.close()
    logger.removeHandler(handler)
    logging.shutdown()

if __name__ == "__main__":
    run_experiment1("2 Cavities 1 Resonator Drive Simulation N3 ga and Ab V2")
    
    

    

