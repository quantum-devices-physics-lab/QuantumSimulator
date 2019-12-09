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


def calculate_n_th(T,ω):
    return 1/(np.exp(sc.hbar*ω/(sc.k*T))-1)

def drive_Hamiltonian(a,ωa,b,ωb,r,ωr,ga,gb,ωd,A):
    H= (ωa-ωd)*a.dag()*a + ωb*b.dag()*b + ωr*r.dag()*r + A*(a.dag()+a) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
    return H

def add_c_ops(c_ops,rate_excitation,rate_relaxation,op):
    if rate_excitation > 0.0:
        c_ops.append(np.sqrt(rate_excitation)*op.dag())

    if rate_relaxation > 0.0:
        c_ops.append(np.sqrt(rate_relaxation)*op)
        
    return c_ops

class SimulationData():
    def __init__(self,task,a,expect_a,purity,b,r,rho):
        self.task = task
        self.rho = rho
        self.purity  = purity
        self.expect_a = expect_a
        self.a = a
        self.b = b
        self.r = r
        
def create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd_begin,ωd_end,n_points,idx,name):
    return {"N":N,
            "ωa":ωa,
            "ωb":ωb,
            "ωr":ωr,
            "ga":ga,
            "gb":gb,
            "κa":κa,
            "κb":κb,
            "κr":κr,
            "T":T,
            "A":A,
            "ωd_begin":ωd_begin,
            "ωd_end":ωd_end,
            "n_points":n_points,
            "idx":idx,
            "name":name}

def execute(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd_begin,ωd_end,n_points,name,dirName):
    
    n_th_a = calculate_n_th(T,ωa)
    n_th_b = calculate_n_th(T,ωb)
    n_th_r = calculate_n_th(T,ωr)

    rate_relaxation_a = κa*(1+n_th_a)
    rate_relaxation_b = κb*(1+n_th_b)
    rate_relaxation_r = κr*(1+n_th_r)

    rate_excitation_a = κa*(n_th_a)
    rate_excitation_b = κb*(n_th_b)
    rate_excitation_r = κr*(n_th_r)
    
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
        
   
    
    ωds = np.linspace(ωd_begin,ωd_end,n_points)*factor
    for idx,ωd in enumerate(ωds):
        H = drive_Hamiltonian(a,ωa,b,ωb,r,ωr,ga,gb,ωd,A)
        rho_ss = steadystate(H, c_ops)
        purity = (rho_ss*rho_ss).tr()
        expect_a = (rho_ss*a).tr()
        
        data = SimulationData(create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd_begin,ωd_end,n_points,idx,name),
                              a,
                              expect_a,
                              purity,
                              b,
                              r,
                              rho_ss)
        
        filedata_name = "{}/data_{}_{}".format(dirName,data.task["name"],data.task["idx"])
        file = open(filedata_name,"wb")
        print("Saving acquired data {} point {}".format(data.task["name"],data.task["idx"]))
        pickle.dump(data,file)
        file.close()
        
    return data

def simulate(name,tasks):
    
    filename = name.replace(" ","_")
    
    logging.basicConfig(level=logging.DEBUG) 
    logger = logging.getLogger(name)

    handler = logging.FileHandler('log_{}_{}.log'.format(filename,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    
    task_count = len(tasks)
    
    cpu_count = mp.cpu_count()
        
    logger.info("Starting Simulation")
    
    dirName = "data_{}_{}".format(filename,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # Create target Directory if don't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        logger.info("Directory {} Created ".format(dirName))
    else:    
        logger.info("Directory {} already exists".format(dirName))

    logger.info("#CPU {}".format(cpu_count))
    
    try:
        t_start = time.time()
        time_1 = []

        pool = mp.Pool(processes = cpu_count)

        results = [pool.apply_async(execute,args=(task["N"],
                                                  task["ωa"],
                                                  task["ωb"],
                                                  task["ωr"],
                                                  task["ga"],
                                                  task["gb"],
                                                  task["κa"],
                                                  task["κb"],
                                                  task["κr"],
                                                  task["T"],
                                                  task["A"],
                                                  task["ωd_begin"],
                                                  task["ωd_end"],
                                                  task["n_points"],
                                                  task["name"],
                                                  dirName),callback=None,error_callback=None) for task in tasks]

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

        while not all([ar.ready() for ar in results]):
            for ar in results:
                ar.wait(timeout=0.1)


        pool.close()
        pool.join

        
        
        for ar in results:
            data = ar.get()
        #    logger.info("Saving acquired data {} point {}".format(data.task["name"],data.task["idx"]))
        #    filedata_name = "{}/data_{}_{}".format(dirName,data.task["name"],data.task["idx"])
        #    file = open(filedata_name,"wb")
        #    pickle.dump(ar.get(),file)
        #    file.close()

    except Exception as e:
        logger.exception(e)
        pool.terminate()
        pool.join()
        raise e
    
    logger.info("End of simulation")

    handler.close()
    logger.removeHandler(handler)
    logging.shutdown()
  
def unpack_simulation_data(filename):
    file = open(filename,"rb")
    data = pickle.load(file)
    file.close()
    return data

def add_simulation_experiment(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,A,T,n_points,begin_ω,end_ω,name,tasks,factor=2.0*np.pi*1e9):
    ωa = ωa * factor
    ωb = ωb * factor
    ωr = ωr * factor
    ga = ga *factor
    gb = gb *factor
    A = A*factor

    ωds = np.linspace(begin_ω,end_ω,n_points)*factor
    for idx,ωd in enumerate(ωds):
        tasks.append(create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd,idx,name))
    

if __name__ == "__main__":

    
    tasks = []
    
    factor = 2.0*np.pi*1e9
    
    N = 5
    ωa = 5.1* factor
    ωb = 5.7* factor
    ωr = 1.0* factor
    κa = 1.0e-4
    κb = 1.0e-4
    κr = 0.01
    gb = 0.05* factor
    ωd_begin = 5.08
    ωd_end = 5.12
    T = 10e-3
    n_points = 2000
    n_case = 0    
    
    gas = np.linspace(0.0,0.2,10)* factor
    A = 0.005* factor
    
    for ga in gas:
        tasks.append(create_task(N,
                                 ωa,
                                 ωb,
                                 ωr,
                                 ga,
                                 gb,
                                 κa,
                                 κb,
                                 κr,
                                 T,
                                 A,
                                 ωd_begin, # begin_ω
                                 ωd_end, # end_ω
                                 n_points,
                                 0, #idx
                                 "case_w1_ga{}".format(n_case))) #name
        
        n_case = n_case + 1
        
        
    ga = 0.1
    As = np.linspace(1.0e-4,0.01,10)
    
    for A in As:
        tasks.append(create_task(N,
                                 ωa,
                                 ωb,
                                 ωr,
                                 ga,
                                 gb,
                                 κa,
                                 κb,
                                 κr,
                                 T,
                                 A,
                                 ωd_begin, # begin_ω
                                 ωd_end, # end_ω
                                 n_points,
                                 0, #idx
                                 "case_w1_A{}".format(n_case))) #name
        n_case = n_case + 1
    

    simulate("2 Cavities 1 Resonator Drive Simulation",tasks)
    