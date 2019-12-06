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
    def __init__(self,task,a,b,r,rho):
        self.task = task
        self.rho = rho
        self.a = a
        self.b = b
        self.r = r
        
def create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd,idx,name):
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
            "ωd":ωd,
            "idx":idx,
            "name":name}

def execute(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd,idx,name):
    
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

    H = drive_Hamiltonian(a,ωa,b,ωb,r,ωr,ga,gb,ωd,A)
    rho_ss = steadystate(H, c_ops)
    
    return SimulationData(create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd,idx,name),a,b,r,rho_ss)

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
    
    cpu_count = mp.cpu_count()-1
        
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
                                                  task["ωd"],
                                                  task["idx"],
                                                  task["name"]),callback=None,error_callback=None) for task in tasks]

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
            logger.info("Saving acquired data {}".format(data.task["idx"]))
            filedata_name = "{}/data_{}_{}".format(dirName,data.task["name"],data.task["idx"])
            file = open(filedata_name,"wb")
            pickle.dump(ar.get(),file)
            file.close()

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



if __name__ == "__main__":
    N=4
    ωa=5.1 * 2*np.pi *1e9
    ωb= 5.7 * 2*np.pi *1e9
    ωr = 4.0 * 2*np.pi *1e9
    ga = 0.09 * 2*np.pi*1e9 #0 até 0.1
    gb = 0.06 * 2*np.pi*1e9
    κa = 10e-4
    κb = 10e-4
    κr = 10e-4
    A = 6*7.0e-5*2*np.pi*1e9
    T=10e-3
    n_points = 2000
    begin_ω = 5.08
    end_ω = 5.12
    
    tasks = []

    name= "wds"

    ωds = np.linspace(begin_ω,end_ω,n_points)*2*np.pi*1e9
    for idx,ωd in enumerate(ωds):
        tasks.append(create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd,idx,name))

    simulate("teste_4",tasks)
    