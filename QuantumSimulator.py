#
# Quantum Simulation Script
# This script simulates two cavities, both coupled to a resonator (all three harmonic oscillators).
# The cavities are labeled cavity a and cavity b while the resonator is labeled resonator r.
#
# The function that actually compute the simulation is called execute.
# It receives the simulation parameters, for example, cavities frequency, fock number, etc, and
# returns an object instantiated from the class SimulationData, which holds the parameters given
# and the density operator, purity, and the destruction operators for a, b and r.
#
# The function execute can be called directly, however to use parallelism, one must call
# the simulate function, which in turn calls the execute internally. The function simulate receives as paremeter
# a string that labels all simulation data, for example, "Simulation 1", and also a list of tasks, which are
# dictionary holding the parameters for the execute function.
#
# Each task in the task list will be computed by the execute function in parallel.
#
# This an example of a task
#
#   {"N":N, ----------------> fock number
#    "ωa":ωa, --------------> cavity a frequency
#    "ωb":ωb, --------------> cavity b frequency
#    "ωr":ωr, --------------> resonator r frequency
#    "ga":ga, --------------> cavity a coupling coefficient
#    "gb":gb, --------------> cavity b coupling coefficient
#    "κa":κa, --------------> cavity a dissipation rate
#    "κb":κb, --------------> cavity b dissipation rate
#    "κr":κr, --------------> resonator r dissipation rate
#    "T":T, ----------------> System temperature
#    "A":A, ----------------> Drive frequency Amplitude
#    "ωd_begin":ωd_begin, --> drive sweep initial frequency
#    "ωd_end":ωd_end, ------> drive sweep final frequency
#    "n_points":n_points, --> number of sweep frequency points
#    "idx":idx, ------------> frequency points number
#    "name":name} ----------> label for this specific task
#
# 

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
def drive_Hamiltonian(a,wa,b,wb,r,wr,ga,gb,wd_a,Aa,wd_b,Ab):
    H= (wa-wd_a)*a.dag()*a + (wb-wd_b)*b.dag()*b + wr*r.dag()*r + Aa*(a.dag()+a) + Ab*(b.dag()+b) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
    return H

# Add the collapse operator for the Master equation.
def add_c_ops(c_ops,rate_excitation,rate_relaxation,op):
    if rate_excitation > 0.0:
        c_ops.append(np.sqrt(rate_excitation)*op.dag())

    if rate_relaxation > 0.0:
        c_ops.append(np.sqrt(rate_relaxation)*op)
        
    return c_ops

# A class that is actually a struct of the data produced by the simulation.
# The really important variable is rho. The rest is for management simplicity.
class SimulationData():
    def __init__(self,task,a,expect_a,purity,b,r,rho):
        self.task = task
        self.rho = rho
        self.purity  = purity
        self.expect_a = expect_a
        self.a = a
        self.b = b
        self.r = r


# Call this function to create a new task.        
def create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_begin,wd_end,wd_b,n_points,idx,name):
    return {"N":N,
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
            "wd_begin":wd_begin,
            "wd_end":wd_end,
            "wd_b": wd_b,
            "n_points":n_points,
            "idx":idx,
            "name":name}


# Main processing function execute.
def execute(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_begin,wd_end,wd_b,n_points,name,dirName):

    # Calculate the average number of photons. Used to calculate the excitation and relaxation rate below.
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

    # Creating the list of collapse operators
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
        
   
    # For each n_points between ωd_begin and ωd_end, the steady state is calculated.
    # The purity and expectation value of a are also calculated.    
    wds = np.linspace(wd_begin,wd_end,n_points)
    for idx,wd_a in enumerate(wds):
        H = drive_Hamiltonian(a,wa,b,wb,r,wr,ga,gb,wd_a,Aa,wd_b,Ab)

        # The steadysate function from QuTiP
        rho_ss = steadystate(H, c_ops)
        
        purity = (rho_ss*rho_ss).tr()
        expect_a = (rho_ss*a).tr()

        # Create an instance of SimulationData class
        data = SimulationData(create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_begin,wd_end,wd_b,n_points,idx,name),
                              a,
                              expect_a,
                              purity,
                              b,
                              r,
                              rho_ss)

        # Save the data. The name of the file will be a combination of the name of the simulation, plus and the
        # name of the task and the number of the point calculated.
        filedata_name = "{}/data_{}_{}".format(dirName,data.task["name"],data.task["idx"])
        file = open(filedata_name,"wb")
        print("Saving acquired data {} point {}".format(data.task["name"],data.task["idx"]))
        pickle.dump(data,file)
        file.close()
        
    return data

# The parallel function simulate.
def simulate(name,tasks,dirName):

    # This is a setup for the logging system
    filename = name.replace(" ","_")
    logging.basicConfig(level=logging.DEBUG) 
    logger = logging.getLogger(name)
    handler = logging.FileHandler('log_{}_{}.log'.format(filename,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Obtain the number of cpus to be used
    task_count = len(tasks)
    cpu_count = mp.cpu_count()

    
    logger.info("Starting Simulation")

    # Create target Directory if one does not exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        logger.info("Directory {} Created ".format(dirName))
    else:    
        logger.info("Directory {} already exists".format(dirName))

    logger.info("#CPU {}".format(cpu_count))

    # Parellel code
    try:
        t_start = time.time()
        time_1 = []

        # get a pool object and apply the function execute for each task.
        pool = mp.Pool(processes = cpu_count)

        results = [pool.apply_async(execute,args=(task["N"],
                                                  task["wa"],
                                                  task["wb"],
                                                  task["wr"],
                                                  task["ga"],
                                                  task["gb"],
                                                  task["ka"],
                                                  task["kb"],
                                                  task["kr"],
                                                  task["T"],
                                                  task["Aa"],
                                                  task["Ab"],
                                                  task["wd_begin"],
                                                  task["wd_end"],
                                                  task["wd_b"],
                                                  task["n_points"],
                                                  task["name"],
                                                  dirName),callback=None,error_callback=None) for task in tasks]

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

    except Exception as e:
        logger.exception(e)
        pool.terminate()
        pool.join()
        raise e
    
    logger.info("End of simulation")
    

    handler.close()
    logger.removeHandler(handler)
    logging.shutdown()

    
