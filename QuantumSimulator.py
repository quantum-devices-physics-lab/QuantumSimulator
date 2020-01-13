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
def calculate_n_th(T,ω):
    return 1/(np.exp(sc.hbar*ω/(sc.k*T))-1)

# Return the system hamiltonian.
def drive_Hamiltonian(a,ωa,b,ωb,r,ωr,ga,gb,ωd,A):
    H = (ωa-ωd)*a.dag()*a + ωb*b.dag()*b + ωr*r.dag()*r + A*(a.dag()+a) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
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


# Main processing function execute.
def execute(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd_begin,ωd_end,n_points,name,dirName):

    # Calculate the average number of photons. Used to calculate the excitation and relaxation rate below.
    n_th_a = calculate_n_th(T,ωa)
    n_th_b = calculate_n_th(T,ωb)
    n_th_r = calculate_n_th(T,ωr)

    rate_relaxation_a = κa*(1+n_th_a)
    rate_relaxation_b = κb*(1+n_th_b)
    rate_relaxation_r = κr*(1+n_th_r)

    rate_excitation_a = κa*(n_th_a)
    rate_excitation_b = κb*(n_th_b)
    rate_excitation_r = κr*(n_th_r)

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
    ωds = np.linspace(ωd_begin,ωd_end,n_points)
    for idx,ωd in enumerate(ωds):

        H = drive_Hamiltonian(a,ωa,b,ωb,r,ωr,ga,gb,ωd,A)

        # The steadysate function from QuTiP
        rho_ss = steadystate(H, c_ops)
        
        purity = (rho_ss*rho_ss).tr()
        expect_a = (rho_ss*a).tr()

        # Create an instance of SimulationData class
        data = SimulationData(create_task(N,ωa,ωb,ωr,ga,gb,κa,κb,κr,T,A,ωd_begin,ωd_end,n_points,idx,name),
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
def simulate(name,tasks):

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

    # Define the name of the folder where the files are going to be stored
    dirName = "data_{}_{}".format(filename,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
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

def get_graphs_case(dirname,case,n_points=0,n_system):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dirName):
        for file in f:
            files.append(os.path.join(r, file))
            
    casefiles = []

    for file in files:
        result = re.search(case,file)
        if result:
            casefiles.append(file)

    if n_points == 0 :
        n_points = len(casefiles)

    rhos = np.array([])
    na = np.array([])
    purity = np.array([])
    


    data = unpack_simulation_data(casefiles[0])
    


    ωds = np.linspace(data.task["ωd_begin"],data.task["ωd_end"],data.task["n_points"])
    for i in range(0,n_points):
        filename = dirName + "data_case_" + case + "_" + str(i);
        data = unpack_simulation_data(filename)
        rhos = np.append(rhos,data.rho)
        na = np.append(na,data.expect_a)
        purity = np.append(purity,data.purity)
    data.task
    dim = data.task["N"]**n_system
    #rhos = rhos.reshape(n_points,dim,dim)
    
    return (n_points,data.task,rhos,na,purity,data.a,data.b,data.r)

def plot_graph(X,Y,title="plot",xlabel="x",ylabel="y",leg="leg",figsize=(14,7),fontsize=20,savename="fig",color="red",toSave=False,useGivenFig=False,fig=-1,axes=-1):
    if not useGivenFig:
        fig, axes = plt.subplots(1,1, figsize=figsize)
    axes.plot(X, Y, color = color,label=leg, lw=1.5)
    axes.legend(loc=0,fontsize=20)
    axes.set_xlabel(xlabel,rotation=0,fontsize= 20)
    axes.set_ylabel(ylabel,rotation=90,fontsize= 20)
    axes.set_title(title, fontsize=16)
    axes.tick_params(axis='both',which='major',labelsize='16')
    if(toSave):
        plt.savefig(savename)
    return fig,axes
    

def convert_task(task):
    factor = 2*np.pi*1e9
    task["A"] = task["A"]/factor
    task["ωr"] = task["ωr"]/factor
    task["ωa"] = task["ωa"]/factor
    task["ωb"] = task["ωb"]/factor
    task["ωd_begin"] = task["ωd_begin"]/factor
    task["ωd_end"] = task["ωd_end"]/factor
    task["ga"] = task["ga"]/factor
    task["gb"] = task["gb"]/factor
    return task

def plot_color(X,Y,C,title="plot",xlabel="x",ylabel="y",figsize=(14,7),fontsize=20,colorbarlabel="Arb. Units",savename="fig",toSave=False):
    fig, axes = plt.subplots(1,1, figsize=figsize)
    plt.pcolor(X,Y,C)
    plt.title(title,fontsize=fontsize)
    plt.xlabel(xlabel,fontsize=fontsize)
    plt.ylabel(ylabel,fontsize=fontsize)
    plt.colorbar(label=colorbarlabel)
    
    if(toSave):
        plt.savefig(savename)

if __name__ == "__main__":

    
    tasks = []
    
    factor = 2.0*np.pi*1e9
    
    N = 5
    ωa = 5.1* factor
    ωb = 5.7* factor
    ωr = 4.0* factor
    κa = 1.0e-4
    κb = 1.0e-4
    κr = 1.0e-4
    gb = 0.05* factor
    ωd_begin = 5.04*factor
    ωd_end = 5.14*factor
    T = 10e-3
    n_points = 1000
    n_case = 0
    
    gas = np.linspace(0.0,0.2,5)* factor
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
        
        
    ga = 0.1*factor
    As = np.linspace(1.0e-4,0.01,5)*factor
    
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
    
