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


def calculate_n_th(T,w):
    return 1/(np.exp(sc.hbar*w/(sc.k*T))-1)

def drive_Hamiltonian(a,wa,b,wb,r,wr,ga,gb,wd,A):
    H= (wa-wd)*a.dag()*a + wb*b.dag()*b + wr*r.dag()*r + A*(a.dag()+a) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
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

        
def create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,A,wd_begin,wd_end,n_points,idx,name):
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
            "A":A,
            "wd_begin":wd_begin,
            "wd_end":wd_end,
            "n_points":n_points,
            "idx":idx,
            "name":name}

def execute(N,wa,wb,wr,ga,gb,ka,kb,kr,T,A,wd_begin,wd_end,n_points,name,dirName):
    
    n_th_a = calculate_n_th(T,wa)
    n_th_b = calculate_n_th(T,wb)
    n_th_r = calculate_n_th(T,wr)

    rate_relaxation_a = ka*(1+n_th_a)
    rate_relaxation_b = kb*(1+n_th_b)
    rate_relaxation_r = kr*(1+n_th_r)

    rate_excitation_a = ka*(n_th_a)
    rate_excitation_b = kb*(n_th_b)
    rate_excitation_r = kr*(n_th_r)
    
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
        
   
    
    wds = np.linspace(wd_begin,wd_end,n_points)
    for idx,wd in enumerate(wds):
        H = drive_Hamiltonian(a,wa,b,wb,r,wr,ga,gb,wd,A)
        rho_ss = steadystate(H, c_ops)
        purity = (rho_ss*rho_ss).tr()
        expect_a = (rho_ss*a).tr()
        
        data = SimulationData(create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,A,wd_begin,wd_end,n_points,idx,name),
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
                                                  task["wa"],
                                                  task["wb"],
                                                  task["wr"],
                                                  task["ga"],
                                                  task["gb"],
                                                  task["ka"],
                                                  task["kb"],
                                                  task["kr"],
                                                  task["T"],
                                                  task["A"],
                                                  task["wd_begin"],
                                                  task["wd_end"],
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

def add_simulation_experiment(N,wa,wb,wr,ga,gb,ka,kb,kr,A,T,n_points,begin_w,end_w,name,tasks,factor=2.0*np.pi*1e9):
    wa = wa * factor
    wb = wb * factor
    wr = wr * factor
    ga = ga *factor
    gb = gb *factor
    A = A*factor

    wds = np.linspace(begin_w,end_w,n_points)*factor
    for idx,wd in enumerate(wds):
        tasks.append(create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,A,wd,idx,name))

def get_graphs_case(dirname,case,n_system,n_points=0):
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
    


    wds = np.linspace(data.task["wd_begin"],data.task["wd_end"],data.task["n_points"])
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
    task["wr"] = task["wr"]/factor
    task["wa"] = task["wa"]/factor
    task["wb"] = task["wb"]/factor
    task["wd_begin"] = task["wd_begin"]/factor
    task["wd_end"] = task["wd_end"]/factor
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
    
    N = 4
    wa = 5.1* factor
    wb = 5.7* factor
    wr = 1.0* factor
    ka = 1.0e-4
    kb = 1.0e-4
    kr = 1.0e-2
    gb = 0.05* factor
    wd_begin = 5.089*factor
    wd_end = 5.105*factor
    T = 10e-3
    n_points = 1000
    n_case = 0
    
    gas = np.linspace(0.0,0.01,10)* factor
    A = 0.0001* factor
    
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
                                 "case_w1_ga{}".format(n_case))) #name
        
        n_case = n_case + 1
        
        
    ga = 0.001*factor
    As = np.linspace(1.0e-5,0.003,10)*factor
    
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
                                 "case_w1_A{}".format(n_case))) #name
        n_case = n_case + 1
    

    simulate("2 Cavities 1 Resonator Drive Simulation",tasks)
    
