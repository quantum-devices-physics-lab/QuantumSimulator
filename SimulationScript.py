
# The code simulates in parallel the steady state of an arbitrary Hamiltonian.
#
# We utilize QuTiP here to calculatate the steadystate. Our main function
# which is called in parallel needs two parameters: A Hamiltonian and a list of colapse operators.
# The basic simulation process is: To define the Hamiltonian and the colapse operators, to simulate the dynamic
# in parallel, to save the data returned by the simulation, which is a density matrices list.
# 
# There is one main data structure, Task, and one main function,
# simulate. Task, a dictionary of everything related to the creation of the Hamiltonian and
# colapse operatores. For example, frequencies, dissipations constants, destruction operators etc.
#
# Of course, a single task gives just one density state which is not really useful for our need.
# Therefore, we use an collections of tasks, called here a sweep. In a sweep all task holds the same values
# except for one parameter, for example the drive power on a cavity.
#
# A single simulation is usually organized like this:
#
# - Create a many number of sweeps.
# - Each task from all sweeps is passed to execute function which is run on its on process in parallel.
# - The return data of all sweeps is saved on disk.
#
# For example, suppose we are interested the following system: Two coupled harmonic oscillators,
# a and b. We will sweep the drive on cavity a for different values of the coupling coefficient.
# On the task, drive on cavity a is named "wd_a" and the coupling coefficient is named "g".
# We would like to make 3 sweep to 3 different values of g from 0 to 2.
#
# Here is the example of the sweeps:
#
# Sweep "g0"
# |-----------| |-----------|     |----------|
# | task 0    | | task 1    |     | task 20  |
# | wd_a=-5.0 | | wd_a=-4.5 | ... | wd_a=5.0 |
# | g=0       | | g=0       |     | g=0      |
# |   ...     | |   ...     |     |   ...    |
# |-----------| |-----------|     |----------|
#
# Sweep "g1"
# |-----------| |-----------|     |----------|
# | task 0    | | task 1    |     | task 20  |
# | wd_a=-5.0 | | wd_a=-4.5 | ... | wd_a=5.0 |
# | g=1       | | g=1       |     | g=1      |
# |   ...     | |   ...     |     |   ...    |
# |-----------| |-----------|     |----------|
#
# Sweep "g2"
# |-----------| |-----------|     |----------|
# | task 0    | | task 1    |     | task 20  |
# | wd_a=-5.0 | | wd_a=-4.5 | ... | wd_a=5.0 |
# | g=2       | | g=2       |     | g=2      |
# |   ...     | |   ...     |     |   ...    |
# |-----------| |-----------|     |----------|
#
#########################################################################################################################

from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
from multiprocessing import  freeze_support
from scipy.constants import *
import scipy.sparse as sp
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy as spy
import time
import csv
import psutil

import qutip.settings as qset
qset.has_mkl = False


def hamiltonian(task,a,b,r):
    '''returns Hamiltonian defined in QuTiP to be simulated. a,b and r are the destroy operators.'''
    wa = task['wa']
    wb = task['wb']
    wr = task['wr']
    ga = task['ga']
    gb = task['gb']
    Ea = task['Ea']
    Eb = task['Eb']
    wda = task['wda']
    wdb = task['wdb']
    H = 0
    H += (wa-wda)*a.dag()*a
    H += (wb-wdb)*b.dag()*b
    H +=  wr*r.dag()*r
    H += Ea*(a.dag()+a)
    H += Eb*(b.dag()+b)
    H += ga*a.dag()*a*(r.dag()+r)
    H += gb*b.dag()*b*(r.dag()+r)
    
    
    
    return H

def collapse_operators(task,a,b,r):
    '''returns a list of collapse operators to be used on the master solver. a,b and r are the destruction operators.'''
    wa = task['wa']
    wb = task['wb']
    wr = task['wr']
    ka = task['ka']
    kb = task['kb']
    kr = task['kr']
    T = task['T']
    
    n_th_a = utilities.n_thermal(wa*1e9,T*k/h)
    n_th_b = utilities.n_thermal(wb*1e9,T*k/h)
    n_th_r = utilities.n_thermal(wr*1e9,T*k/h)
    
    rate_relaxation_a = ka*(1+n_th_a)
    rate_relaxation_b = kb*(1+n_th_b)
    rate_relaxation_r = kr*(1+n_th_r)

    rate_excitation_a = ka*(n_th_a)
    rate_excitation_b = kb*(n_th_b)
    rate_excitation_r = kr*(n_th_r)
    
    c_ops = []

    
    # only add the list if is not zero
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
        
    return c_ops


def save_csv(filename,l,units):
    '''saves the data to an csv file. To save in csv, define a list of column names, then create multiple dictionaries whose key values are the same as the list of column names.'''
    task = l[0]['task']
    Na = task['Na']
    Nb = task['Nb']
    Nr = task['Nr']

    # Saving a file which holds the units of the tasks parameters
    with open(filename+'_units.csv', mode='w',newline='') as csv_file:
        fieldnames = list(units.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(units)

    # There is header file which will be used in the load_csv function to know how to interpret the data saved as density matrix.
    with open(filename+'_header.csv', mode='w',newline='') as csv_file:
        fieldnames = ['sweep_variable_name',
                      'sweep_variable_length',
                      'sweep_variable_range_initial_value',
                      'sweep_variable_range_final_value',
                      'main_variable_name', 
                      'main_variable_length',
                      'main_variable_range_initial_value',
                      'main_variable_range_final_value',
                      'Na',
                      'Nb',
                      'Nr']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'sweep_variable_name':task['sweep_variable_name'],
                         'sweep_variable_length': task['sweep_variable_length'],
                          'sweep_variable_range_initial_value':task['sweep_variable_range_initial_value'],
                          'sweep_variable_range_final_value':task['sweep_variable_range_final_value'],
                         'main_variable_name':task['main_variable_name'],
                         'main_variable_length':task['main_variable_length'],
                          'main_variable_range_initial_value':task['main_variable_range_initial_value'],
                          'main_variable_range_final_value':task['main_variable_range_final_value'],
                         'Na':Na,
                         'Nb':Nb,
                         'Nr':Nr})

    #Saving all the tasks and density matrices.
    with open(filename+'.csv', mode='w',newline='') as csv_file:
        fieldnames = ['sweep_idx', 
                      'task_idx',
                      'wa',
                      'wb',
                      'wr',
                      'ka',
                      'kb',
                      'kr',
                      'ga',
                      'gb',
                      'T',
                      'Ea',
                      'Eb',
                      'wda',
                      'wdb']

        for i in range(0,(Na*Nb*Nr)**2):
            fieldnames.append('rho'+str(i))

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for result in l:
            sweep_idx = result['sweep_idx']
            task_idx = result['task_idx']
            task = result['task']
            rho_ss = result["rho_ss"]
            tarefa = {'sweep_idx': sweep_idx,
                      'task_idx': task_idx,
                      'wa':task['wa'],
                      'wb':task['wb'],
                      'wr':task['wr'],
                      'ka':task['ka'],
                      'kb':task['kb'],
                      'kr':task['kr'],
                      'ga':task['ga'],
                      'gb':task['gb'],
                      'T':task['T'],
                      'Ea':task['Ea'],
                      'Eb':task['Eb'],
                      'wda':task['wda'],
                      'wdb':task['wdb']}

            rho = rho_ss.full().reshape((Na*Nb*Nr)**2,)
            for j in range(0,(Na*Nb*Nr)**2):
                tarefa['rho'+str(j)] = rho[j]

            writer.writerow(tarefa)
    
def load_csv(filename):
    '''It reads the data from an csv file.'''

    #Loads the header to acquire mainly the dimensions of the coupled systems.
    with open(filename+'_header.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        header = next(readCSV)
        data = next(readCSV)
        sweep_variable_name = data[0]
        sweep_variable_length = int(data[1])
        sweep_variable_range_initial_value = float(data[2])
        sweep_variable_range_final_value = float(data[3])
        main_variable_name = data[4] 
        main_variable_length = int(data[5])
        main_variable_range_initial_value = float(data[6])
        main_variable_range_final_value = float(data[7])
        Na = int(data[8])
        Nb = int(data[9])
        Nr = int(data[10])      

    # Define a matrices to be returned holding the data.
    tasks = np.zeros((main_variable_length,sweep_variable_length),dtype=dict)
    rhos = np.zeros((main_variable_length,sweep_variable_length),dtype=Qobj)

    with open(filename+'.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        header = next(readCSV)
        for row in readCSV:
            sweep_idx = int(row[0])
            task_idx = int(row[1])
            wa = float(row[2])
            wb = float(row[3])
            wr = float(row[4])
            ka = float(row[5])
            kb = float(row[6])
            kr = float(row[7])
            ga = float(row[8])
            gb = float(row[9])
            T  = float(row[10])
            Ea = float(row[11])
            Eb = float(row[12])

            wda = float(row[13])
            wdb = float(row[14])

            rho = Qobj(np.array(row[15:]).reshape((Na*Nb*Nr,Na*Nb*Nr)),dims=[[Na,Nb,Nr],[Na,Nb,Nr]])

            task = {
                'wa':wa,
                'wb':wb,
                'wr':wr,
                'ka':ka,
                'kb':kb,
                'kr':kr,
                'ga':ga,
                'gb':gb,
                'T':T,
                'Ea':Ea,
                'Eb':Eb,
                'wda':wda,
                'wdb':wdb,
                'Na':Na,
                'Nb':Nb,
                'Nr':Nr,
                'main_variable_name': main_variable_name,
                'main_variable_length': main_variable_length,
                'main_variable_range_initial_value': main_variable_range_initial_value,
                'main_variable_range_final_value': main_variable_range_final_value,
            
                'sweep_variable_name': sweep_variable_name,
                'sweep_variable_length': sweep_variable_length,
                'sweep_variable_range_initial_value': sweep_variable_range_initial_value,
                'sweep_variable_range_final_value': sweep_variable_range_final_value}

            tasks[sweep_idx][task_idx] = task
            rhos[sweep_idx][task_idx] = rho
                
    return (tasks,rhos)

def simulate(task):
    '''calculates a density matrix using the master solver steadystate. It uses the iterative method lgmres.'''

    # The destruction operator
    a = tensor(destroy(task['Na']),qeye(task['Nb']),qeye(task['Nr']))
    b = tensor(qeye(task['Na']),destroy(task['Nb']),qeye(task['Nr']))
    r = tensor(qeye(task['Na']),qeye(task['Nb']),destroy(task['Nr']))

    c_ops = collapse_operators(task,a,b,r)
    
    H = hamiltonian(task,a,b,r)
    
    result = {}

    # this will be used when loading the saved data to organize the list of tasks and density matrices.
    result['task_idx'] = task['task_idx']
    result['sweep_idx'] = task['sweep_idx']


    # This the function which returns the density matrices for the steady state of our system.
    # A brief explanation of the parameters used (some copied verbatim from the source code):
    # H: the Hamiltonian
    # c_ops:  a list of operators
    # method: the type of method we will use. There are many, but we have found the
    #    iterative-lgmres to be the fastest and still being accurate.
    # use_precond: ITERATIVE ONLY. Use an incomplete sparse LU decomposition as a
    #    preconditioner for the 'iterative' GMRES and BICG solvers.
    #    Speeds up convergence time by orders of magnitude in many cases.
    # use_rcm: Use reverse Cuthill-Mckee reordering to minimize fill-in in the
    #    LU factorization of the Liouvillian.
    # tol: ITERATIVE ONLY. Tolerance used for terminating solver.
    # return_info: Return a dictionary of solver-specific infomation about the
    #    solution and how it was obtained.
    rho_ss,info = steadystate(H, c_ops, method='iterative-lgmres',use_precond=True, 
                use_rcm=True, tol=1e-15,return_info=True)
    
    result['rho_ss'] = rho_ss
    result['info'] = info
    result['task'] = task

    print('sweep {} task {}   \
          Time used (seconds) for task: {} \
          percent memory used {}'.format(result['sweep_idx'],
                                         result['task_idx'],
                                         info['solution_time'],
                                         psutil.virtual_memory().percent))
   
    return result

if __name__ == "__main__":

    
    name = 'Nr15_p1000'

    # The same parameters for all tasks. Unit is GHz.
    task = {
        'Na':4, # Destroy operator dimension of cavity a
        'Nb':4, # Destroy operator dimension of cavity b
        'Nr':15, # Destroy operator dimension of cavity r
        'wa':5.1, # Resonance frequency of cavity a
        'wb':5.7, # Resonance frequency of cavity b
        'wr':0.1, # Resonance frequency of cavity r
        'ka': 1.275e-4, # dissipation coefficient of cavity a. Chosen so that cavity a quality factor is 40k
        'kb': 1.425e-4, # dissipation coefficient of cavity b. Chosen so that cavity b quality factor is 40k
        'kr': 6.75e-4, # ka = 5*(kb+kr)/2. Quality factor for cavity r is ~150
        'ga':6.0e-3, # Coupling coefficient between cavity a and resonator r.
        'gb':4.0e-3, # Coupling coefficient between cavity b and resonator r.
        'T':10.0e-3, # Temperatura of the system
        'wdb':5.7, # Drive frequency on cavity b
        'Ea':5.0e-6, #Drive strength of cavity a
        'main_variable_name': 'Eb', # the name of the parameter that is different to each sweep
        'main_variable_length': 10, # number of sweeps
        'sweep_variable_name': 'wda', # the name of the parameter that is swept.
        'sweep_variable_length': 1000, # the length of each sweep
    }

    # The units for each parameter
    task_units = {
        'wa':'GHz',
        'wb':'GHz',
        'wr':'GHz',
        'ka':'GHz',
        'kb':'GHz',
        'kr':'GHz',
        'ga':'GHz',
        'gb':'GHz',
        'T':'K',
        'wda':'GHz',
        'wdb':'GHz',
        'Ea':'GHz',
        'Eb':'GHz',
        'main_variable_range':'GHz',
        'sweep_variable_range':'GHz'
    }
    
    print('Configuring tasks')

    # define wda
    n_points = task['sweep_variable_length']
    shift = task['ga']*task['ga']/task['wr']
    wda_i = task['wa'] - 250e-6 - shift
    wda_f = task['wa'] + 50e-6 - shift
    wdas = np.linspace(wda_i,wda_f,n_points,endpoint=True)

    # define Eb
    n_sweeps = task['main_variable_length']
    Eb_i = 1.0e-6
    Eb_f = 160.0e-6
    Ebs = np.linspace(Eb_i,Eb_f,n_sweeps,endpoint=True)
    
    task['main_variable_range_initial_value'] = Eb_i
    task['main_variable_range_final_value'] = Eb_f
    task['sweep_variable_range_initial_value'] = wda_i
    task['sweep_variable_range_final_value'] = wda_f
    
    sweeps = []
    for ((sweep_idx,Eb),(task_idx,wda)) in itertools.product(enumerate(Ebs),enumerate(wdas)):
        newtask = task.copy()
        newtask['Eb'] = Eb
        newtask['wda'] = wda
        newtask['sweep_idx'] = sweep_idx
        newtask['task_idx'] = task_idx

        sweeps.append(newtask)
    
    time_start = time.time()
    
    print('Simulating')
    freeze_support()
    l = parallel_map(simulate,sweeps,num_cpus=40)    

    time_end = time.time()
    print('Simulation time taken(seconds): {}'.format(time_end-time_start))
    
    print('Saving data to csv file')
    save_csv('D:\\'+name,l,task_units)
    
    time_end2 = time.time()
    print('Saving data time taken(seconds): {}'.format(time_end2-time_end))
