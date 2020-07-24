
# The code simulates in parallel the steady state of an arbitrary Hamiltonian.
#
# We utilize QuTiP here to do the heavy calculations. The function used is steadystate,
# which receives two parameters: A Hamiltonian and an array of colapse operators.
# The basic simulation process is: To define the Hamiltonian and the colapse operators, to simulate the dynamic
# in parallel, to save the data returned by the simulation, which is the density matrix.
# 
# There is one main data structure, Task, and one main function,
# execute. Task, a dictionary everything  related to the creation of the Hamiltonian and
# colapse operatores, for example,  frequencies, dissipations constants, destruction operators etc.
#
# Of course, a single task gives just one density state which is not useful that useful Therefore we
# use an collections of task, called here a sweep. In a sweep all task are holds the same values
# except for one parameter, for example drive on cavity.
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
# Every sweep should be named as the coupling coefficient plus a number, for example "g0".
# It is important to name sweep the same name used in the task.
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
# Another important structure is the Experiment class which holds all the sweeps and format the
# data returned to be saved. The experiment class is used to access the data simulated to be analyzed.
#
# Now that a general picture was drawn, here is the following sequence of function:
# - In the main, the run_experiment1 function is called
# - In run_experiment1, the sweeps are created as well as an Experiment class instance called experiment
# - In run_experiment1, experiment is passed to the simulate function which creates the multiple process
# - In simulate, a new process is created to each sweep. After all sweep are simulated, experiment instance format
#   the list of data returned and simulate returns the new experiment instance
# - In run_experiment1, experiment instance is saved
#
#########################################################################################################################

from qutip import *
import numpy as np
import scipy.constants as sc
from multiprocessing import  freeze_support
import pickle
import os
import re
import sys
import csv
import itertools
import psutil
import time

# Qutip and MKL are not working well
# https://github.com/qutip/qutip/issues/975
import qutip.settings as qset
qset.has_mkl = False

def drive_Hamiltonian(a,wa,ga,wd_a,Aa,
                      b,wb,gb,wd_b,Ab,
                      r,wr):
    '''
    Returns a QObj which is the Hamiltonian to be simulated.

    Parameters
    ----------

    a : *Qobj*
        Destruction operator of cavity a.

    wa : *float*
         Frequency of cavity a, in GHz.

    ga : *float*
         Coupling frequency of cavity a to resonator r, in GHz.

    wd_a : *float*
           Drive frequency on cavity a, in GHz.

    Aa : *float*
         Drive amplitude of cavity a, in GHz;

    b : *Qobj*
        Destruction operator of cavity b.

    wb : *float*
         Frequency of cavity b, in GHz.

    gb : *float*
         Coupling frequency of cavity b to resonator r, in GHz.

    wd_b : *float*
           Drive frequency on cavity b, in GHz.

    Ab : *float*
         Drive amplitude of cavity b, in GHz;

    r : *Qobj*
        Destruction operator of cavity r.

    wr : *float*
         Frequency of cavity b, in GHz.

    Returns
    -------

    H : *QObj*
        Hamiltonian to be simulated

    '''
    H= (wa-wd_a)*a.dag()*a + (wb-wd_b)*b.dag()*b + wr*r.dag()*r + Aa*(a.dag()+a) + Ab*(b.dag()+b) - ga*a.dag()*a*(r.dag()+r) - gb*b.dag()*b*(r.dag()+r)
    return H



def create_task(Na,Nb,Nr,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_a,wd_b,idx,sweep_idx):
    '''
    The function create_task builds the hamiltonian and the collapse operators, which are returned inside a dictionary.
    It also adds to the dictionary other all the variables used to build the hamiltonian.

    Parameters
    ----------

    Na : *int*
        Fock number cavity a

    Nb : *int*
        Fock number cavity b

    Nr : *int*
        Fock number cavity r

    T : *float*
        Temperature of the system.

    wa : *float*
         Frequency of cavity a, in GHz.

    ka : *float*
         Dissipation rate of cavity a, in GHz.

    ga : *float*
         Coupling frequency of cavity a to resonator r, in GHz.

    wd_a : *float*
           Drive frequency on cavity a, in GHz.

    Aa : *float*
         Drive amplitude of cavity a, in GHz.

    wb : *float*
         Frequency of cavity b, in GHz.

    kb : *float*
         Dissipation rate of cavity b, in GHz.

    gb : *float*
         Coupling frequency of cavity b to resonator r, in GHz.

    wd_b : *float*
           Drive frequency on cavity b, in GHz.

    Ab : *float*
         Drive amplitude of cavity b, in GHz;

    r : *Qobj*
        Destruction operator of cavity r.

    wr : *float*
         Frequency of cavity r, in GHz.

    kr : *float*
         Dissipation rate of cavity r, in GHz.

    idx : *int*
          Id of this task. Defines its order on the sweep this task is part of.

    sweep_idx : *ind*
           sweep indice
    

    Returns
    -------

    *dict*
        dictionary with all the the parameters and other parameters created in the function.

    '''
    n_th_a = utilities.n_thermal(wa*1e9,T*sc.k/sc.h)
    n_th_b = utilities.n_thermal(wb*1e9,T*sc.k/sc.h)
    n_th_r = utilities.n_thermal(wr*1e9,T*sc.k/sc.h)    

    rate_relaxation_a = ka*(1+n_th_a)
    rate_relaxation_b = kb*(1+n_th_b)
    rate_relaxation_r = kr*(1+n_th_r)

    rate_excitation_a = ka*(n_th_a)
    rate_excitation_b = kb*(n_th_b)
    rate_excitation_r = kr*(n_th_r)

    
    task = {"Na":Na,
            "Nb":Nb,
            "Nr":Nr,
            "n_th_a":n_th_a,
            "rate_relaxation_a":rate_relaxation_a,
            "rate_excitation_a":rate_excitation_a,
            "n_th_b":n_th_b,
            "rate_relaxation_b":rate_relaxation_b,
            "rate_excitation_b":rate_excitation_b,
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

    task["idx"] = idx;
    task["sweep_idx"] = sweep_idx;
    return task;


def create_wd_a_sweep(Na,Nb,Nr,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_as,wd_b,n_points,sweep_idx):
    '''
    Returns an array of tasks. It is a sweep on the drive frequency at cavity a.

    Parameters
    ----------

    Na : *int*
        Fock number cavity a

    Nb : *int*
        Fock number cavity b

    Nr : *int*
        Fock number cavity r

    T : *float*
        Temperature of the system.

    wa : *float*
         Frequency of cavity a, in GHz.

    ka : *float*
         Dissipation rate of cavity a, in GHz.

    ga : *float*
         Coupling frequency of cavity a to resonator r, in GHz.

    wd_a : *float*
           Drive frequency on cavity a, in GHz.

    Aa : *float*
         Drive amplitude of cavity a, in GHz.

    wb : *float*
         Frequency of cavity b, in GHz.

    kb : *float*
         Dissipation rate of cavity b, in GHz.

    gb : *float*
         Coupling frequency of cavity b to resonator r, in GHz.

    wd_b : *float*
           Drive frequency on cavity b, in GHz.

    Ab : *float*
         Drive amplitude of cavity b, in GHz.

    r : *Qobj*
        Destruction operator of cavity r.

    wr : *float*
         Frequency of cavity r, in GHz.

    kr : *float*
         Dissipation rate of cavity r, in GHz.

    n_points : *int*
               Number of tasks on the sweep.

    wd_a_initial : *float*
                   Initial value of drive on cavity a.

    wd_a_final : *float*
                  Final value of drive on cavity a.

    name : *string*
           Name of this sweep.

    Returns
    -------

    *list*
         return a list of tasks

    '''
    sweep = np.array([])

    for (idx,wd_a) in enumerate(wd_as):
        task = create_task(Na,Nb,Nr,wa,wb,wr,ga,gb,ka,kb,kr,T,Aa,Ab,wd_a,wd_b,idx,sweep_idx)
        sweep = np.append(sweep,task)
    return sweep


def simulate_steadystate(task):
    
    time_start = time.time()

    # The destruction operator
    a = tensor(destroy(task['Na']),qeye(task['Nb']),qeye(task['Nr']))
    b = tensor(qeye(task['Na']),destroy(task['Nb']),qeye(task['Nr']))
    r = tensor(qeye(task['Na']),qeye(task['Nb']),destroy(task['Nr']))

    c_ops = []

    if task['rate_excitation_a'] > 0.0:
        c_ops.append(np.sqrt(task['rate_excitation_a'])*a.dag())

    if task['rate_relaxation_a'] > 0.0:
        c_ops.append(np.sqrt(task['rate_relaxation_a'])*a)
        
    if task['rate_excitation_b'] > 0.0:
        c_ops.append(np.sqrt(task['rate_excitation_b'])*b.dag())

    if task['rate_relaxation_b'] > 0.0:
        c_ops.append(np.sqrt(task['rate_relaxation_b'])*b)
        
    if task['rate_excitation_r'] > 0.0:
        c_ops.append(np.sqrt(task['rate_excitation_r'])*r.dag())

    if task['rate_relaxation_r'] > 0.0:
        c_ops.append(np.sqrt(task['rate_relaxation_r'])*r)

    H = drive_Hamiltonian(a,task['wa'],task['ga'],task['wd_a'],task['Aa'],
                          b,task['wb'],task['gb'],task['wd_b'],task['Ab'],
                          r,task['wr'])
    
    result = {}
    result['task_idx'] = task['idx']
    result['sweep_idx'] = task['sweep_idx']
    
    del task

    rho_ss = steadystate(H, c_ops, method='iterative-gmres',use_precond=True, 
                use_rcm=True, tol=1e-15)

    del c_ops
    del H

    
    purity = (rho_ss*rho_ss).tr()
    purity_a = (rho_ss.ptrace(0)*rho_ss.ptrace(0)).tr()
    purity_b = (rho_ss.ptrace(1)*rho_ss.ptrace(1)).tr()
    purity_r = (rho_ss.ptrace(2)*rho_ss.ptrace(2)).tr()
    
    expect_a = (rho_ss*a).tr()
    expect_a_dag = (rho_ss*a.dag()).tr()
    expect_na = (a.dag()*a*rho_ss).tr()
    
    expect_b = (rho_ss*b).tr()
    expect_b_dag = (rho_ss*b.dag()).tr()
    expect_nb = (b.dag()*b*rho_ss).tr()
    
    expect_r = (rho_ss*r).tr()
    expect_r_dag = (rho_ss*r.dag()).tr()
    expect_nr = (r.dag()*r*rho_ss).tr()

    del a
    del b
    del r

    result['purity'] = purity
    result['purity_a'] = purity_a
    result['purity_b'] = purity_b
    result['purity_r'] = purity_r

    result['expect_a'] = expect_a
    result['expect_a_dag'] = expect_a_dag
    result['expect_na'] = expect_na

    result['expect_b'] = expect_b
    result['expect_b_dag'] = expect_b_dag
    result['expect_nb'] = expect_nb

    result['expect_r'] = expect_r
    result['expect_r_dag'] = expect_r_dag
    result['expect_nr'] = expect_nr

    result['rho_ss'] = rho_ss
    
    time_end = time.time()
    
    print('sweep {} task {}  Time used (seconds) for task: {} percent memory used {}'.format(result['sweep_idx'],result['task_idx'],time_end - time_start,psutil.virtual_memory().percent))

    return result


class Experiment():

    def __init__(self,f=0,a=0,b=0,r=0,name='',sweeps=np.array([]), sweep_variable = np.array([]), main_variable=np.array([]),main_variable_name='',sweep_variable_name='',units={}):

        self.name = name
        self.n_points = len(sweep_variable)
        self.sweeps = sweeps
        self.sweep_variable = sweep_variable
        self.main_variable = main_variable
        self.main_variable_name = main_variable_name
        self.sweep_variable_name = sweep_variable_name
        self.f = f
        self.a = a
        self.b = b
        self.r = r
        self.units = units
        

        
    def save_csv(self,filename):
        Na = self.tasks[0][0]['Na']
        Nb = self.tasks[0][0]['Nb']
        Nr = self.tasks[0][0]['Nr']

        with open(filename+'_header.csv', mode='w',newline='') as csv_file:
            fieldnames = ['len_main_variable', 'n_points', 'main_variable_name', 'sweep_variable_name', 'Na', 'Nb', 'Nr']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'len_main_variable':len(self.main_variable),
                             'n_points': self.n_points,
                             'main_variable_name':self.main_variable_name,
                             'sweep_variable_name':self.sweep_variable_name,
                             'Na':Na,
                             'Nb':Nb,
                             'Nr':Nr})

        with open(filename+'.csv', mode='w',newline='') as csv_file:
            fieldnames = ['sweep_idx', 
                          'main_variable '+self.main_variable_name+' ('+self.units['main_variable']+')',
                          'idx', 
                          'sweep_variable '+self.sweep_variable_name+' ('+self.units['sweep_variable']+')',
                          'wa ('+self.units['wa']+')',
                          'wb ('+self.units['wb']+')',
                          'wr ('+self.units['wr']+')',
                          'ka ('+self.units['ka']+')',
                          'kb ('+self.units['ka']+')',
                          'kr ('+self.units['ka']+')',
                          'ga ('+self.units['ga']+')',
                          'gb ('+self.units['ga']+')',
                          'T ('+self.units['T']+')',
                          'Aa ('+self.units['Aa']+')',
                          'Ab ('+self.units['Ab']+')',
                          'wd_a ('+self.units['wd_a']+')',
                          'wd_b ('+self.units['wd_b']+')',
                          'expect_a (Arb. Units)',
                          'expect_a_dag (Arb. Units)',
                          'expect_na (Arb. Units)',
                          'expect_b (Arb. Units)',
                          'expect_b_dag (Arb. Units)',
                          'expect_nb (Arb. Units)',
                          'expect_r (Arb. Units)',
                          'expect_r_dag (Arb. Units)',
                          'expect_nr (Arb. Units)',
                          'purity (Arb. Units)',
                          'purity_a (Arb. Units)',
                          'purity_b (Arb. Units)',
                          'purity_r (Arb. Units)']

            for i in range(0,(Na*Nb*Nr)**2):
                fieldnames.append('rho'+str(i))

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            for ((sweep_idx,sweep_var),(idx,var)) in itertools.product(enumerate(self.main_variable),enumerate(self.sweep_variable)):
                tarefa = {'sweep_idx': sweep_idx,
                          'main_variable '+self.main_variable_name+' ('+self.units['main_variable']+')':sweep_var,
                          'idx': idx,
                          'sweep_variable '+self.sweep_variable_name+' ('+self.units['sweep_variable']+')':var,
                          'wa ('+self.units['wa']+')':self.tasks[sweep_idx][idx]['wa'],
                          'wb ('+self.units['wb']+')':self.tasks[sweep_idx][idx]['wb'],
                          'wr ('+self.units['wr']+')':self.tasks[sweep_idx][idx]['wr'],
                          'ka ('+self.units['ka']+')':self.tasks[sweep_idx][idx]['ka'],
                          'kb ('+self.units['kb']+')':self.tasks[sweep_idx][idx]['kb'],
                          'kr ('+self.units['kr']+')':self.tasks[sweep_idx][idx]['kr'],
                          'ga ('+self.units['ga']+')':self.tasks[sweep_idx][idx]['ga'],
                          'gb ('+self.units['gb']+')':self.tasks[sweep_idx][idx]['gb'],
                          'T ('+self.units['T']+')':self.tasks[sweep_idx][idx]['T'],
                          'Aa ('+self.units['Aa']+')':self.tasks[sweep_idx][idx]['Aa'],
                          'Ab ('+self.units['Ab']+')':self.tasks[sweep_idx][idx]['Ab'],
                          'wd_a ('+self.units['wd_a']+')':self.tasks[sweep_idx][idx]['wd_a'],
                          'wd_b ('+self.units['wd_b']+')':self.tasks[sweep_idx][idx]['wd_b'],
                          'expect_a (Arb. Units)': self.expect_a[sweep_idx][idx],
                          'expect_a_dag (Arb. Units)': self.expect_a_dag[sweep_idx][idx],
                          'expect_na (Arb. Units)': self.expect_na[sweep_idx][idx],
                          'expect_b (Arb. Units)': self.expect_b[sweep_idx][idx],
                          'expect_b_dag (Arb. Units)': self.expect_b_dag[sweep_idx][idx],
                          'expect_nb (Arb. Units)': self.expect_nb[sweep_idx][idx],
                          'expect_r (Arb. Units)': self.expect_r[sweep_idx][idx],
                          'expect_r_dag (Arb. Units)': self.expect_r_dag[sweep_idx][idx],
                          'expect_nr (Arb. Units)': self.expect_nr[sweep_idx][idx],
                          'purity (Arb. Units)': self.purity[sweep_idx][idx],
                          'purity_a (Arb. Units)': self.purity_a[sweep_idx][idx],
                          'purity_b (Arb. Units)': self.purity_b[sweep_idx][idx],
                          'purity_r (Arb. Units)': self.purity_r[sweep_idx][idx]}

                rho = self.rhos[sweep_idx][idx].full().reshape((Na*Nb*Nr)**2,)
                for j in range(0,(Na*Nb*Nr)**2):
                    tarefa['rho'+str(j)] = rho[j]

                writer.writerow(tarefa)
    
    def load_csv(self,filename):
        
        
        with open(filename+'_header.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            header = next(readCSV)
            data = next(readCSV)
            len_main_variable = int(data[0])
            n_points = int(data[1])
            main_variable_name = data[2] 
            sweep_variable_name = data[3]
            Na = int(data[4])
            Nb = int(data[5])
            Nr = int(data[6])
            
        self.n_points = n_points
        self.main_variable_name = main_variable_name
        self.sweep_variable_name = sweep_variable_name
        self.rhos = np.zeros((len_main_variable,self.n_points),dtype=Qobj)
        
        self.sweep_variable = np.zeros((self.n_points,),dtype=float)
        self.main_variable = np.zeros((len_main_variable,),dtype=float)
        
        self.expect_a = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_a_dag = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_na = np.zeros((len_main_variable,self.n_points),dtype=float)

        self.expect_b = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_b_dag = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_nb = np.zeros((len_main_variable,self.n_points),dtype=float)

        self.expect_r = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_r_dag = np.zeros((len_main_variable,self.n_points),dtype=complex)
        self.expect_nr = np.zeros((len_main_variable,self.n_points),dtype=float)

        self.purity = np.zeros((len_main_variable,self.n_points),dtype=float)
        self.purity_a = np.zeros((len_main_variable,self.n_points),dtype=float)
        self.purity_b = np.zeros((len_main_variable,self.n_points),dtype=float)
        self.purity_r = np.zeros((len_main_variable,self.n_points),dtype=float)        
        
        self.tasks = np.zeros((len(self.main_variable),self.n_points),dtype=dict)

        with open(filename+'.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            header = next(readCSV)
            for row in readCSV:
                sweep_idx = int(row[0])
                main_variable = float(row[1])
                idx = int(row[2])
                sweep_variable = float(row[3])
                wa = float(row[4])
                wb = float(row[5])
                wr = float(row[6])
                ka = float(row[7])
                kb = float(row[8])
                kr = float(row[9])
                ga = float(row[10])
                gb = float(row[11])
                T  = float(row[12])
                Aa = float(row[13])
                Ab = float(row[14])

                wd_a = float(row[15])
                wd_b = float(row[16])

                expect_a = complex(row[17])
                expect_a_dag = complex(row[18])
                expect_na = float(row[19])

                expect_b = complex(row[20])
                expect_b_dag = complex(row[21])
                expect_nb = float(row[22])
                
                expect_r = complex(row[23])
                expect_r_dag = complex(row[24])
                expect_nr = float(row[25])
                
                purity = complex(row[26])
                purity_a = complex(row[27])
                purity_b = complex(row[28])
                purity_r = complex(row[29])
                
                rho = Qobj(np.array(row[30:]).reshape((Na*Nb*Nr,Na*Nb*Nr)),dims=[[Na,Nb,Nr],[Na,Nb,Nr]])

                self.expect_a[sweep_idx][idx] = expect_a
                self.expect_a_dag[sweep_idx][idx] = expect_a_dag
                self.expect_na[sweep_idx][idx] = expect_na
                
                self.expect_b[sweep_idx][idx] = expect_b
                self.expect_b_dag[sweep_idx][idx] = expect_b_dag
                self.expect_nb[sweep_idx][idx] = expect_nb
                
                self.expect_r[sweep_idx][idx] = expect_r
                self.expect_r_dag[sweep_idx][idx] = expect_r_dag
                self.expect_nr[sweep_idx][idx] = expect_nr
                
                self.purity[sweep_idx][idx] = np.real(purity)
                self.purity_a[sweep_idx][idx] = np.real(purity_a)
                self.purity_b[sweep_idx][idx] = np.real(purity_b)
                self.purity_r[sweep_idx][idx] = np.real(purity_r)
                
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
                  'Aa':Aa,
                  'Ab':Ab,
                  'wd_a':wd_a,
                  'wd_b':wd_b,
                  'Na':Na,
                  'Nb':Nb,
                  'Nr':Nr}

                self.tasks[sweep_idx][idx] = task
                self.rhos[sweep_idx][idx] = rho
                self.sweep_variable[idx] = sweep_variable
                self.main_variable[sweep_idx] = main_variable
        

    def save(self,filename):
        file = open(filename,"wb")
        pickle.dump(self,file)
        file.close()
        
    def load(self,filename):
        file = open(filename,"rb")
        data = pickle.load(file)
        file.close()

        self.n_points =  data.n_points
        self.sweep_variable =  data.sweep_variable
        self.main_variable =  data.main_variable
        self.main_variable_name =  data.main_variable_name
        self.sweep_variable_name = data.sweep_variable_name
        self.f = data.f
        self.a = data.a
        self.b = data.b
        self.r = data.r
        self.name = data.name
        self.tasks = data.tasks
        self.rhos = data.rhos
        self.units = data.units

        self.expect_a = data.expect_a
        self.expect_a_dag = data.expect_a_dag
        self.expect_na = data.expect_na

        self.expect_b = data.expect_b
        self.expect_b_dag = data.expect_b_dag
        self.expect_nb = data.expect_nb

        self.expect_r = data.expect_r
        self.expect_r_dag = data.expect_r_dag
        self.expect_nr = data.expect_nr

        self.purity = data.purity
        self.purity_a = data.purity_a
        self.purity_b = data.purity_b
        self.purity_r = data.purity_r
        
        
    def simulate(self,serial=False,**kwargs):
        l = []
        if serial:
            l = serial_map(self.f,self.sweeps)
        else:
            l = parallel_map(self.f,self.sweeps,**kwargs)
        
        return l

    
    def process(self, l):

        self.rhos = np.zeros((len(self.main_variable),self.n_points),dtype=Qobj)
        
        self.expect_a = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_a_dag = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_na = np.zeros((len(self.main_variable),self.n_points),dtype=float)

        self.expect_b = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_b_dag = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_nb = np.zeros((len(self.main_variable),self.n_points),dtype=float)

        self.expect_r = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_r_dag = np.zeros((len(self.main_variable),self.n_points),dtype=complex)
        self.expect_nr = np.zeros((len(self.main_variable),self.n_points),dtype=float)

        self.purity = np.zeros((len(self.main_variable),self.n_points),dtype=float)
        self.purity_a = np.zeros((len(self.main_variable),self.n_points),dtype=float)
        self.purity_b = np.zeros((len(self.main_variable),self.n_points),dtype=float)
        self.purity_r = np.zeros((len(self.main_variable),self.n_points),dtype=float)        
        
        self.tasks = np.zeros((len(self.main_variable),self.n_points),dtype=dict)
        
        sweep_idx=task_idx=0
        for result in l:
            sweep_idx = result['sweep_idx']
            task_idx = result['task_idx']
            
            self.rhos[sweep_idx][task_idx] = result['rho_ss']
            
            self.expect_a[sweep_idx][task_idx] = result['expect_a']
            self.expect_a_dag[sweep_idx][task_idx] = result['expect_a_dag']
            self.expect_na[sweep_idx][task_idx] = result['expect_na']
            
            self.expect_b[sweep_idx][task_idx] = result['expect_b']
            self.expect_b_dag[sweep_idx][task_idx] = result['expect_b_dag']
            self.expect_nb[sweep_idx][task_idx] = result['expect_nb']
            
            self.expect_r[sweep_idx][task_idx] = result['expect_r']
            self.expect_r_dag[sweep_idx][task_idx] = result['expect_r_dag']
            self.expect_nr[sweep_idx][task_idx] = result['expect_nr']

            self.purity[sweep_idx][task_idx] = result['purity']
            self.purity_a[sweep_idx][task_idx] = result['purity_a']
            self.purity_b[sweep_idx][task_idx] = result['purity_b']
            self.purity_r[sweep_idx][task_idx] = result['purity_r']

        for task in self.sweeps:
            self.tasks[task['sweep_idx']][task['idx']] = task


        del self.sweeps
    
        

    


    
def simulation1mk():
    time_start =time.time()
    
    print('Configuring tasks')

    sweeps = np.array([])
    Na= 4
    Nb = 4
    Nr = 4
    wa = 5.1
    wb = 5.7
    wr = 0.1
    ka = kb = 1.0e-4
    kr = 5.0e-4
    gb = 4e-3
    ga = 6e-3
    wd_begin = wa - 50*1e-6-360e-6
    wd_end = wa + 50*1e-6-360e-6
    T = 1e-3
    n_points = 1000
    number_of_cases_Ab = 10
    Abs = np.linspace(1e-6,10e-6,number_of_cases_Ab)
    Aa = 5.0e-6

    wd_b = wb - 0.0001
    
    units = {'wa':'GHz',
             'wb':'GHz',
             'wr':'GHz',
             'ka':'GHz',
             'kb':'GHz',
             'kr':'GHz', 
             'ga':'GHz', 
             'gb':'GHz', 
             'Aa':'GHz', 
             'Ab':'GHz', 
             'T': 'K', 
             'wd_a': 'GHz',
             'wd_b': 'GHz',
             'main_variable':'GHz',
             'sweep_variable':'GHz'}

    wd_as = np.linspace(wd_begin,wd_end,n_points)

    a = tensor(destroy(Na),qeye(Nb),qeye(Nr))
    b = tensor(qeye(Na),destroy(Nb),qeye(Nr))
    r = tensor(qeye(Na),qeye(Nb),destroy(Nr))    

    for idx,Ab in enumerate(Abs):
        print('Creating sweep {}'.format(idx))
        sweeps=np.append(sweeps,create_wd_a_sweep(Na,Nb,Nr,
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
                                        wd_as,
                                        wd_b,
                                        n_points,
                                        idx))

    name = "shift_high_Aa5kHz_ka&kb100kHz_Na4_Nb4_Nr4_T1mk_solver_iterative_gmres_npoints_wd_a1000_Ab10points_range100_serial"
    experiment = Experiment(simulate_steadystate,a,b,r,
                            name,
                            sweeps,
                            wd_as,
                            Abs,
                            'Ab',
                            'wd_a',
                            units)

    print('Simulating')
    l = experiment.simulate(serial=False)
    
    print('Processing')
    experiment.process(l)
    
    time_end = time.time()
    print('Simulation time taken(seconds): {}'.format(time_end-time_start))
    
    print('Saving instance')
    experiment.save('D:\\'+name+'.data')
    
    time_end2 = time.time()
    print('Saving data time taken(seconds): {}'.format(time_end2-time_end))
    
    print('Saving data to csv file')
    experiment.save_csv('D:\\'+name)
    
    time_end3 = time.time()
    print('Saving csv data time taken(seconds): {}'.format(time_end3-time_end2))
    
def simulation3mk():
    time_start =time.time()
    
    print('Configuring tasks')

    sweeps = np.array([])
    Na= 4
    Nb = 4
    Nr = 6
    wa = 5.1
    wb = 5.7
    wr = 0.1
    ka = kb = 1.0e-4
    kr = 5.0e-4
    gb = 4e-3
    ga = 6e-3
    wd_begin = wa - 2000*1e-6
    wd_end = wa + 2000*1e-6
    T = 3e-3
    n_points = 1000
    number_of_cases_Ab = 10
    Abs = np.linspace(1e-6,10e-6,number_of_cases_Ab)
    Aa = 5.0e-6

    wd_b = wb - 0.0001
    
    units = {'wa':'GHz',
             'wb':'GHz',
             'wr':'GHz',
             'ka':'GHz',
             'kb':'GHz',
             'kr':'GHz', 
             'ga':'GHz', 
             'gb':'GHz', 
             'Aa':'GHz', 
             'Ab':'GHz', 
             'T': 'K', 
             'wd_a': 'GHz',
             'wd_b': 'GHz',
             'main_variable':'GHz',
             'sweep_variable':'GHz'}

    wd_as = np.linspace(wd_begin,wd_end,n_points)

    a = tensor(destroy(Na),qeye(Nb),qeye(Nr))
    b = tensor(qeye(Na),destroy(Nb),qeye(Nr))
    r = tensor(qeye(Na),qeye(Nb),destroy(Nr))    

    for idx,Ab in enumerate(Abs):
        print('Creating sweep {}'.format(idx))
        sweeps=np.append(sweeps,create_wd_a_sweep(Na,Nb,Nr,
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
                                        wd_as,
                                        wd_b,
                                        n_points,
                                        idx))

    name = "shift_high_Aa5kHz_ka&kb100kHz_Na4_Nb4_Nr6_T3mk_solver_iterative_gmres_npoints_wd_a1000_Ab10points"
    experiment = Experiment(simulate_steadystate,a,b,r,
                            name,
                            sweeps,
                            wd_as,
                            Abs,
                            'Ab',
                            'wd_a',
                            units)

    print('Simulating')
    l = experiment.simulate(serial=False)
    
    print('Processing')
    experiment.process(l)
    
    time_end = time.time()
    print('Simulation time taken(seconds): {}'.format(time_end-time_start))
    
    print('Saving instance')
    experiment.save('D:\\'+name+'.data')
    
    time_end2 = time.time()
    print('Saving data time taken(seconds): {}'.format(time_end2-time_end))
    
    print('Saving data to csv file')
    experiment.save_csv('D:\\'+name)
    
    time_end3 = time.time()
    print('Saving csv data time taken(seconds): {}'.format(time_end3-time_end2))
    
def simulation5mk():
    time_start =time.time()
    
    print('Configuring tasks')

    sweeps = np.array([])
    Na= 4
    Nb = 4
    Nr = 4
    wa = 5.1
    wb = 5.7
    wr = 0.1
    ka = kb = 1.0e-4
    kr = 5.0e-4
    gb = 4e-3
    ga = 6e-3
    wd_begin = wa - 50*1e-6 -360e-6
    wd_end = wa + 50*1e-6 -360e-6
    T = 5e-3
    n_points = 1000
    number_of_cases_Ab = 10
    Abs = np.linspace(1e-6,10e-6,number_of_cases_Ab)
    Aa = 5.0e-6

    wd_b = wb - 0.0001
    
    units = {'wa':'GHz',
             'wb':'GHz',
             'wr':'GHz',
             'ka':'GHz',
             'kb':'GHz',
             'kr':'GHz', 
             'ga':'GHz', 
             'gb':'GHz', 
             'Aa':'GHz', 
             'Ab':'GHz', 
             'T': 'K', 
             'wd_a': 'GHz',
             'wd_b': 'GHz',
             'main_variable':'GHz',
             'sweep_variable':'GHz'}

    wd_as = np.linspace(wd_begin,wd_end,n_points)

    a = tensor(destroy(Na),qeye(Nb),qeye(Nr))
    b = tensor(qeye(Na),destroy(Nb),qeye(Nr))
    r = tensor(qeye(Na),qeye(Nb),destroy(Nr))    

    for idx,Ab in enumerate(Abs):
        print('Creating sweep {}'.format(idx))
        sweeps=np.append(sweeps,create_wd_a_sweep(Na,Nb,Nr,
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
                                        wd_as,
                                        wd_b,
                                        n_points,
                                        idx))

    name = "shift_high_Aa5kHz_ka&kb100kHz_Na4_Nb4_Nr4_T5mk_solver_iterative_gmres_npoints_wd_a1000_Ab10points_range100"
    experiment = Experiment(simulate_steadystate,a,b,r,
                            name,
                            sweeps,
                            wd_as,
                            Abs,
                            'Ab',
                            'wd_a',
                            units)

    print('Simulating')
    l = experiment.simulate(serial=False)
    
    print('Processing')
    experiment.process(l)
    
    time_end = time.time()
    print('Simulation time taken(seconds): {}'.format(time_end-time_start))
    
    print('Saving instance')
    experiment.save('D:\\'+name+'.data')
    
    time_end2 = time.time()
    print('Saving data time taken(seconds): {}'.format(time_end2-time_end))
    
    print('Saving data to csv file')
    experiment.save_csv('D:\\'+name)
    
    time_end3 = time.time()
    print('Saving csv data time taken(seconds): {}'.format(time_end3-time_end2))
    

if __name__ == "__main__":
    freeze_support()
    simulation1mk()
