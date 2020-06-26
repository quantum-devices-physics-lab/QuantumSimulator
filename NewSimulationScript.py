from qutip import *
import numpy as np
import scipy.constants as sc
from multiprocessing import  freeze_support
import pickle
import os
import re
import sys

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

    print('sweep {} task {}'.format(task['sweep_idx'],task['idx']))
    
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

    rho_ss = steadystate(H, c_ops)
    
    purity = (rho_ss*rho_ss).tr()
    purity_a = (rho_ss.ptrace(0)*rho_ss.ptrace(0)).tr()
    purity_b = (rho_ss.ptrace(1)*rho_ss.ptrace(1)).tr()
    purity_r = (rho_ss.ptrace(2)*rho_ss.ptrace(2)).tr()
    
    expect_a = (rho_ss*a).tr()
    expect_a_dag = (rho_ss*a.dag()).tr()
    expect_na = (a*a.dag()*rho_ss).tr()
    
    expect_b = (rho_ss*b).tr()
    expect_b_dag = (rho_ss*b.dag()).tr()
    expect_nb = (b*b.dag()*rho_ss).tr()
    
    expect_r = (rho_ss*r).tr()
    expect_r_dag = (rho_ss*r.dag()).tr()
    expect_nr = (r*r.dag()*rho_ss).tr()

    result = {}
    result['task_idx'] = task['idx']
    result['sweep_idx'] = task['sweep_idx']

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

    return result


class Experiment():

    def __init__(self,f=0,a=0,b=0,r=0,name='',sweeps=np.array([]), sweep_range = np.array([]), sweep_variable=np.array([]),sweep_variable_name=''):

        self.name = name
        self.n_points = len(sweep_range)
        self.sweeps = sweeps
        self.sweep_range = sweep_range
        self.sweep_variable = sweep_variable
        self.sweep_variable_name = sweep_variable_name
        self.f = f
        self.a = a
        self.b = b
        self.r = r

    def save(self,filename):
        file = open(filename,"wb")
        pickle.dump(self,file)
        file.close()
        
    def load(self,filename):
        file = open(filename,"rb")
        data = pickle.load(file)
        file.close()

        self.n_points =  data.n_points
        self.sweep_range =  data.sweep_range
        self.sweep_variable =  data.sweep_variable
        self.sweep_variable_name =  data.sweep_variable_name
        self.f = data.f
        self.a = data.a
        self.b = data.b
        self.r = data.r
        self.name = data.name
        self.tasks = data.tasks
        self.rhos = data.rhos

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
        
        
    def simulate(self,serial=False):
        l = []
        if serial:
            l = serial_map(self.f,self.sweeps)
        else:
            l = parallel_map(self.f,self.sweeps)
        self.process(l)

    
    def process(self, l):
        self.rhos = np.zeros((len(self.sweep_variable),self.n_points),dtype=Qobj)
        
        self.expect_a = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_a_dag = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_na = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)

        self.expect_b = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_b_dag = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_nb = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)

        self.expect_r = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_r_dag = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.expect_nr = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)

        self.purity = np.zeros((len(self.sweep_variable),self.n_points),dtype=complex)
        self.purity_a = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)
        self.purity_b = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)
        self.purity_r = np.zeros((len(self.sweep_variable),self.n_points),dtype=float)        
        
        self.tasks = np.zeros((len(self.sweep_variable),self.n_points),dtype=dict)

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
    
        

    

if __name__ == "__main__":
    freeze_support()

    sweeps = np.array([])
    Na=Nb=Nr = 3
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
    n_points = 1000
    number_of_cases_ga = 10
    gas = np.linspace(0,10e-3,number_of_cases_ga)
    Aa = 5.0e-6
    Ab = 5.0e-6
    wd_b = wb - 0.0001

    wd_as = np.linspace(wd_begin,wd_end,n_points)

    a = tensor(destroy(Na),qeye(Nb),qeye(Nr))
    b = tensor(qeye(Na),destroy(Nb),qeye(Nr))
    r = tensor(qeye(Na),qeye(Nb),destroy(Nr))    

    for idx,ga in enumerate(gas):
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

    name = "test"
    experiment = Experiment(simulate_steadystate,a,b,r,
                            name,
                            sweeps,
                            wd_as,
                            gas,
                            'ga')

    experiment.simulate()

    experiment.save('test.data')

    
