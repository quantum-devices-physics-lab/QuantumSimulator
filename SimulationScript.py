from qutip import *
import numpy as np
from QuantumSimulator import *
import datetime


def unpack_simulation_data(filename):
    file = open(filename,"rb")
    data = pickle.load(file)
    file.close()
    return data

def add_simulation_experiment(N,wa,wb,wr,ga,gb,ka,kb,kr,A,T,n_points,begin_w,end_w,name,tasks):
    wa = wa 
    wb = wb
    wr = wr 
    ga = ga
    gb = gb
    A = A

    wds = np.linspace(begin_w,end_w,n_points)*factor
    for idx,wd in enumerate(wds):
        tasks.append(create_task(N,wa,wb,wr,ga,gb,ka,kb,kr,T,A,wd,idx,name))



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
    task["kr"] = task["kr"]/factor
    task["ka"] = task["ka"]/factor
    task["kb"] = task["kb"]/factor
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

def get_graphs_case(dirName,case,n_system,n_points=0):
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
    
    return (n_points,data.task,rhos,na,purity,data.a,data.b,data.r)

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
    
        
        
    def process(self,
                dirname,
                number_of_cases,
                n_points,
                name_cases,
                N,
                n_oscillators):
        
        self.number_of_cases = number_of_cases
        self.n_points = n_points
        self.name_cases = name_cases
        self.fock = N
        self.n_oscillators = n_oscillators
        
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
                n_points,task,rhos,na,p,a,b,r = get_graphs_case(dirname,name_case+str(i),self.fock,n_points=self.n_points)
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
    number_of_oscillators = 3
    wa = 5.1
    wb = 5.7
    wr = 1.0
    ka = 1.0e-4
    kb = 1.0e-4
    kr = 1.0e-2
    gb = 0.010
    ga = 0.005
    wd_begin = wa - 50*1e-6
    wd_end = wa + 50*1e-6
    T = 10e-3
    n_points = 10
    n_case = 0
    number_of_cases = 5
    
    Abs = np.linspace(0.0,3,number_of_cases) *1e-3
    wd_b = 5.7
    Aa = 1e-6
    
    for Ab in Abs:
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
                                 Aa,
                                 Ab,
                                 wd_begin, # begin_w
                                 wd_end, # end_w
                                 wd_b,
                                 n_points,
                                 0, #idx
                                 "case_Ab{}".format(n_case))) #name task
        
        n_case = n_case + 1
      
    name = "2 Cavities 1 Resonator Drive Simulation"
    filename = name.replace(" ","_")
    dirName = "data_{}_{}/".format(filename,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    experiment = Experiment()
    

    simulate(name,tasks,dirName)
    
    print("Beginning Post-processing");
    
    experiment.process(dirName,
                       number_of_cases,
                       n_points,
                       ['Ab'],
                       N,
                       number_of_oscillators)
    
    experiment.save(name.replace(" ","_")+'.data')
    
    print("Finished Post-processing");
    
    
    
