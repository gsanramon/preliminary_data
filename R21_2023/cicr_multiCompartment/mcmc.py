import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg') # needed if running as standalone 
import matplotlib.pyplot as plt
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import arviz as az
import sunode
from sunode.wrappers.as_theano import solve_ivp


def cicr(y,t,p):  
    cyto=y[0]
    ER1=y[1]
    ER2=y[2]
    
    kSC=p[0]
    kRyr=p[1]
    kcomp=p[2]
    
    dcytodt = kRyr*ER2 - kSC*cyto
    dER1dt = kSC*cyto - kcomp*ER1
    dER2dt = kcomp*ER1 - kRyr*ER2
    
    return [dcytodt,dER1dt,dER2dt]

# dydt function
def cicrSunode(t,y,p): 
    return {
    'cyto': p.kRyr * y.ER2 - p.kSC * y.cyto,
    'ER1': p.kSC * y.cyto - p.kcomp * y.ER1,
    'ER2': p.kcomp * y.ER1 - p.kRyr * y.ER2}

# run
def run():
    # read in expt data
    dat = np.genfromtxt('exptData.csv', delimiter=',',skip_header=1)
    frameRate=6e-3 # sec
    
    # to remove nan values
    cytoExpt=[]
    for i in dat[:,XXX]:
        if i != i:  # if i is nan, ignore
            continue
        else:
            cytoExpt.append(i)
            
    erExpt=[]
    for i in dat[:,YYY]:
        if i != i:  # if i is nan, ignore
            continue
        else:
            erExpt.append(i)
    
    # normalization
    #cytoExpt-=np.min(cytoExpt)
    cytoMax=np.max(cytoExpt)
    cytoExpt/=cytoMax    
    #erExpt-=np.min(erExpt)
    erExpt/=cytoMax
    
    
    datLen=len(cytoExpt)
    tEnd=datLen*frameRate
    ts=np.linspace(0,tEnd,datLen)
    
    # actually running MCMC
    with pm.Model() as model1:
        # proposal distributions for each parameter
        kSC = pm.Lognormal('kSC', mu=pm.math.log(3.0e0), sigma=1e-1)
        kRyr = pm.Lognormal('kRyr', mu=pm.math.log(2.0e-1), sigma=1e-1)
        kcomp = pm.Lognormal('kcomp', mu=pm.math.log(2.0e0), sigma=1e-1)
        
        # initial values
        y0={'cyto': (np.array(cytoExpt[0]), ()),
            'ER1': (np.array(3.0e-1), ()),
            'ER2': (np.array(erExpt[0]), ())}
        
        # params
        params = {'kSC': (kSC, ()),
        'kRyr': (kRyr, ()),
        'kcomp': (kcomp, ()),
        'useless': (np.array(1.0e0), ())} # some useless fixed param to work around a sunode bug
        
        # running simulations of proposed param sets
        solution, *_ = solve_ivp(
        y0=y0,
        params=params,
        rhs=cicrSunode,
        tvals=ts,
        t0=ts[0],)     
        
        cyto = pm.Deterministic('cyto', solution['cyto'])
        ER2 = pm.Deterministic('ER2', solution['ER2'])
        
        #normalization
        cytoMax=cytoExpt.max()
        #cyto-=cyto.min()
        cyto/=cytoMax
        #ER2-=ER2.min()
        ER2/=cytoMax

        # compute likelihood
        sigma = pm.HalfCauchy('sigma',0.1) 
        Y1 = pm.Normal('Y1', mu=cyto, sigma=sigma, observed=cytoExpt)
        Y2 = pm.Normal('Y2', mu=ER2, sigma=sigma, observed=erExpt)
        
        # run MCMC and store data
        numOfChains=5
        step = pm.DEMetropolisZ()
        trace = pm.sample(2000, tune=1000, target_accept=0.99, cores=numOfChains, compute_convergence_checks=False)
    
    pm.save_trace(trace, directory="OUTPATH", overwrite=True)
       
    results={}
    toFit=['kSC','kRyr','kcomp','sigma']
    for key in toFit:
        vals=[]
        for chain in range(numOfChains):
            dat=trace.get_values(key,chains=[chain])
            vals.append(dat)
        results[key]=vals
    
    pickle.dump( results , open('results.pkl','wb'))



run()

