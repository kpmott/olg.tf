#AUTHOR: KEVIN MOTT
#Implement long-lived stochastic OLG exchange economy with two assets, use TensorFlow for policy functions

from signal import Sigmasks
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
tf.config.run_functions_eagerly(True)

from functools import reduce  # Required in Python 3
import operator

from tqdm import tqdm
from tqdm.keras import TqdmCallback
import time

#from itertools import product
import datetime

import matplotlib.pyplot as plt

#from sklearn import metrics
from scipy.optimize import fsolve
from scipy.stats import norm

#import io
import os
import platform

if platform.system() == 'Windows':
    os.chdir("C:\\Users\\kpmott\Dropbox\\CMU\Research\\NeuralNets\\tf.olg")
elif platform.system() == 'Linux':
	os.chdir("/home/kpmott/Dropbox/CMU/Research/NeuralNets/tf.olg")
else:
    os.chdir("/Users/kpmott/Dropbox/CMU/Research/NeuralNets/tf.olg")

#import requests

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#-----------------------------------------------------------------------------------------------------------------
#Declare economic primitives -- preferences etc 

#ti = time.time()

#Lifespan 
L = 60
wp = int(L*2/3)
rp = L - wp

#Time discount
β = 1.#0.98**(60/L)

#Risk-aversion coeff
γ = 2.

#Stochastic elements
probs = [0.5, 0.5]
S = len(probs) 

#share of total resources
ωGuess = 7/8*norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45) \
       / sum(norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45))

#share of total resources: 1/16 to dividend; the rest to endowment income
ls = np.array([*[1/8], *ωGuess, *np.zeros(rp)])

#total resources
wbar = 1

#shock perturbation vector
ζtrue = 0.03
wvec = [wbar - ζtrue, wbar + ζtrue]            #total income in each state
δvec = np.multiply(ls[0],wvec)                    #dividend in each state
ωvec = [ls[1:]*w for w in wvec]

#plt.plot(ωvec[0])
#plt.plot(ωvec[1])
#plt.show()

#mean-center all shock-contingent values
δ = ls[0]
ω = ls[1:]

#net supply of assets: for later
equitysupply = 1
bondsupply = 0

#-----------------------------------------------------------------------------------------------------------------
#utility
def u(x):
    if γ == 1:
        return np.log(x)
    else:
        return (x**(1-γ)-1)/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
@tf.function
def upinv(x):
    a = 1e-16
    if x <= a:
        return -(x-a) + a**(-1/γ)
    else:
        return x**(-1/γ)

def upinv_tf(x):
    return x**(1/γ)
    #a = 1e-16
    #return tf.where(tf.less_equal(x,a),-(x-a)+a**(-1/γ), x**(1/γ))

#-----------------------------------------------------------------------------------------------------------------
#det-SS calculations
#compute lifetime consumption based on equity holdings e0 and prices p0
def c_eq(e0,p0):
    #p0 ∈ ℜ^L:      prices from birth to death 
    #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
    
    #vector of consumption in each period 
    cons = np.zeros(L)
    cons[0] = ω[1]-p0[1]*e0[1]
    cons[-1] = ω[-1]+(p0[-1]+δ)*e0[-1]
    for i in range(1,L-1):
        cons[i] = ω[i]+(p0[i]+δ)*e0[i-1]-p0[i]*e0[i]
    
    return cons

def x_eq(e0,p0):
    #p0 ∈ ℜ^L:      prices from birth to death 
    #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
    
    #vector of consumption in each period 
    x = np.zeros(L)
    x[0] = ω[1]
    x[-1] = ω[-1]+(p0[-1]+δ)*e0[-1]
    for i in range(1,L-1):
        x[i] = ω[i]+(p0[i]+δ)*e0[i-1]
    
    return x

def ss_eq(x):
    #equity holdings for 1:(L-1)
    e = x[0:-1]
    #price
    p = x[-1]
    
    #consumption must be nonnegative
    cons = np.maximum(1e-12*np.ones(L),c_eq(e,p*np.ones(L)))

    #Euler equations
    ssVec = np.zeros(L)
    for i in range(0,L-1):
        ssVec[i] = p*up(cons[i]) - β*(p+δ)*up(cons[i+1])
    #market clearing
    ssVec[-1] = equitysupply - sum(e)

    return ssVec

#Guess equity is hump-shaped
eguess = norm.pdf(range(1,L),.8*wp,L/3)
eguess = [x/sum(eguess) for x in eguess]
pguess = .5

#solve
fsolve(ss_eq,[*eguess,*[pguess]],full_output=1)
bar = fsolve(ss_eq,[*eguess,*[pguess]],full_output=0)
ebar = bar[0:-1]
bbar = ebar*0
pbar = bar[-1]
qbar = 1/β
xbar = x_eq(ebar,pbar*np.ones(L))
cbar = c_eq(ebar,pbar*np.ones(L))

#-----------------------------------------------------------------------------------------------------------------
#time and such for neurals 
T = 2000
burn = int(T/10)
train = T - burn
time = slice(burn,T,1)

shocks = range(S)
svec = np.random.choice(shocks,T,probs)
Ωvec = [ωvec[s] for s in svec]
Δvec = δvec[svec]
Ω = tf.convert_to_tensor(Ωvec,dtype='float32')
Δ = tf.reshape(tf.convert_to_tensor(Δvec,dtype='float32'),(T,1))
ω = tf.convert_to_tensor(ωvec,dtype='float32')
δ = tf.reshape(tf.convert_to_tensor(δvec,dtype='float32'),(S,1))
#-----------------------------------------------------------------------------------------------------------------
input = 2*(L-2)+1
output = 2*(L-1)+2

OUT =       slice(0     ,output,1)
equity =    slice(0     ,L-1   ,1)
bond =      slice(L-1   ,2*L-2 ,1)
price =     slice(2*L-2 ,2*L-1 ,1)
ir =        slice(2*L-1 ,2*L ,1)

def activation_final(tensorOut):
    #τ = tf.shape(tensorOut)[0]
    out_e = tf.keras.activations.tanh(tensorOut[:,equity])
    out_b = tf.keras.activations.tanh(tensorOut[:,bond])
    out_p = tf.keras.activations.softplus(tensorOut[:,price])
    out_q = tf.keras.activations.softplus(tensorOut[:,ir])
    return tf.concat([out_e,out_b,out_p,out_q],axis=1)

model = Sequential()
model.add(Dense(128, input_dim=input,    activation='sigmoid')) # Hidden 1
model.add(Dense(128,                     activation='sigmoid')) # Hidden 2
model.add(Dense(128,                     activation='sigmoid')) # Hidden 3
model.add(Dense(output,                 activation=activation_final)) # Output
model.compile(loss='mean_squared_error', optimizer='adam')


#PART ZERO: c = c̄/̄x*x
for t in range(1):
    X = []
    C = []
    P = []
    E = []

    counter = 0
    p = pbar*1e10
    p_old = pbar
    
    for counter in range(500):
        x = Ωvec[t]+ [*[0],*ebar*(p_old + Δvec[t])]
        c = x*cbar/xbar
        p = sum(x-c)
        if np.abs(p-p_old) < 1e-12:
            e = (x[:-1]-c[:-1])/p
            #print(counter)
            break   
        else:
            counter += 1
            p_old = p                             
    
    X.append(x)
    C.append(c)
    P.append(p)
    E.append(e)

for t in range(1,T):
    counter = 0
    p = pbar*1e10
    p_old = pbar
    
    for counter in range(500):
        x = Ωvec[t]+ [*[0],*E[t-1]*(p_old + Δvec[t])]
        c = x*cbar/xbar
        p = sum(x-c)
        if np.abs(p-p_old) < 1e-12:
            e = (x[:-1]-c[:-1])/p
            #print(counter)
            break   
        else:
            counter += 1
            p_old = p                             
    
    X.append(x)
    C.append(c)
    P.append(p)
    E.append(e)

X = tf.convert_to_tensor(X,dtype='float32')
C = tf.convert_to_tensor(C,dtype='float32')
E = tf.convert_to_tensor(E,dtype='float32')
P = tf.convert_to_tensor(P,dtype='float32')
Σ = tf.concat([E[:-1,:-1],E[:-1,:-1]*0,tf.reshape(tf.convert_to_tensor(svec[:-1],'float32'),(T-1,1))],1)
y_train = tf.concat([E[1:],E[1:]*0,tf.reshape(P[1:],(T-1,1)),tf.reshape(P[1:],(T-1,1))*0],1)
model.fit(Σ[time],y_train[time],batch_size=T,epochs=500,verbose=0,callbacks=[TqdmCallback()])

#-----------------------------------------------------------------------------------------------------------------
#PART ONE: ADD BOND AND TRAIN NETWORK
#Forward simulation and training

"""
for thyme in tqdm(range(5)):
    for t in range(1):
        #t=0
        Σ = []
        Y = []
        Σ.append([*ebar[:-1],*bbar[:-1],*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity][:-1]
        b = Y[t][bond][:-1]
        
    for t in tqdm(range(1,T)):
        Σ.append([*e,*b,*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity][:-1]
        b = Y[t][bond][:-1]

    # Y = model(Σ,training=False)[time]
    # E = Y[:,equity]
    # B = Y[:,bond]
    # P = Y[:,price]
    # Q = Y[:,ir]
    # x0 = tf.reshape(tf.concat([tf.reshape(Ω[time][0,0],(1,)),Ω[time][0,1:] + (P[0] + Δ[time][0])*ebar + bbar],axis=0),(1,L))
    # xrest = tf.concat([tf.reshape(Ω[time][1:,0],(train-1,1)),Ω[time][1:,1:] + (P[1:] + Δ[time][1:])*E[0:-1] + B[0:-1]],axis=1)
    # X = tf.concat([x0,xrest],axis=0)
    # C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(train,1))],axis=1)

    # phat = tf.reshape(tf.math.reduce_sum(X-C,axis=1),(train,1))
    # qhat = tf.constant(qbar,shape=(train,1))
    # chat = cbar/xbar*X
    # ehat = (X[:,:-2]-chat[:,:-2])/phat
    # train_y = tf.concat([ehat,ehat*0,phat,qhat],axis=1)

    train_y = tf.repeat([[*ebar,*bbar,*[pbar],*[qbar]]],repeats=train,axis=0)
    model.fit(tf.convert_to_tensor(Σ)[time],train_y,batch_size=T,epochs=500,verbose=0,callbacks=[TqdmCallback()])


model.summary()
Y = model(tf.convert_to_tensor(Σ),training=False)[time]
E = Y[:,equity]
B = Y[:,bond]
P = Y[:,price]
Q = Y[:,ir]
x0 = tf.reshape(tf.concat([tf.reshape(Ω[time][0,0],(1,)),Ω[time][0,1:] + (P[0] + Δ[time][0])*ebar + bbar],axis=0),(1,L))
xrest = tf.concat([tf.reshape(Ω[time][1:,0],(train-1,1)),Ω[time][1:,1:] + (P[1:] + Δ[time][1:])*E[0:-1] + B[0:-1]],axis=1)
X = tf.concat([x0,xrest],axis=0)
C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(train,1))],axis=1)
"""

#abs tol    
ϵ = 1e-8

#custom loss function
@tf.function
def euler_loss(y_true,y_pred):
    #ISSUE IS SIZE(T,) BECAUSE INPUT BATCH IS 32 OR WHATEVER TF 
    τ = tf.shape(y_pred).numpy()[0]
    #Today
    E = y_pred[:,equity]
    B = y_pred[:,bond]
    P = y_pred[:,price]
    Q = y_pred[:,ir]
    x0 = tf.reshape(tf.concat([tf.reshape(Ω[time][0,0],(1,)),Ω[time][0,1:] + (P[0] + Δ[time][0])*ebar + bbar],axis=0),(1,L))
    xrest = tf.concat([tf.reshape(Ω[time][1:,0],(τ-1,1)),Ω[time][1:,1:] + (P[1:] + Δ[time][1:])*E[0:-1] + B[0:-1]],axis=1)
    X = tf.concat([x0,xrest],axis=0)
    C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(τ,1))],axis=1)

    #Forecast
    Σf = tf.concat(
            [tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(τ,1))],axis=1) for s in range(S)]
            ,axis=0)
    yf = model(Σf,training=False)
    Yf = tf.stack([yf[s*τ:(s+1)*τ,:] for s in range(S)],axis=1)
    Ef = Yf[:,:,equity]
    Bf = Yf[:,:,bond]
    Pf = Yf[:,:,price]
    Qf = Yf[:,:,ir]
    xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[τ],axis=0),(τ,S,1))
    xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
    Xf = tf.concat([xf0,xfrest],2)
    Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(τ,S,1))],axis=2)
    
    Cpos = tf.greater(C,0)
    Cfpos = tf.math.reduce_all(tf.greater(Cf,0),axis=1)
    Cpen = tf.math.logical_and(Cpos,Cfpos)
    negC = tf.math.minimum(tf.reduce_min(Cf,1),C)
    #Cf = tf.math.maximum(Cf,ϵ)

    #Equity Expectation
    E_exp = tf.tensordot((Pf + δ)*up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #Bond Expectation
    B_exp = tf.tensordot(up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #"Euler" penalty for oldest
    Eul_old = tf.where(Cpen[:,-1],0.,10000000.-C[:,-1])

    #Market clearing
    E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=1))
    B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=1))

    #Loss = Euler for each agent
    # B_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*B_exp/Q)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*E_exp/P)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul_sum = tf.math.reduce_sum(E_eul,axis=1)
    # B_eul_sum = tf.math.reduce_sum(B_eul,axis=1)
    # return E_eul_sum + B_eul_sum + E_mc_sum + B_mc_sum + Eul_old

    #Loss = Divide Eulers
    Eul = tf.reduce_sum(tf.where(Cpen[:,:-1],tf.math.abs(P/Q - E_exp/B_exp),10000000.-negC[:,:-1]),axis=1)
    return Eul + B_mc_sum + E_mc_sum + Eul_old

model.compile(loss=euler_loss, optimizer='adam')

for t in range(1):
    #t=0
    Σ = []
    Y = []
    Σ.append([*ebar[0:-1],*bbar[0:-1],*[svec[t]]])
    Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
    e = Y[t][equity]
    b = Y[t][bond]
    
for t in tqdm(range(1,T)):
    Σ.append([*ebar[0:-1],*bbar[0:-1],*[svec[t]]])
    Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
    e = Y[t][equity]
    b = Y[t][bond]

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(tf.convert_to_tensor(Σ)[time],tf.zeros((train,output)),batch_size=train,epochs=500,verbose=0,callbacks=[TqdmCallback()])#,tbc])

skip = False
for thyme in tqdm(range(2500)):
    for t in range(1):
        #t=0
        Σ = []
        Y = []
        Σ.append([*ebar[0:-1],*bbar[0:-1],*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity]
        b = Y[t][bond]
        
    for t in tqdm(range(1,T)):
        Σ.append([*ebar[0:-1],*bbar[0:-1],*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity]
        b = Y[t][bond]

    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(tf.convert_to_tensor(Σ)[time],tf.zeros((train,output)),batch_size=train,epochs=500,verbose=0,callbacks=[TqdmCallback()])#,tbc])

    skip = tf.math.reduce_mean(euler_loss(tf.zeros((train,output)),tf.convert_to_tensor(Y,dtype='float32')[time])) <= np.log(ϵ)
    if skip.numpy():
        break

#Economy
Y = model(tf.convert_to_tensor(Σ),training=False)[time]
E = Y[:,equity]
B = Y[:,bond]
P = Y[:,price]
Q = Y[:,ir]
x0 = tf.reshape(tf.concat([tf.reshape(Ω[time][0,0],(1,)),Ω[time][0,1:] + (P[0] + Δ[time][0])*ebar + bbar],axis=0),(1,L))
xrest = tf.concat([tf.reshape(Ω[time][1:,0],(train-1,1)),Ω[time][1:,1:] + (P[1:] + Δ[time][1:])*E[0:-1] + B[0:-1]],axis=1)
X = tf.concat([x0,xrest],axis=0)
C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(train,1))],axis=1)
#Forecasts
τ = train
Σf = tf.concat([tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(τ,1))],axis=1) for s in range(S)],axis=0)
yf = model(Σf,training=False)
Yf = tf.stack([yf[s*τ:(s+1)*τ,:] for s in range(S)],axis=1)
Ef = Yf[:,:,equity]
Bf = Yf[:,:,bond]
Pf = Yf[:,:,price]
Qf = Yf[:,:,ir]
xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[τ],axis=0),(τ,S,1))
xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
Xf = tf.concat([xf0,xfrest],2)
Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(τ,S,1))],axis=2)

"""
"""