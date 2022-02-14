#AUTHOR: KEVIN MOTT
#Implement long-lived stochastic OLG exchange economy with two assets, use TensorFlow for policy functions

import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#, Input, concatenate, Activation
import keras.backend as K
#from tensorflow.keras.initializers import glorot_uniform
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import Model
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

#-----------------------------------------------------------------------------------------------------------------
#Declare economic primitives -- preferences etc 

ti = time.time()

#Lifespan 
L = 5
wp = int(L*2/3)
rp = L - wp

#Time discount
β = 0.98**(60/L)

#Risk-aversion coeff
γ = 2

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
    return (x**(1-γ)-1)/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
@tf.function
def upinv(x):
    a = 0.001
    if x <= a:
        return -(x-a) + a**(-1/γ)
    else:
        return x**(-1/γ)


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
    cons = c_eq(e,p*np.ones(L))#max.(1e-3,c_eq(e,p*ones(L)));

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

#solve
bar = fsolve(ss_eq,[*eguess,*[0.5]])
ebar = bar[0:-1]
bbar = ebar*0
pbar = bar[-1]
xbar = x_eq(ebar,pbar*np.ones(L))
cbar = c_eq(ebar,pbar*np.ones(L))

# plt.plot(eguess)
# plt.plot(ebar)
# plt.show()

#-----------------------------------------------------------------------------------------------------------------
#Now somehow let's do neural networks? 

#Forward simulation and training
T = 500

shocks = range(S)
svec = np.random.choice(shocks,T,probs)
Ωvec = [ωvec[s] for s in svec]
Δvec = δvec[svec]

input = 2*L-3
output = 2*L-2

OUT =       slice(0     ,output,1)
equity =    slice(0     ,L-2   ,1)
bond =      slice(L-2   ,2*L-4 ,1)
price =     slice(2*L-4 ,2*L-3 ,1)
ir =        slice(2*L-3 ,2*L-2 ,1)



""" 
Here's the architecture: 
-We don't give any fucks about dimensionality 
-So let's use the actual state variable: 
    -1-pd lagged asset holdings for all agents (except one, market-clearing)

-Input: [(e_i^t)_{i=1}^{L-2}, (b_i^t)_{i=1}^{L-2}, s^t] ∈ ℜ^{2L-3}

-Output: [(e_i^t)_{i=1}^{L-2},(b_i^t)_{i=1}^{L-2}, p^t, q^t ] ∈ ℜ^{2L-2} 
    b.c. for (b_i^t)
-Activations: c(ReLU) e(tanh) p(ReLU) q(ReLU)
"""

model = Sequential()
model.add(Dense(25, input_dim=input, activation='tanh')) # Hidden 1
model.add(Dense(10, activation='tanh')) # Hidden 2
model.add(Dense(output, activation='linear')) # Output

#upinv_list = lambda a: map(upinv,a)

#def prod(iterable):
#    return reduce(operator.mul, iterable, 1)

@tf.function
def dot(K, L):
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))
    

#custom loss function
@tf.function
def euler_loss(y_true,y_pred):
    #Today
    E = tf.stack([[*y_pred[t][equity],*[equitysupply-sum(y_pred[t][equity])]] for t in range(T)],axis=0)
    B = tf.stack([[*y_pred[t][bond],*[bondsupply-sum(y_pred[t][bond])]] for t in range(T)],axis=0)
    P = tf.stack([y_pred[t][price] for t in range(T)],axis=0)
    Q = tf.stack([y_pred[t][ir] for t in range(T)],axis=0)
    x0 = tf.stack([[*[Ωvec[0][0]], *Ωvec[0][1:] + (P[0] + Δvec[0]) * ebar + Q[0] * bbar]],axis=0)
    xrest = tf.stack([[*[Ωvec[t][0]], *Ωvec[t][1:] + (P[t] + Δvec[t]) * E[t-1] + Q[t] * B[t-1]] for t in range(1,T)], axis=0)
    X = tf.stack([*x0,*xrest],axis=0)
    C = tf.stack([[*X[t][0:L-1] - P[t]*E[t] - Q[t]*B[t],*[X[t][-1]]] for t in range(T)], axis=0)

    #Forecast
    Yf = model(tf.stack([[[*E[t][0:-1],*B[t][0:-1],*[s]] for s in range(S)] for t in range(T)],axis=0),training=False)
    Ef = tf.stack([[[*Yf[t][s][equity],*[equitysupply-sum(Yf[t][s][equity])]] for s in range(S)] for t in range(T)])
    Bf = tf.stack([[[*Yf[t][s][bond],*[bondsupply-sum(Yf[t][s][bond])]] for s in range(S)] for t in range(T)])
    Pf = tf.stack([[Yf[t][s][price] for s in range(S)] for t in range(T)])
    Qf = tf.stack([[Yf[t][s][ir] for s in range(S)] for t in range(T)])
    Xf = tf.stack([[[*[ωvec[s][0]],*ωvec[s][1:] + (Pf[t][s] + δvec[s])*E[t] + B[t]] for s in range(S)] for t in range(T)],axis=0)
    Cf = tf.stack([[[*Xf[t][s][0:L-1] - Pf[t][s]*Ef[t][s] - Qf[t][s]*Bf[t][s],*[Xf[t][s][-1]]] for s in range(S)] for t in range(T)], axis=0)

    #Equity Eulers
    E_state_by_state = tf.stack([[(Pf[t][s]+δvec[s])*up(Cf[t][s][1:L]) for s in range(S)] for t in range(T)],axis=0)
    E_exp = tf.stack([[dot(probs,[E_state_by_state[t][s][ℓ] for s in range(S)]) for ℓ in range(L-1)] for t in range(T)],axis=0)
    E_eul = tf.math.abs(tf.stack([[upinv(β*E_exp[t][ℓ]/P[t])/C[t][ℓ] for ℓ in range(L-1)] for t in range(T)],axis=0))

    #Bond Eulers
    B_state_by_state = tf.stack([[up(Cf[t][s][1:L]) for s in range(S)] for t in range(T)],axis=0)
    B_exp = tf.stack([[dot(probs,[B_state_by_state[t][s][ℓ] for s in range(S)]) for ℓ in range(L-1)] for t in range(T)],axis=0)
    B_eul = tf.math.abs(tf.stack([[upinv(β*B_exp[t][ℓ]/Q[t])/C[t][ℓ] for ℓ in range(L-1)] for t in range(T)],axis=0))

    #return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    E_eul_sum = tf.math.reduce_sum(E_eul,axis=1)
    B_eul_sum = tf.math.reduce_sum(B_eul,axis=1)
    
    return tf.reshape(E_eul_sum + B_eul_sum, (T,))

    
model.compile(loss=None, optimizer='adam')

#---------------------------
for t in range(1):
    #make sure we start fresh
    X = []
    C = []
    E = [] 
    B = []
    P = []
    Q = []
    Σ = []
    Y = []

    #t=0
    Σ.append(np.concatenate((ebar[0:-1],bbar[0:-1],[svec[t]]),axis=0))
    Y.append(model(Σ[t].reshape(1,input), training=False)[0].numpy())
    e = Y[t][equity]
    b = Y[t][bond]
    E.append(np.concatenate((e,[equitysupply-sum(e)]),axis=0))
    B.append(np.concatenate((b,[bondsupply-sum(b)]),axis=0))
    
for t in tqdm(range(1,T)):
    Σ.append(np.concatenate((E[t-1][0:-1],B[t-1][0:-1],[svec[t]]),axis=0))
    Y.append(model(Σ[t].reshape(1,input), training=False)[0].numpy())
    e = Y[t][equity]
    b = Y[t][bond]
    E.append(np.concatenate((e,[equitysupply-sum(e)]),axis=0))
    B.append(np.concatenate((b,[bondsupply-sum(b)]),axis=0))
    

model.add_loss(lambda: euler_loss(tf.zeros((T,output)),model(tf.convert_to_tensor(Σ),training=False)))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(tf.convert_to_tensor(Σ),tf.zeros((T,output)),batch_size=T,epochs=25,verbose=1,callbacks=[TqdmCallback(),tbc])

print('Runtime =',time.time()-ti)
