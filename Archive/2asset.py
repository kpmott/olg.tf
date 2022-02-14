#AUTHOR: KEVIN MOTT
#Implement long-lived stochastic OLG exchange economy with two assets, use TensorFlow for policy functions

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense#, Input, concatenate, Activation
#from tensorflow.keras.initializers import glorot_uniform
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import Model
tf.config.run_functions_eagerly(True)

#from itertools import product
#import time
#import datetime

import matplotlib.pyplot as plt

#from sklearn import metrics
from scipy.optimize import fsolve
from scipy.stats import norm

#import io
import os
import platform

if platform.system() == 'Windows':
    os.chdir("C:\\Users\\kpmott\Dropbox\\CMU\Research\\NeuralNets\\tf.olg")
else:
    os.chdir("/Users/kpmott/Dropbox/CMU/Research/NeuralNets/tf.olg")

#import requests

#-----------------------------------------------------------------------------------------------------------------
#Declare economic primitives -- preferences etc 

#Lifespan 
L = 6
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
ωvec = pd.DataFrame(np.transpose([ls[1:]*w for w in wvec]))

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
    return x**γ

#utility derivative
def up(x):
    return γ*x**(γ-1)

#inverse of utility derivative
def upinv(x):
    if x <= 0:
        return 99999
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
T = 1000
X = pd.DataFrame(0., index=range(T),columns=range(L)); X.iloc[0] = xbar
C = pd.DataFrame(0., index=range(T),columns=range(L)); C.iloc[0] = cbar
E = pd.DataFrame(0., index=range(T),columns=range(L-1)); E.iloc[0] = ebar
B = pd.DataFrame(0., index=range(T),columns=range(L-1))
P = pd.Series(0.,index=range(T)); P.iloc[0] = pbar
Q = pd.Series(0.,index=range(T))



shocks = range(S)
svec = pd.Series(np.random.choice(shocks,T,probs))
svec_not = (svec + 1) % 2
Ωvec = pd.DataFrame.transpose(ωvec[svec])
Δvec = pd.Series(δvec[svec])

input = 2*L-3
output = 2*L-2

OUT = range(output)
equity = range(0,L-2)
bond = range(L-2,2*L-4)
price = range(2*L-4,2*L-3)
ir = range(2*L-3,2*L-2)


Σ = pd.DataFrame(0.,index=range(T),columns=range(input))
Y = pd.DataFrame(0.,index=range(T),columns=range(output))

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

#x = np.ones((T,input))
#y = np.zeros((T,output))


model = Sequential()
model.add(Dense(25, input_dim=input, activation='tanh')) # Hidden 1
model.add(Dense(10, activation='tanh')) # Hidden 2
model.add(Dense(output, activation='linear')) # Output
model.compile(loss='mean_squared_error', optimizer='adam')
#model.predict(np.ones((1,input)))
#model.fit(x,y,verbose=2,epochs=10)


 


#---------------------------
for t in range(1):
    Σ.iloc[t] = np.concatenate((ebar[0:-1],bbar[0:-1],[svec.iloc[t]]),axis=0)
    Y.iloc[t] = model.predict(np.array(Σ.iloc[t]).reshape(1,input))[0]
    e = Y.iloc[t][equity]
    b = Y.iloc[t][bond]
    E.iloc[t] = np.concatenate((e,[equitysupply-sum(e)]),axis=0)
    B.iloc[t] = np.concatenate((b,[bondsupply-sum(b)]),axis=0)
    P.iloc[t] = Y.iloc[t][price]
    Q.iloc[t] = Y.iloc[t][ir]
    X.iloc[t] = np.concatenate(([Ωvec.iloc[t][0]], Ωvec.iloc[t][1:] + (P.iloc[t] + Δvec.iloc[t]) * ebar + Q.iloc[t] * bbar),axis=0)
    C.iloc[t] = X.iloc[t] - np.concatenate((P.iloc[t] * E.iloc[t] - Q.iloc[t] * B.iloc[t],[0]),axis=0)

    #Build per-period forecast



for t in range(1,T):
    Σ.iloc[t] = np.concatenate((E.iloc[t-1][0:-1],B.iloc[t-1][0:-1],[svec.iloc[t]]),axis=0)
    Y.iloc[t] = model.predict(np.array(Σ.iloc[t]).reshape(1,input))[0]
    e = Y.iloc[t][equity]
    b = Y.iloc[t][bond]
    E.iloc[t] = np.concatenate((e,[equitysupply-sum(e)]),axis=0)
    B.iloc[t] = np.concatenate((b,[bondsupply-sum(b)]),axis=0)
    P.iloc[t] = Y.iloc[t][price]
    Q.iloc[t] = Y.iloc[t][ir]
    X.iloc[t] = np.concatenate(([Ωvec.iloc[t][0]], np.array(Ωvec.iloc[t][1:]) + np.array((P.iloc[t] + Δvec.iloc[t]) * E.iloc[t-1] + Q.iloc[t] * B.iloc[t-1])),axis=0)
    C.iloc[t] = np.array(X.iloc[t]) - np.concatenate((np.array(P.iloc[t] * E.iloc[t] - Q.iloc[t] * B.iloc[t]),[0]),axis=0)



#Now use Euler Equations to build "true" Y, use this to train, update
