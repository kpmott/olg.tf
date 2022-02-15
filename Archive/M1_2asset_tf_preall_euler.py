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

# from tqdm import tqdm
# from tqdm.keras import TqdmCallback
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
L = 10
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
    a = 1e-16
    if x <= a:
        return -(x-a) + a**(-1/γ)
    else:
        return x**(-1/γ)

def upinv_tf(x):
    a = 1e-16
    return tf.where(tf.less_equal(x,a),-(x-a)+a**(-1/γ), x**(1/γ))

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

Ω = tf.convert_to_tensor(Ωvec,dtype='float32')
Δ = tf.reshape(tf.convert_to_tensor(Δvec,dtype='float32'),(T,1))
ω = tf.convert_to_tensor(ωvec,dtype='float32')
δ = tf.reshape(tf.convert_to_tensor(δvec,dtype='float32'),(S,1))



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
model.add(Dense(64, input_dim=input, activation='sigmoid')) # Hidden 1
model.add(Dense(64, activation='sigmoid')) # Hidden 2
model.add(Dense(32, activation='sigmoid')) # Hidden 3
model.add(Dense(output, activation='selu')) # Output
    
#custom loss function
@tf.function
def euler_loss(y_true,y_pred):
    #ISSUE IS SIZE(T,) BECAUSE INPUT BATCH IS 32 OR WHATEVER TF 
    τ = tf.shape(y_pred).numpy()[0]
    #Today
    E = tf.concat([y_pred[:,equity],tf.reshape(equitysupply-tf.reduce_sum(y_pred[:,equity],axis=1),(τ,1))],axis=1)
    B = tf.concat([y_pred[:,bond],tf.reshape(bondsupply-tf.reduce_sum(y_pred[:,bond],axis=1),(τ,1))],axis=1)
    P = y_pred[:,price]
    Q = y_pred[:,ir]
    x0 = tf.reshape(tf.concat([tf.reshape(Ω[0,0],(1,)),Ω[0,1:] + (P[0] + Δ[0])*ebar + bbar],axis=0),(1,L))
    xrest = tf.concat([tf.reshape(Ω[1:,0],(τ-1,1)),Ω[1:,1:] + (P[1:] + Δ[1:])*E[0:-1] + B[0:-1]],axis=1)
    X = tf.concat([x0,xrest],axis=0)
    C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(τ,1))],axis=1)

    #Forecast
    Σf = tf.stack([tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(τ,1))],axis=1) for s in range(S)],axis=1) #how to avoid list comprehension? 
    Yf = model(Σf,training=False)    
    Ef = tf.concat([Yf[:,:,equity],equitysupply-tf.reshape(tf.reduce_sum(Yf[:,:,equity],axis=2),(τ,S,1))],axis=2)
    Bf = tf.concat([Yf[:,:,bond],bondsupply-tf.reshape(tf.reduce_sum(Yf[:,:,bond],axis=2),(τ,S,1))],axis=2)
    Pf = Yf[:,:,price]
    Qf = Yf[:,:,ir]
    xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[τ],axis=0),(τ,S,1))
    xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
    Xf = tf.concat([xf0,xfrest],2)
    Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(τ,S,1))],axis=2)
    
    ϵ = 1e-8
    #Equity Eulers
    E_exp = tf.tensordot((Pf + δ)*up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    E_eul = tf.math.abs(upinv_tf(β*E_exp/(P+ϵ))/(C[:,:-1] + ϵ))
    #E_eul = tf.math.abs(β*E_exp/(P+ϵ) - up(C[:,:-1]))

    #Bond Eulers
    B_exp = tf.tensordot(up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    B_eul = tf.math.abs(upinv_tf(β*B_exp/(Q+ϵ))/(C[:,:-1] + ϵ))
    #B_eul = tf.math.abs(β*B_exp/(Q+ϵ) - up(C[:,:-1]))

    #sums
    E_eul_sum = tf.math.reduce_sum(E_eul,axis=1)
    B_eul_sum = tf.math.reduce_sum(B_eul,axis=1)
    return tf.math.log(tf.math.add(E_eul_sum + B_eul_sum, ϵ), (τ,))

    
model.compile(loss=euler_loss, optimizer='adam')

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
    
for t in range(1,T):
    Σ.append(np.concatenate((E[t-1][0:-1],B[t-1][0:-1],[svec[t]]),axis=0))
    Y.append(model(Σ[t].reshape(1,input), training=False)[0].numpy())
    e = Y[t][equity]
    b = Y[t][bond]
    E.append(np.concatenate((e,[equitysupply-sum(e)]),axis=0))
    B.append(np.concatenate((b,[bondsupply-sum(b)]),axis=0))
    

#model.add_loss(lambda: euler_loss(tf.zeros((T,output)),model(tf.convert_to_tensor(Σ),training=False)))

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# for thyme in tqdm(range(5000)):
#     for t in range(1):
#         #make sure we start fresh
#         X = []
#         C = []
#         E = [] 
#         B = []
#         P = []
#         Q = []
#         Σ = []
#         Y = []

#         #t=0
#         Σ.append(np.concatenate((ebar[0:-1],bbar[0:-1],[svec[t]]),axis=0))
#         Y.append(model(Σ[t].reshape(1,input), training=False)[0].numpy())
#         e = Y[t][equity]
#         b = Y[t][bond]
#         E.append(np.concatenate((e,[equitysupply-sum(e)]),axis=0))
#         B.append(np.concatenate((b,[bondsupply-sum(b)]),axis=0))
        
#     for t in range(1,T):
#         Σ.append(np.concatenate((E[t-1][0:-1],B[t-1][0:-1],[svec[t]]),axis=0))
#         Y.append(model(Σ[t].reshape(1,input), training=False)[0].numpy())
#         e = Y[t][equity]
#         b = Y[t][bond]
#         E.append(np.concatenate((e,[equitysupply-sum(e)]),axis=0))
#         B.append(np.concatenate((b,[bondsupply-sum(b)]),axis=0))
        

#     #model.add_loss(lambda: euler_loss(tf.zeros((T,output)),model(tf.convert_to_tensor(Σ),training=False)))

#     log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     class haltCallback(tf.keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs={}):
#             if(logs.get('loss') <= 1.):
#                 print("\n\n\nReached 1 loss value so cancelling training!\n\n\n")
#                 self.model.stop_training = True
#     trainingStopCallback = haltCallback()

#     model.fit(tf.convert_to_tensor(Σ),tf.zeros((T,output)),batch_size=T,epochs=500,verbose=0,callbacks=[TqdmCallback(),tbc,trainingStopCallback])


# Y = tf.convert_to_tensor(Y)
# E = tf.concat([Y[:,equity],tf.reshape(equitysupply-tf.reduce_sum(Y[:,equity],axis=1),(T,1))],axis=1)
# B = tf.concat([Y[:,bond],tf.reshape(bondsupply-tf.reduce_sum(Y[:,bond],axis=1),(T,1))],axis=1)
# P = Y[:,price]
# Q = Y[:,ir]
# x0 = tf.reshape(tf.concat([tf.reshape(Ω[0,0],(1,)),Ω[0,1:] + (P[0] + Δ[0])*ebar + bbar],axis=0),(1,L))
# xrest = tf.concat([tf.reshape(Ω[1:,0],(T-1,1)),Ω[1:,1:] + (P[1:] + Δ[1:])*E[0:-1] + B[0:-1]],axis=1)
# X = tf.concat([x0,xrest],axis=0)
# C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(T,1))],axis=1)
# Σf = tf.stack([tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(T,1))],axis=1) for s in range(S)],axis=1) #how to avoid list comprehension? 
# Yf = model(Σf,training=False)    
# Ef = tf.concat([Yf[:,:,equity],equitysupply-tf.reshape(tf.reduce_sum(Yf[:,:,equity],axis=2),(T,S,1))],axis=2)
# Bf = tf.concat([Yf[:,:,bond],bondsupply-tf.reshape(tf.reduce_sum(Yf[:,:,bond],axis=2),(T,S,1))],axis=2)
# Pf = Yf[:,:,price]
# Qf = Yf[:,:,ir]
# xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[T],axis=0),(T,S,1))
# xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
# Xf = tf.concat([xf0,xfrest],2)
# Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(T,S,1))],axis=2)

model.fit(tf.convert_to_tensor(Σ),tf.zeros((T,output)),batch_size=T,epochs=100,verbose=1,callbacks=[tbc]) #TqdmCallback(),