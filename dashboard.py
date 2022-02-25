#Load packages
from packages import *

#Load paramaters
from params import *

#Load neural network architecture
from nn import *

import detSS
#load in detSS allocs
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#Import fit function
import nn_fit
Σ, Y = nn_fit.fit_euler(num_epochs=1,num_iters=150,tb=False,batchsize=T) 



for t in range(1):
    #t=0
    Σ = []
    Y = []
    Σ.append([*ebar[0:-1],*bbar[0:-1],*[rvec[t]]])
    Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
    e = Y[t][equity]
    b = Y[t][bond]

for t in tqdm(range(1,T)):
    Σ.append([*e[:-1],*b[:-1],*[rvec[t]]])
    Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
    e = Y[t][equity]
    b = Y[t][bond]

Y = tf.convert_to_tensor(Y)

#Economy
C = Y[...,cons]
E = Y[...,equity]
B = Y[...,bond]
P = Y[...,price]
Q = Y[...,ir]

#Forecast
Σf = tf.stack([tf.pad(tf.concat([E[...,:-1],B[...,:-1]],-1),tf.constant([[0,0],[0,1]]),constant_values=s) for s in np.unique(rvec)],0)
Yf = tf.stack([model(Σf[s],training=False) for s in range(S)],0)
Cf = Yf[...,cons]
Pf = Yf[...,price]

#Market clearing
E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))

#Budget constraints: c = ω + (p+δ)e^{-1} + b^{-1} - pe - qb
Elag = tf.concat([tf.constant([ebar],'float32'),E[:-1]],0)
Blag = tf.concat([tf.constant([bbar],'float32'),B[:-1]],0)
bc_1 = tf.math.abs(Ω[...,0] - tf.squeeze(P)*E[...,0] - tf.squeeze(Q)*B[...,0] - C[...,0])
bc_i   = tf.math.abs(Ω[...,1:-1] - P*E[...,1:] - Q*B[...,1:] + (P+Δ)*Elag[...,1:] + Blag[...,1:] - C[...,1:-1])
bc_L = tf.math.abs(Ω[...,-1] + tf.squeeze(P+Δ)*Elag[...,-1] + Blag[...,-1] -C[...,-1])
bc = bc_1 + tf.math.reduce_sum(bc_i,-1) + bc_L

#Loss = Divide Excess Returns
Eul = tf.math.reduce_sum(
    tf.math.abs(
        β/up(C)*
        (tf.tensordot(up(Cf)*(Pf+tf.expand_dims(δ,-1)),tf.convert_to_tensor(probs),axes=[[0],[0]])/P 
        - tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[0],[0]])/Q)
        )
        ,1)
Err = Eul + B_mc_sum + E_mc_sum + bc
Err_mean_train = tf.constant(tf.reduce_mean(Err[time]),shape=(burn,))
Err_Ergodic = tf.concat([Err_mean_train,Err[time]],0)

E_mc_sum
B_mc_sum
bc
Eul