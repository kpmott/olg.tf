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
Σ, Y = nn_fit.fit_euler(num_epochs=500,num_iters=25,tb=False)

#Economy
E = Y[:,equity]
B = Y[:,bond]
P = Y[:,price]
Q = Y[:,ir]
x0 = tf.reshape(tf.concat([tf.reshape(Ω[0,0],(1,)),Ω[0,1:] + (P[0] + Δ[0])*ebar + bbar],axis=0),(1,L))
xrest = tf.concat([tf.reshape(Ω[1:,0],(T-1,1)),Ω[1:,1:] + (P[1:] + Δ[1:])*E[0:-1] + B[0:-1]],axis=1)
X = tf.concat([x0,xrest],axis=0)
C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(T,1))],axis=1)

#Forecast
Σf = tf.concat(
        [tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,dtype='float32',shape=(T,1))],axis=1) for s in np.unique(rvec)]
        ,axis=0)
yf = model(Σf,training=False)
Yf = tf.stack([yf[s*T:(s+1)*T,:] for s in range(S)],axis=1)
Ef = Yf[:,:,equity]
Bf = Yf[:,:,bond]
Pf = Yf[:,:,price]
Qf = Yf[:,:,ir]
xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[T],axis=0),(T,S,1))
xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
Xf = tf.concat([xf0,xfrest],2)
Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(T,S,1))],axis=2)