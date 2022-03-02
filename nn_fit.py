#Load packages
from packages import *

#Load paramaters
from params import *

#Load neural network architecture
from nn import *

#Pre-fit neural network to get good starting point
#import nn_prefit
#nn_prefit.prefit()

import detSS
#load in detSS allocs
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#Custom loss function
import cust_loss #; cust_loss.euler_loss(Y,Y)

#new model -- recompile with custom loss
model.compile(loss=None, optimizer='adam')

def fit_euler(num_epochs=5,tb=False,batchsize=1):
    skip = False
    Σ = tf.expand_dims(tf.convert_to_tensor([rvec],'float32'),-1)
    model.add_loss(lambda: cust_loss.euler_loss(model(Σ,training=False),model(Σ,training=False)))
    if tb:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        model.fit(Σ,[tf.zeros((batchsize,T,output)),tf.zeros((batchsize,T,S,outputF)),tf.zeros((batchsize,T,1))],batch_size=batchsize,epochs=num_epochs,verbose=0,callbacks=[TqdmCallback(),tbc])
    else:
        model.fit(Σ,[tf.zeros((batchsize,T,output)),tf.zeros((batchsize,T,S,outputF)),tf.zeros((batchsize,T,1))],batch_size=batchsize,epochs=num_epochs,verbose=0,callbacks=[TqdmCallback()])#,tbc])

    Y = model(Σ, training = False)
    
    return Σ,Y

Σ,Y = fit_euler(10000,False,1)

y = Y[0]
E = y[...,equity]
B = y[...,bond]
P = y[...,price]
Q = y[...,ir]
state = svec
#Resources
Ω = tf.convert_to_tensor([ωvec[s] for s in state],'float32')
Δ = tf.convert_to_tensor([δvec[s] for s in state],'float32')

#Budget constraints -- find consumption 
Elag = tf.concat([tf.expand_dims(tf.constant([ebar],'float32'),0),E[:,:-1]],-2)
Blag = tf.concat([tf.expand_dims(tf.constant([bbar],'float32'),0),B[:,:-1]],-2)
c1 = tf.expand_dims(Ω[...,0] - tf.squeeze(P)*E[...,0] - tf.squeeze(Q)*B[...,0],-1)
ci = tf.expand_dims(Ω[...,1:-1],0) - P*E[:,:,1:] - Q*B[...,1:] + (P+tf.expand_dims(tf.expand_dims(Δ,0),-1))*Elag[...,1:] + Blag[...,1:]
cL = tf.expand_dims(Ω[...,-1] + (tf.squeeze(P)+Δ)*Elag[...,-1] + Blag[...,-1],-1)
Chat = tf.concat([c1,ci,cL],2)
ϵc = 1e-9
C = tf.maximum(ϵc,Chat)


Yf = Y[1]
Cf = Yf[...,consF]
Pf = Yf[...,priceF]

#Market clearing
E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))

#Excess Returns
δf = tf.reshape(tf.constant(δ,'float32'),(1,1,S,1))
Eul = tf.math.reduce_sum(tf.math.abs(
        β/up(C[...,:-1])*
        (tf.tensordot(up(Cf)*(Pf+δf),tf.convert_to_tensor(probs),axes=[[2],[0]])/P 
        - tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[2],[0]])/Q)
        )
        ,-1)
Err = Eul + B_mc_sum + E_mc_sum
Err