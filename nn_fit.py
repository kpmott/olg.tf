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
model.compile(loss=cust_loss.euler_loss, optimizer='adam')

def fit_euler(num_epochs=500,num_iters=10,tb=False):
    skip = False
    for thyme in tqdm(range(num_iters)):
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

        if tb:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tbc = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            model.fit(tf.convert_to_tensor(Σ),tf.zeros((T,output)),batch_size=T,epochs=num_epochs,verbose=0,callbacks=[TqdmCallback(),tbc])
        else:
            model.fit(tf.convert_to_tensor(Σ),tf.zeros((T,output)),batch_size=T,epochs=num_epochs,verbose=0,callbacks=[TqdmCallback()])#,tbc])

        skip = tf.math.reduce_mean(cust_loss.euler_loss(tf.zeros((T,output)),tf.convert_to_tensor(Y,dtype='float32'))) <= ϵ
        
        if skip.numpy():
            break
        
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
    
    return tf.convert_to_tensor(Σ,'float32'), tf.convert_to_tensor(Y,'float32')