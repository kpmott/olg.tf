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

Σ,Y = fit_euler(250,True,1)