#!/home/kpmott/Git/tf.olg/bin/python3

#Load packages
from packages import *

#Load paramaters
from params import *

#Load neural network architecture
from nn import *

import detSS
#load in detSS allocs
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#Pre-fit neural network to get good starting point
import nn_prefit
nn_prefit.prefit()

#Import fit function
import nn_fit
nn_fit.opt.learning_rate.assign(1e-5)
Σ, Y, losses = nn_fit.fit_euler(num_epochs=1,batch=32,num_iters=1000,tb=False,losshist=[]) 

nn_fit.opt.learning_rate.assign(1e-6)
Σ, Y, losses = nn_fit.fit_euler(num_epochs=1,batch=64,num_iters=300,tb=False,losshist=losses) 

# nn_fit.opt.learning_rate.assign(1e-6)
# Σ, Y, losses = nn_fit.fit_euler(num_epochs=1,batch=64,num_iters=150,tb=False,losshist=losses) 

# nn_fit.opt.learning_rate.assign(1e-7)
# Σ, Y, losses = nn_fit.fit_euler(num_epochs=1,batch=64,num_iters=150,tb=False,losshist=losses) 

#SAVE MODEL
np.savetxt('losses.csv', losses, delimiter=',')
model.save("saved_model/my_model")
model.save_weights("saved_model/my_model")