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
Σ, Y = nn_fit.fit_euler(num_epochs=500,num_iters=10,tb=True)

#Economy
