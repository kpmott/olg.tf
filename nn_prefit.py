from packages import *
from params import *
import detSS
from nn import *

#compile with MSE
model.compile(loss='mean_squared_error', optimizer='adam')

#load in detSS allocs
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#run pre-fit to detSS values 
def prefit():
    svec, rvec, Ωvec, Δvec, Ω, Δ, ω, δ = SHOCKS()
    # skE = tf.random.uniform((T,L-1))*ζtrue - ζtrue/2
    # skP = tf.random.uniform((T,1))*ζtrue - ζtrue/2
    
    #inputs: e,b,w,ω,t
    E = tf.ones((T,L-1))*ebar #+ skE
    P = tf.ones((T,1))*pbar #+ skP
    Σ = tf.concat([E,E*0,tf.reshape(tf.convert_to_tensor(rvec,'float32'),(T,1)),Ω],1)
    Σ = tf.concat([Σ,tf.expand_dims(tf.convert_to_tensor(range(T),'float32'),-1)],-1)
    #Σ = tf.random.normal((T,input),dtype='float32')
    
    #output: e,b,p,q,t
    y_train = tf.concat([E,E*0,P,qbar*tf.ones((T,1))],1)
    y_train = tf.concat([y_train,tf.expand_dims(tf.convert_to_tensor(range(T),'float32'),-1)],-1)
    
    #train: prefit to "ergodic" detSS whatever
    model.fit(Σ,y_train,batch_size=T,epochs=1500,verbose=0,callbacks=[TqdmCallback()]) #int(T/4)

# def prefit():
#     def bias(x):
#         diff = tf.math.abs(activation_final(tf.convert_to_tensor(x,dtype='float32')) - tf.cast(tf.concat([ebar,ebar*0,[pbar],[qbar]],0),'float32'))
#         return diff.numpy()
#     model.layers[-3].set_weights([model.layers[-3].get_weights()[0],fsolve(bias,[*ebar,*ebar*0,*[pbar],*[qbar]],full_output=True)[0]])    