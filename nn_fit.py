#Load packages
from packages import *

#Load paramaters
from params import *

#Load neural network architecture
from nn import *

import detSS
#load in detSS allocs
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

#Custom loss function
import cust_loss

#Learning rate
opt = tf.keras.optimizers.Adam(learning_rate=10e-6)
#new model -- recompile with no loss
model.compile(loss=None, optimizer=opt)    

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

def fit_euler(num_epochs=500,batch=50,num_iters=10,tb=False,losshist=[]):
    skip = False
    for thyme in tqdm(range(num_iters)):
        #re-draw shocks
        svec, rvec, Ωvec, Δvec, Ω, Δ, ω, δ  = SHOCKS()

        #training data
        R =         tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(rvec,'float32'),-1),-1)
        timelist =  tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.array(range(T)),'float32'),-1),-1)
        Ω =         tf.expand_dims(Ω,-2)
        Σ = tf.Variable(tf.zeros((T,1,input)),dtype='float32')
        Y = tf.Variable(tf.zeros((T,1,output)),dtype='float32')
        for t in range(1):
            Σ[t].assign([[*ebar,*bbar,*[rvec[t]],*Ωvec[t],*[t]]])
            Y[t].assign(model(Σ[t],training=False))
            e = Y[t,:,equity]
            b = Y[t,:,bond]
        for t in range(1,T):
            Σ[t].assign(tf.concat([e,b,R[t],Ω[t],timelist[t]],1))
            Y[t].assign(model(Σ[t],training=False))
            e = Y[t,:,equity]
            b = Y[t,:,bond]
        Σ = tf.squeeze(tf.convert_to_tensor(Σ))
        Y = tf.squeeze(tf.convert_to_tensor(Y))
        Elag = tf.squeeze(tf.concat([[ebar],Y[:-1,equity]],0))
        Blag = tf.squeeze(tf.concat([[bbar],Y[:-1,bond]],0))

        #Recompile with loss (including lags)
        model.compile(loss=cust_loss.euler_loss(Elag=Elag,Blag=Blag,rvec=rvec,Ωvec=Ωvec,Ω=Ω,Δ=Δ,ω=ω,δ=δ), optimizer=opt)    
        
        #tensorboard callback? 
        model.fit(Σ,tf.zeros((T,output)),batch_size=batch,epochs=num_epochs,verbose=0,callbacks=[TqdmCallback(),history])
        
        losshist.append(history.losses[-1])
        n_MA = 20
        ma = [np.sum(losshist[np.maximum(i-n_MA+1,0):i+1])/np.minimum(i+1,n_MA) for i in range(np.shape(losshist)[0])]

        #losses = [item for sublist in losshist for item in sublist]
        plt.title('Loss Function');plt.plot(list(losshist),label='Loss Function');plt.plot(ma,'k--',label=str(n_MA)+'-Period Moving Average');plt.legend();plt.xlabel('Epoch');plt.ylabel('Loss Value');plt.yscale('log');plt.savefig('Losses.png');plt.clf()
        
        if thyme == num_iters - 1:
            E = Y[...,equity]
            B = Y[...,bond]
            P = Y[...,price]
            Q = Y[...,ir]
            times = tf.cast(Y[...,time],'int32')

            Elag = tf.pad(tf.concat([ebar*tf.ones((1,L-1)),E[:-1]],0),[[0,0],[1,0]])
            Blag = tf.pad(tf.concat([bbar*tf.ones((1,L-1)),B[:-1]],0),[[0,0],[1,0]])

            #Consumption
            E_ = Elag
            B_ = Blag
            Ωt = tf.squeeze(tf.gather(Ω,times))
            Δt = tf.expand_dims(tf.squeeze(tf.gather(Δ,times)),-1)

            Chat = Ωt + (P+Δt)*E_ + B_ - P*tf.pad(E,[[0,0],[0,1]]) - Q*tf.pad(B,[[0,0],[0,1]])
            ϵc = 1e-12
            C = tf.maximum(ϵc,Chat) 
            conspen = tf.reduce_sum(1/ϵc*tf.math.abs(tf.minimum(Chat,0)),-1)

            #Forecast
            SS = tf.convert_to_tensor([[*[rvec[s]],*Ωvec[s]] for s in range(S)],'float32')
            statecont = tf.transpose(tf.repeat(tf.expand_dims(SS,0),repeats=tf.shape(times)[0].numpy(),axis=0),[1,0,2])
            timesf = tf.repeat(tf.expand_dims(times,0),repeats=S,axis=0)
            statecont = tf.concat([statecont,tf.cast(timesf,'float32')],-1)

            assets = tf.repeat(tf.expand_dims(tf.concat([E,B],-1),0),repeats=S,axis=0)
            Σf = tf.concat([assets,statecont],-1)
            #Σf = tf.stack([tf.pad(tf.concat([E,B],-1),tf.constant([[0,0],[0,1]]),constant_values=s) for s in np.unique(rvec)],0)
            Yf = tf.stack([model(Σf[s],training=False) for s in range(S)],0)
            Yf = tf.transpose(Yf,[1,0,2])
            Pf = Yf[...,price]
            Qf = Yf[...,ir]
            Ef = Yf[...,equity]
            Bf = Yf[...,bond]
            Eflag = tf.repeat(tf.expand_dims(tf.pad(E,[[0,0],[1,0]]),-2),repeats=S,axis=-2)
            Bflag = tf.repeat(tf.expand_dims(tf.pad(B,[[0,0],[1,0]]),-2),repeats=S,axis=-2)
            Cfhat = ω + (Pf+δ)*Eflag + Bflag - Pf*tf.pad(Ef,[[0,0],[0,0],[0,1]]) - Qf*tf.pad(Bf,[[0,0],[0,0],[0,1]])
            Cf = tf.maximum(ϵc,Cfhat)

            #Market clearing
            E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
            B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))

            #Euler Losses
            Eul_Eq      = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])*(Pf + δ)   ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(P*up(C[...,:-1])) - 1.),-1)
            Eul_Bond    = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])            ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(Q*up(C[...,:-1])) - 1.),-1)

            #Lp norm
            p = 1.
            Err = (Eul_Eq**p + Eul_Bond**p + B_mc_sum**p + E_mc_sum**p + conspen**p)**(1/p)

            eqRet = ((P[1:] + Δ[1:])/P[:-1])**(L/60) - 1.
            bondRet = (1/Q[1:])**(L/60) - 1.

            exRet = eqRet - bondRet

            plt.plot(   exRet[-125:]);plt.xlabel('t');plt.ylabel('');plt.legend();plt.title('Excess Return');   plt.savefig('exret.png');plt.clf()
            plt.plot(       C[-125:]);plt.xlabel('t');plt.ylabel('');plt.legend();plt.title('Consumption');     plt.savefig('cons.png');plt.clf()
            plt.plot(       E[-125:]);plt.xlabel('t');plt.ylabel('');plt.legend();plt.title('Equity Holdings'); plt.savefig('e.png');plt.clf()
            plt.plot(       B[-125:]);plt.xlabel('t');plt.ylabel('');plt.legend();plt.title('Bond Holdings');   plt.savefig('b.png');plt.clf()
        
    return tf.convert_to_tensor(Σ,'float32'), tf.convert_to_tensor(Y,'float32'), losshist