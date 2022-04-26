from packages import *
from params import *
from nn import *

#load in detSS allocs
import detSS
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

# Y = tf.convert_to_tensor(Y)
# idxs = tf.range(T)
# ridxs = tf.random.shuffle(idxs)
# y_pred = tf.gather(Y,ridxs)[:32]

def euler_loss(Elag,Blag,rvec,Ωvec,Ω,Δ,ω,δ):
    def loss(y_true,y_pred):
        #Today
        E = y_pred[...,equity]
        B = y_pred[...,bond]
        P = y_pred[...,price]
        Q = y_pred[...,ir]
        times = tf.cast(y_pred[...,time],'int32')

        #Consumption
        E_ = tf.pad(tf.squeeze(tf.gather(Elag,times)),[[0,0],[1,0]])
        B_ = tf.pad(tf.squeeze(tf.gather(Blag,times)),[[0,0],[1,0]])
        Ωt = tf.squeeze(tf.gather(Ω,times))
        Δt = tf.expand_dims(tf.squeeze(tf.gather(Δ,times)),-1)

        Chat = Ωt + (P+Δt)*E_ + B_ - P*tf.pad(E,[[0,0],[0,1]]) - Q*tf.pad(B,[[0,0],[0,1]])
        ϵc = 1e-6
        C = tf.maximum(ϵc,Chat) 
        conspen = -tf.where(tf.less(tf.reduce_min(Chat,-1),0),1/ϵc*tf.reduce_min(Chat,-1),0)#tf.reduce_sum(1/ϵc*tf.math.abs(tf.minimum(Chat,0)),-1)

        #Forecast
        SS = tf.convert_to_tensor([[*[rvec[s]],*Ωvec[s]] for s in range(S)],'float32')
        statecont = tf.transpose(tf.repeat(tf.expand_dims(SS,0),repeats=tf.shape(times)[0],axis=0),[1,0,2])
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
        Bf = Yf[...,bond]*0
        Eflag = tf.repeat(tf.expand_dims(tf.pad(E,[[0,0],[1,0]]),-2),repeats=S,axis=-2)
        Bflag = tf.repeat(tf.expand_dims(tf.pad(B,[[0,0],[1,0]]),-2),repeats=S,axis=-2)
        Cfhat = ω + (Pf+δ)*Eflag + Bflag - Pf*tf.pad(Ef,[[0,0],[0,0],[0,1]]) - Qf*tf.pad(Bf,[[0,0],[0,0],[0,1]])
        Cf = tf.maximum(ϵc,Cfhat)

        #Market clearing
        E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
        B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))
        #B_mc_sum = (B_mc_sum + 1.)**3 - 1.

        #Euler Losses
        Eul_Eq      = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])*(Pf + δ)   ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(P*up(C[...,:-1])) - 1.),-1)
        Eul_Bond    = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])            ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(Q*up(C[...,:-1])) - 1.),-1)*0

        #upinv
        # Eul_Eq      = tf.math.reduce_sum(tf.math.abs(upinv_tf(tf.tensordot(up(Cf[...,1:])*(Pf + δ)   ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/P)/C[...,:-1] - 1.),-1)
        # Eul_Bond    = tf.math.reduce_sum(tf.math.abs(upinv_tf(tf.tensordot(up(Cf[...,1:])            ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/Q)/C[...,:-1] - 1.),-1)

        #Pricing errors
        # qerr = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])       ,tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(up(C[...,:-1])) - Q),-1)
        # perr = tf.math.reduce_sum(tf.math.abs(tf.tensordot(up(Cf[...,1:])*(Pf+δ),tf.convert_to_tensor(probs),axes=[[1],[0]])*β/(up(C[...,:-1])) - P),-1)

        #Forecast error
        # lkups = [[t,svec[t]] for t in range(T)]
        # Bfhat = tf.gather_nd(indices=lkups,params=Bf)
        # Efhat = tf.gather_nd(indices=lkups,params=Ef)
        # Pfhat = tf.gather_nd(indices=lkups,params=Pf)
        # Qfhat = tf.gather_nd(indices=lkups,params=Qf)
        # Eferr = tf.reduce_sum(tf.abs(E - Efhat),-1)
        # Bferr = tf.reduce_sum(tf.abs(B - Bfhat),-1)
        # Pferr = tf.reduce_sum(tf.abs(P - Pfhat),-1)
        # Qferr = tf.reduce_sum(tf.abs(Q - Qfhat),-1)
        # ferr = Bferr + Eferr + Pferr + Qferr

        #Lp norm
        p = 1.
        Err = (Eul_Eq**p + Eul_Bond**p + B_mc_sum**p + E_mc_sum**p + conspen**p)**(1/p)     # + ferr**p + qerr**p + perr**p)**(1/p)
        return tf.where(tf.math.less(tf.squeeze(times),burn),0.,Err)
        #return tf.math.log(tf.where(tf.math.less(tf.squeeze(times),burn),1.,Err))/tf.math.log(10.)
    return loss