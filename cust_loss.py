from packages import *
from params import *
from nn import *

#load in detSS allocs
import detSS
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

def euler_loss(y_true,y_pred):
    #ISSUE IS SIZE(T,) BECAUSE INPUT BATCH IS 32 OR WHATEVER TF 
    τ = tf.shape(y_pred).numpy()[0]
    #Today
    E = y_pred[:,equity]
    B = y_pred[:,bond]
    P = y_pred[:,price]
    Q = y_pred[:,ir]
    x0 = tf.reshape(tf.concat([tf.reshape(Ω[time][0,0],(1,)),Ω[time][0,1:] + (P[0] + Δ[time][0])*ebar + bbar],axis=0),(1,L))
    xrest = tf.concat([tf.reshape(Ω[time][1:,0],(τ-1,1)),Ω[time][1:,1:] + (P[1:] + Δ[time][1:])*E[0:-1] + B[0:-1]],axis=1)
    X = tf.concat([x0,xrest],axis=0)
    C = tf.concat([X[:,0:L-1]-P*E-Q*B,tf.reshape(X[:,-1],(τ,1))],axis=1)

    #Forecast
    Σf = tf.concat(
            [tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(τ,1))],axis=1) for s in range(S)]
            ,axis=0)
    yf = model(Σf,training=False)
    Yf = tf.stack([yf[s*τ:(s+1)*τ,:] for s in range(S)],axis=1)
    Ef = Yf[:,:,equity]
    Bf = Yf[:,:,bond]
    Pf = Yf[:,:,price]
    Qf = Yf[:,:,ir]
    xf0 = tf.reshape(tf.repeat([tf.reshape(ω[:,0],shape=(S,))],repeats=[τ],axis=0),(τ,S,1))
    xfrest = ω[:,1:] + (Pf + δ)*Ef + Bf
    Xf = tf.concat([xf0,xfrest],2)
    Cf = tf.concat([Xf[:,:,0:L-1]-Pf*Ef-Qf*Bf,tf.reshape(Xf[:,:,-1],(τ,S,1))],axis=2)
    
    Cpos = tf.greater(C,0)
    Cfpos = tf.math.reduce_all(tf.greater(Cf,0),axis=1)
    Cpen = tf.math.logical_and(Cpos,Cfpos)
    negC = tf.math.minimum(tf.reduce_min(Cf,1),C)
    #Cf = tf.math.maximum(Cf,ϵ)

    #Equity Expectation
    E_exp = tf.tensordot((Pf + δ)*up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #Bond Expectation
    B_exp = tf.tensordot(up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #"Euler" penalty for oldest
    Eul_old = tf.where(Cpen[:,-1],0.,10000000.-C[:,-1])

    #Market clearing
    E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=1))
    B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=1))

    #Loss = Euler for each agent
    # B_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*B_exp/Q)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*E_exp/P)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul_sum = tf.math.reduce_sum(E_eul,axis=1)
    # B_eul_sum = tf.math.reduce_sum(B_eul,axis=1)
    # return E_eul_sum + B_eul_sum + E_mc_sum + B_mc_sum + Eul_old

    #Loss = Divide Eulers
    Eul = tf.reduce_sum(tf.where(Cpen[:,:-1],tf.math.abs(P/Q - E_exp/B_exp),10000000.-negC[:,:-1]),axis=1)
    return Eul + B_mc_sum + E_mc_sum + Eul_old