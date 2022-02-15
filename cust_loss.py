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
    C = y_pred[:,cons]
    E = y_pred[:,equity]
    P = y_pred[:,price]
    Q = y_pred[:,ir]

    #compute B and X
    zeroE = tf.concat([tf.constant(0.,shape=(T,1)),E],1)
    B = (Ω[:,:-1] + tf.concat([tf.reshape([*[0],*ebar[:-1]],(1,L-1)),zeroE[:-1,:-1]],0) - P*E - C)/Q
    zeroB = tf.concat([tf.constant(0.,shape=(T,1)),B],1)
    X = Ω + tf.concat([tf.reshape([*[0],*ebar],(1,L)),zeroE[:-1]],0) + tf.concat([tf.reshape([*[0],*bbar],(1,L)),zeroB[:-1]],0)
    
    #Forecast
    Σf = tf.concat(
            [tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,shape=(τ,1))],axis=1) for s in range(S)]
            ,axis=0)
    yf = model(Σf,training=False)
    Yf = tf.stack([yf[s*τ:(s+1)*τ,:] for s in range(S)],axis=1)
    Cf = Yf[:,:,cons]
    Ef = Yf[:,:,equity]
    Pf = Yf[:,:,price]
    Qf = Yf[:,:,ir]
    
    #compute Bf and Xf
    zeroEf = tf.concat([tf.constant(0.,shape=(T,S,1)),Ef],2)
    Bf = (ω[:,:-1] + (Pf*δ)*tf.concat([tf.reshape(tf.convert_to_tensor([[*[*[0],*ebar[:-1]]],[*[*[0],*ebar[:-1]]]]),(1,S,L-1)),zeroEf[:-1,:,:-1]],0) - Cf)/Qf
    zeroBf = tf.concat([tf.constant(0.,shape=(T,S,1)),Bf],2)
    Xf = ω + tf.concat([tf.reshape(tf.convert_to_tensor([[*[*[0],*ebar]],[*[*[0],*ebar]]]),(1,S,L)),zeroEf[:-1,:]],0) + \
        tf.concat([tf.reshape(tf.convert_to_tensor([[*[*[0],*bbar]],[*[*[0],*bbar]]]),(1,S,L)),zeroBf[:-1,:]],0)

    #Equity Expectation
    E_exp = tf.tensordot((Pf + δ)*up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #Bond Expectation
    B_exp = tf.tensordot(up(Cf[:,:,1:L]),tf.convert_to_tensor(probs),axes=[[1],[0]])
    
    #"Euler" penalty for oldest
    Xpen = tf.math.minimum(tf.math.reduce_min(Xf,1)[:,-1],X[:,-1],1)
    Eul_old = tf.where(tf.math.logical_and(tf.reduce_all(Xf[:,:,-1]>0,1),X[:,-1]>0),0.,10000000.-Xpen)

    #Market clearing
    MCpen = 10.
    E_mc_sum = MCpen*tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=1))
    B_mc_sum = MCpen*tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=1))

    #Loss = Euler for each agent
    # B_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*B_exp/Q)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul = tf.where(Cpen[:,:-1],tf.math.abs(upinv_tf(β*E_exp/P)/C[:,:-1] -1),10000000.-negC[:,:-1])
    # E_eul_sum = tf.math.reduce_sum(E_eul,axis=1)
    # B_eul_sum = tf.math.reduce_sum(B_eul,axis=1)
    # return E_eul_sum + B_eul_sum + E_mc_sum + B_mc_sum + Eul_old

    #Loss = Divide Eulers
    Eul = tf.reduce_sum(tf.math.abs(P/Q - E_exp/B_exp),axis=1)
    Err = Eul + B_mc_sum + E_mc_sum + Eul_old
    Err_mean_train = tf.constant(tf.reduce_mean(Err[time]),shape=(burn,))
    Err_Ergodic = tf.concat([Err_mean_train,Err[time]],0)
    return Err_Ergodic