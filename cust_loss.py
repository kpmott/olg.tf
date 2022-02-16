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
    B = y_pred[:,bond]
    P = y_pred[:,price]
    Q = y_pred[:,ir]
    Elag = tf.concat([[[*[0],*ebar]],tf.concat([tf.zeros((τ-1,1)),E[1:]],1)],0)
    Blag = tf.concat([[[*[0],*bbar]],tf.concat([tf.zeros((τ-1,1)),B[1:]],1)],0)

    #Forecast
    Σf = tf.concat([tf.concat([E[:,0:-1],B[:,0:-1],tf.constant(s*1.,dtype='float32',shape=(τ,1))],axis=1) for s in np.unique(rvec)],axis=0)
    yf = model(Σf,training=False)
    Yf = tf.stack([yf[s*τ:(s+1)*τ,:] for s in range(S)],axis=1)
    Cf = Yf[:,:,cons]
    #Ef = Yf[:,:,equity]
    #Bf = Yf[:,:,bond]
    Pf = Yf[:,:,price]
    #Qf = Yf[:,:,ir]

    #Market clearing
    E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=1))
    B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=1))

    #Budget constraints: c = ω + (p+δ)e^{-1} + b^{-1} - pe - qb
    E0 = tf.concat([E,tf.zeros((τ,1))],axis=1)
    B0 = tf.concat([B,tf.zeros((τ,1))],axis=1)
    bc = tf.reduce_sum(tf.math.abs(Ω + (P+Δ)*Elag + Blag - P*E0 - Q*B0 - C),1)

    #Loss = Divide Excess Returns
    Eul = tf.math.reduce_sum(tf.math.abs(β/up(C)*(tf.tensordot(up(Cf)*(Pf+δ),tf.convert_to_tensor(probs),axes=[[1],[0]])/P - tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[1],[0]])/Q)),1)
    Err = Eul + B_mc_sum + E_mc_sum + bc

    #Remove noise from burn period
    Err_mean_train = tf.constant(tf.reduce_mean(Err[time]),shape=(burn,))
    Err_Ergodic = tf.concat([Err_mean_train,Err[time]],0)
    return Err_Ergodic
#do I need penalties for forecast, too? 