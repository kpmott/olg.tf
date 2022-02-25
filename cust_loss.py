from packages import *
from params import *
from nn import *

#load in detSS allocs
import detSS
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

def euler_loss(y_true,y_pred):
    #Today
    C = y_pred[...,cons]
    E = y_pred[...,equity]
    B = y_pred[...,bond]
    P = y_pred[...,price]
    Q = y_pred[...,ir]

    #Forecast
    Σf = tf.stack([tf.pad(tf.concat([E[...,:-1],B[...,:-1]],-1),tf.constant([[0,0],[0,1]]),constant_values=s) for s in np.unique(rvec)],0)
    Yf = tf.stack([model(Σf[s],training=False) for s in range(S)],0)
    Cf = Yf[...,cons]
    Pf = Yf[...,price]

    #Market clearing
    E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
    B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))

    #Budget constraints: c = ω + (p+δ)e^{-1} + b^{-1} - pe - qb
    Elag = tf.concat([tf.constant([ebar],'float32'),E[:-1]],0)
    Blag = tf.concat([tf.constant([bbar],'float32'),B[:-1]],0)
    bc_1 = tf.math.abs(Ω[...,0] - tf.squeeze(P)*E[...,0] - tf.squeeze(Q)*B[...,0] - C[...,0])
    bc_i   = tf.math.abs(Ω[...,1:-1] - P*E[...,1:] - Q*B[...,1:] + (P+Δ)*Elag[...,1:] + Blag[...,1:] - C[...,1:-1])
    bc_L = tf.math.abs(Ω[...,-1] + tf.squeeze(P+Δ)*Elag[...,-1] + Blag[...,-1] -C[...,-1])
    bc = bc_1 + tf.math.reduce_sum(bc_i,-1) + bc_L

    #Eulers
    # Eul_eq = tf.math.abs(upinv_tf(β*tf.tensordot(up(Cf)*(Pf+tf.expand_dims(δ,-1)),tf.convert_to_tensor(probs),axes=[[0],[0]])/P)/C - 1. )
    # Eul_b  = tf.math.abs(upinv_tf(β*tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[0],[0]])/Q)/C - 1. )
    # Eul = tf.reduce_sum(Eul_eq+Eul_b,-1)

    #Excess Returns
    Eul = tf.math.reduce_sum(tf.math.abs(
            β/up(C)*
            (tf.tensordot(up(Cf)*(Pf+tf.expand_dims(δ,-1)),tf.convert_to_tensor(probs),axes=[[0],[0]])/P 
            - tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[0],[0]])/Q)
            )
            ,1)
    Err = Eul + B_mc_sum + E_mc_sum + bc

    #Remove noise from burn period
    Err_mean_train = tf.constant(tf.reduce_mean(Err[time]),shape=(burn,))
    Err_Ergodic = tf.concat([Err_mean_train,Err[time]],0)
    return tf.math.sqrt(Err_Ergodic)
#do I need penalties for forecast, too? 