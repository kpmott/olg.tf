from packages import *
from params import *
from nn import *

#load in detSS allocs
import detSS
ebar,bbar,pbar,qbar,xbar,cbar = detSS.detSS_allocs()

def rankSort(arr):
    sorted_list = sorted(arr)
    rank = 0
    sorted_rank_list = [1]
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            rank += 1
        sorted_rank_list.append(rank)
    
    rank_list = [] 
    # zip function returns iterator of tuple pairs of matching values in two inputss
    # dict function casts 1nd value in tuple as key and 2nd value in tuple as value
    item_rank_dict = dict(zip(sorted_list, sorted_rank_list))
    for item in arr:
        item_rank = item_rank_dict[item]
        rank_list.append(item_rank)
    return rank_list


def euler_loss(y_true,y_pred):
    #Today's state
    #state = rankSort(tf.squeeze(y_pred[2]).numpy())
    state = svec
    #Today's Economy
    ytoday = y_pred[0]
    E = ytoday[...,equity]
    B = ytoday[...,bond]
    P = ytoday[...,price]
    Q = ytoday[...,ir]
    
    #Resources
    Ω = tf.convert_to_tensor([ωvec[s] for s in state],'float32')
    Δ = tf.convert_to_tensor([δvec[s] for s in state],'float32')

    #Budget constraints -- find consumption 
    Elag = tf.concat([tf.expand_dims(tf.constant([ebar],'float32'),0),E[:,:-1]],-2)
    Blag = tf.concat([tf.expand_dims(tf.constant([bbar],'float32'),0),B[:,:-1]],-2)
    c1 = tf.expand_dims(Ω[...,0] - tf.squeeze(P)*E[...,0] - tf.squeeze(Q)*B[...,0],-1)
    ci = tf.expand_dims(Ω[...,1:-1],0) - P*E[:,:,1:] - Q*B[...,1:] + (P+tf.expand_dims(tf.expand_dims(Δ,0),-1))*Elag[...,1:] + Blag[...,1:]
    cL = tf.expand_dims(Ω[...,-1] + (tf.squeeze(P)+Δ)*Elag[...,-1] + Blag[...,-1],-1)
    Chat = tf.concat([c1,ci,cL],2)
    ϵc = 1e-9
    C = tf.maximum(ϵc,Chat)

    #Forecast
    Σf = tf.expand_dims(tf.convert_to_tensor([rvec],'float32'),-1)
    Yf = y_pred[1]
    Cf = Yf[...,consF]
    Pf = Yf[...,priceF]

    #Market clearing
    E_mc_sum = tf.math.abs(equitysupply - tf.math.reduce_sum(E,axis=-1))
    B_mc_sum = tf.math.abs(bondsupply - tf.math.reduce_sum(B,axis=-1))

    #Cons penalty
    conspen = tf.math.reduce_sum(1/ϵc*tf.math.abs(tf.minimum(0.,Chat)),-1)

    #Forecast penalties
    lkups = [[t,svec[t]] for t in range(T)]
    Cfhat = tf.expand_dims(tf.gather_nd(indices=lkups,params=tf.squeeze(Cf)),0)
    Cferr = tf.abs(C[:,1:,:-1] - Cfhat[:,:-1])
    Cferr = tf.pad(Cferr,[[0,0],[0,1],[0,0]])

    Pfhat = tf.expand_dims(tf.expand_dims(tf.gather_nd(indices=lkups,params=tf.squeeze(Pf)),0),-1)
    Pferr = tf.abs(P[:,1:,:] - Pfhat[:,:-1])
    Pferr = tf.pad(Pferr,[[0,0],[0,1],[0,0]])

    forecastErr = tf.reduce_sum(Cferr + Pferr, -1)


    #Eulers
    # Eul_eq = tf.math.abs(upinv_tf(β*tf.tensordot(up(Cf)*(Pf+tf.expand_dims(δ,-1)),tf.convert_to_tensor(probs),axes=[[0],[0]])/P)/C - 1. )
    # Eul_b  = tf.math.abs(upinv_tf(β*tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[0],[0]])/Q)/C - 1. )
    # Eul = tf.reduce_sum(Eul_eq+Eul_b,-1)

    #***************************************** THE DOT PRODUCTS AND ADDING Pf+δf
    #Excess Returns
    δf = tf.reshape(tf.constant(δ,'float32'),(1,1,S,1))
    Eul = tf.math.reduce_sum(tf.math.abs(
            β/up(C[...,:-1])*
            (tf.tensordot(up(Cf)*(Pf+δf),tf.convert_to_tensor(probs),axes=[[2],[0]])/P 
            - tf.tensordot(up(Cf),tf.convert_to_tensor(probs),axes=[[2],[0]])/Q)
            )
            ,-1)
    Err = Eul + B_mc_sum + E_mc_sum + conspen + forecastErr
    return Err

    #Remove noise from burn period
    #Err_mean_train = tf.constant(tf.reduce_mean(Err[time]),shape=(burn,))
    #Err_Ergodic = tf.concat([Err_mean_train,Err[time]],0)
    #return tf.math.sqrt(Err_Ergodic)
#do I need penalties for forecast, too? 