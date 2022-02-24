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
    #PART ZERO: c = cbar/xbar * x
    for t in range(1):
        X = []
        C = []
        P = []
        E = []

        counter = 0
        p = pbar*1e10
        p_old = pbar
        
        for counter in range(500):
            x = Ωvec[t]+ [*[0],*ebar*(p_old + Δvec[t])]
            c = x*cbar/xbar
            p = sum(x-c)
            if np.abs(p-p_old) < 1e-12:
                e = (x[:-1]-c[:-1])/p
                #print(counter)
                break   
            else:
                counter += 1
                p_old = p                             
        
        X.append(x)
        C.append(c)
        P.append(p)
        E.append(e)

    for t in range(1,T):
        counter = 0
        p = pbar*1e10
        p_old = pbar
        
        for counter in range(500):
            x = Ωvec[t]+ [*[0],*E[t-1]*(p_old + Δvec[t])]
            c = x*cbar/xbar
            p = sum(x-c)
            if np.abs(p-p_old) < 1e-12:
                e = (x[:-1]-c[:-1])/p
                #print(counter)
                break   
            else:
                counter += 1
                p_old = p                             
        
        X.append(x)
        C.append(c)
        P.append(p)
        E.append(e)

    X = tf.convert_to_tensor(X,dtype='float32')
    C = tf.convert_to_tensor(C,dtype='float32')
    E = tf.convert_to_tensor(E,dtype='float32')
    P = tf.convert_to_tensor(P,dtype='float32')

    #input: e,b,w
    Σ = tf.concat([E[:-1,:-1],E[:-1,:-1]*0,tf.reshape(tf.convert_to_tensor(rvec[:-1],'float32'),(T-1,1))],1)
    
    #output: c,e,b,p,q
    y_train = tf.concat([C[1:],E[1:],E[1:],tf.reshape(P[1:],(T-1,1)),tf.constant(1/β,shape=(T-1,1))],1)
    
    #train: prefit to "ergodic" detSS whatever
    model.fit(Σ,y_train,batch_size=T-1,epochs=500,verbose=0,callbacks=[TqdmCallback()])

    #fill values 
    for t in range(1):
        #t=0
        Σ = []
        Y = []
        Σ.append([*ebar[0:-1],*bbar[0:-1],*[rvec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity]
        b = Y[t][bond]

    for t in tqdm(range(1,T)):
        Σ.append([*e[:-1],*b[:-1],*[rvec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        e = Y[t][equity]
        b = Y[t][bond]

    Σ = tf.convert_to_tensor(Σ,'float32')
    Y = tf.convert_to_tensor(Y,'float32')
    
    return Σ,Y