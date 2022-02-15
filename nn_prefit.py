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
    Σ = tf.concat([E[:-1,:-1],E[:-1,:-1]*0,tf.reshape(tf.convert_to_tensor(svec[:-1],'float32'),(T-1,1))],1)
    y_train = tf.concat([C[1:,:-1],E[1:],tf.reshape(P[1:],(T-1,1)),tf.reshape(P[1:],(T-1,1))*0],1)
    model.fit(Σ,y_train,batch_size=T,epochs=500,verbose=0,callbacks=[TqdmCallback()])

    for t in range(1):
        #t=0
        Σ = []
        Y = []
        Σ.append([*ebar[0:-1],*bbar[0:-1],*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        c = Y[t][cons]
        e = Y[t][equity]
        p = Y[t][price]
        q = Y[t][ir]
        x = Ωvec[t]+ [*[0],*ebar*(p + Δvec[t])+bbar]
        b = (x[:-1] - c[:-1] - p*e)/q

    for t in tqdm(range(1,T)):
        e_old = e
        b_old = b
        Σ.append([*e_old[:-1],*b_old[:-1],*[svec[t]]])
        Y.append(model(tf.convert_to_tensor([Σ[t]]),training=False).numpy()[0])
        c = Y[t][cons]
        e = Y[t][equity]
        p = Y[t][price]
        q = Y[t][ir]
        x = Ωvec[t]+ [*[0],*e_old*(p + Δvec[t])+b_old]
        b = (x[:-1] - c[:-1] - p*e)/q

    Σ = tf.convert_to_tensor(Σ,'float32')
    Y = tf.convert_to_tensor(Y,'float32')
    
    return Σ,Y
