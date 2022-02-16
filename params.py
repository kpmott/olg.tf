from packages import *

os.chdir("/home/kpmott/Git/tf.olg")

#Lifespan 
L = 3
wp = int(L*2/3)
rp = L - wp

#Time discount
β = 0.98**(60/L)

#Risk-aversion coeff
γ = 2.

#Stochastic elements
probs = [0.5, 0.5]
S = len(probs) 

#share of total resources
ωGuess = 7/8*norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45) \
       / sum(norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45))

#share of total resources: 1/16 to dividend; the rest to endowment income
ls = np.array([*[1/8], *ωGuess, *np.zeros(rp)])

#total resources
wbar = 1

#shock perturbation vector
ζtrue = 0.03
wvec = [wbar - ζtrue, wbar + ζtrue]            #total income in each state
δvec = np.multiply(ls[0],wvec)                    #dividend in each state
ωvec = [ls[1:]*w for w in wvec]

#plt.plot(ωvec[0])
#plt.plot(ωvec[1])
#plt.show()

#mean-center all shock-contingent values
δ_scalar = ls[0]
ω_scalar = ls[1:]

#net supply of assets: for later
equitysupply = 1
bondsupply = 0

#-----------------------------------------------------------------------------------------------------------------
#utility
def u(x):
    if γ == 1:
        return np.log(x)
    else:
        return (x**(1-γ)-1)/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
@tf.function
def upinv(x):
    a = 1e-16
    if x <= a:
        return -(x-a) + a**(-1/γ)
    else:
        return x**(-1/γ)

def upinv_tf(x):
    return x**(1/γ)
    #a = 1e-16
    #return tf.where(tf.less_equal(x,a),-(x-a)+a**(-1/γ), x**(1/γ))



#-----------------------------------------------------------------------------------------------------------------
#time and such for neurals 
T = 2000
burn = int(T/10)
train = T - burn
time = slice(burn,T,1)

shocks = range(S)
svec = np.random.choice(shocks,T,probs)
Ωvec = [ωvec[s] for s in svec]
Δvec = δvec[svec]
rvec = np.sum(Ωvec,1)+Δvec
Ω = tf.convert_to_tensor(Ωvec,dtype='float32')
Δ = tf.reshape(tf.convert_to_tensor(Δvec,dtype='float32'),(T,1))
ω = tf.convert_to_tensor(ωvec,dtype='float32')
δ = tf.reshape(tf.convert_to_tensor(δvec,dtype='float32'),(S,1))

#tolerance
ϵ = 1e-8
#-----------------------------------------------------------------------------------------------------------------
input = 2*(L-2)+1
output = 2*(L-1)+2

OUT =       slice(0     ,output,1)
equity =      slice(0     ,L-1   ,1)
bond =    slice(L-1   ,2*L-2 ,1)
price =     slice(2*L-2 ,2*L-1 ,1)
ir =        slice(2*L-1 ,2*L ,1)