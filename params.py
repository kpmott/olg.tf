from packages import *
os.chdir("/home/kpmott/Git/tf.olg")

#Lifespan 
L = 3

#working periods and retirement periods 
wp = int(L*2/3)
rp = L - wp

#Time discount rate
β = 1.#0.95**(60/L)

#Risk-aversion coeff
γ = 2.

#Stochastic elements
probs = [0.5, 0.5]
S = len(probs) 

divshare = .1

#total resources
wbar = 1

#share of total resources
#ωGuess = norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.25)
#ωGuess = (1-divshare)*ωGuess/np.sum(ωGuess)

#share of total resources: 1/8 to dividend; the rest to endowment income
#ls = wbar*np.array([*[divshare], *ωGuess, *np.zeros(rp)])
ls = np.array([1/3, 1/4, 5/12, 0])


#shock perturbation vector
ζtrue = 0.05
wvec = [wbar*(1 - ζtrue), wbar*(1+ + ζtrue)]     #total resources in each state
δvec = np.multiply(ls[0],wvec)/wbar          #dividend in each state
ωvec = [ls[1:]*w/wbar for w in wvec]         #endowment process

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
        return (x**(1-γ))/(1-γ)

#utility derivative
def up(x):
    return x**-γ

#inverse of utility derivative
def upinv_tf(x):
    return x**(1/γ)

#-----------------------------------------------------------------------------------------------------------------
#time and such for neurals 
T = 7500
burn = int(T/100)            #burn period: this is garbage
train = T - burn            #how many periods are "counting?"
time = slice(burn,T,1)      #period in which we care

def SHOCKS():
    #shocks
    shocks = range(S)
    svec = np.random.choice(shocks,T,probs)

    #endowments and dividends and total resources
    Ωvec = [ωvec[s] for s in svec]
    Δvec = δvec[svec]
    rvec = np.sum(Ωvec,1)+Δvec

    #convert to tensors now for easier operations later
    Ω = tf.convert_to_tensor(Ωvec,dtype='float32')
    Δ = tf.reshape(tf.convert_to_tensor(Δvec,dtype='float32'),(T,1))
    ω = tf.convert_to_tensor(ωvec,dtype='float32')
    δ = tf.reshape(tf.convert_to_tensor(δvec,dtype='float32'),(S,1))

    return svec, rvec, Ωvec, Δvec, Ω, Δ, ω, δ 

#machine tolerance
ϵ = 1e-8
#-----------------------------------------------------------------------------------------------------------------
"""
input   = [(e_i^{t-1})_i,(b_i^{t-1})_i,w^t,(ω_i^t)_i,t]
output  = [((e_i^{t})_{i=1}^{L-1},(b_i^{t})_{i=1}^{L-1},p^t,q^t]   ∈ ℜ^{2L}
"""
#input/output dims
#        assets     + endowments    + time
input = 2*(L-1)+1   + L             + 1

#        assets + prices    + time
output = 2*(L-1) + 2        + 1

#slices to grab output 
equity =    slice(0     ,L-1    ,1)
bond =      slice(L-1   ,2*L-2  ,1)
price =     slice(2*L-2 ,2*L-1  ,1)
ir =        slice(2*L-1 ,2*L    ,1)
time =      slice(2*L   ,2*L+1  ,1)