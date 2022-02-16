from packages import *
os.chdir("/home/kpmott/Git/tf.olg")

#Lifespan 
L = 3

#working periods and retirement periods 
wp = int(L*2/3)
rp = L - wp

#Time discount rate
β = 0.98**(60/L)

#Risk-aversion coeff
γ = 2.

#Stochastic elements
probs = [0.5, 0.5]
S = len(probs) 

#share of total resources
ωGuess = 7/8*norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45) \
       / sum(norm.pdf(np.linspace(1,wp,wp),0.8*wp,L*0.45))

#share of total resources: 1/8 to dividend; the rest to endowment income
ls = np.array([*[1/8], *ωGuess, *np.zeros(rp)])

#total resources
wbar = 1

#shock perturbation vector
ζtrue = 0.03
wvec = [wbar - ζtrue, wbar + ζtrue]     #total resources in each state
δvec = np.multiply(ls[0],wvec)          #dividend in each state
ωvec = [ls[1:]*w for w in wvec]         #endowment process

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
def upinv_tf(x):
    return x**(1/γ)

#-----------------------------------------------------------------------------------------------------------------
#time and such for neurals 
T = 2000
burn = int(T/10)            #burn period: this is garbage
train = T - burn            #how many periods are "counting?"
time = slice(burn,T,1)      #period in which we care

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

#machine tolerance
ϵ = 1e-8
#-----------------------------------------------------------------------------------------------------------------
"""
input   = [(e_i^{t-1})_{i=1}^{L-2},(b_i^{t-1})_{i=1}^{L-2},w^t]                         ∈ ℜ^{2L-3}
output  = [(c_i^{t})_{i=1}^{L},(e_i^{t})_{i=1}^{L-1},(b_i^{t})_{i=1}^{L-1},p^t,q^t]   ∈ ℜ^{3L-1}
"""
#input/output dims
input = 2*(L-2)+1
output = L+2*(L-1)+2

#slices to grab output 
cons =      slice(0     ,L)
equity =    slice(L   ,2*L-1   ,1)
bond =      slice(2*L-1 ,3*L-2 ,1)
price =     slice(3*L-2 ,3*L-1 ,1)
ir =        slice(3*L-1 ,3*L ,1)