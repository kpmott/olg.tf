from packages import *
from params import *
#-----------------------------------------------------------------------------------------------------------------
def detSS_allocs():
    #det-SS calculations
    #compute lifetime consumption based on equity holdings e0 and prices p0
    def c_eq(e0,p0):
        #p0 ∈ ℜ^L:      prices from birth to death 
        #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        cons = np.zeros(L)
        cons[0] = ω_scalar[1]-p0[1]*e0[1]
        cons[-1] = ω_scalar[-1]+(p0[-1]+δ_scalar)*e0[-1]
        for i in range(1,L-1):
            cons[i] = ω_scalar[i]+(p0[i]+δ_scalar)*e0[i-1]-p0[i]*e0[i]
        
        return cons

    def x_eq(e0,p0):
        #p0 ∈ ℜ^L:      prices from birth to death 
        #e0 ∈ ℜ^{L-1}:  equity holdings from birth to (death-1)
        
        #vector of consumption in each period 
        x = np.zeros(L)
        x[0] = ω_scalar[1]
        x[-1] = ω_scalar[-1]+(p0[-1]+δ_scalar)*e0[-1]
        for i in range(1,L-1):
            x[i] = ω_scalar[i]+(p0[i]+δ_scalar)*e0[i-1]
        
        return x

    def ss_eq(x):
        #equity holdings for 1:(L-1)
        e = x[0:-1]
        #price
        p = x[-1]
        
        #consumption must be nonnegative
        cons = np.maximum(1e-12*np.ones(L),c_eq(e,p*np.ones(L)))

        #Euler equations
        ssVec = np.zeros(L)
        for i in range(0,L-1):
            ssVec[i] = p*up(cons[i]) - β*(p+δ_scalar)*up(cons[i+1])
        #market clearing
        ssVec[-1] = equitysupply - sum(e)

        return ssVec

    #Guess equity is hump-shaped
    eguess = norm.pdf(range(1,L),.8*wp,L/3)
    eguess = [x/sum(eguess) for x in eguess]
    pguess = .5

    #solve
    fsolve(ss_eq,[*eguess,*[pguess]],full_output=1)
    bar = fsolve(ss_eq,[*eguess,*[pguess]],full_output=0)
    ebar = bar[0:-1]
    bbar = ebar*0
    pbar = bar[-1]
    qbar = 1/β
    xbar = x_eq(ebar,pbar*np.ones(L))
    cbar = c_eq(ebar,pbar*np.ones(L))

    return ebar,bbar,pbar,qbar,xbar,cbar