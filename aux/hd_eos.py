# Joe Nyhan, 30 June 2021# Joe Nyhan, 30 June 2021
# Equation of state for hydrodynamical simulation of neutron star.

from hd_params import *

# ========== NEWTON RHAPSON

# necessary functions for root finding
@njit
def f_rho(u, rho):
    Pi, Phi = u
    p = P(rho)
    return (Pi + Phi - 2 * rho) * (Pi+ Phi + 2 * p) - (Pi - Phi)**2

@njit
def df_drho(u, rho):
    Pi, Phi = u
    p = P(rho)
    return -2*(Pi + Phi + 2*p) + (Pi + Phi - 2*rho) * (2 * dP_drho(rho,p))


# root finding algorithm

@njit
def rootFinder(u, rho0):

    def tol(rho_new, rho_old):
        return np.abs(rho_new - rho_old)/(rho_new + rho_old)

    rho_old = rho0
    rho_new = 0

    for i in range(NR_MAX_ITERATIONS):
        rho_new = rho_old - f_rho(u, rho_old) / df_drho(u, rho_old)
        if rho_new < 0:
            rho_new = rho_old / 2
            # break
        elif tol(rho_new, rho_old) < NR_TOL:
            break
        else:
            rho_old = rho_new
    
    return rho_new


# ========== EOS implementations

if not eos_UR:
    # numerically solve for rho
    @njit
    def rho(u,rho0):
        return rootFinder(u,rho0)


if eos_UR:

    @njit
    def P(rho):
        return (Gamma - 1) * rho

    @njit
    def rho(u,rho0):
        Pi, Phi = u
        first_term = -(Pi+Phi)* (2-Gamma)/(4*(Gamma-1))
        second_term = 1/(Gamma-1)
        under_sqrt = (Phi+Phi)**2*((2-Gamma)/4)**2 + (Gamma-1)*Pi*Phi
        return first_term + second_term*np.sqrt(under_sqrt)


if eos_polytrope:

    @njit
    def P(rho):
        return K * rho**Gamma

    @njit
    def dP_drho(rho,p):
        return K * Gamma * rho**(Gamma-1)


if eos_SLy:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.005925
        a4  = 0.16326
        a5  = 6.48
        a6  = 11.4971
        a7  = 19.105
        a8  = 0.8938
        a9  = 6.54
        a10 = 11.4950
        a11 = -22.775
        a12 = 1.5707
        a13 = 4.3
        a14 = 14.08
        a15 = 27.80
        a16 = -1.653
        a17 = 1.50
        a18 = 14.67

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 + (a7+a8*xi)*f0_9_10 + (a11+a12*xi)*f0_13_14 + (a15+a16*xi)*f0_17_18

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.005925
        a4  = 0.16326
        a5  = 6.48
        a6  = 11.4971
        a7  = 19.105
        a8  = 0.8938
        a9  = 6.54
        a10 = 11.4950
        a11 = -22.775
        a12 = 1.5707
        a13 = 4.3
        a14 = 14.08
        a15 = 27.80
        a16 = -1.653
        a17 = 1.50
        a18 = 14.67

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6));
        d_f0_9_10  = + f0_9_10**2  * a9  * np.exp(a9 *(a10-xi));
        d_f0_13_14 = + f0_13_14**2 * a13 * np.exp(a13*(a14-xi));
        d_f0_17_18 = + f0_17_18**2 * a17 * np.exp(a17*(a18-xi));

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_10   + (a7+a8*xi)*d_f0_9_10 \
        + a12*f0_13_14 + (a11+a12*xi)*d_f0_13_14 \
        + a16*f0_17_18 + (a15+a16*xi)*d_f0_17_18

        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho

if eos_FPS:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.006004
        a4  = 0.16345
        a5  = 6.50
        a6  = 11.8440
        a7  = 17.24
        a8  = 1.065
        a9  = 6.54
        a10 = 11.8421
        a11 = -22.003
        a12 = 1.5552
        a13 = 9.3
        a14 = 14.19
        a15 = 23.73
        a16 = -1.508
        a17 = 1.79
        a18 = 15.13

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 + (a7+a8*xi)*f0_9_10 + (a11+a12*xi)*f0_13_14 + (a15+a16*xi)*f0_17_18

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.006004
        a4  = 0.16345
        a5  = 6.50
        a6  = 11.8440
        a7  = 17.24
        a8  = 1.065
        a9  = 6.54
        a10 = 11.8421
        a11 = -22.003
        a12 = 1.5552
        a13 = 9.3
        a14 = 14.19
        a15 = 23.73
        a16 = -1.508
        a17 = 1.79
        a18 = 15.13

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6));
        d_f0_9_10  = + f0_9_10**2  * a9  * np.exp(a9 *(a10-xi));
        d_f0_13_14 = + f0_13_14**2 * a13 * np.exp(a13*(a14-xi));
        d_f0_17_18 = + f0_17_18**2 * a17 * np.exp(a17*(a18-xi));

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_10   + (a7+a8*xi)*d_f0_9_10 \
        + a12*f0_13_14 + (a11+a12*xi)*d_f0_13_14 \
        + a16*f0_17_18 + (a15+a16*xi)*d_f0_17_18

        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho

if eos_BSk19:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 3.916
        a2  = 7.701
        a3  = 0.00858
        a4  = 0.22114
        a5  = 3.269
        a6  = 11.964
        a7  = 13.349
        a8  = 1.3683
        a9  = 3.254
        a10 = -12.953
        a11 = 0.9237
        a12 = 6.20
        a13 = 14.383
        a14 = 16.693
        a15 = -1.0514
        a16 = 2.486
        a17 = 15.362
        a18 = 0.085
        a19 = 6.23
        a20 = 11.68
        a21 = -0.029
        a22 = 20.1
        a23 = 14.19

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 \
            + (a7+a8*xi)*f0_9_6 + (a10+a11*xi)*f0_12_13 + (a14+a15*xi)*f0_16_17 \
            + (a18/(1 + a19*(xi-a20)**2)) + (a21/(1 + a22*(xi-a23)**2))

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 3.916
        a2  = 7.701
        a3  = 0.00858
        a4  = 0.22114
        a5  = 3.269
        a6  = 11.964
        a7  = 13.349
        a8  = 1.3683
        a9  = 3.254
        a10 = -12.953
        a11 = 0.9237
        a12 = 6.20
        a13 = 14.383
        a14 = 16.693
        a15 = -1.0514
        a16 = 2.486
        a17 = 15.362
        a18 = 0.085
        a19 = 6.23
        a20 = 11.68
        a21 = -0.029
        a22 = 20.1
        a23 = 14.19


        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6))
        d_f0_9_6   = + f0_9_6**2   * a9  * np.exp(a9 *(a6-xi))
        d_f0_12_13 = + f0_12_13**2 * a12 * np.exp(a12*(a13-xi))
        d_f0_16_17 = + f0_16_17**2 * a16 * np.exp(a16*(a17-xi))

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_6    + (a7+a8*xi)*d_f0_9_6 \
        + a11*f0_12_13 + (a10+a11*xi)*d_f0_12_13 \
        + a15*f0_16_17 + (a14+a15*xi)*d_f0_16_17 \
        + (-2) * a18 * a19 * (xi - a20) * (1 + a19*(xi - a20)**2)**(-2) \
        + (-2) * a21 * a22 * (xi - a23) * (1 + a22*(xi - a23)**2)**(-2)
        
        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho

if eos_BSk20:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 4.078
        a2  = 7.587
        a3  = 0.00839
        a4  = 0.21695
        a5  = 3.614
        a6  = 11.942
        a7  = 13.751
        a8  = 1.3373
        a9  = 3.606
        a10 = -22.996
        a11 = 1.6229
        a12 = 4.88
        a13 = 14.274
        a14 = 23.560
        a15 = -1.5564
        a16 = 2.095
        a17 = 15.294
        a18 = 0.084
        a19 = 6.36
        a20 = 11.67
        a21 = -0.042
        a22 = 14.8
        a23 = 14.18

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 \
            + (a7+a8*xi)*f0_9_6 + (a10+a11*xi)*f0_12_13 + (a14+a15*xi)*f0_16_17 \
            + (a18/(1 + a19*(xi-a20)**2)) + (a21/(1 + a22*(xi-a23)**2))

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 4.078
        a2  = 7.587
        a3  = 0.00839
        a4  = 0.21695
        a5  = 3.614
        a6  = 11.942
        a7  = 13.751
        a8  = 1.3373
        a9  = 3.606
        a10 = -22.996
        a11 = 1.6229
        a12 = 4.88
        a13 = 14.274
        a14 = 23.560
        a15 = -1.5564
        a16 = 2.095
        a17 = 15.294
        a18 = 0.084
        a19 = 6.36
        a20 = 11.67
        a21 = -0.042
        a22 = 14.8
        a23 = 14.18


        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6))
        d_f0_9_6   = + f0_9_6**2   * a9  * np.exp(a9 *(a6-xi))
        d_f0_12_13 = + f0_12_13**2 * a12 * np.exp(a12*(a13-xi))
        d_f0_16_17 = + f0_16_17**2 * a16 * np.exp(a16*(a17-xi))

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_6    + (a7+a8*xi)*d_f0_9_6 \
        + a11*f0_12_13 + (a10+a11*xi)*d_f0_12_13 \
        + a15*f0_16_17 + (a14+a15*xi)*d_f0_16_17 \
        + (-2) * a18 * a19 * (xi - a20) * (1 + a19*(xi - a20)**2)**(-2) \
        + (-2) * a21 * a22 * (xi - a23) * (1 + a22*(xi - a23)**2)**(-2)
        
        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho

if eos_BSk21:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 4.857
        a2  = 6.981
        a3  = 0.00706
        a4  = 0.19351
        a5  = 4.085
        a6  = 12.065
        a7  = 10.521
        a8  = 1.5905
        a9  = 4.104
        a10 = -28.726
        a11 = 2.0845
        a12 = 4.89
        a13 = 14.302
        a14 = 22.881
        a15 = -1.7690
        a16 = 0.989
        a17 = 15.313
        a18 = 0.091
        a19 = 4.68
        a20 = 11.65
        a21 = -0.086
        a22 = 10.0
        a23 = 14.15

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 \
            + (a7+a8*xi)*f0_9_6 + (a10+a11*xi)*f0_12_13 + (a14+a15*xi)*f0_16_17 \
            + (a18/(1 + a19*(xi-a20)**2)) + (a21/(1 + a22*(xi-a23)**2))

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 4.857
        a2  = 6.981
        a3  = 0.00706
        a4  = 0.19351
        a5  = 4.085
        a6  = 12.065
        a7  = 10.521
        a8  = 1.5905
        a9  = 4.104
        a10 = -28.726
        a11 = 2.0845
        a12 = 4.89
        a13 = 14.302
        a14 = 22.881
        a15 = -1.7690
        a16 = 0.989
        a17 = 15.313
        a18 = 0.091
        a19 = 4.68
        a20 = 11.65
        a21 = -0.086
        a22 = 10.0
        a23 = 14.15

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_6   = f0(a9 *(a6-xi))
        f0_12_13 = f0(a12*(a13-xi))
        f0_16_17 = f0(a16*(a17-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6))
        d_f0_9_6   = + f0_9_6**2   * a9  * np.exp(a9 *(a6-xi))
        d_f0_12_13 = + f0_12_13**2 * a12 * np.exp(a12*(a13-xi))
        d_f0_16_17 = + f0_16_17**2 * a16 * np.exp(a16*(a17-xi))

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_6    + (a7+a8*xi)*d_f0_9_6 \
        + a11*f0_12_13 + (a10+a11*xi)*d_f0_12_13 \
        + a15*f0_16_17 + (a14+a15*xi)*d_f0_16_17 \
        + (-2) * a18 * a19 * (xi - a20) * (1 + a19*(xi - a20)**2)**(-2) \
        + (-2) * a21 * a22 * (xi - a23) * (1 + a22*(xi - a23)**2)**(-2)
        
        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho