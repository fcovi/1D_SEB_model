''' TURBULENT FLUXES MODULE
    This is part of the Model() class

    Contains functions to calculate the turbulent fluxes '''
# external libraries
import numpy as np
# local libraries
import params


def turbulentfluxes(self, ii):
    ''' Main turbulent fluxes function
        Calls all other functions '''
    # INPUT
    method_turbul = params.method_turbul

    # MAIN
    # 1) Ambach Method
    if method_turbul == 1:
        turbulentfluxes_Ambach(self, ii)

    # 2) Monin-Obukhov Theory
    if method_turbul == 2:
        turbulentfluxes_MoninObukhov(self, ii)

    # OUTPUT
    return


def turbulentfluxes_Ambach(self, ii):
    ''' Compute turbulent fluxes using Ambach method
        Assume neutral stability of atmosphere '''
    # INPUT
    P = self.P    # air pressure from AWS elevation
    P0 = params.P0  # standard atmospheric pressure
    K = params.K    # von Karman's constant
    z = params.z    # instruments height
    cp = params.cp  # specific heat of air
    Ls = params.Ls  # latent heat of sublimation
    Lv = params.Lv  # latent heat of evaporation
    dens0 = params.dens0         # density of air at standard atmospheric pressure
    Tair = self.surf.Tair[ii]    # air temperature
    Tsurf = self.surf.Tsurf[ii]  # surface temperature
    wind = self.surf.Wind[ii]    # wind velocity

    # MAIN
    # assume neutral condition of atmospheric stability
    psiM = 0
    psiH = 0
    psiE = 0
    # surface roughness lenght
    z0w, z0T, z0e = roughness_length(self, psiM, ii)

    # SENSIBLE heat flux
    dragcoeffH = (P/P0) * K**2 / ((np.log(z/z0w) - psiM) * (np.log(z/z0T) - psiH))
    QH = dragcoeffH * dens0 * cp * wind * (Tair-Tsurf)

    # LATENT heat flux
    # Vapor pressure of the air and of the surface
    ez = vaporpressure_air(self, ii)
    es = vaporpressure_surf(self, ii)
    # Latent heat of evaporation or sublimation, see Greuel and Konzelmann (1994)
    L = Ls  # sublimation
    if ((ez - es) > 0.) & (Tsurf == 0.):  # condensation, if surface is melting
        L = Lv
    dragcoeffL = (0.623/P0) * K**2 / ((np.log(z/z0w) - psiM) * (np.log(z/z0T) - psiE))
    QL = dragcoeffL * dens0 * L * wind * (ez-es)

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'QH'] = QH
    self.surf.loc[self.surf.index[ii], 'QL'] = QL
    return


def turbulentfluxes_MoninObukhov(self, ii):
    ''' Compute turbulent fluxes using Monin-Obukhov Theory
        Include an iterative procedure to determin stability functions
        First convergence of QH is computed and then QL is computed assuming psiE=psiH'''
    # INPUT
    P = self.P  # air pressure from AWS elevation
    P0 = params.P0  # standard atmospheric pressure
    K = params.K  # von Karman's constant
    z = params.z  # instruments height
    cp = params.cp  # specific heat of air
    Ls = params.Ls  # latent heat of sublimation
    Lv = params.Lv  # latent heat of evaporation
    dens0 = params.dens0  # density of air at standard atmospheric pressure
    Tair = self.surf.Tair[ii]  # air temperature
    Tsurf = self.surf.Tsurf[ii]  # surface temperature
    wind = self.surf.Wind[ii]  # wind velocity

    # MAIN
    # SENSIBLE HEAT FLUX (ITERATIVE PROCEDURE)
    # first assume neutral condition of atmospheric stability
    psiM = 0
    psiH = 0
    # surface roughness lenght
    z0w, z0T, z0e = roughness_length(self, psiM, ii)
    # SENSIBLE heat flux
    dragcoeffH = (P / P0) * K ** 2 / ((np.log(z / z0w) - psiM) * (np.log(z / z0T) - psiH))
    QH = dragcoeffH * dens0 * cp * wind * (Tair - Tsurf)

    # iteration to determine flux convergence
    iterstep = 0
    while True:
        # stability computation not needed if QH = 0
        # this also avoid division for 0 when calculating L
        if QH == 0:
            break

        # Monin-Obukhov length
        L = moninobukhov_length(self, QH, psiM, ii)
        # stability functions, depends on atmospheric stability and L only!
        if L > 0:  # stable stratification
            psiM, psiH = stabilityfunc_stable(L)
        else:  # unstable stratification
            psiM, psiH = stabilityfunc_unstable(L)
        # roughness length (Andreas uses new stability function psiM)
        z0w, z0T, z0e = roughness_length(self, psiM, ii)

        # calculate new SENSIBLE heat flux
        dragcoeffH = (P / P0) * K ** 2 / ((np.log(z / z0w) - psiM) * (np.log(z / z0T) - psiH))
        QHnew = dragcoeffH * dens0 * cp * wind * (Tair - Tsurf)

        # determine flux convergence condition
        QHdiff = abs(QHnew - QH)
        QHdiffMAX = abs(QH) / 100 * 1  # 1% of sensible heat flux
        QH = QHnew
        if QHdiff < QHdiffMAX:
            break

        # if fluxes do not converge
        iterstep += 1
        if iterstep > 100:  # only allow 100 iterations
            print('%s - Monin-Obulhov stability iteration do not converge, computing turbulent fluxes using '
                  'Ambach Method' % self.surf.index[ii].strftime('%Y-%m-%d %H'))
            # compute flux without stability
            psiM = 0
            psiH = 0
            z0w, z0T, z0e = roughness_length(self, psiM, ii)
            dragcoeffH = (P / P0) * K ** 2 / ((np.log(z / z0w) - psiM) * (np.log(z / z0T) - psiH))
            QH = dragcoeffH * dens0 * cp * wind * (Tair - Tsurf)
            break

    # LATENT HEAT FLUX
    psiE = psiH  # assume stability function is the same
    # Vapor pressure of the air and of the surface
    ez = vaporpressure_air(self, ii)
    es = vaporpressure_surf(self, ii)
    # Latent heat of evaporation or sublimation, see Greuel and Konzelmann (1994)
    L = Ls  # sublimation
    if ((ez - es) > 0.) & (Tsurf == 0.):  # condensation, if surface is melting
        L = Lv
    dragcoeffL = (0.623 / P0) * K ** 2 / ((np.log(z / z0w) - psiM) * (np.log(z / z0T) - psiE))
    QL = dragcoeffL * dens0 * L * wind * (ez - es)

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'QH'] = QH
    self.surf.loc[self.surf.index[ii], 'QL'] = QL
    return


def roughness_length(self, psiM, ii):
    ''' Compute surface roughness length. 2 methods available:
        1) fixed ratio: z0T = z0e = z0w/z0Te_div
        2) according to Andreas (1987), Boundary Layer Meteor. 38, 159-184 '''
    # INPUT
    method_z0Te = params.method_z0Te
    z0w = params.z0w_init  # always user prescribed, never change!
    z0T = None
    z0e = None

    # MAIN
    # 1) Fixed ratio method
    if method_z0Te == 1:
        z0Te_div = params.z0Te_div
        z0T = z0w/z0Te_div
        z0e = z0T

    # 2) Andreas (1987) surface renewal theory, see also Munro 1990
    elif method_z0Te == 2:
        viscos = 0.000015   # (m^2/s) kinematic viscosity of air for 0 degrees
        # viscos = 0.0000139  # (m^2/s) kinematic viscosity of air for 5 degrees

        ustar = friction_velocity(self, psiM, ii)  # friction velocity, needed for Reynolds number
        Re = ustar * z0w / viscos    # Reynolds number

        # Smooth regime
        if Re <= 0.135:
            z0T = z0w * np.exp(1.250)
            z0e = z0w * np.exp(1.610)
        # Transitional regime
        elif (Re > 0.135) & (Re < 2.5):
            z0T = z0w * np.exp(0.149-0.550 * np.log(Re))
            z0e = z0w * np.exp(0.351-0.628 * np.log(Re))
        # Rough regime
        elif Re >= 2.5:
            z0T = z0w * np.exp(0.317-0.565 * np.log(Re)-0.183 * (np.log(Re)**2))
            z0e = z0w * np.exp(0.396-0.512 * np.log(Re)-0.180 * (np.log(Re)**2))

    # ERROR otherwise
    else:
        print('ERROR in function roughness_length(): method_z0Te not valid')
        quit()

    # OUTPUT
    return z0w, z0T, z0e


def friction_velocity(self, psiM, ii):
    ''' Calculate friction velocity '''
    # INPUT
    K = params.K    # von Karman's constant
    z = params.z    # instruments height
    z0w = params.z0w_init      # always user prescribed, never change!
    wind = self.surf.Wind[ii]  # wind velocity

    # MAIN
    ustar = (K * wind) / (np.log(z / z0w) - psiM)

    # OUTPUT
    return ustar


def moninobukhov_length(self, QH, psiM, ii):
    ''' Compute Monin-Obukhov length
        NB: NOT TO BE used if QH = 0 '''
    # INPUT
    g = params.g    # acceleration of gravity
    cp = params.cp  # specific heat of air
    K = params.K    # von Karman's constant
    dens0 = params.dens0       # density of air at standard atmospheric pressure
    Tair = self.surf.Tair[ii]  # air temperature

    # MAIN
    # minimum length to avoid crashing in case of too high stability
    # Regine says this value comes from discussion with M. Rotach
    Lmin = 0.3

    # friction velocity, needed for Monin-Obukhov length
    ustar = friction_velocity(self, psiM, ii)
    # Monin-Obukhov length
    L = dens0 * cp * ustar**3 * (Tair + 273.15) / (K * g * QH)
    if (L < Lmin) & (L > 0):
        L = Lmin

    # OUTPUT
    return L


def stabilityfunc_stable(L):
    ''' Calculate stability functions for a stable atmospheric
        stratification according to Beljaar and Holtslag (1991) '''
    # INPUT
    z = params.z  # instruments height
    # constants from Beljaar and Holtslag (1991)
    a = 1
    b = 2/3
    c = 5
    d = 0.35

    # MAIN
    psiM = -(a*z/L + b*(z/L - c/d) * np.exp(-d*z/L) + b*c/d)
    psiH = -((1 + 2*a*z/(3*L))**1.5 + b*(z/L - c/d)*np.exp(-d*z/L) + b*c/d - 1)

    # OUTPUT
    return psiM, psiH


def stabilityfunc_unstable(L):
    ''' Calculate stability functions for an unstable atmospheric
        stratification according to Panofsky and Dutton (1984)
        CHECK PAPER TO BE SURE ABOUT THIS!!!! '''
    # INPUT
    z = params.z  # instruments height

    # MAIN
    xx = (1 - 16*z/L)**0.25
    psiM = np.log((1+xx**2)/2 * ((1+xx)/2)**2) - 2*np.arctan(xx) + np.pi/2
    psiH = 2 * np.log(0.5 * (1 + (1-16*z/L)**0.5))

    # OUTPUT
    return psiM, psiH


def vaporpressure_air(self, ii):
    ''' Compute vapor pressure of the air '''
    # INPUT
    Tair = self.surf.Tair[ii]  # air temperature
    RH = self.surf.RH[ii]      # relative humidity

    # MAIN
    # first calculate saturation vapour pressure of the air
    Ez = 610.78 * np.exp((17.08085 * Tair) / (234.15 + Tair))
    ez = RH * Ez / 100

    # OUTPUT
    return ez


def vaporpressure_surf(self, ii):
    ''' Compute vapor pressure of the air '''
    # INPUT
    Tsurf = self.surf.Tair[ii]  # surface temperature

    # MAIN
    # first calculate saturation vapour pressure of the surface
    Es = 610.78 * np.exp((17.08085 * Tsurf) / (234.15 + Tsurf))
    es = Es

    # OUTPUT
    return es
