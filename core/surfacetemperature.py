""" SURFACE TEMPERTAURE MODULE
    This is part of the Model() class

    Contains functions to calculate the surface temperature """
# external libraries
import numpy as np
# local libraries
import params
import pdb


def surfacetemperature(self, ii):
    """ Compute surface temperature, options available:
        1) Tsurf from measured outgoing longwave radiation
        2) Tsurf from skin layer formulation
        3) Tsurf from subsurface model (as in Gruell & Konzelmann) NOT ADVISED """
    # INPUT

    # MAIN
    # 1) from measured outgoing longwave radiation
    if params.method_Tsurf == 1:
        surftemp_fromLWout(self, ii)
    # 2) skin layer formulation
    elif params.method_Tsurf == 2:
        surftemp_skin(self, ii)
    # 3) linear extrapolation of upper 2 subsurface layer
    elif params.method_Tsurf == 3:
        surftemp_model(self, ii)
    # print error
    else:
        print('ERROR in fucntion surfacetemperature(): method_Tsurf not valid')
        quit()

    # OUTPUT
    return


def surftemp_fromLWout(self, ii):
    """ Compute surface temperature from measured longwave outgoing radiation
        using Stefan-Boltzman """
    # INPUT
    LWout = self.surf.LWout[ii]  # measured outgoing longwave radiation
    eps = params.eps             # surface emissivity
    sigma = params.sigma         # Stefan-Boltzmann constant

    # MAIN
    # Stefan-Boltzman equation
    Tsurf = np.power(LWout / (eps * sigma), 1 / 4) - 273.15
    # Tsurf cannot exceed 0C
    if Tsurf > 0:
        Tsurf = 0.

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    return


def surftemp_skin(self, ii):
    """ Compute surface temperature using skin layer formulation and
        bisection method """
    # INPUT
    Taccur = 0.005  # Tskin accuracy
    Tinterv = 40.   # Tskin interval, for bisection method
    # subsurface current state
    dz = self.iisubs.dz.values  # current thickness
    temp = self.iisubs.temp.values  # current temperature

    # MAIN
    # Tsurf INITIALIZATION
    # for the first time step calculate first guess of Tsurf from
    # linear interpolation of upper 2 subusrface layers
    if ii == 0:
        Tgrad = (temp[1] - temp[0]) / ((dz[1]+dz[0])*0.5)
        Tsurf = temp[0] - Tgrad*dz[0]*0.5
        if Tsurf > 0:
            Tsurf = 0.
        self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    # for other time steps use Tsurf of previous time step
    else:
        Tsurf = self.surf.Tsurf[ii-1]
        self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf

    Tsurf1 = Tsurf - Tinterv
    Tsurf2 = Tsurf + Tinterv

    Tsurf = bisection(self, Tsurf1, Tsurf2, Taccur, ii)
    if Tsurf >= 0:
        Tsurf = 0

        # pdb.set_trace()
    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    return


def surftemp_model(self, ii):
    """ Compute surface temperature from the 2 upper layers of the subsurface
        this was the initial appraoch of Greuell and Konzelmann (1994)
        NB: NOT SURE IT'S A GOOD APPROACH, in this way Tsurf always lags
        one timestep behind. NOT RECCOMENDED!! """
    # INPUT
    # subsurface current state
    dz = self.iisubs.dz.values  # current thickness
    temp = self.iisubs.temp.values  # current temperature

    # MAIN
    Tgrad = (temp[1] - temp[0]) / ((dz[1] + dz[0]) * 0.5)
    Tsurf = temp[0] - Tgrad * dz[0] * 0.5
    if Tsurf > 0:
        Tsurf = 0.

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    return


def bisection(self, Tsurf1, Tsurf2, Taccur, ii):
    """ Bisection method to determin surface temperature
        that closes the energy balance """
    # INPUT
    nnmax = 40
    # MAIN

    fmid = skintemp_energybalance(self, Tsurf2, ii)
    ff = skintemp_energybalance(self, Tsurf1, ii)

    if ff*fmid >= 0:
        print('ERROR: Function bisection() root must be bracketed for bisection! day: %s'
              % self.surf.index[ii].strftime('%Y-%m-%d'))
        pdb.set_trace()
        quit()
    if ff < 0:
        rtb = Tsurf1
        dTsurf = Tsurf2 - Tsurf1
    else:
        rtb = Tsurf2
        dTsurf = Tsurf1 - Tsurf2

    for nn in range(0, nnmax):
        dTsurf = dTsurf * 0.5
        tmid = rtb + dTsurf
        fmid = skintemp_energybalance(self, tmid, ii)
        if fmid <= 0:
            rtb = tmid
        if (abs(dTsurf) < Taccur) or (fmid == 0.0):
            break  # exit loop

    if nn == nnmax:
        print('WARNING: Function bisection() maximum number of bisections!! day: %s'
               % self.surf.index[ii].strftime('%Y-%m-%d'))

    Tbisection = rtb

    # OUTPUT
    return Tbisection


def skintemp_energybalance(self, Tsurf, ii):
    """ Calculate the energy balance as a function of the
        given surface temperature """
    # INPUT
    # trick to invoke the already prescribed function changing only
    # temporary the stored surface temperature
    Tcurrent_holder = self.surf.Tsurf[ii]
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf

    # MAIN
    # Longwave radiation
    self.longwavenet(ii)
    # Turbulent fluxes
    self.turbulentfluxes(ii)
    # Rain energy
    self.rainenergy(ii)
    # Ground heat flux
    self.groundheatflux(ii)
    # Net radiation
    Qnet = self.surf.SWnet[ii] + self.surf.LWnet[ii]
    # Energy balance
    if params.method_radpen == 2:  # radiation penetration
        Q = Qnet + self.surf.QH[ii] + self.surf.QL[ii] + self.surf.QR[ii] + self.surf.QG[ii] - self.surf.SWpen[ii]
    else:
        Q = Qnet + self.surf.QH[ii] + self.surf.QL[ii] + self.surf.QR[ii] + self.surf.QG[ii]

    # OUTPUT
    # change current store surface temperature back
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tcurrent_holder
    return Q


def surftemp_skin_CARLEEN(self, ii):
    """ Compute surface temperature using skin layer formulation and
        bisection method """
    # INPUT
    Taccur = 0.005
    Tinterv = 40.
    # subsurface current state
    dz = self.iisubs.dz.values  # current thickness
    temp = self.iisubs.temp.values  # current temperature

    # MAIN
    # Tsurf INITIALIZATION
    # for the first time step calculate first guess of Tsurf from
    # linear interpolation of upper 2 subusrface layers
    if ii == 0:
        Tgrad = (temp[1] - temp[0]) / ((dz[1]+dz[0])*0.5)
        Tsurf = temp[0] - Tgrad*dz[0]*0.5
        if Tsurf > 0:
            Tsurf = 0.
        self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    # for other time steps use Tsurf of previous time step
    else:
        Tsurf = self.surf.Tsurf[ii-1]

    Tsurf1 = Tsurf - Tinterv
    Tsurf2 = Tsurf + Tinterv
    dTsurf = 2. * Taccur
    dTsurf1 = 2. * Taccur
    Tsurfold = Tsurf1

    Tskiniter = 0
    TskiniterMAX = 40

    while (dTsurf > Taccur) & (dTsurf1 > 0.):
        Tskiniter += 1

        Tsurfold1 = Tsurfold
        Tsurfold = Tsurf

        Tsurf = bisection(self, Tsurf1, Tsurf2, Taccur, ii)
        # sourceskin = skintemp_energybalance(self, Tsurf, ii)
        if Tsurf >= 0:
            Tsurf = 0
        # NOT SURE WHAT THIS SOURCESKIN IS NEEDED FOR IN DEBAM
        #     sourceskin = skintemp_energybalance(self, 0, ii)
        # if sourceskin < 0:
        #     sourceskin = 0

        dTsurf = abs(Tsurf - Tsurfold)
        dTsurf1 = abs(Tsurf - Tsurfold1)

        # NO SOLUTION FOUND
        if Tskiniter >= TskiniterMAX:
            print('WARNING: function surftemp_skin() maximum number of iterations reached!! day: %s'
                   % self.surf.index[ii].strftime('%Y-%m-%d'))
            Tsurf = 0.5 * (Tsurf + Tsurfold)
            dTsurf = 0.

    # NO SOLUTION FOUND
    if (dTsurf > Taccur) & (dTsurf1 == 0.) & (Tskiniter > TskiniterMAX):
        print('WARNING: function surftemp_skin() nosolution found!! day: %s'
               % self.surf.index[ii].strftime('%Y-%m-%d'))
        Tsurf = 0.5 * (Tsurf + Tsurfold)

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'Tsurf'] = Tsurf
    return
