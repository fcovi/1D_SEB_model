""" ENERGY BALANCE MODULE
    This is part of the Model() class

    Contains functions to calculate the surface energy balance
    NB: surface temperature functions are given in surfacetemperature.py
    NB: turbulent fluxes functions are given in turbulence.py """
# external libraries
import pandas as pd
import numpy as np
# local libraries
import params
from core.turbulence import turbulentfluxes
from core.subsurface import thermalconductivity
import pdb


def energybalance(self, ii):
    """ Compute energy balance as a residual of the other fluxes.
        Requires all fluxes to be computed before calling this"""
    # INPUT
    Tsurf = self.surf.Tsurf[ii]
    SWnet = self.surf.SWnet[ii]
    LWnet = self.surf.LWnet[ii]
    QH = self.surf.QH[ii]
    QL = self.surf.QL[ii]
    QR = self.surf.QR[ii]
    QG = self.surf.QG[ii]

    # MAIN
    # net radiation
    Qnet = SWnet + LWnet
    Qres = 0.
    # melt energy
    if params.method_radpen == 2:  # radiation penetration
        Q = Qnet + QH + QL + QR + QG - self.surf.SWpen[ii]
    else:
        Q = Qnet + QH + QL + QR + QG
    # only available if surface temperature < 0
    if Tsurf < 0:
        Q = 0.
        if params.method_radpen == 2:  # radiation penetration
            Qres = Qnet + QH + QL + QR + QG - self.surf.SWpen[ii]
        else:
            Qres = Qnet + QH + QL + QR + QG

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'Qnet'] = Qnet
    self.surf.loc[self.surf.index[ii], 'Qres'] = Qres
    self.surf.loc[self.surf.index[ii], 'Q'] = Q
    return


def shortwavenet(self, ii):
    """ Compute shortwave net radiation from measurements and shortwave
        radiation penetration. Depends only on timestep, for this reason
        it is called out of the main energy balance function to avoid the
        skin surface temperature formulation to do the SW penetration
        calculations for every bisection method iteration """
    # INPUT
    SWin = self.surf.SWin[ii]   # measured incoming shortwave radiation
    SWout = self.surf.SWout[ii]  # measured outgoing shortwave radiation

    # MAIN
    SWnet = SWin - SWout
    # radiation penetration
    if params.method_radpen == 2:  # radiation penetration
        swradpenetration(self, ii)
    # print error
    if params.method_radpen != 1:
        if params.method_radpen != 2:
            print('ERROR in function shortwavenet(): method_radpen not valid')
            quit()

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'SWnet'] = SWnet
    return


def longwavenet(self, ii):
    """ Compute longwave net radiation, 2 options
        1) LWout from measurements
        2) LWout from simulated skin surface temperature """
    # INPUT
    LWin = self.surf.LWin[ii]    # measured incoming longwave radiation

    # NET LONGWAVE RADIATION
    # 1) LWout from measurements
    if params.method_Tsurf == 1:
        LWout = self.surf.LWout[ii]  # measured outgoing longwave radiation
        LWnet = LWin - LWout
    # 2) LWout from Tsurf using Stefan-Boltzmann law
    elif params.method_Tsurf == 2 or params.method_Tsurf == 3:
        eps = params.eps
        sigma = params.sigma
        Tsurf = self.surf.Tsurf[ii]
        LWout = eps * sigma * np.power(Tsurf + 273.15, 4)
        LWnet = LWin - LWout
    # print error
    else:
        LWnet = None
        print('ERROR in function longwavenet(): method_Tsurf not valid')
        quit()

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'LWout'] = LWout
    self.surf.loc[self.surf.index[ii], 'LWnet'] = LWnet
    return


def groundheatflux(self, ii):
    """ Compute the ground heat flux
        1) QG as SEB closure when Tsurf < 0
        2) QG from subsurface model """
    # INPUT
    method_QG = params.method_QG
    Tsurf = self.surf.Tsurf[ii]

    # MAIN
    # 1) QG as SEB closure
    if method_QG == 1:
        # all other fluxes need to be calculated first
        Qnet = self.surf.Qnet[ii]
        QH = self.surf.QH[ii]
        QL = self.surf.QL[ii]
        QR = self.surf.QR[ii]
        # close SEB
        QG = -(Qnet + QH + QL + QR)
        # if Tsurf = 0 all energy goes into melt, assume QG = 0
        if Tsurf == 0:
            QG = 0.

    # 2) QG from subsurface model
    elif method_QG == 2:
        # INPUT
        dens = self.iisubs['dens']
        temp = self.iisubs['temp']
        dz = self.iisubs['dz']
        conduct = thermalconductivity(self, dens)  # calculate thermal conductivity (subsurface call is later)
        QG_fromsubs_opt = 1  # use the 1 or 2 top most layers to calculate QG
        # MAIN
        if QG_fromsubs_opt == 1:
            QG = -(conduct[0] * (Tsurf-temp[0])) / (dz[0]*0.5)
        else:
            Ga = (conduct[0] * (Tsurf-temp[0])) / (dz[0]*0.5)
            conduct_mean = (conduct[0]*dz[0] + conduct[1]*dz[1]) / (dz[0]+dz[1])  # conduct between layer 1 and 2
            dTdz = (temp[0] - temp[1]) / (0.5 * (dz[0] + dz[1]))
            Gb = conduct_mean * dTdz
            QG = -(dz[0] * (2.*Ga - Gb) + dz[1]*Ga) / (dz[0] + dz[1])

    # print error
    else:
        QG = None
        print('ERROR in function groundheatflux(): method_QG not valid')
        quit()

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'QG'] = QG
    return


def rainenergy(self, ii):
    """ Compute energy flux from rain
        -) rain has to be mm/timestep
        -) rain temp assumed to be air temp """
    # COMMENT IN DEBAM: this should be inside iteration loop for
    # surface temperature, neglected because error is considered small (CHECK???)
    # INPUT
    cw = params.cw  # specific heat of water
    dt = self.dt    # (s) timestep length
    Tair = self.surf.Tair[ii]
    Tsurf = 0  # Surface temperature forced to 0 in DEBAM (WHY?)

    # MAIN
    # Determine rain/snow
    snowrain_ratio(self, ii)
    rain = self.surf.rain[ii]
    # Calculate QR (W/m2)
    QR = cw*rain/dt * (Tair-Tsurf)
    # NB: here there would be density of water but rain is in mm so the 1000 cancels out

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'QR'] = QR
    return


def snowrain_ratio(self, ii):
    """ Detarmination of rain or snow """
    ''' -) ALL snow if Tair < Train_treshold - Train_halfrange
        -) ALL rain if Tair > Train_treshold + Train_halfrange
        -) linear mixture in between
        Train_treshold and Train_halfrange defined in params_SiteJ2017_percUNI5m.py
        Typical values in DEBAM:
        -) Train_treshold = 1.0 C
        -) Train_halfrange = 1.0 C (hardcoded in DEBAM) '''
    # INPUT
    Train_treshold = params.Train_treshold
    Train_halfrange = params.Train_halfrange
    prec = self.surf.Prec[ii]
    Tair = self.surf.Tair[ii]

    # MAIN
    prec = prec * params.prec_correction  # apply precipitation correction
    rain = 0.
    snow = 0.

    # only snow
    if Tair <= (Train_treshold-Train_halfrange):
        snow = prec
        rain = 0.
    # only rain
    if Tair >= (Train_treshold+Train_halfrange):
        snow = 0.
        rain = prec
    # snow/rain mixture
    if (Tair > (Train_treshold-Train_halfrange)) & (Tair < (Train_treshold+Train_halfrange)):
        snow = (Train_treshold + Train_halfrange - Tair) / (2 * Train_halfrange) * prec
        rain = prec - snow

    # make sure rain and snow are not negative
    if rain < 0:
        rain = 0.
    if snow < 0:
        snow = 0.
    # make sure rain and snow are 0 if precipitation is 0
    if prec == 0:
        snow = 0.
        rain = 0.

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'snow'] = snow
    self.surf.loc[self.surf.index[ii], 'rain'] = rain
    return


def swradpenetration(self, ii):
    """ Compute shortwave radiation penetration into the subsurface.
        Method described by Brandt and Warren (1993), applied by Van den Broeke et al. (2008),
        uses two stream approach from Schlatter (1972).
        Files needed are initialized by swrad_penetration_initialization() """
    # INPUT
    densice = params.densice     # density of ice
    SWin = self.surf.SWin[ii]    # measured incoming shortwave radiation
    SWout = self.surf.SWout[ii]  # measured outgoing shortwave radiation
    # # MIE SCATTERING coefficients
    # dlambda = self.MieScatt.dlambda.values            # delta wavelength
    # asym = self.MieScatt.asym.values                  # asymetric coefficient
    # qext = self.MieScatt.Qext.values                  # extinction efficiency
    # cosinglescat = self.MieScatt.cosinglescat.values  # cosinglescat
    # subsurface current state
    z = self.iisubs.z.values        # layer depth
    dz = self.iisubs.dz.values      # layer thickness
    dens = self.iisubs.dens.values  # layer density

    # initialize to 0 some variables
    SWnet = SWin - SWout  # net radiation
    SWz = np.zeros_like(z)    # SW absorbed at depth z
    dSWdz = np.zeros_like(z)  # SW absorbed by each subsurface layer (=SWz*dz)
    SWpen = 0.                # sum of all energy penetrated to deeper layers

    # MAIN
    # SPECTRAL INCOMING RADIATION
    SWin_spectrum = solarspectrum(self, ii)

    # FIRST LAYER (zz = 0)
    # MIE SCATTERING coefficients
    # constant snow grain size
    if params.method_grainsize == 1:
        radius = params.grain_size  # (m) snow grain size radius in mm in params_SiteJ2017_percUNI5m.py
        dlambda, asym, qext, cosinglescat = miescattering(self, radius)
    # snow grain size parameterized
    elif params.method_grainsize == 2:
        radius = self.iisubs.re.values[0]   # (m) snow grain size radius in mm in params_SiteJ2017_percUNI5m.py
        dlambda, asym, qext, cosinglescat = miescattering(self, radius)
    else:
        print('ERROR in function swradpenetration(): method_grainsize not valid')
        quit()

    # compute multiscattering albedo and extinction coeff for every wavelenght
    sigext = (qext * 3. * dens[0]) / (4. * densice * radius)
    aext = sigext * cosinglescat
    rcoef = 0.5 * sigext * (1.0 - asym) * (1.0 - cosinglescat)
    klam = np.sqrt(aext * aext + 2.0 * aext * rcoef)  # depends on dens, radius, and wavelenght
    mlam = (aext + rcoef - klam) / rcoef  # doens't depend on dens (the same for every layer)
    # SURFACE RADIATION (this stay in the energy balance equation)
    SW0 = np.sum(dlambda * SWin_spectrum * (1.0 - mlam)) / 1000  # basically SWnet using the multispectral albedo
    if SW0 == 0:
        SWsfc = 0
        scaling = 0
    else:
        SWsfc = SW0 - np.sum(dlambda * SWin_spectrum * (1.0 - mlam) * np.exp(-klam * params.radsfc_dz)) / 1000
        SWsfc = SWnet / SW0 * SWsfc      # scale to incoming SWnet
        scaling = (SWnet - SWsfc) / SW0  # scaling for subs layers

    # LAYERS LOOP
    for zz in range(0, self.nz):

        # MIE SCATTERING coefficients if grain size is parameterized
        if params.method_grainsize == 2:
            radius = self.iisubs.re.values[zz]  # (m) snow grain size radius in mm in params_SiteJ2017_percUNI5m.py
            dlambda, asym, qext, cosinglescat = miescattering(self, radius)

        # compute multiscattering albedo and extinction coeff for every wavelenght
        sigext = (qext * 3. * dens[zz]) / (4. * densice * radius)
        aext = sigext * cosinglescat
        rcoef = 0.5 * sigext * (1.0 - asym) * (1.0 - cosinglescat)
        klam = np.sqrt(aext * aext + 2.0 * aext * rcoef)  # depends on dens, radius, and wavelenght
        mlam = (aext + rcoef - klam) / rcoef              # doens't depend on dens (the same for every layer)

        sumUP = np.sum(dlambda * SWin_spectrum * (1.0 - mlam) * np.exp(-klam * (z[zz] - 0.5 * dz[zz]))) / 1000
        sumDOWN = np.sum(dlambda * SWin_spectrum * (1.0 - mlam) * np.exp(-klam * (z[zz] + 0.5 * dz[zz]))) / 1000
        SWz[zz] = -abs(scaling * (sumUP - sumDOWN))
        dSWdz[zz] = SWz[zz] / dz[zz]
        SWpen = SWpen - SWz[zz]

        # loop EXIT condition (optimize perfomance)
        if sumDOWN * scaling < 0.0001:
            self.radpen_nz = zz  # number of layers affected by radpen (used in subs temp)
            break

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'SWsfc'] = SWsfc  # store SW that stays at the surface
    self.surf.loc[self.surf.index[ii], 'SWpen'] = SWpen  # store total SW penetrated
    self.iisubs['dSWdz'] = dSWdz  # store SW absorbed by each subsurface layer
    return


def solarspectrum(self, ii):
    """ Compute spectral incoming shortwave radiation using
        input reference files see Brand and Warren J. Glac. 1993.
        Requires cloudiness [0, 1], which is calculated from LWin and Tair
        following Munneke (2011) approach, requires polynomial fitting of
        AWS data to be done prior to model run """
    # INPUT
    SWin = self.surf.SWin[ii]
    Tair = self.surf.Tair[ii]
    LWin = self.surf.LWin[ii]
    sigma = params.sigma
    # REFERENCE SPECTRA
    SolarPlateauClear = self.SolarSpectrum_file.PlateauClear
    SolarPlateauCloudy = self.SolarSpectrum_file.PlateauCloud
    SolarSeaClear = self.SolarSpectrum_file.SeaClear
    dlambda = self.MieScatt100mu.dlambda.values  # delta wavelength (SAME for SolarSPectrum and MieScatt files)
    pres = self.P        # pressure at the AWS from elevation (IT WOULD BE BETTER IF PRES is MEASURED!)
    presplat = 60900.0   # pressure plateau for standard solar spectrum
    pressea = 98000.0    # presure sea level for standard solar spectrum

    # MAIN
    # CLOUD FACTOR
    # first determine cloud cover from air temperature and longwave radiation
    # POLYNOMIAL coefficient (from AWS data fit, give in params_SiteJ2017_percUNI5m.py)
    pfit = params.LWin_min_fit
    TairK = Tair + 273.15  # depends on how fit is done (K or C)
    LWin_max = sigma * (TairK ** 4)
    LWin_min = pfit[2] + pfit[1] * TairK + pfit[0] * TairK ** 2
    cloud = (LWin - LWin_min) / (LWin_max - LWin_min)
    if cloud > 1:
        cloud = 1
    if cloud < 0:
        cloud = 0

    # calculate cloud factor for every wavelenght
    cloudf = SolarPlateauClear.copy()
    cloudf.loc[SolarPlateauClear == 0] = 0.
    cloudf.loc[SolarPlateauClear != 0] = 1. - cloud * (1. - SolarPlateauCloudy/SolarPlateauClear)
    # determine solar spectrum
    dsdp = (SolarSeaClear - SolarPlateauClear) / (pressea - presplat)
    SolarAWSClear = SolarPlateauClear + dsdp * (pres - presplat)
    spectrum = SolarAWSClear * cloudf
    # scale it with SWin
    spectrum_sum = np.sum(dlambda * spectrum) / 1000
    SWin_spectrum = (SWin / spectrum_sum) * spectrum

    # OUTPUT
    return SWin_spectrum


def miescattering(self, radius):
    """ Select Mie scattering coefficient given snow grain size """
    # INPUT
    # radius = grain size (m)

    # MAIN
    if radius < 150/10**6:
        dlambda = self.MieScatt100mu.dlambda.values
        asym = self.MieScatt100mu.asym.values
        qext = self.MieScatt100mu.Qext.values
        cosinglescat = self.MieScatt100mu.cosinglescat.values
    elif (radius >= 150/10**6) & (radius < 275/10**6):
        dlambda = self.MieScatt200mu.dlambda.values
        asym = self.MieScatt200mu.asym.values
        qext = self.MieScatt200mu.Qext.values
        cosinglescat = self.MieScatt200mu.cosinglescat.values
    elif (radius >= 275/10**6) & (radius < 425/10**6):
        dlambda = self.MieScatt350mu.dlambda.values
        asym = self.MieScatt350mu.asym.values
        qext = self.MieScatt350mu.Qext.values
        cosinglescat = self.MieScatt350mu.cosinglescat.values
    elif (radius >= 425/10**6) & (radius < 750/10**6):
        dlambda = self.MieScatt500mu.dlambda.values
        asym = self.MieScatt500mu.asym.values
        qext = self.MieScatt500mu.Qext.values
        cosinglescat = self.MieScatt500mu.cosinglescat.values
    elif (radius >= 750/10**6) & (radius < 1750/10**6):
        dlambda = self.MieScatt1000mu.dlambda.values
        asym = self.MieScatt1000mu.asym.values
        qext = self.MieScatt1000mu.Qext.values
        cosinglescat = self.MieScatt1000mu.cosinglescat.values
    # elif radius >= 1750/10**6:
    #     dlambda = self.MieScatt2500mu.dlambda.values
    #     asym = self.MieScatt2500mu.asym.values
    #     qext = self.MieScatt2500mu.Qext.values
    #     cosinglescat = self.MieScatt2500mu.cosinglescat.values
    else:
        dlambda = self.MieScatt2500mu.dlambda.values
        asym = self.MieScatt2500mu.asym.values
        qext = self.MieScatt2500mu.Qext.values
        cosinglescat = self.MieScatt2500mu.cosinglescat.values

    # OUTPUT
    return dlambda, asym, qext, cosinglescat


def swradpenetration_initialization(self):
    """ Read and initialize files needed for shortwave radiation penetrataion.
        Called in model class __init__ """
    # INPUT
    path = params.radpen_folder     # folder path to radiation files

    # MAIN
    # Read input files into class attributes
    # SOLAR SPECTRUM
    self.SolarSpectrum_file = pd.read_csv(path+'/IN_SolarSpectrum.txt', sep='\t')
    # MIE SCATTERING: depends on grain size, from Brands and Warren
    # asym = asymmetric factor g, Qext = extinction efficiency, cosinglescat = (1 - omega)
    names = ['lambda', 'dlambda', 'asym', 'Qext', 'cosinglescat']
    self.MieScatt100mu = pd.read_csv(path + '/IN_Mie(r=0.1mm).txt', sep='\t', names=names)
    self.MieScatt200mu = pd.read_csv(path + '/IN_Mie(r=0.2mm).txt', sep='\t', names=names)
    self.MieScatt350mu = pd.read_csv(path + '/IN_Mie(r=0.35mm).txt', sep='\t', names=names)
    self.MieScatt500mu = pd.read_csv(path + '/IN_Mie(r=0.5mm).txt', sep='\t', names=names)
    self.MieScatt1000mu = pd.read_csv(path + '/IN_Mie(r=1.0mm).txt', sep='\t', names=names)
    self.MieScatt2500mu = pd.read_csv(path + '/IN_Mie(r=2.5mm).txt', sep='\t', names=names)

    # OUTPUT
    return
