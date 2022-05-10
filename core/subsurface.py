""" SUBSURFACE MODULE
    This is part of the Model() class

    Contains functions to simulate the subsurface processes """
# external libraries
import pandas as pd
import xarray as xr
import numpy as np
import pdb
# internal libraries
import params


def subsurface(self, ii):
    """ Main subsurface function, calls all other functions.
        Simulatining subsurface processes rquires both loops
        over timesteps and subsurface layers

        NB: the 1D heat equation solver ofter requires a timestep
        smaller than 1 hour to be stable """
    # INPUT
    self.surf.loc[self.surf.index[ii], 'surfmelt'] = 0.
    self.surf.loc[self.surf.index[ii], 'subsmelt'] = 0.
    # grain size fractions (initialize to 0)
    if params.method_radpen == 2 & params.method_grainsize == 2:
        self.iisubs.fnew = np.zeros_like(self.iisubs.fnew)  # new snow fraction
        self.iisubs.frfz = np.zeros_like(self.iisubs.frfz)  # refreezing fraction

    # MAIN
    # TEMPERATURE
    temperature(self, ii)
    # SURFACE ACCUMULATION (snowfall + rain) (only if there is any input precipitation)
    if self.surf.Prec[ii] > 0.:
        accumulation(self, ii)

    # LAYERS LOOP: MELTING + REFREEZING + PERCOLATION + DENSIFICATION
    for zz in range(0, self.nz):
        # MELTING (only if there is energy available for melt)
        if self.surf.Q[ii] > 0:
            melting_surface(self, ii, zz)
        if self.iisubs['temp'][zz] > 0:
            melting_subsurface(self, zz)
        # DEEP WATER PERCOLATION
        if params.method_perc == 2:
            deep_percolation(self, zz)
        # REFREEZING (only if there is water and layer temp < 0)
        if (self.iisubs['water'][zz] > 0) & (self.iisubs['temp'][zz] < 0):
            refreezing(self, zz)
        # PERCOLATION (only if there is water in the layer)
        if self.iisubs['water'][zz] > 0:
            percolation(self, zz)
        # DENSIFICATION
        if params.method_densif != 0:  # only if densification is allowed
            densification(self, ii, zz)

    # GRAIN SIZE
    if params.method_radpen == 2 & params.method_grainsize == 2:
        grainsize(self, ii)

    surfaceheight(self, ii)  # surface height change
    # UPDATE LAYERS (uses only current state of subsurface)
    update_layers(self)

    # UPDATE CONTAINERS
    self.surf.loc[self.surf.index[ii], 'nsfclayersmelted'] = self.layersmelted  # keep an eye to melted sfc layers
    self.iisubs['drfzdt_c'] += self.iisubs['drfzdt']  # update cumulative container
    self.iisubs['drfzdt'] = np.zeros(len(self.iisubs['drfzdt']))  # reset timestep container
    if params.method_radpen == 2:
        self.iisubs['dzdt_c'] += self.iisubs['dzdt']  # update cumulative container
        self.iisubs['dzdt'] = np.zeros(len(self.iisubs['dzdt']))  # reset timestep container
        self.iisubs['dmeltdt_c'] += self.iisubs['dmeltdt']  # update cumulative container
        self.iisubs['dmeltdt'] = np.zeros(len(self.iisubs['dmeltdt']))  # reset timestep container
    if params.method_perc == 2:
        self.surfacemelt_deepperc = 0  # reset surface melt counter for deep percolation
    # OUTPUT: write output only every hour
    if self.surf.index[ii].minute == 0 and self.surf.index[ii].second == 0:
        self.subs['z'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['z']
        self.subs['dz'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dz']
        self.subs['temp'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['temp']
        self.subs['dens'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dens']
        self.subs['mass'][:, int(ii / params.subs_timestep)] = self.iisubs['mass']
        self.subs['water'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['water']
        self.subs['sh'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.surfaceheight
        self.subs['k'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['k']
        self.subs['z_rel'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['z'] - self.surfaceheight
        self.subs['drfzdt'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['drfzdt_c']
        self.iisubs['drfzdt_c'] = np.zeros(len(self.iisubs['drfzdt_c']))  # reset cumulative container
        if params.method_radpen == 2:
            self.subs['dzdt'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dzdt_c']
            self.iisubs['dzdt_c'] = np.zeros(len(self.iisubs['dzdt_c']))  # reset cumulative container
            self.subs['dmeltdt'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dmeltdt_c']
            self.iisubs['dmeltdt_c'] = np.zeros(len(self.iisubs['dmeltdt_c']))  # reset cumulative container
            self.subs['dSWdz'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dSWdz']
            if params.method_extraoutput == 2:
                self.subs['dT_radpen'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dT_radpen']
                self.subs['dT_conduct'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['dT_conduct']
            if params.method_grainsize == 2:
                self.subs['re'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['re']
                self.subs['re_dry'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['re_dry']
                self.subs['re_wet'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['re_wet']
                self.subs['re_new'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['re_new']
                self.subs['re_rfz'].loc[dict(TIMESTAMP=self.surf.index[ii])] = self.iisubs['re_rfz']

    return


def temperature(self, ii):
    """ Compute subsurface temperature using 1D heat equation.
        A Forward in Time Central in Space (FTCS) finite difference method
        is implemented here. """
    # INPUT
    cpice = params.cpice  # specific heat capacity of ice
    dt = self.dt  # (s) timestep length
    Tsurf = self.surf.Tsurf[ii]  # surface temperature current timestep
    # subsurface current state
    dz = self.iisubs.dz.values  # current layers thickness
    dens = self.iisubs.dens.values  # current density
    mass = self.iisubs.mass.values  # current mass (here don't need to do the nan trick?)
    Told = self.iisubs.temp.values  # current temperature
    # exclude nan values from temperature computation
    dz = dz[~np.isnan(dz)]
    dens = dens[~np.isnan(dens)]
    Told = Told[~np.isnan(Told)]
    Tnew = np.zeros_like(Told)  # temperature to be calculated

    # MAIN
    # calculate thermal conductivity
    conduct = thermalconductivity(self, dens)
    conduct = conduct[~np.isnan(dz)]  # got error operands could not be broadcast together
    # radaition penetration
    if params.method_radpen == 2:
        dSWdz = self.iisubs.dSWdz.values
    else:
        dSWdz = None

    # 1D HEAT EQUATION
    # weighted average of layer variables (there is a double division by 2 which disappear (half layer considered)
    # layer thickness (distance from centers of adiajents layers)
    dz_up = (dz[:-2] + dz[1:-1]) / 2  # upper layer
    dz_dn = (dz[1:-1] + dz[2:]) / 2  # lower layer
    # conductivity
    conduct_up = (conduct[:-2] * dz[:-2] + conduct[1:-1] * dz[1:-1]) / (2 * dz_up)  # upper layer
    conduct_dn = (conduct[1:-1] * dz[1:-1] + conduct[2:] * dz[2:]) / (2 * dz_dn)  # lower layer
    # density
    dens_up = (dens[:-2] * dz[:-2] + dens[1:-1] * dz[1:-1]) / (2 * dz_up)  # upper layer
    dens_dn = (dens[1:-1] * dz[1:-1] + dens[2:] * dz[2:]) / (2 * dz_dn)  # lower layer

    # boundary condition bottom: zero flux
    Tnew[-1] = Told[-1]
    # boundary condition surface: Tsurf used to calculate first level temperature
    Tnew[0] = Told[0] + dt / (cpice * dz[0]) * (conduct_up[0] / (dens_up[0] * dz[0] / 2) * (Tsurf - Told[0]) -
                                                conduct_up[0] / (dens_up[0] * dz_up[0]) * (Told[0] - Told[1]))
    # heat equation
    Tnew[1:-1] = Told[1:-1] + dt / (cpice * dz[1:-1]) * (conduct_up / (dens_up * dz_up) * (Told[:-2] - Told[1:-1]) -
                                                         conduct_dn / (dens_dn * dz_dn) * (Told[1:-1] - Told[2:]))

    # effect of radiation penetration (add penetrated energy)
    if params.method_radpen == 2:
        SWnz = self.radpen_nz  # layers affected by SW penetration
        Tnew[0:SWnz] = Tnew[0:SWnz] - dSWdz[0:SWnz] * dz[0:SWnz] * dt / (mass[0:SWnz] * cpice)
        # Tnew[0:SWnz] = Tnew[0:SWnz] - dSWdz[0:SWnz] * dz[0:SWnz] * dt / (dens[0:SWnz] * cpice)
        # OUTPUT (save extra output)
        if params.method_extraoutput == 2:
            self.iisubs['dT_radpen'][0:SWnz] = - dSWdz[0:SWnz] * dz[0:SWnz] * dt / (mass[0:SWnz] * cpice)
            # self.iisubs['dT_radpen'][0:SWnz] = - dSWdz[0:SWnz] * dz[0:SWnz] * dt / (dens[0:SWnz] * cpice)
            dT_conduct = np.zeros_like(Tnew)
            dT_conduct[0] = dt / (cpice * dz[0]) * (conduct_up[0] / (dens_up[0] * dz[0] / 2) *
                                                    (Tsurf - Told[0]) - conduct_up[0] / (dens_up[0] * dz_up[0]) * (
                                                                Told[0] - Told[1]))
            dT_conduct[1:-1] = dt / (cpice * dz[1:-1]) * (conduct_up / (dens_up * dz_up) * (Told[:-2] - Told[1:-1]) -
                                                          conduct_dn / (dens_dn * dz_dn) * (Told[1:-1] - Told[2:]))
            self.iisubs['dT_conduct'][~np.isnan(self.iisubs['temp'])] = dT_conduct

    # OUTPUT
    self.iisubs['temp'][~np.isnan(self.iisubs['temp'])] = Tnew  # update current temperature
    return


def accumulation(self, ii):
    """ Account for SNOWFALL and RAINFALL at the surface
    -) snowfall is added to the first top layer mass
    -) rainfall is added to the first layer water content """
    # INPUT
    dens_newsnow = params.dens_newsnow
    snowfall = self.surf.snow[ii]  # (mm w.e.)
    rainfall = self.surf.rain[ii]  # (mm w.e.)
    # subsurface current state
    dz = self.iisubs.dz.values  # current thickness
    dens = self.iisubs.dens.values  # current density
    mass = self.iisubs.mass.values  # current mass
    water = self.iisubs.water.values  # current water

    # MAIN
    # SNOWFALL
    dz[0] = dz[0] + snowfall / dens_newsnow  # (m) add snowfall to thickness of first layer
    mass[0] = mass[0] + snowfall  # (g) add snowfall to mass of first layer
    dens[0] = mass[0] / dz[0]  # (kg/m3) adjust density of first layer
    # new snow fraction for grain size computations
    if params.method_radpen == 2 & params.method_grainsize == 2:
        self.iisubs['fnew'][0] = snowfall / mass[0]  # update new snow fraction

    # RAINFALL
    water[0] = water[0] + rainfall  # add rainfall to water content of first layer

    # OUTPUT
    self.iisubs['dz'] = dz  # update current thickness
    self.iisubs['dens'] = dens  # update current density
    self.iisubs['mass'] = mass  # update current mass
    self.iisubs['water'] = water  # update current water
    return


def melting_surface(self, ii, zz):
    """ Calculate melting of surface layers
    -) all melt comes from the surface
    -) it accounts for melting of more than 1 layer (if energy is enough) WORKING ON IT see UPDATE_LAYERS """
    # INPUT
    Qmelt = self.surf.Q[ii]  # energy available for melt (= SEB)
    dt = self.dt  # (s) timestep length
    Lf = params.Lf  # latent heat of fusion
    method_perc = params.method_perc  # percolation method
    # subsurface current state
    z = self.iisubs.z.values  # current depth
    dz = self.iisubs.dz.values  # current thickness
    dens = self.iisubs.dens.values  # current density
    temp = self.iisubs.temp.values  # current temperature
    mass = self.iisubs.mass.values  # current mass
    water = self.iisubs.water.values  # current water

    # MAIN
    # FIRST LAYER
    if zz == 0:  # surface melt calculate only at the top layer
        surfacemelt = Qmelt * dt / Lf  # surface melt from energy balance
        self.surfacemelt = surfacemelt  # class attribute for next layer iteration
        if method_perc == 2:
            self.surfacemelt_deepperc = surfacemelt  # counter for deep water percolazion
        self.layersmelted = 0  # melted layers counter
    if self.surfacemelt > 0:  # if there is any surface melt
        if method_perc == 1:  # only with TIP BUCKET percolation
            water[zz] = water[zz] + self.surfacemelt  # add melt to first layer water content
        mass[zz] = mass[zz] - self.surfacemelt    # remove melt from layer mass
        dz[zz] = mass[zz] / dens[zz]              # adjust thickness of layer (keep same density)
        # REMOVE LAYER if it's all melted (e.g. negative mass)
        if mass[zz] < 0:
            self.surfacemelt = -mass[zz]
            water[zz + 1] = water[zz]  # add water to the next layer
            # layer doens't exist anymore
            self.layersmelted += 1  # updated melted layers counter
            z[zz] = np.nan
            mass[zz] = np.nan
            dz[zz] = np.nan
            dens[zz] = np.nan
            water[zz] = np.nan
            temp[zz] = np.nan
        else:
            self.surfacemelt = 0.

    # OUTPUT
    self.iisubs['z'] = z  # update current depth
    self.iisubs['dz'] = dz  # update current thickness
    self.iisubs['dens'] = dens  # update current density
    self.iisubs['temp'] = temp  # update current temperature
    self.iisubs['mass'] = mass  # update current mass
    self.iisubs['water'] = water  # update current water
    return


def melting_subsurface(self, zz):
    """ Calculate melting of subsurface layers
        -) subsurface layers can melt if their temperatures is > 0
        -) due to ground heat flux or radiation penetration """
    # INPUT
    Lf = params.Lf        # latent heat of fusion
    cpice = params.cpice  # specific heat capacity of ice
    # subsurface current state
    dz = self.iisubs.dz.values        # current thickness
    dens = self.iisubs.dens.values    # current density
    temp = self.iisubs.temp.values    # current temperature
    mass = self.iisubs.mass.values    # current mass
    water = self.iisubs.water.values  # current water

    # MAIN
    energy = abs(cpice * mass[zz] * temp[zz])  # energy to bring layer back to T=0C
    layermelt = energy / Lf
    water[zz] = water[zz] + layermelt  # add layer melt to water content
    mass[zz] = mass[zz] - layermelt    # remove layer melt from layer mass
    dz[zz] = mass[zz] / dens[zz]       # update layer thickness (keep density the same)
    temp[zz] = 0.                      # update layer temperature
    # OUTPUT
    self.iisubs['dzdt'][zz] = - layermelt / dens[zz]  # thickness change to update surface height
    self.iisubs['dmeltdt'][zz] = layermelt
    self.iisubs['dz'] = dz  # update current thickness
    self.iisubs['dens'] = dens  # update current density
    self.iisubs['temp'] = temp  # update current temperature
    self.iisubs['mass'] = mass  # update current mass
    self.iisubs['water'] = water  # update current water
    return


def refreezing(self, zz):
    """ Compute refreezing in subsurface layers """
    # INPUT
    densice = params.densice  # density of ice
    Lf = params.Lf  # latent heat of fusion
    cpice = params.cpice  # specific heat capacity of ice
    # subsurface current state
    z = self.iisubs.z.values  # current depth
    dz = self.iisubs.dz.values  # current thickness
    dens = self.iisubs.dens.values  # current density
    temp = self.iisubs.temp.values  # current temperature
    mass = self.iisubs.mass.values  # current mass
    water = self.iisubs.water.values  # current water
    drfz = self.iisubs.drfzdt.values  # delta refreezing

    # MAIN
    drfz[zz] = 0  # set layer delta refreezing to zero

    if dens[zz] < densice:
        energywater = water[zz] * Lf  # energy to freeze all water
        energytemp = abs(cpice * mass[zz] * temp[zz])  # energy to bring layer at T=0C
        energydens = (densice - dens[zz]) * dz[zz] * Lf  # energy to bring layer at densice
        # select smaller energy
        energy = energywater
        if energy > energytemp:
            energy = energytemp
        if energy > energydens:
            energy = energydens
        # refreeze the selected amount of energy
        water[zz] = water[zz] - energy / Lf  # update water
        if water[zz] < 0:
            water[zz] = 0  # cannot be negative
        mass[zz] = mass[zz] + energy / Lf  # update mass
        drfz[zz] = energy / Lf  # update refreezing
        dens[zz] = mass[zz] / dz[zz]  # update density
        temp[zz] = temp[zz] + energy / (mass[zz] * cpice)  # update temperature (rises due to latent heat release)
        if temp[zz] > 0:
            temp[zz] = 0  # cannot be > 0

    else:  # if dens layer > densice
        # allow surface layer to refreeze even on top of ice
        if zz == 0:
            energywater = water[zz] * Lf  # energy to freeze all water
            energytemp = abs(cpice * mass[zz] * temp[zz])  # energy to bring layer at T=0C
            # select smaller energy
            energy = energywater
            if energy > energytemp:
                energy = energytemp
            # refreeze the selected amount of energy
            water[zz] = water[zz] - energy / Lf  # update water
            if water[zz] < 0:
                water[zz] = 0  # cannot be negative
            mass[zz] = mass[zz] + energy / Lf  # update mass
            dz[zz] = dz[zz] + (energy / Lf) / densice  # update thickness
            drfz[zz] = energy / Lf  # update refreezing
            dens[zz] = mass[zz] / dz[zz]  # update density
            temp[zz] = temp[zz] + energy / (mass[zz] * cpice)  # update temperature (rises due to latent heat release)
            if temp[zz] > 0:
                temp[zz] = 0  # cannot be > 0

    # fraction of refreeezing for snow grain size calculations
    if params.method_radpen == 2 & params.method_grainsize == 2:
        self.iisubs['frfz'] = drfz[zz] / mass[zz]  # update refreezing fraction

    # OUTPUT
    self.iisubs['z'] = z  # update current depth
    self.iisubs['dz'] = dz  # update current thickness
    self.iisubs['dens'] = dens  # update current density
    self.iisubs['temp'] = temp  # update current temperature
    self.iisubs['mass'] = mass  # update current mass
    self.iisubs['water'] = water  # update current water
    self.iisubs['drfzdt'] = drfz  # update current water
    return


def percolation(self, zz):
    """ Compute percolation through subsurface layers """
    # INPUT
    densice = params.densice  # density ice
    denswater = params.denswater  # density water
    # subsurface current state
    dz = self.iisubs.dz.values  # current thickness
    dens = self.iisubs.dens.values  # current density
    water = self.iisubs.water.values  # current water

    # MAIN
    # first compute irreducible water content
    irrwc = irreduciblewatercontent(self, zz)
    airvolumeice = ((densice - dens[zz]) / densice) * dz[zz]  # available layer pore space
    minwatercont = airvolumeice * irrwc * denswater  # minimum water content
    if dens[zz] >= densice:  # no water content allowed if ice
        minwatercont = 0
    # percolation (only if more water than irrwc)
    if water[zz] > minwatercont:
        deltawater = water[zz] - minwatercont
        water[zz] = minwatercont  # set layer water to minimum water content
        if water[zz] < 0:
            water[zz] = 0.  # water content cannot be negative
        if zz != (self.nz - 1):  # for all layer but not the last one
            water[zz + 1] = water[zz + 1] + deltawater  # add excess water to next layer
        else:  # do nothing for the moment
            print('percolation(): bottom layer reached')

    # OUTPUT (only water content changed)
    self.iisubs['water'] = water  # update current water
    return


def deep_percolation(self, zz):
    """ Compute deep percolation following Marchenko et al. (2017)
        Compute directly how much water needs to be added to each layer """
    # INPUT
    method_PDF = params.method_PDF  # Probability Distribution Function type
    zlim = params.perc_zlim         # (m) max percolation depth
    # subsurface current state
    z = self.iisubs.z.values  # current depth
    dz = self.iisubs.dz.values  # current thickness
    water = self.iisubs.water.values  # current water

    if method_PDF == 1:  # uniform PDF
        if z[zz] <= zlim:
            pdf = 1/zlim
        else:
            pdf = 0
    elif method_PDF == 2:  # linear PDF
        if z[zz] <= zlim:
            pdf = 2*(zlim-z[zz])/(zlim**2)
        else:
            pdf = 0
    elif method_PDF == 3:  # normal PDF
        sig = zlim/3  # PDF standard deviation (99.7% of water stays above zlim)
        pdf = 2*np.exp(-1*(z[zz]**2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))
    else:  # print error
        print('ERROR in function deep_percolation(): method_PDF not valid')
        quit()

    water[zz] = water[zz] + pdf * dz[zz] * self.surfacemelt_deepperc  # add melt to layer

    # OUTPUT
    self.iisubs['water'] = water  # update current water
    return


def densification(self, ii, zz):
    """ Compute densification of the dry snowpack
        For the moment it follow Herron & Langway, adapted by Li & Zwally """
    # INPUT
    densice = params.densice  # density ice
    dt = self.dt  # (s) timestep length
    tkel = 273.16  # 0C in Kelvin
    universal = 8.3144  # universal gas constant in J/K/mol
    tauyear = 31536000.  # amount of seconds in one year
    Tsurf = self.surf.Tsurf[ii]  # surface temperature current timestep
    # subsurface current state
    z = self.iisubs.z.values  # layer depth
    dens = self.iisubs.dens.values  # current density
    temp = self.iisubs.temp.values  # current temperature

    # MAIN
    # 1) Herron & Langway, adapted by Li & Zwally
    if params.method_densif == 1:
        # parameterizaation constants
        Deff = 0.000110
        p0 = 610.5
        lc = 2838000
        rv = 461.5
        beta = 2.0  # comment from DEBAM: was 18 ??
        # annual accumulation (for the moment define it as constant!
        accyear = 1.0 * 31.50 / 100  # (m) | DEBAM: accyear[i][j] = 1.0*SNOW[i][j]/100.
        # first determine some factors based on density and temperature
        if dens[zz] < densice:  # layer is not ice
            if temp[zz] < -0.132:
                factk0G = 8.36 * pow(abs(temp[zz]), -2.061)
                factE = 883.8 * pow(abs(temp[zz]), -0.885)
            else:
                factk0G = 542.88
                factE = 5304.5
            factk0 = beta * factk0G
            factk = factk0 * np.exp(-factE / (universal * (temp[zz] + tkel)))
            factpow = 1.
        else:  # layer is ice
            factk = 0.
            factpow = 1.
        # calculate temperature gradient
        if zz == 0:  # surface layer
            tgrad = (Tsurf - temp[zz]) / z[zz]
        else:
            tgrad = (temp[zz] - temp[zz - 1]) / (z[zz] - z[zz - 1])  # GRADIENTS HERE IN DEBAM ARE SWAPPED?
        Ra = (factk * pow(accyear, factpow) * ((densice - dens[zz]) / densice))

        if zz == 0:  # surface layer
            J1 = -Deff * (p0 / pow(rv, 2)) * ((lc - rv * (Tsurf + tkel)) / pow((Tsurf + tkel), 3)) * \
                 np.exp((lc / rv) * ((1 / tkel) - (1 / (Tsurf + tkel)))) * tgrad
            J2 = -Deff * (p0 / pow(rv, 2)) * ((lc - rv * (temp[zz] + tkel)) / pow((temp[zz] + tkel), 3)) * \
                 np.exp((lc / rv) * ((1 / tkel) - (1 / (temp[zz] + tkel)))) * tgrad
            Rv = -(J2 - J1) / z[zz]

        else:
            J1 = -Deff * (p0 / pow(rv, 2)) * ((lc - rv * (temp[zz - 1] + tkel)) / pow((temp[zz - 1] + tkel), 3)) * \
                 np.exp((lc / rv) * ((1 / tkel) - (1 / (temp[zz - 1] + tkel)))) * tgrad
            J2 = -Deff * (p0 / pow(rv, 2)) * ((lc - rv * (temp[zz] + tkel)) / pow((temp[zz] + tkel), 3)) * \
                 np.exp((lc / rv) * ((1 / tkel) - (1 / (temp[zz] + tkel)))) * tgrad
            Rv = -(J2 - J1) / (z[zz] - z[zz - 1])

        dens[zz] = dens[zz] + (Ra + Rv) * (dt / tauyear)

    # print error
    else:
        print('ERROR in function densification(): method_densif not valid')
        quit()

    # OUTPUT
    self.iisubs['dens'] = dens  # update current density
    return


def grainsize(self, ii):
    """ Compute snow/firn/ice grain size
        Follows the approach presented in Munneke (2011): compute evolution of
        effective snow grain size (the surface area‐weighted mean grain size of
        a collection of ice particles), considering dry and wet snow metamorphism and
        new snow and refrozen snow fractions.
        -) needs snow grain size to be initialized for first time step
        -) needs new snow and refreezing to be calculated first
        -) can be compute out of layers loop (faster) at the end of subsurface routine """
    # INPUT
    dt = self.dt  # (s) timestep length
    re0 = 54.4 / 10 ** 6
    rer = 1500 / 10 ** 6
    # subsurface current state
    re_old = self.iisubs.re.values  # current effective snow grain size
    fnew = self.iisubs.fnew.values  # fraction of new snow
    frfz = self.iisubs.frfz.values  # fraction of refrozen snow
    fold = 1 - fnew - frfz  # fraction of old snow

    # MAIN
    # DRY SNOW METAMORPHISM (parametric curves from SNICAR (Flanner and Zender, 2006)
    if params.method_drymetamorphism == 1:
        dre_dry_dt = 0
    elif params.method_drymetamorphism == 2:
        dre_dry_dt = drysnowmetamorphism(self, ii)
    else:
        print('ERROR in function grainsize(): method_drymetamorphism not valid')
        quit()

    # WET SNOW METAMORPHISM
    C = 4.22 * pow(10, -13)  # (m3/s) constant from Brun (1989)
    # fliq = self.iisubs.water.values/1000  # (m) liquid water content, need to convert to m?
    fliq = self.iisubs.water.values / (self.iisubs.mass.values + self.iisubs.water.values)  # from Carleen
    dre_wet_dt = C * pow(fliq, 3) / (4 * np.pi * pow(re_old, 2))

    # EFFECTIVE SNOW GRAIN SIZE
    re_new = (re_old + dre_dry_dt * dt + dre_wet_dt * dt) * fold + re0 * fnew + rer * frfz

    # OUTPUT
    self.iisubs['re'] = re_new  # update current effective snow grain size
    self.iisubs['re_dry'] = dre_dry_dt * dt * fold
    self.iisubs['re_wet'] = dre_wet_dt * dt * fold
    self.iisubs['re_new'] = re0 * fnew
    self.iisubs['re_rfz'] = rer * frfz
    return


def drysnowmetamorphism(self, ii):
    """ Compute dry snow metamorphism using the lookup table from ... """
    # INPUT
    table = self.drygrainsizeTABLE
    Tsurf = self.surf.Tsurf[ii]  # surface temperature current timestep
    dz = self.iisubs.dz.values  # current layers thickness
    temp = self.iisubs.temp.values  # current temperature
    dens = self.iisubs.dens.values  # current density
    re = self.iisubs.re.values  # current grain size

    # MAIN
    # TEMPERATURE GRADIENT: needed for lookup table
    dTdz = np.zeros_like(temp)
    # surface layer
    dTdz[0] = (Tsurf - (temp[0] * dz[0] + temp[1] * dz[1]) / (dz[0] + dz[1])) / dz[0]
    # all layers
    dTdz[1:-1] = ((temp[:-2] * dz[:-2] + temp[1:-1] * dz[1:-1]) / (dz[:-2] + dz[1:-1]) -
                  (temp[1:-1] * dz[1:-1] + temp[2:] * dz[2:]) / (dz[1:-1] + dz[2:])) / dz[1:-1]
    # bottom layer, not relevant assume = to second last layer
    dTdz[-1] = dTdz[-2]
    dTdz = np.abs(dTdz)  # take absolute value

    # LOOKUP TABLE: extract proper values depending on temp, dens, and dTdz
    # unfortunately a loop over the layers is needed here (BUT ONLY OVER LAYER AFFECTED by RADPEN!)
    # 2 methods in testing here:
    # 1) as Carleen model: do 3D linear interpolation form original lookup tables (computational expensive)
    # 2) use regrided lookup tables (e.g. regrid lookup table, using the same interpolation as in Carleen model,
    #    then select here the closest neighboor)

    # 1) 3D linear interpolation method ----------------------------------------
    if params.method_drylookuptable == 1:
        tau = np.zeros_like(temp)
        kappa = np.zeros_like(temp)
        dr0 = np.zeros_like(temp)
        drdry = np.zeros_like(temp)
        for zz in range(0, self.radpen_nz):
            # find indeces of the grid point nearest to specific values
            abstemp = np.abs(table.TVals - (temp[zz] + 273.15))
            absdTdz = np.abs(table.DTDZVals - dTdz[zz])
            absdens = np.abs(table.DENSVals - dens[zz])
            itemp = int(np.where(abstemp == np.min(abstemp))[0][0])
            idTdz = int(np.where(absdTdz == np.min(absdTdz))[0][0])
            idens = int(np.where(absdens == np.min(absdens))[0][0])

            if temp[zz] + 273.15 < table.TVals[itemp].values:
                itemp -= 1
            if dTdz[zz] < table.DTDZVals[idTdz].values:
                idTdz -= 1
            if dens[zz] < table.DENSVals[idens].values:
                idens -= 1

            # retrieve values (sort of linear interpolation, copied by Carleen model)
            # avoid end of array issues here
            if temp[zz] + 273.15 >= np.max(table.TVals.values):
                itemp -= 1
                fractemp = 1
            else:
                fractemp = (temp[zz] + 273.15 - table.TVals[itemp].values) / \
                           (table.TVals[itemp + 1].values - table.TVals[itemp].values)
            if dTdz[zz] >= np.max(table.DTDZVals.values):
                idTdz -= 1
                fracdTdz = 1
            else:
                fracdTdz = (dTdz[zz] - table.DTDZVals[idTdz].values) / \
                           (table.DTDZVals[idTdz + 1].values - table.DTDZVals[idTdz].values)
            if dens[zz] >= np.max(table.DENSVals.values):
                idens -= 1
                fracdens = 1
            else:
                fracdens = (dens[zz] - table.DENSVals[idens].values) / \
                           (table.DENSVals[idens + 1].values - table.DENSVals[idens].values)

            tau[zz] = tau[zz] + (1.0 - fracdens) * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).taumat.values
            tau[zz] = tau[zz] + (1.0 - fracdens) * (1.0 - fractemp) * fracdTdz * \
                      table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens).taumat.values
            tau[zz] = tau[zz] + (1.0 - fracdens) * fractemp * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens).taumat.values
            tau[zz] = tau[zz] + (1.0 - fracdens) * fractemp * fracdTdz * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens).taumat.values
            tau[zz] = tau[zz] + fracdens * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens + 1).taumat.values
            tau[zz] = tau[zz] + fracdens * (1.0 - fractemp) * fracdTdz * \
                      table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens + 1).taumat.values
            tau[zz] = tau[zz] + fracdens * fractemp * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens + 1).taumat.values
            tau[zz] = tau[zz] + fracdens * fractemp * fracdTdz * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens + 1).taumat.values

            kappa[zz] = kappa[zz] + (1.0 - fracdens) * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                        table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).kapmat.values
            kappa[zz] = kappa[zz] + (1.0 - fracdens) * (1.0 - fractemp) * fracdTdz * \
                        table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens).kapmat.values
            kappa[zz] = kappa[zz] + (1.0 - fracdens) * fractemp * (1.0 - fracdTdz) * \
                        table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens).kapmat.values
            kappa[zz] = kappa[zz] + (1.0 - fracdens) * fractemp * fracdTdz * \
                        table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens).kapmat.values
            kappa[zz] = kappa[zz] + fracdens * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                        table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens + 1).kapmat.values
            kappa[zz] = kappa[zz] + fracdens * (1.0 - fractemp) * fracdTdz * \
                        table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens + 1).kapmat.values
            kappa[zz] = kappa[zz] + fracdens * fractemp * (1.0 - fracdTdz) * \
                        table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens + 1).kapmat.values
            kappa[zz] = kappa[zz] + fracdens * fractemp * fracdTdz * \
                        table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens + 1).kapmat.values

            dr0[zz] = dr0[zz] + (1.0 - fracdens) * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).dr0mat.values
            dr0[zz] = dr0[zz] + (1.0 - fracdens) * (1.0 - fractemp) * fracdTdz * \
                      table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens).dr0mat.values
            dr0[zz] = dr0[zz] + (1.0 - fracdens) * fractemp * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens).dr0mat.values
            dr0[zz] = dr0[zz] + (1.0 - fracdens) * fractemp * fracdTdz * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens).dr0mat.values
            dr0[zz] = dr0[zz] + fracdens * (1.0 - fractemp) * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens + 1).dr0mat.values
            dr0[zz] = dr0[zz] + fracdens * (1.0 - fractemp) * fracdTdz * \
                      table.isel(TVals=itemp, DTDZVals=idTdz + 1, DENSVals=idens + 1).dr0mat.values
            dr0[zz] = dr0[zz] + fracdens * fractemp * (1.0 - fracdTdz) * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz, DENSVals=idens + 1).dr0mat.values
            dr0[zz] = dr0[zz] + fracdens * fractemp * fracdTdz * \
                      table.isel(TVals=itemp + 1, DTDZVals=idTdz + 1, DENSVals=idens + 1).dr0mat.values

            # calculate dry snow metamorphism
            if kappa[zz] <= 0 or tau[zz] <= 0:
                drdry[zz] = 0.0
            else:
                if abs(re[zz] - params.grain_size) < 1.0E-5:
                    drdry[zz] = (dr0[zz] * 1E-6 * ((tau[zz] / (tau[zz] + 1.0)) ** (1. / kappa[zz]))) / 3600.
                else:
                    drdry[zz] = (dr0[zz] * 1E-6 * ((tau[zz] / (tau[zz] + 1E6 * (re[zz] - params.grain_size))) **
                                                   (1. / kappa[zz]))) / 3600.

    # 2) high res lookup tables method ----------------------------------------
    elif params.method_drylookuptable == 2:
        tau = np.zeros_like(temp)
        kappa = np.zeros_like(temp)
        dr0 = np.zeros_like(temp)
        drdry = np.zeros_like(temp)
        for zz in range(0, self.radpen_nz):  # use plus 10 so this is never going to be a real problem
            # find indeces of the grid point nearest to specific values
            abstemp = np.abs(table.TVals - (temp[zz] + 273.15))
            absdTdz = np.abs(table.DTDZVals - dTdz[zz])
            absdens = np.abs(table.DENSVals - dens[zz])
            itemp = int(np.where(abstemp == np.min(abstemp))[0][0])
            idTdz = int(np.where(absdTdz == np.min(absdTdz))[0][0])
            idens = int(np.where(absdens == np.min(absdens))[0][0])
            # select nearest lookup tables values
            tau[zz] = table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).taumat.values
            kappa[zz] = table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).kapmat.values
            dr0[zz] = table.isel(TVals=itemp, DTDZVals=idTdz, DENSVals=idens).dr0mat.values
            # calculate dry snow metamorphism
            if kappa[zz] <= 0 or tau[zz] <= 0:
                drdry[zz] = 0.0
            # TRY THIS beacuse of runtimewarning: invalid value encountered in double_scalars
            elif kappa[zz] <= 1.0E-8:
                drdry[zz] = 0.0
            else:
                if abs(re[zz] - params.grain_size) < 1.0E-5:
                    drdry[zz] = (dr0[zz] * 1E-6 * ((tau[zz] / (tau[zz] + 1.0)) ** (1. / kappa[zz]))) / 3600.
                else:
                    if (tau[zz] + 1E6 * (re[zz] - params.grain_size)) <= 1.0E-8:
                        drdry[zz] = 0.0
                    else:
                        drdry[zz] = (dr0[zz] * 1E-6 * ((tau[zz] / (tau[zz] + 1E6 * (re[zz] - params.grain_size))) **
                                                       (1. / kappa[zz]))) / 3600.

    else:
        print('ERROR in function drygrainsize_initialization(): method_drylookuptable not valid')
        quit()

    # OUTPUT
    return drdry


def thermalconductivity(self, density):
    """ Compute thermal conductivity of snow and ice as a function of density.
        Different options:
        1) Von Dussen, presented in Mellor, J. Glaciol, 19(81), 15-66, 1977
                       or Sturm, J. Glaciol., 43(143), 26-41, 1997
        2) Sturm, J. Glaciol., 43(143), 26-41, 1997
        3) Douville et al., Clim. Dynamics, 12, 21-35, 1995
        4) Jansson, presented in Sturm, J. Glaciol., 43(143), 26-41, 1997
        5) Ostin & Andersson, Int. J. Heat Mass Transfer, 34(4-5), 1009-1017, 1991
                       or presented in Sturm, J. Glaciol., 43(143), 26-41, 1997 """
    # INPUT
    method_conduct = params.method_conduct
    densice = params.densice

    # MAIN
    # 1) Von Duseen
    if method_conduct == 1:
        conduct = 0.21e-01 + 0.42e-03 * density + 0.22e-08 * density ** 3
    # 2) Sturm (1997)
    elif method_conduct == 2:
        conduct = 0.138 - 1.01e-3 * density + 3.233e-6 * density ** 2
    # 3) Douville (1995)
    elif method_conduct == 3:
        conduct = 2.2 * pow((density / densice), 1.88)
    # 4) Jansson
    elif method_conduct == 4:
        conduct = 0.02093 + 0.7953e-3 * density + 2.512e-12 * density ** 4
    # 5) Ostin & Andersson (1991)
    elif method_conduct == 5:
        conduct = -0.00871 + 0.439e-3 * density + 1.05e-6 * density ** 2
    # print error
    else:
        conduct = None
        print('ERROR in function thermalconductivity(): method_conduct not valid')
        quit()

    # OUTPUT
    self.iisubs['k'][:len(conduct)] = conduct  # update current thermal conductivity
    return conduct


def irreduciblewatercontent(self, zz):
    """ Compute the irreducible water content, 2 methods:
        1) Schneider and Jansson (2004), J. Glaciol., 50(168), 25-34
        2) Coleou and Lesaffre (1998), Ann. Glaciol., 26, 64-68
        dencap: density of capilary water (kg/m3)
        denpor: density of water when all pores are filled completely (kg/m3)
        dencol: density of water when maximum amount according to Coleou is filled (kg/m3)
        irrwc: irreducible water content in % of mass according to Schneider """
    # INPUT
    densice = params.densice  # density of ice
    denswater = params.denswater  # density of water
    # subsurface current state
    dens = self.iisubs.dens.values  # current density

    # MAIN
    # 1) Schneider and Jansson (2004)
    if params.method_irrwc == 1:
        if dens[zz] >= densice:  # layer of ice => irrwc=0
            irrwc = 0.
        else:
            porosity = (densice - dens[zz]) / densice
            irrwater = 0.0143 * np.exp(3.3 * porosity)
            dencol = irrwater / (1. - irrwater) * dens[zz]
            denpor = porosity * denswater
            dencap = dencol
            if dencap > denpor:
                dencap = denpor
            irrwc = dencap / (porosity * denswater)
    # 2) Coleou and Lesaffre (1998)
    elif params.method_irrwc == 2:
        if dens[zz] >= densice:  # layer of ice => irrwc=0
            irrwc = 0.
        else:
            porosity = (densice - dens[zz]) / densice
            irrwater = (0.057 * porosity) / (1.0 - porosity) + 0.017
            dencol = irrwater / (1. - irrwater) * dens[zz]
            denpor = porosity * denswater
            dencap = dencol
            if dencap > denpor:
                dencap = denpor
            irrwc = dencap / (porosity * denswater)
    # print error
    else:
        irrwc = None
        print('ERROR in function irreduciblewatercontent(): method_irrwc not valid')
        quit()

    # OUTPUT
    return irrwc


def surfaceheight(self, ii):
    """ Compute surface height change for every timestep """
    # INPUT
    dens_newsnow = params.dens_newsnow
    snowfall = self.surf.snow[ii]  # (mm w.e.)
    Qmelt = self.surf.Q[ii]  # energy available for melt (= SEB)
    dt = self.dt  # (s) timestep length
    Lf = params.Lf  # latent heat of fusion
    surfheight = self.surfaceheight
    dens_firstlayer = self.iisubs.dens.values[0]  # current first layer density density

    # MAIN
    # SNOWFALL
    surfheight += snowfall / dens_newsnow  # (m) add snowfall to surface height
    # SURFACE MELTING
    melt = Qmelt * dt / Lf  # surface melt from energy balance
    surfheight -= melt / dens_firstlayer  # (m) subtract melt to surface height
    # SUBSURFACE MELTING (if radiation penetration
    if params.method_radpen == 2:
        surfheight += np.nansum(self.iisubs['dzdt'])

    # OUTPUT
    self.surf.loc[self.surf.index[ii], 'sh'] = surfheight
    self.surf.loc[self.surf.index[ii], 'surfmelt'] = melt
    self.surf.loc[self.surf.index[ii], 'subsmelt'] = np.nansum(self.iisubs['dmeltdt'])
    self.surf.loc[self.surf.index[ii], 'dshdz_newsnow'] = snowfall / dens_newsnow
    self.surf.loc[self.surf.index[ii], 'dshdz_surfmelt'] = - melt / dens_firstlayer
    if params.method_radpen == 2:
        self.surf.loc[self.surf.index[ii], 'dshdz_subsmelt'] = np.nansum(self.iisubs['dzdt'])
    self.surfaceheight = surfheight  # update current surface depth
    return


def update_layers(self):
    """ Update layers: replace, merge, split
        NB: STILL IN TESTING PHASE """
    # INPUT
    dz_opt = self.subs_init.layerthickness  # optimal layer thickness (keep the initialization grid!)
    # subsurface current state
    z = self.iisubs.z.values  # current depth
    dz = self.iisubs.dz.values  # current thickness
    dens = self.iisubs.dens.values  # current density
    temp = self.iisubs.temp.values  # current temperature
    mass = self.iisubs.mass.values  # current mass
    water = self.iisubs.water.values  # current water
    if params.method_radpen == 2:
        re = self.iisubs.re.values  # current grain size

    # MAIN
    # LAYERS REMOVAL
    # if layers melted completely in melting()
    # NB maybe this could be removed and taken care already by MERGING?
    # CAREFULL THIS COULD BE A PROBLEM IN CURRENT FORMULATION!!!!
    if self.layersmelted != 0:
        # slide arrays up
        mass[:-self.layersmelted] = mass[self.layersmelted:]
        dz[:-self.layersmelted] = dz[self.layersmelted:]
        dens[:-self.layersmelted] = dens[self.layersmelted:]
        temp[:-self.layersmelted] = temp[self.layersmelted:]
        water[:-self.layersmelted] = water[self.layersmelted:]
        re[:-self.layersmelted] = re[self.layersmelted:]
        # set last layer to nan (there is one less layers)
        z[-self.layersmelted:] = np.nan
        mass[-self.layersmelted:] = np.nan
        water[-self.layersmelted:] = np.nan
        dz[-self.layersmelted:] = np.nan
        dens[-self.layersmelted:] = np.nan
        temp[-self.layersmelted:] = np.nan
        if params.method_radpen == 2:
            re[-self.layersmelted:] = np.nan
        print('Melted layers removed')

    # if np.isnan(dz).any():
    #     z = z[~np.isnan(z)]  # depth
    #     dz = dz[~np.isnan(dz)]           # thickness
    #     mass = mass[~np.isnan(mass)]     # mass
    #     dens = dens[~np.isnan(dens)]     # dens
    #     temp = temp[~np.isnan(temp)]     # temp
    #     water = water[~np.isnan(water)]  # water
    # LAYER MERGING
    # to avoid too small layers for stability of 1D heat equation solver
    # only surface layer to be merged
    if dz[0] < 0.5 * dz_opt[0]:  # half optimal thickness
        mass[1] = mass[1] + mass[0]
        water[1] = water[1] + water[0]
        dens[1] = (dz[1] * dens[1] + dz[0] * dens[0]) / (dz[1] + dz[0])  # could also be computed from new mass/dz
        temp[1] = (dz[1] * temp[1] + dz[0] * temp[0]) / (dz[1] + dz[0])
        if params.method_radpen == 2:
            re[1] = (dz[1] * re[1] + dz[0] * re[0]) / (dz[1] + dz[0])
        dz[1] = dz[1] + dz[0]  # change thickness last because of weighted averages
        # first layer doens't exist anymore
        # slide arrays up
        mass[:-1] = mass[1:]
        dz[:-1] = dz[1:]
        dens[:-1] = dens[1:]
        temp[:-1] = temp[1:]
        water[:-1] = water[1:]
        if params.method_radpen == 2:
            re[:-1] = re[1:]
        # set last layer to nan (there is one less layers)
        z[-1] = np.nan
        mass[-1] = np.nan
        water[-1] = np.nan
        dz[-1] = np.nan
        dens[-1] = np.nan
        temp[-1] = np.nan
        if params.method_radpen == 2:
            re[-1] = np.nan
    # LAYER SPLITTING
    # to avoid too big layers
    # only surface layer is split
    if dz[0] > 1.5 * dz_opt[0]:  # 1.5 times optimal thickness
        # slide arrays down
        mass[1:] = mass[:-1]
        dz[1:] = dz[:-1]
        dens[1:] = dens[:-1]
        temp[1:] = temp[:-1]
        water[1:] = water[:-1]
        if params.method_radpen == 2:
            re[1:] = re[:-1]
        # assign new layer properties
        mass[0] = mass[1] / 2  # split mass
        mass[1] = mass[1] / 2  # split mass
        dz[0] = dz[1] / 2  # split thickness
        dz[1] = dz[1] / 2  # split thickness
        water[0] = water[1] / 2  # split water
        water[1] = water[1] / 2  # split water
        dens[0] = dens[1]  # density stays the same
        temp[0] = temp[1]  # temperature stays the same
        if params.method_radpen == 2:
            re[0] = re[1]      # grain size stays the same
        # UPDATE DEPTH, reconstructed from the layer thickness
    # every loop, only where thickness is not nan
    z[~np.isnan(dz)] = np.cumsum(dz[~np.isnan(dz)]) - 0.5 * dz[~np.isnan(dz)]

    # OUTPUT
    self.iisubs['z'] = z  # update current depth
    self.iisubs['dz'] = dz  # update current thickness
    self.iisubs['dens'] = dens  # update current density
    self.iisubs['temp'] = temp  # update current temperature
    self.iisubs['mass'] = mass  # update current mass
    self.iisubs['water'] = water  # update current water
    if params.method_radpen == 2:
        self.iisubs['re'] = re  # update current water
    return


def iisubs_initialization(self):
    """ Initialize the iisubs dataframe.
        It's pandas dataframe which contains the current state of the subsurface
        this is computed at every time step and stored in the xarray dataset only hourly
        (TO AVOID storing big files, not all sub timesteps are kept) """
    # surfaceheight counter for every timestep (DO I REALLY NEED THIS? could I just use self.surf.sh[ii]?
    self.surfaceheight = self.subs['sh'][0].values.copy()
    # subsurface current state dataframe
    self.iisubs = pd.DataFrame()  # make it a class attribute
    # initialize it with the first values (which comes from init file)
    self.iisubs['z'] = self.subs.z[:, 0].values
    self.iisubs['dz'] = self.subs.dz[:, 0].values
    self.iisubs['temp'] = self.subs.temp[:, 0].values
    self.iisubs['dens'] = self.subs.dens[:, 0].values
    self.iisubs['mass'] = self.subs.mass[:, 0].values
    self.iisubs['water'] = self.subs.water[:, 0].values
    self.iisubs['k'] = np.zeros_like(self.subs.water[:, 0].values)
    self.iisubs['fnew'] = np.zeros_like(self.subs.water[:, 0].values)
    self.iisubs['frfz'] = np.zeros_like(self.subs.water[:, 0].values)
    # cumulative variables (not instantaneous, they need a counter)
    self.iisubs['drfzdt'] = np.zeros_like(self.subs.water[:, 0].values)
    self.iisubs['drfzdt_c'] = np.zeros_like(self.subs.water[:, 0].values)  # counter (to save hourly cumulative)
    self.iisubs['dzdt'] = np.zeros_like(self.subs.z[:, 0].values)
    self.iisubs['dzdt_c'] = np.zeros_like(self.subs.z[:, 0].values)  # counter (to save hourly cumulative)
    self.iisubs['dmeltdt'] = np.zeros_like(self.subs.z[:, 0].values)
    self.iisubs['dmeltdt_c'] = np.zeros_like(self.subs.z[:, 0].values)  # counter (to save hourly cumulative)
    # when radiation penetration is used
    if params.method_radpen == 2:
        self.iisubs['dzdt'] = np.zeros_like(self.subs.z[:, 0].values)
        self.iisubs['melt'] = np.zeros_like(self.subs.z[:, 0].values)
        if params.method_grainsize == 2:
            self.iisubs['re'] = self.subs.re[:, 0].values
            self.iisubs['re_dry'] = self.subs.re_dry[:, 0].values
            self.iisubs['re_wet'] = self.subs.re_wet[:, 0].values
            self.iisubs['re_new'] = self.subs.re_new[:, 0].values
            self.iisubs['re_rfz'] = self.subs.re_rfz[:, 0].values
        if params.method_extraoutput == 2:
            self.iisubs['dT_radpen'] = self.subs.re_rfz[:, 0].values
            self.iisubs['dT_conduct'] = self.subs.re_rfz[:, 0].values

    return


def drygrainsize_initialization(self):
    """ Read and initialize the lookup tables needed for dry snow grain metamorphism grainsize().
        Called in model class __init__ """
    # INPUT
    path = params.radpen_folder  # folder path to radiation files

    # MAIN
    # load lookup tables for Carleen method (linear interpolation within the model)
    if params.method_drylookuptable == 1:
        if params.ssa_in == 60:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=60).nc').load()
        elif params.ssa_in == 80:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=80).nc').load()
        elif params.ssa_in == 100:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=100).nc').load()
        else:
            print('ERROR in function drygrainsize_initialization(): ssa_in not valid')
            quit()
    # laod high res. interpolated lookup tables (faster computation times)
    elif params.method_drylookuptable == 2:
        if params.ssa_in == 60:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=60)_INT.nc').load()
        elif params.ssa_in == 80:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=80)_INT.nc').load()
        elif params.ssa_in == 100:
            self.drygrainsizeTABLE = xr.open_dataset(path + '/drygrainsize(SSAin=100)_INT.nc').load()
        else:
            print('ERROR in function drygrainsize_initialization(): ssa_in not valid')
            quit()
    else:
        print('ERROR in function drygrainsize_initialization(): method_drylookuptable not valid')
        quit()

    # OUTPUT
    return
