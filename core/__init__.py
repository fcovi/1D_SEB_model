# external libraries
import pandas as pd
import xarray as xr
import numpy as np

import pdb
# local libraries
import params
from .energybalance import swradpenetration_initialization
from .subsurface import drygrainsize_initialization
from .subsurface import iisubs_initialization


class Model(object):
    """ PyFrEM Model Class
        Create a model object which contains:
        -) climate forcing data as class attribute (pandas dataframe)
        -) subsurface initialization as class attribute (xarray dataser)
        -) model parameters as class attribute
        -) model functions as class methods """

    # MODEL INITIALIZATION start -------------------------------------------------------------------------------
    def __init__(self):
        """ Model Initialization
            -) read climate forcing data
            -) read subsurface initialization data
            -) initialize subsurface
            -) initialize relevant model parameters """

        # CLIMATE FORCINGS (pandas dataframe)
        # read climate forcinga data file
        self.met = pd.read_csv(params.met_file, index_col=0, parse_dates=True, header=1)
        # linear interpolation : the subsurface model requires a timestep smaller than 1 hour
        # to maintain the stability of solver. Linearly interpolate to prescribed subtimestep
        freq = params.timestep / params.subs_timestep
        self.surf = self.met.resample(str(freq) + 'H').mean().interpolate()
        self.surf.Prec = self.surf.Prec * freq  # adjust precipitation (not a mean value but a cumulated one!)
        # initialize some variables
        self.surf.loc[self.surf.index[0], 'sh'] = 0.  # relative surface height
        self.surf.loc[self.surf.index[0], 'surfmelt'] = 0.  # surface melt
        self.surf.loc[self.surf.index[0], 'subsmelt'] = 0.  # subsurface melt

        # SUBSURFACE INITIALIZATION  (xarray dataset)
        # read subsurface initialization data file
        self.subs_init = pd.read_csv(params.firn_file, sep=' ', header=1)
        # SUBSURFACE INITIALIZATION (xarray dataset)
        subs = xr.Dataset()
        # define subsurface coordinates
        subs.coords['nr'] = self.subs_init.index  # subs layers number
        subs.coords['TIMESTAMP'] = self.met.index  # timestamps
        # allocate subsurface variables
        subs['z'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['z_rel'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['dz'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['temp'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['dens'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['mass'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['water'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['drfzdt'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['k'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        subs['sh'] = (['TIMESTAMP'], np.full((len(self.met.index)), np.nan))
        if params.method_radpen == 2:
            subs['dSWdz'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
            subs['dzdt'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
            subs['dmeltdt'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
            if params.method_grainsize == 2:
                subs['re'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
                subs['re_dry'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
                subs['re_wet'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
                subs['re_new'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
                subs['re_rfz'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
            if params.method_extraoutput == 2:
                subs['dT_radpen'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
                subs['dT_conduct'] = (['nr', 'TIMESTAMP'], np.full((len(self.subs_init.index), len(self.met.index)), np.nan))
        # initialize first timestep of dataset
        subs['z'][self.subs_init.index, 0] = self.subs_init.layerdepth
        subs['dz'][self.subs_init.index, 0] = self.subs_init.layerthickness
        subs['temp'][self.subs_init.index, 0] = self.subs_init.layertemperature
        subs['dens'][self.subs_init.index, 0] = self.subs_init.layerdensity
        subs['mass'][self.subs_init.index, 0] = self.subs_init.layerdensity * self.subs_init.layerthickness
        subs['water'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
        subs['drfzdt'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
        subs['k'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
        subs['sh'][0] = 0
        subs['z_rel'][self.subs_init.index, 0] = subs['z'][self.subs_init.index, 0]
        if params.method_radpen == 2:
            subs['dSWdz'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
            subs['dzdt'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
            subs['dmeltdt'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
            if params.method_grainsize == 2:
                subs['re'][self.subs_init.index, 0] = np.full(len(self.subs_init.index), params.grain_size)
                subs['re_dry'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
                subs['re_wet'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
                subs['re_new'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
                subs['re_rfz'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
            if params.method_extraoutput == 2:
                subs['dT_radpen'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))
                subs['dT_conduct'][self.subs_init.index, 0] = np.zeros(len(self.subs_init.index))

        # set subs xarray dataset as class attribute
        self.subs = subs  # subsurface xarray dataset
        # initialize current state of the subsurface dataframe (see function for description)
        iisubs_initialization(self)

        # MODEL PARAMETERS set as class attribute
        self.dt = freq * 60 * 60             # (s) timestep length
        self.nt = len(self.surf.index)       # number of time steps
        self.nz = len(self.subs_init.index)  # number of subsurface layers
        self.surfacemelt = 0   # surface melt, initialize it to 0

        self.layersmelted = 0  # melted layers counter, initialize it to 0
        self.P = params.P0 * np.exp(-0.0001184 * params.elev)  # air pressure from AWS elevation

        # SHORTWAVE RADIATION PENETRATION INITIALIZATION
        if params.method_radpen == 2:
            swradpenetration_initialization(self)
            self.radpen_nz = 0  # number of layers affected by SW penetration
            if params.method_grainsize == 2:
                drygrainsize_initialization(self)  # initialize dry metamorphism lookup table

        # DEEP PERCOLATION INITIALIZATION
        if params.method_perc == 2:
            self.surfacemelt_deepperc = 0  # surface melt counter for deep percolation, initialize it to 0
    # MODEL INITIALIZATION end ---------------------------------------------------------------------------------

    # MODEL FUNCTIONS ---------------------------------------------------------------------------------
    # NB import only methods that are called in the main function model()
    from .surfacetemperature import surfacetemperature
    from .energybalance import energybalance
    from .energybalance import shortwavenet
    from .energybalance import longwavenet
    from .energybalance import rainenergy
    from .energybalance import groundheatflux
    from .turbulence import turbulentfluxes
    from .subsurface import subsurface


    def model(self):
        """ Main Model function
            Calling this function run a simulation with the
            input files and parameters specified in params.py """

        # LOOP over TIMESTEP
        # noinspection PyArgumentList
        for ii in range(0, self.nt):

            # Print day to screen at midnight
            if self.surf.index[ii].hour == 0 and self.surf.index[ii].minute == 0 and self.surf.index[ii].second == 0:
                print('MODEL: Computing day %s' % self.surf.index[ii].strftime('%Y-%m-%d'))

            # ENERGY BALANCE --------------
            # Shortwave net radiation (independet from Tsurf)
            self.shortwavenet(ii)
            # Surface temperature
            self.surfacetemperature(ii)
            # Net raiation
            self.longwavenet(ii)
            # Turbulent fluxes
            self.turbulentfluxes(ii)
            # Rain energy
            self.rainenergy(ii)
            # Ground heat flux
            self.groundheatflux(ii)
            # Energy balance
            self.energybalance(ii)

            # SUBSURFACE PROCESSES --------
            if ii > 0:  # skip first timestep (== initialization)
                self.subsurface(ii)

        return
