""" Configuration file for model parameters """

# INPUT DATA
# run_name = 'EKT2018_rad30_5cm35m'
# met_file = '/Users/fcovi/Greenland/PyFrEM_runs/EKT/init/met/EKT20182019_metinit.txt'
# firn_file = '/Users/fcovi/Greenland/PyFrEM_runs/EKT/init/firn/EKT2018_firninit_5cm35m.txt'
run_name = 'SiteJ2019_nopen'
met_file = '/Users/fcovi/Greenland/Paper1X/model_runs/SiteJ/init/met/SiteJ2019_metinit.txt'
firn_file = '/Users/fcovi/Greenland/Paper1X/model_runs/SiteJ/init/firn/SiteJ2019_firninit_5cm35m_Ts.txt'


### AWS information
elev = 2060  # elevation (m), used to calculate atmospheric pressure
# elev = 2355  # elevation (m), used to calculate atmospheric pressure
z = 2  # (m)  instruments height, used in turbulent fluxes calculation


### GENERAL MODEL PARAMETERS
timestep = 1        # (hours) timestep in hours (hourly simulations HIGHLY suggested)
subs_timestep = 15  # Number of subtimesteps per main timestep (to mantain subsurface solver stability)


### TURBULENCE OPTIONS
method_turbul = 2  # 1=Ambach | 2=Monin-Obukhov Stability
z0w_init = 1e-4    # (m) surface roughness length for wind speed, depends on the surface
                   # (e.g. snow or ice), always user prescribed!
method_z0Te = 2    # 1=fixed ratio | 2=according to Andreas (1987)
                   # method to compute surface roughness length for temperature and water vapor
z0Te_div = 100     # fixed ration for method_z0Te | z0T = z0e = z0w/z0Te_div


### PRECIPITATION OPTIONS
dens_newsnow = 350      # (kg/m^3) density of snowfall
prec_correction = 0.55  # 55%  precipitation correction factor (to control e.g. wind drift etc TO BE BETTER TESTED)
Train_treshold = 1.0    # (C)  threshold temperature rain/snow precipitation
Train_halfrange = 1.0   # (C)  half range temperature rain/snow precipation
                        #      (e.g. Train_treshold +/- Train_halfrange is rain/snow mixture)

### RADIATION PENTRATION OPTION
method_radpen = 1           # Shortwave radiation penetration method: 1=NO | 2=YES
radpen_folder = '/Users/fcovi/Greenland/PyFrEM/radpen'
radsfc_dz = 0.01            # (m) thickness of fictitious surface layer
method_grainsize = 2        # Grain size method: 1=consant | 2=Munneke(2011), only used in radiation penetration
method_drymetamorphism = 2  # Dry snow metamorphism method: 1=NO | 2=Flanner(2006)
method_drylookuptable = 2   # Lookup table method: 1=lin. interp. | 2=high res. lookup table
grain_size = 0.1 / 1000     # (m) constant snow grain size, used with method_grainsize = 1 or as initialization
                            # with method_grainsize = 2
ssa_in = 100                # (microns) initial grain size for dry snow metamorphism
# longwave equivalent cloudiness polynomial fit for clear sky conditions
# calculated from Tair and LWin data following Munneke (2011)
# EKT
# LWin_min_fit = [1.79460887e-02, -6.74378886e+00, 7.12412954e+02]
# Site J
LWin_min_fit = [2.50331935e-02, -1.04124049e+01, 1.18639468e+03]
method_extraoutput = 2  # save to output extra parameters: 1=NO | 2=YES
                        # increase in layer temperature due to radiation penetration
                        # increase in layer temperature due to convection


### SUBSURFACE OPTIONS
method_Tsurf = 2     # Surface temp method: 1=from measured LWout | 2=skin layer formulation
                     #                      3=from model
method_QG = 2        # Ground heat flux: 1=SEB closure assumption | 2=subsurface model
method_conduct = 3   # Snow/Ice conductivity method: 1=Van Dussen, in Sturm (1997)  | 2=Sturm(1997)
                     #      3=Douville(1995) | 4=Jansson, in Sturm(1997) | 5=Ostin & Anderson, in Sturm(1997)
method_irrwc = 1     # Irrwc method: 1=Schneider(2004) | 2=Coleou(1998)
method_densif = 1    # Densification method: 0=no dens. | 1=Herron&Langway, adapted by Li & Zwally


### PHYSICAL CONSTANTS (should not really be changed)
dens0 = 1.29      # (kg/m^3)  density of air at standard atmospheric pressure
densice = 900     # (kg/m^3)  density of ice
denswater = 1000  # (kg/m^3)  density of water
cp = 1005         # (J/kgK)   specific heat of air
cpice = 2090      # (J/kgC )  specific heat capacity of ice CAREFUL HERE
cw = 4180         # (J/kgK)   specific heat of water
g = 9.80665       # (m/s^2)   acceleration of gravity
Lf = 0.334e6      # (J/kg)    latent heat of fusion
Lv = 2.514e6      # (J/kg)    latent heat of evaporation
Ls = 2.848e6      # (J/kg)    latent heat of sublimation
P0 = 101325       # (Pa)      standard atmospheric pressure
K = 0.4           # von Karman's constant (turbulence)
eps = 1           # emissivity of the surface
sigma = 5.6703e-8  # (W m^-2 K^-4) Stefan-Boltzmann constant
