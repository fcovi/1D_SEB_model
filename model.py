#!/usr/bin/env python3
""" MAIN MODEL """
from shutil import copyfile
import time
# internal libraries
import params
from core import Model

start_time = time.time()
# COPY param.py RUN file
copyfile('params.py', 'output/params_%s.py' % params.run_name)


model = Model()
model.model()


# WRITE OUTPUT TO FILE
print('--- Writing output to file ---')
# FOR LATER COMBINE ALL IN ONE NetCDF FILE
# Surface (txt file)
surf_output_file = open('output/surface_%s.txt' % params.run_name, 'w')
model.surf.to_csv(surf_output_file, header=True, float_format='%.6f', na_rep='nan')
surf_output_file.close()
# Subsurface (NetCDF file IN THE FURTURE, for the moment having issues with writing to NetCDF)
model.subs.to_dataframe().to_csv('output/subsurface_%s.txt' % params.run_name, header=True, float_format='%.6f', na_rep='nan')
# subs.to_netcdf('output_data/subsurface.nc')

print("--- %s seconds ---" % (time.time() - start_time))
runtime_file = open('output/runtime_%s.txt' % params.run_name, 'w')
runtime_file.write('%.f' % (time.time() - start_time))
runtime_file.close()
