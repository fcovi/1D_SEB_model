# Surface Energy Balance and Subsurface Model
Federico Covi - [fcovi@alaska.edu](fcovi@alaska.eud)

One dimensional surface energy balance and subsurface model used in Covi and others (2022): *Challenges in modeling the energy balance and melt in the percolation zone of the Greenland ice sheet*. 

***

## Required Dependencies
All the scripts are developed in Python3.9. \
The model requieres the following dependencies:
* [numpy](https://numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [xarray](https://docs.xarray.dev/en/stable/)

## Running the Model
To run the model clone this repository and change the "path" variable in params.py according to your local filepath and run model.py. 
An example simulation should start with data from Site J (in the percolation zone of southwest Greenland) for the summer of 2019. 
* params.py: is where parameters and parameterizations are chosen.
* input/met/: is where the meteorological forcing files are located. Example files used in Covi and others (2022) are given. 
* input/firn/: is where the subsurface initialization files are located. Example files used in Covi and others (2022) are given.
* output/: is where the model results are stored. Files for the example simulation at Site J are given. 


