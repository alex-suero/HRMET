# README
Author:  Sam Zipper (samzipper@ku.edu)\
Python Port and Optimizations: Alexis Suero (alexis.esmb@gmail.com)

This repository contains the code for the High Resolution Mapping of 
EvapoTranspiration (HRMET; pronounced "hermit") model, ported to Python 
with optimized vectorized functions for enhanced performance.

HRMET is introduced and described in the following publication:

> Zipper, S.C. & S.P. Loheide II (2014). Using evapotranspiration to
assess drought sensitivity on a subfield scale with HRMET, a high
resolution surface energy balance model. Agricultural & Forest
Meteorology 197: 91-102. DOI: 10.1016/j.agrformet.2014.06.009

Link: http://dx.doi.org/10.1016/j.agrformet.2014.06.009

## Repository Contents ##
- `HRMET.py`: this is the vectorized HRMET code.

- `HRMET_single.py`: this is the original HRMET code translated to Python.

- `HRMET Example.ipynb`: a Python Notebook that provides an example of
HRMET use.

## Key Assumptions of HRMET ##
- HRMET calculated the 1D surface energy balance. However, it is typically
applied over fields to produce raster maps of ET. In order to do this, 
you simply have to define the relevant inputs at all locations you want
to map ET, and then run HRMET at each grid point. 
 
- Thus, assumptions of spatial homogeneity of inputs should be made with 
care. For example, in Zipper et al. (2014), we assume uniform meteorological 
conditions over our relatively small (~600 x 600 m) field. This assumption 
gets increasingly problematic as your spatial scale increases. HRMET was 
designed for precision-agriculture scale applications; however, the physical 
principles should work at larger scales, so long as the the input data is 
sufficiently high-resolution.

## Known Issues ##
- HRMET does not work well in extremely short canopies or deserts (canopy 
height approaching 0 meters).

- HRMET does not work well when the canopy height exceeds the height of 
temperature and wind speed measurements.

- The G_Tw coefficient (used in cloudiness estimation) takes a summer value 
by default; future versions should automatically select this 
based on DOY.