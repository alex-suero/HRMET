# High Resolution Mapping of EvapoTranspiration (HRMET)
Author:  Sam Zipper (samzipper@ku.edu)\
Python Port and Optimizations: Alexis Suero (alexis.esmb@gmail.com)

**HRMET** is a model designed for high-resolution mapping 
of evapotranspiration (ET), using surface temperature and weather data 
for precision-agriculture and drought sensitivity assessments.


## Table of Contents
- [Installation](#installation)
- [Repository Contents](#repository-contents)
- [Key Assumptions of HRMET](#key-assumptions-of-hrmet)
- [Known Issues](#known-issues)
- [Citation](#citation)
- [License](#license)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/alex-suero/HRMET
    cd HRMET
    ```

2. **Install the required Python libraries:**

   - Install `rasterio` package:
   ```sh
    pip install rasterio
   ```

    - Install `scikit-image` package:
   ```sh
    pip install scikit-image
   ```

## Repository Contents
- `HRMET.py`: this is the vectorized HRMET code.

- `HRMET_single.py`: this is the original HRMET code translated to Python.

- `HRMET Example.ipynb`: a Python Notebook that provides an example of
HRMET use. *(Note: data used in this notebook is not included in this 
repository.)*

## Key Assumptions of HRMET
- **Spatial Homogeneity:** HRMET calculates a 1D surface energy balance
 and can be applied over fields to create raster maps of ET. When 
 generating ET maps, assumptions of spatial homogeneity should be 
 considered carefully. For example, Zipper et al. (2014) assumes uniform 
 meteorological conditions across a relatively small field (~600 x 600 m). 
 This assumption may become less valid as the spatial scale increases.

- **Precision-Agriculture Scale:** HRMET is designed for small, 
precision-agriculture applications. While the physical principles may 
extend to larger scales, sufficiently high-resolution input data is 
essential for accurate results.

## Known Issues
- **Short Canopies:** HRMET does not perform well in areas with extremely 
short canopies or desert-like environments (canopy height approaching 0 
meters).

- **Canopy Height vs. Measurement Height:** The model may produce 
inaccurate results if the canopy height exceeds the height of temperature 
and wind speed measurements.

- **G_Tw Coefficient:** The G_Tw coefficient (used in cloudiness 
estimation) defaults to a summer value. Future versions should include 
automatic selection based on the day of the year (DOY).

## Citation
HRMET is introduced and described in the following publication:

> Zipper, S.C. & S.P. Loheide II (2014). *Using evapotranspiration to
assess drought sensitivity on a subfield scale with HRMET, a high
resolution surface energy balance model*. Agricultural & Forest
Meteorology 197: 91-102. DOI: 10.1016/j.agrformet.2014.06.009

Link: http://dx.doi.org/10.1016/j.agrformet.2014.06.009

## License
This project is licensed under the GNU General Public License v3.0.
