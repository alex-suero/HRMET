"""
Copyright (C) 2024 Alexis Emanuel Suero Mirabal

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np

def HRMET(datetime:pd.Timestamp, longitude:float, latitude:float, T_air:float, 
          SW_in:float, wind_s:float, ea:float, pa:float, z_air:float, 
          z_wind:float, LAI:float, height:float, T_surf:float, alb_soil:float, 
          alb_veg:float, emiss_soil:float, emiss_veg:float, daylight_savings:int=1
          ) -> float:
    """
    High Resolution Surface Energy Balance Model (HRMET)
    
    Calculates evapotranspiration for a single point at a single instant using an energy 
    balance approach.
    
    Original work citation:
    Zipper, S.C. & S.P. Loheide II (2014). Using evapotranspiration to assess drought 
    sensitivity on a subfield scale with HRMET, a high resolution surface energy balance 
    model. Agricultural & Forest Meteorology 197: 91-102. 
    DOI: 10.1016/j.agrformet.2014.06.009
    
    Author:
        Sam Zipper
        University of Victoria
        samuelczipper@gmail.com
    
    Python port author:
        Alexis Emanuel Suero Mirabal
        alexis.esmb@gmail.com

    Args:
        datetime (pd.Timestamp): Date and time
        longitude (float): Longitude in decimal degrees
        latitude (float): Latitude in decimal degrees
        T_air (float): Air temperature [°C]
        SW_in (float): Incoming shortwave radiation [W/m2]
        wind_s (float): Wind speed [m/s]
        ea (float): Air vapor pressure [kPa]
        pa (float): Atmospheric pressure [kPa]
        z_air (float): Height of air temperature measurements [m]
        z_wind (float): Height of wind speed measurements [m]
        LAI (float): Leaf Area Index [m2/m2]
        height (float): Canopy height [m]
        T_surf (float): Canopy surface temperature [°C]
        alb_soil (float): Albedo of soil [0-1]
        alb_veg (float): Albedo of vegetation [0-1]
        emiss_soil (float): Emissivity of soil [0-1]
        emiss_veg (float): Emissivity of vegetation [0-1]
        daylight_savings (int, optional): daylight savings time [1=yes (summer), 0=no]. 
            Defaults to 1.

    Returns:
        ET (float): Evapotranspiration rate [mm/hr]
    """
    # Define trigonometric functions that take degrees as input
    sind = lambda x: np.sin(np.radians(x))
    cosd = lambda x: np.cos(np.radians(x))
    asind = lambda x: np.degrees(np.arcsin(x))
    acosd = lambda x: np.degrees(np.arccos(x))

    # Check for missing inputs
    inputs = [longitude, latitude, T_air, SW_in, wind_s, ea, pa, LAI, height, 
              T_surf, alb_soil, alb_veg, emiss_soil, emiss_veg]
    if any(x is None for x in inputs):
        raise ValueError("Missing input data")
    
    # Error check: is wind measurement height < plant height?
    if z_wind < height:
        raise ValueError(
            "z_wind < height; provide wind speed measurement from different height")
    if z_air < height:
        raise ValueError(
            "z_air < height; provide air temperature measurement from different height")
    
    # Ensure minimum values
    wind_s = max(0.1, wind_s)
    ea = max(0.01, ea)
    LAI = max(0.01, LAI)
    height = max(0.01, height)
    
    # Define constants and unit conversions
    T_k = T_surf + 273.16           # surface temperature [K]
    T_air_k = T_air + 273.16        # air temperature [K]
    stef_boltz = 5.67e-8            # Stefan-Boltzmann Constant [W m-2 K-4]
    Cp = 29.3                       # specific heat of air [J mol-1 °C-1]
    water_mw = 0.018                # molecular mass of water [kg mol-1]
    air_mw = 0.029                  # molecular mass of air [kg mol-1]
    water_p = 1000                  # density of water [kg m-3]
    air_p = ((44.6 * pa * 273.15) / 
             (101.3 * T_air_k))     # molar density of air [mol m-3]
    es_a = 0.611                    # constant 'a' for es equations [kPa]
    es_b = 17.502                   # constant 'b' for es equations [-]
    es_c = 240.97                   # constant 'c' for es equations [°C]
    von_karman = 0.4                # von Karman's constant [-]
    
    # Get datetime components
    datetime = pd.Timestamp(datetime)
    year = datetime.year
    julian_day = datetime.timetuple().tm_yday           # julian day (Jan 1 = 1)
    julian_time = datetime.hour + datetime.minute/60    # julian time in hours (0-24)
    
    # Calculate Energy Balance
    # Solar position calculations from Campbell & Norman, chapter 11
    f = 279.575 + 0.9856*julian_day
    eq_time = (1/3600) * (- 104.7*sind(f) 
                          + 596.2*sind(2*f) 
                          + 4.3*sind(3*f)
                          - 12.7*sind(4*f) 
                          - 429.3*cosd(f) 
                          - 2*cosd(2*f) 
                          + 19.3*cosd(3*f)) # equation of time (C&N 11.4)
    
    # Figure out # of degrees west of standard meridian 
    # (if negative, that means you're east!)
    while longitude > 7.5:
        longitude -= 15
    
    # Longitude correction 
    lc = -longitude/15
    
    # Calculate solar noon time [hrs]
    solar_noon = 12 + daylight_savings - lc - eq_time
    
    # Solar declination angle calculation [deg] (C&N eq. 11.2)
    solar_dec = asind(0.39785*sind(278.97 + 0.9856*julian_day +
                                   1.9165*sind(356.6 + 0.9856*julian_day)))
    
    # Solar zenith angle calculation [deg] (C&N eq. 11.1)
    zenith = acosd(sind(latitude)*sind(solar_dec) +
                   cosd(latitude)*cosd(solar_dec) *
                   cosd(15*(julian_time - solar_noon)))
    
    # Calculate downwelling LW radiation, following approach of Crawford &
    # Duchon (1999). Includes cloudiness effects.
    sol_const = 1361.5  # average annual solar constant [W m-2]
    # Calculate airmass number
    m = 35*cosd(zenith)*(1224*(cosd(zenith))**2 + 1)**(-0.5)
    # Calculate airmass number (using Kasten and Young formula)
    # m = 1 / (cosd(zenith) + 0.50572*(96.07995 - zenith)**-1.6364)
    
    # Look up table for G_Tw (delta) in Tw calculation, from Smith (1966) Table 1
    # Using only summer values here, but table also has spring/winter/fall
    # (potential future improvement- select values based on julianDay)
    def get_G_Tw(latitude):
        # List of (latitude threshold, G_Tw value) pairs
        thresholds = [(10, 2.80), (20, 2.70), (30, 2.98),
                      (40, 2.92), (50, 2.77), (60, 2.67),
                      (70, 2.61), (80, 2.24), (90, 1.94)]
        
        # Check if latitude is within valid range
        if not (0 <= latitude <= 90):
            raise ValueError("Latitude > 90 degrees or < 0 degrees")
        
        # Find the appropriate G_Tw based on latitude
        for threshold, G_Tw in thresholds:
            if latitude < threshold:
                return G_Tw
    G_Tw = get_G_Tw(latitude)
    
    # Dewpoint temperature [°C], from C&N eq. 3.14
    T_dew = es_c*np.log(ea/es_a) / (es_b - np.log(ea/es_a))
    
    # Atmospheric precipitable water, from Crawford & Duchon
    atm_water = np.exp(0.1133 - np.log(G_Tw+1) + 0.0393*T_dew)
    
    # Corrections
    # for Rayleigh scattering, absorption by permanent gases
    Tr_Tpg = 1.021 - 0.084*(m*(0.000949*pa*10+0.051))**0.5
    # for absorption by water vapor
    Tw = 1 - 0.077*(atm_water*m)**0.3
    # for scattering by aerosols
    Ta = 0.935**m
    
    # Calculate LWin based on cloudiness
    # Clear sky shortwave irradiance [W m-2]
    Rso = sol_const * cosd(zenith) * Tr_Tpg * Tw * Ta
    # Cloudiness factor [-]
    clf = max(0, min((1-SW_in/Rso), 1))
    # Emissivity based on cloud fraction, from Crawford & Duchon
    emiss_sky = (clf + (1 - clf)*(1.22 + 0.06*np.sin((datetime.month + 2)*np.pi/6)) *
                 (ea*10/T_air_k)**(1/7))
    # Total incoming absorbed longwave radiation from atmosphere [W m-2]
    LW_in = emiss_sky * stef_boltz * (T_air_k**4)
    
    # Calculate LWout based on separate vegetation and soil components 
    # (two-source model)
    # Fractional plant cover based on LAI, Norman et al. (1995)
    fc = 1 - np.exp(-0.5*LAI) 
    # Outgoing LW from vegetation [W m-2]
    LW_out_veg = emiss_veg * stef_boltz * (T_k**4) * (1 - np.exp(0.9*np.log(1 - fc)))
    # Outgoing LW from soil [W m-2]
    LW_out_soil = emiss_soil * stef_boltz * (T_k**4) * np.exp(0.9*np.log(1 - fc))
    # Total outgoing longwave radiation [W m-2]
    LW_out = LW_out_veg + LW_out_soil
    
    # Calculate total SWout as the sum of vegetation and soil components 
    # (two-source model)
    # Outgoing shortwave radiation from vegetation [W m-2],
    # based on amount of SW radiation reaching ground (Norman et al. (1995) Eq. 13)
    SW_out_veg = SW_in*(1 - np.exp(0.9*np.log(1 - fc)))*alb_veg
    # Outgoing shortwave radiation from soil surface [W m-2], 
    # based on amount of SW radiation reaching ground (Norman et al. (1995) Eq. 13)
    SW_out_soil = SW_in*np.exp(0.9*np.log(1 - fc))*alb_soil
    # Outgoing SW radiation as sum of soil and vegetation components [W m-2]
    SW_out = SW_out_veg + SW_out_soil
    
    # Calculate net radiation budget (R) [W m-2]
    R = SW_in - SW_out + LW_in - LW_out
    
    # Calculate Ground heat flux (G), based on the amount of radiation reaching the 
    # ground
    # Eq. 13 - calculate G as 35% of R reaching soil (Norman et al., 1995)
    G = 0.35*R*np.exp(0.9*np.log(1 - fc))
    
    # Iterative H calculation
    # Raupach (1994) z0m, d values as function of LAI, h
    cw = 2      # empirical coefficient
    cr = 0.3    # empirical coefficient
    cs = 0.003  # empirical coefficient
    cd1 = 7.5   # empirical coefficient
    u_max = 0.3 # empirical coefficient
    sub_rough = np.log(cw)-1+(1/cw) # roughness-sublayer influence function
    
    # Ratio of u*/uh
    u_uh = min(u_max, ((cs+cr*(LAI/2))**0.5))
    # Zero-plane displacement height [m]
    d = height * (1-(1-np.exp(-np.sqrt(cd1*LAI)))/np.sqrt(cd1*LAI))
    # Roughness length for momentum transfer [m]
    z0m = height*(1 - d/height)*np.exp(-von_karman*(1/u_uh) - sub_rough)
    # kB^-1 factor from Bastiaansen SEBAL paper to convert from z0m to z0h kB1=2.3 
    # means z0h = 0.1*z0m, which corresponds to C&N empirical equation
    kb1 = 2.3
    # Roughness length for heat transfer
    z0h = z0m/np.exp(kb1)
    
    # Iterative solution to H, rH, etc
    def iterate_H(initial_zeta, initial_h):
        # Initial stability factor for diabatic correction (zeta from C&N sec 7.5) 
        # >0 when surface cooler than air
        zeta = initial_zeta 
        H_iter = [initial_h]    # placeholder for H during iterative process
        change_perc = 0.5       # arbitrary starting value
        i = 0                   # starting i for iterations
        while abs(change_perc) > 0.001:
            i += 1
            
            # Calculate diabatic correction factors based on zeta. From C&N
            if zeta > 0:
                # stable flow, C&N equation 7.27
                diab_M = 6 * np.log(1 + zeta)
                diab_H = diab_M
            else:
                # unstable flow, C&N equation 7.26
                diab_H = -2*np.log((1 + (1 - 16*zeta)**0.5)/2)
                diab_M = 0.6*diab_H
            
            # Calculate u*, gHa based on diabatic correction factors. from C&N
            # Friction velocity [m], C&N eq. 7.24
            u_star = wind_s*von_karman / (np.log((z_wind - d)/z0m) + diab_M) 
            r_ha = 1/((von_karman**2) * wind_s * air_p / 
                      (1.4 * ((np.log((z_wind - d)/z0m) + diab_M) * 
                              (np.log((z_air - d)/z0h) + diab_H)))) # [m2 s mol-1]
            
            # By including r_excess, we can ignore the difference between 
            # T_surf and T_air
            # Excess resistance [m2 s mol-1], from Norman & Becker (1995)
            r_excess = air_mw*np.log(z0m/z0h) / (air_p*von_karman* u_star)
            # Total resistance [m2 s mol-1]
            r_htot = r_ha + r_excess
            
            # Calculate H and zeta for next iteration
            # Sensible heat flux
            H_new = Cp * (T_surf - T_air)/r_htot
            H_iter.append(H_new)
            # Updated zeta value
            zeta = -von_karman*9.8*z_wind*H_iter[-1] / (air_p*Cp*T_air_k*(u_star**3))
            
            # Calculate the percentage change between iterations
            if i > 2:
                change_perc = (H_iter[i] - H_iter[i-1]) / H_iter[i-1]
            if i == 10_000:
                raise ValueError("10,000 iterations, will not converge")
        
        # Final value of H after convergence
        H_final = 0.02 if H_iter[-1]==0 else H_iter[-1]
        return H_final
    
    # Calculate H
    H_low = iterate_H(0.5, 0.5)     # positive stability
    H_high = iterate_H(-0.5, 500)   # negative stability
    if abs((H_low - H_high)/H_low) <= 0.01:
        H = (H_low + H_high)/2
    else:
        raise ValueError("H_low and H_high are too far apart!")
    
    # Calculate ET rate
    # Latent heat flux rate as residual of energy budget [W m-2]
    LE = R - G - H
    # Latent heat of vaporization of water, dependent on temperature (°C) [J mol-1].
    # Formula B-21 from Dingman 'Physical Hydrology'
    lamda = (2.495 - (2.36e-3)*T_surf)*water_mw*1_000_000
    # Evaporation rate [mm hr-1]
    ET = (water_mw/water_p) * (60*60) * 1000 * LE/lamda
    # Set equal to 0 in values below 0 (may happen in low vegetation areas)
    ET = max(0, ET)
    
    return ET
