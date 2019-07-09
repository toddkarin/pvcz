# Photovoltaic Climate Zones (PVCZ)

This package provides the photovoltaic climate zones (PVCZ) and climate stressor data which describes the degree of environmental degradation expected on a PV module located in different locations on the world. 

# Install
Installation is easy with pip
```python
pip install pvcz
```

# About
The data is calcuated from the global land data accumulation service (GLDAS) at 0.25 degree resolution across the world.

For a full description, see the file 'Karin2019 - Photovoltaic Degradation Climate Zones - PVSC' which describes the methods.

## Climate stressors
This dataset is provided as a csv file and as a pickle file containing climate stressors specific to PV degradation.

- lat: latitude in fractional degrees.
- lon: longitude in fractional degres.
- T_equiv_rack: Arrhenius-weighted module equivalent temperature calculated using open-rack polymer-back temperature model and activation energy 1.1 eV, in C
- T_equiv_roof: Arrhenius-weighted module equiva- lent temperature calculated using close-roof-mount glass- back temperature model and activation energy 1.1 eV, in
C
- specific_humidity_mean: Average specific humidity, in g/kg.
- T_velocity: Average rate of change of module temperature using open-rack polymer-back temperature model, in C/hr.
- GHI_mean: Mean global horizontal irradiance, in kWh/m2/day.
- wind_speed: ASCE wind speed with a mean recurrence interval of 25 years, in m/s.
- T_ambient_min: Minimum ambient temperature, in C
- KG_zone: Koppen Geiger zone.
- T_equiv_rack_zone: Temperature zone for open-rack modules as a number 0 through 9, equivalent to temperature zones T1 through T10 respectively.
- T_equiv_roof_zone: Temperature zone for close- roof-mount modules as a number 0 through 9, equivalent to temperature zones T1 through T10 respectively.
- specific_humidity_mean_zone: Specific humid- ity zone, as a number 0 through 4, equivalent to temperature zones H1 through H5 respectively.
- wind_speed_zone: Wind speed zone as a number 0 through 4, equivalent to wind zones W1 through W5 respectively.
- pvcz: Photovoltaic climate zone, combined Temperature (rack) and humidity zones as a number 0 through 49, corresponding to temperature zones T1:H1, T2:H1, ... , T10:H5, see next variable as well.
- pvcz_labeled: Photovoltaic climate zone, combined Temperature (rack) and humidity zones as an alpha- numeric key, e.g. T5:H2.

## Examples

The following code snippet shows how to find the climate stressors and zones closest to a particular latitude and longitude.

```python
import pvcz

# Get the PVCZ data, df is a dataframe of lat/lon points on land and associated stressors.
df, info = pvcz.get_pvcz_data()

# Point of interest specified by lat/lon coordinates.
lat_poi = 32
lon_poi = -95.23

# Find the closest location on land to a point of interest
closest_index = pvcz.arg_closest_point(lat_poi, lon_poi, df['lat'], df['lon'])

# Get the stressor data from this location
location_data = df.iloc[closest_index]
print(location_data)
```

The following code makes a map of a particular stressor. 

```python
import numpy as np
import pandas as pd
import os
import pvcz

from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



# Get the data.
# Note df is a flattened list of lat/lon values that only includes those over land
df, info = pvcz.get_pvcz_data()

# For some uses (like making a map), it is convenient to have a 2D grid of lat/long values
data = {}
for v in df:
    data[v] = pvcz.convert_flat_to_grid(df[v],
                                        info['keepers'],
                                        info['lon_all'],
                                        info['lat_all'])

# Make a plot

# grid the lat/lon coordinates.
xg, yg = np.meshgrid(info['lon_all'], info['lat_all'])

# Set up the map
fig = plt.figure(0, figsize=(5.5, 4.5))
plt.clf()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
m = Basemap(projection='cyl', llcrnrlat=-60, urcrnrlat=90, \
            llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines(linewidth=0.5)
m.drawcountries()

# Draw the filled contours.
cs = m.contourf(xg, yg, data['T_equiv_rack'],
                levels=40, cmap="jet", latlon=True)

cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('Equivalent Temperature, Rack (C)')

plt.show()

```
 