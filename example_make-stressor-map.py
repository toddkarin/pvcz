"""
Example for importing the PV climate stressors and zones.

"""

import numpy as np
import pvcz
from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Get the data.
# Note df is a flattened list of lat/lon values that only includes those over land
df = pvcz.get_pvcz_data()
info = pvcz.get_pvcz_info()

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

# Draw the filled contour lines (the map).
cs = m.contourf(xg, yg, data['T_equiv_rack_1p1eV'],
                levels=40, cmap="jet", latlon=True)

cbar = m.colorbar(cs,location='bottom',pad="5%")
cbar.set_label('Equivalent Temperature, Rack (C)')

plt.show()
