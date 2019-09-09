"""
Example for looking up stressors at a particular location.

"""

import pvcz

# Note df is a flattened list of lat/lon values that only includes those over land
df = pvcz.get_pvcz_data()

# Point of interest specified by lat/lon coordinates.
lat_poi = 32
lon_poi = -95.23

# Find the closest location on land to a point of interest
closest_index = pvcz.arg_closest_point(lat_poi, lon_poi, df['lat'], df['lon'])

# Get the stressor data from this location
location_data = df.iloc[closest_index]
