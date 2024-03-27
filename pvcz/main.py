# import netCDF4
import fnmatch
import os
import pandas as pd
import datetime
import numpy as np
# import util


def get_pvcz_data():
    """
    Load pvcz climate data

    Examples
    --------

        df = pvcz.get_pvcz_data()

    Returns
    -------
    df : dataframe
        List of lat/lon values and associated PV climate stressors and zones.

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))

    filename = os.path.join(dir_path,
                            'PVCZ-2019_ver0p2_world_PV_climate_stressors_and_zones.pkl')

    df = pd.read_pickle(filename)
    #
    # info_filename = os.path.join(dir_path,'PVCZ-2019_GLDAS_NOAH025_3H_info.npz')
    # info = load_npz(info_filename)

    return df


def get_pvcz_info():
    """
    Load pvcz climate data

    Examples
    --------

        info = pvcz.get_pvcz_info()

    Returns
    -------
    info : dict

        Dictionary containing information on the dataset. Includes units of
        columns in df.

        'lat_all' - List of all latitudes in the GLDAS dataset

        'lon_all' - List of all longitudes in the GLDAS dataset.

        'keepers' - boolean array of whether flattened lat_all/lon_all grid
        is on land or not. Useful for converting the flattened data back to
        gridded data.

        'Ea' - activation energy for calculating equivalent temperatures (eV)

        'dt' - time step in hours for the calculation.

        'time_stamp' - time stamps of the datapoints used in the climate
        calculation.


    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #
    # filename = os.path.join(dir_path,
    #                         'PVCZ-2019_world_PV_climate_stressors_and_zones.pkl')
    #
    # df = pd.read_pickle(filename)

    info_filename = os.path.join(dir_path,'PVCZ-2019_GLDAS_NOAH025_3H_info.npz')
    info = load_npz(info_filename)

    return info


def get_pvcz_zones():
    """
    Get data structure containing information on the zone labeling.

    Returns
    -------

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    zones_filename = os.path.join(dir_path,'PVCZ-2019_ver0p2_zones.npz')

    return load_npz(zones_filename)


def load_npz(filename):
    """
    Load npz file from a local file.

    Parameters
    ----------
    filename

    Returns
    -------
    data : dict
        dictionary of the items in the npz file.

    """
    #
    data = {}
    with np.load(filename,allow_pickle=True) as arr:
        for var in list(arr.keys()):
            try:
                data[var] = arr[var].item()
            except:
                data[var] = arr[var]
    return data


def inspect_database(root_path):
    """Build database for NASA geo files

    Folder contains .cd4 files such as NLDAS or GLDAS.

    Examples
    --------
    inspect_database('data_folder')

    Parameters
    ----------
    root_path
        Folder to inspect.
    Returns
    -------
    files
        pandas DataFrame containing information on files in the root_path.

    """
    # root_path = 'GLDAS_data'
    pattern = '*.nc4'

    # GLDAS_NOAH025_3H.A20160101.0000.021.nc4.SUB.nc4
    # GLDAS_NOAH025_3H.A20160101.1800.021.nc4.SUB.nc4

    # filedata = pd.DataFrame(columns=['model','year','month','day','hour'])
    filename_list = []
    filename_fullpath = []
    model = []
    year = []
    month = []
    day = []
    hour = []
    date = []
    file_size = []

    # Cycle through files in directory, extract info from filename without opening file.
    # Note this would break if NREL changed their naming scheme.
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):

            temp = filename.split('.')

            model

            #
            filename_list.append(filename)
            filename_fullpath.append(os.path.join(root, filename))
            model.append(temp[0])
            year.append(int(temp[1][1:5]))
            month.append(int(temp[1][5:7]))
            day.append(int(temp[1][7:9]))
            hour.append(int(temp[2][0:2]))
            file_size.append(os.path.getsize(filename_fullpath[-1]))
            date.append( datetime.datetime(
                year[-1],month[-1],day[-1],hour[-1],0,
                                           0))
    # Create a DataFrame
    files = pd.DataFrame.from_dict({
        'model': model,
        'filename': filename_list,
        'fullpath': filename_fullpath,
        'year': year,
        'month':month,
        'day': day,
        'hour': hour,
        'datetime': date,
        'file_size': file_size
    })


    files = files.sort_values(by='datetime')

    # Redefine the index.
    files.index = range(files.__len__())
    return files
 

def get_unmasked_indices(nc):
    """Make boolean array for converting to lat/lon pairs.

    Geographic data in gridded dataset often contains many datapoints over
    oceans that are not of current interest. Therefore it can be useful
    instead of dealing with a 2D grid of lat/lon values to instead use a 1D
    list of lat/lon pairs over the region of interest.

    This function returns a boolean array for extracting the unmasked values.

    Automatically picks the variable with the longest number of elements.

    Parameters
    ----------
    nc
        netCDF4._netCDF4.Dataset

    Returns
    -------
    keepers
        Boolean array for extracting flattened list of unmasekd valued.
        keepers has length
    lat_flat
        Latitudes corresponding to True values of keepers
    lon_flat
        Longitudes corresponding to True values of keepers
    """

    # Find variable for getting the mask
    var_list = list(nc.variables)
    var_numel = np.zeros(len(var_list))
    for j in range(len(var_list)):
        var_numel[j] = np.prod(nc[var_list[j]].shape)
    var_index = var_numel.argmax()
    var = var_list[var_index]

    # data_mask = nc[list(nc.variables.keys())[0]] == nc.missing_value
    data_mask = nc[var][:].mask

    # Make a lat and lon grid.
    lon_grid, lat_grid = np.meshgrid(nc['lon'][:].data, nc['lat'][:].data)

    # All data that is not masked should be a vaild datapoint for keeping.
    keepers = np.logical_not(data_mask[0].flatten())

    # Make a list of all land data points (unmasked values).
    lat_flat = lat_grid.flatten()[keepers]
    lon_flat = lon_grid.flatten()[keepers]

    return (keepers, lat_flat, lon_flat)


def convert_grid_to_flat(x,y,z,keepers):
    """
    x_flat, y_flat, z_flat = convert_grid_to_flat(x,y,z,keepers) converts x,
    y,z data and a list of which values to keep 'keepers' into three
    flattened vectors x_flat, y_flat and z_flat.

    Parameters
    ----------
    x : array size (N,1)
        x
    y : array size (M,1)

    z : array size (N,M)

    keepers : boolean array size (N,M)
        Boolean array determining which indices of z are kept after flattening.

    Returns
    -------

    """
    xgrid, ygrid = np.meshgrid(x,y)

    x_flat = xgrid.flatten()[keepers.flatten()]
    y_flat = ygrid.flatten()[keepers.flatten()]
    z_flat = z[keepers].flatten()

    return x_flat, y_flat, z_flat


def convert_flat_to_grid(z_flat, keepers,  lon_all, lat_all):
    """
    Convert the a flattened data vector z_flat into a gridded, masked numpy
    array. keepers denotes the elements of a gridded (l

    Parameters
    ----------
    z_flat : array

        z_flat is an array with size (np.sum(keepers*1),). This represents a
        flattened list of all the valid points (those on land).

    keepers : array of size (N*M,)

        Boolean array describing whether a lat/lon point is on land or not.

    lon_all : array of size (M,1)

        All longitudes in the gridded data.

    lat_all : array size (N,1)

        All latitudes in the gridded data.

    Returns
    -------
    zm : masked array, size (N,M)

         gridded version of z_flat. Mask is true for points where keepers ==
         False.

    """

    # Old method (only numeric data).
    # z_flat_all = np.zeros((len(keepers))) + np.nan
    # z_flat_all[keepers] = np.array(z_flat)
    #
    # z = np.reshape(z_flat_all, (len(lat_all), len(lon_all)))
    # zm = np.ma.masked_invalid(z)
    #


    # This method works for non-numeric data.
    z_flat_all = np.empty(shape=(len(keepers)), dtype=z_flat.dtype)
    z_flat_all[keepers] = np.array(z_flat)

    z = np.reshape(z_flat_all, (len(lat_all), len(lon_all)))

    zm = np.ma.masked_array(data=z, mask=np.logical_not(keepers))
    return zm


def equiv_temp_in_C(temperature, Ea):
    """Calculate Arrhenius-weighted temperature

    Parameters
    ----------
    temperature
        Array of temperature values
    Ea
        Activation energy in eV
    Returns
    -------
    T_equiv
        Equivalent temperature in eV.
    """
    Ea = convert_units(Ea,'eV','J')
    kB = 1.381e-23


    T_equiv = -Ea/kB/np.log( np.mean( np.exp(-Ea/(kB*(temperature+273.15))))) - 273.15
    return T_equiv


def convert_units(input,input_units,output_units):
    """
    Convert energy, frequency and wavelength units

    Description

        Using fundamental physical constants ==  there is a unique way of
        converting an energy into different units. For example ==  we often want
        to know the energy ==  wavelength or frequency of a photon. This
        function converts between these representations easily ==  threading
        over
        vectors automatically.

        This function uses the INDEX OF REFRACTION OF AIR as 1.000268746.
        To use a different index of refraction ==  modify the speed of light in
        the code.

        Frequency units are in cycles per second.

    Example:
        Convert 820 nm into 1.512 eV.

        > convertUnits(820, 'nm', 'eV')

    Available choices of units:

        'nm' (nanometers)
        'm' (meters)
        'eV' (electron Volts) ==
        'cm-1' (inverse centimeters)
        'Hz' (Hertz) ==  'KHz' ==  'MHz' ==  'GHz' ==  'THz' ==
        'K' (kelvin) ==  'J' (Joules)
        'kJ/mol'

    """


    # input_units = input_units.lower()
    # output_units = output_units.lower()

    # Need all the sig figs we can get!

    # Electron charge
    e = 1.602176463e-19
    # planck's constant
    h = 6.62606957e-34
    # speed of light in air for 820 nm light ==  70 F ==  1 atm
    c = 299792458/1.000268746
    # Boltzman constant
    kB = 1.3806503e-23
    # Avogadro's constant
    Na = 6.0221409e23
    # inch to mm
    inch2mm = 25.4

    # convert input to joules
    if input_units == 'nm':
        energy = h*c/(input*1e-9)
    elif input_units == 'mm':
        energy = h*c/(input*1e-3)
    elif input_units == 'm':
        energy = h*c/input
    elif input_units == 'eV':
        energy = input*e
    elif input_units == 'Hz':
        energy = h*input
    elif input_units == 'KHz':
        energy = h*input*1e3
    elif input_units == 'MHz':
        energy = h*input*1e6
    elif input_units == 'GHz':
        energy = h*input*1e9
    elif input_units == 'THz':
        energy = h*input*1e12
    elif input_units == 'cm-1':
        energy = h*c*input*100
    elif input_units == 'J':
        energy = input
    elif input_units == 'K':
        energy = kB*input
    elif input_units == 'kJ/mol':
        energy = input*1000/Na
    else:
        Exception('Input units not recognized')


    # Convert Joules to output
    if output_units == 'GHz':
        output = energy/h/1e9
    elif output_units == 'eV':
        output = energy/e
    elif output_units == 'Hz':
        output = energy/h
    elif output_units == 'KHz':
        output = energy/h/1e3
    elif output_units == 'MHz':
        output = energy/h/1e6
    elif output_units == 'THz':
        output = energy/h/1e12
    elif output_units == 'm':
        output = h*c/energy
    elif output_units == 'in':
        output = h*c/energy
    elif output_units == 'nm':
        output = h*c/energy*1e9
    elif output_units == 'cm-1':
        output = (h*c/energy*100)**(-1)
    elif output_units == 'K':
        output = energy/kB
    elif output_units == 'J':
        output = energy
    elif output_units == 'kJ/mol':
        output = energy/1000*Na
    else:
        Exception('Output units not recognized')

    return output

def closest_degrees(lat_find, lon_find, lat_list, lon_list):
    """
    Finds closest lat/lon using Euclidean distance on a a cylindrical grid.

    Parameters
    ----------
    lat_find : numeric
        latitude to search for
    lon_find : numeric
        longitude to search for
    lat_list : array
        list of latitudes over which to perform the search
    lon_list : array
        list of longitudes over which to perform the search
    Returns
    -------
    closest_index
        argument of the closest index of lat_list, lon_list.
    distance_in_degrees
        distance to closest point in fractional degrees.

    """
    distance = np.sqrt( (lat_find-lat_list)**2 + (lon_find-lon_list)**2 )
    closest_index = np.argmin(np.array(distance))
    distance_in_degrees = distance[closest_index]

    return (closest_index, distance_in_degrees)


def uv_degradation_stressor(UV_irradiance, temperature,
                            relative_humidity, p=1, Ea=1, n=1):
    """Cumulative UV degradation

    Finds the cumulative UV degradation using

        degradation = I^p exp(Ea/(kB T)) RH^n

    where I is the UV_irradiance, T is the module temperature,

    Parameters
    ----------
    UV_irradiance
        UV irradiance in W/m^2
    temperature
        Temperature to use in calculation in C, usually module temperature
    relative_hummidity
        relative humidity in percent (i.e. 0 to 100)
    p
        scaling exponent for irradiance.
    Ea
        Activation energy in eV
    n
        scaling exponent for humidity.

    Returns
    -------
    degradation
        cumulative degradation experienced.

    """
    Ea = convert_units(Ea, 'eV', 'J')
    kB = 1.381e-23

    degradation = np.sum(
        UV_irradiance ** p * \
        np.exp(-Ea / (kB * (temperature + 273.15))) * \
        (relative_humidity / 100) ** n)

    return degradation


def water_saturation_vapor_pressure(temperature_in_C):
    """

    Calculate saturation water vapor pressure from temperature in celsius.
    Pressure returned in units of pascal.


    Parameters
    ----------
    temperature_in_C
        Temperature in celsius.
    Returns
    -------
    Pws
        Saturation water vapor pressure in Pascal
    """
    T = temperature_in_C + 273.15

    Tc = 647.096
    Pc = 22064000
    v = 1 - T/Tc
    C1 = -7.85951783
    C2 = 1.84408259
    C3 = -11.7866497
    C4 = 22.6807411
    C5 = -15.9618719
    C6 = 1.80122502
    Pws = Pc*np.exp( Tc/T*(C1*v + C2*v**1.5 + C3*v**3 + C4*v**3.5 + C5*v**4 + \
                           C6*v**7.5 ))
    return Pws


def calculate_zone_by_threshold(stressor, threshold):
    """

    Parameters
    ----------
    stressor
        Array of climate stressor values
    threshold
        list of threshold values for defining zones.
    Returns
    -------
    zone
        Climate zone

    """
    zone = np.zeros(stressor.shape,dtype=np.int8)
    zone[stressor<threshold[0]] = 1
    for k in range(len(threshold)):
        zone[stressor>threshold[k]] = k+2

    return zone


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance in km between two locations.

    Parameters
    ----------
    lat1 : numeric
        latitude of first point, in degrees.
    lon1 : numeric
        longitude of first point, in degrees.
    lat2 : numeric
        latitude of second point, in degrees.
    lon2 : numeric
        longitude of second point, in degrees.

    Returns
    -------
    numeric: Haversine distance in km.

    """
    p = 0.017453292519943295
    a = 0.5 - np.cos((lat2-lat1)*p)/2 + np.cos(lat1*p)*np.cos(lat2*p) * (1-np.cos((lon2-lon1)*p)) / 2
    return 12742 * np.arcsin(np.sqrt(a))


def arg_closest_point(lat_point, lon_point, lat_list, lon_list):
    """
    Calculate the index of the closest point in the list of coordinates (
    lat_list, lon_list) to the point (lat_point, lon_point). Uses Haversine
    distance formula to calculate the distance.

    Parameters
    ----------
    lat_point : numeric
        latitude of point to search for, in degrees
    lon_point : numeric
        longitude of point to search for, in degrees.
    lat_list : pandas series
        list of latitudes to search within, in degrees.
    lon_list : pandas series
        list of longitudes to search within, in degrees. Must be the same size
        as lat_list

    Returns
    -------
        numeric : distance
    """
    lat_list = lat_list.astype(np.float64)
    lon_list = lon_list.astype(np.float64)
    
    return np.argmin(
        haversine_distance(
            np.array(lat_list), 
            np.array(lon_list),
            lat_point, lon_point)
    )


