# Photovoltaic Climate Zones (PVCZ)

This package provides the photovoltaic climate zones (PVCZ) and climate stressor data which describes the degree of environmental degradation expected on a PV module located in different locations on the world. 

# Install
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

 

 