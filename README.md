### About
An anomaly detection challenge for data with complex dependency 
structure.

Accompanying software package.  

For further information, please visit our challenge website at
http://nicococo.github.io/CropYieldLoss

### Data
There are three states from India: Bihar, MP (Madhya Pradesh), UP (Uttar Pradesh)
Two sets of data from different sources: 

- Yield data per year per district 
Source: ICRISAT http://vdsa.icrisat.ac.in

- Daily weather data (max temperature, min temperature, percipitation, wind, relative humidity, solar) 
from different weather grid points. Weather data is available for several spatial locations within 
each district.  
Source: http://globalweather.tamu.edu

Processed data can be aquired at http://...
and should contain the following information:
 
- WDATA (548, 11284, 6)
['Max Temperature' 'Min Temperature' 'Precipitation' 'Wind' 'Relative Humidity' 'Solar']
- DATES (11284, 3)
['Day' 'Month' 'Year']
- STATIONS (548, 4)
['Id' 'Longitude' 'Latitude' 'Elevation']
- DISTS (93, 5)
['DistCode' 'DistName' 'StateCode' 'StateName' 'StationIds']
- LABEL (93, 31)
[1979 - 2009]

### Installation
You can conveniently install the software package using:
pip install git+https://github.com/nicococo/CropYieldLoss.git

### Basic Example
The most basic example for loading, preparing, training, and testing
is given in `ex1_simple.py`:

`wdata, dates, stations, dists, label = load_data('cylad1.npz')`
`F = get_basic_weather_station_feats(wdata, dates)`
`X, inds, y = get_exms_basic_feats(3, F, stations, dates, dists, label)`
`X_train, inds_train, y_train, X_test, inds_test, y_test = split_train_test(X, inds, y, perc=0.2)`
`mySvm = svm.LinearSVC(C=1., fit_intercept=False, class_weight='balanced')`
`mySvm.fit(X_train, y_train)`
`fpr, tpr, thresholds = metrics.roc_curve(y_test, mySvm.decision_function(X_test), pos_label=+1)`
`print 'AUC: ', metrics.auc(fpr, tpr)`

Note: This example assumes that the processed data 'cylad1.npz' (see website where to 
download) is stored in your current path.

### References
If you use results, software, or data from this challenge in your
own research, please cite our paper: __to appear...__