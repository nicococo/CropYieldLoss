import pandas as pd
import numpy as np
import csv
import sys
import datetime

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

STATES = ['Bihar','MP','UP']
PATH_WEATHER = 'weather'
PATH_YIELD = 'yield'

WEATHER_YEARS = [1979, 2014]
YIELD_YEARS = [1966, 2009]
YEARS_INC = [1979, 2009]


def get_districts_from_yield():
    dists = list()
    for s in STATES:
        xls = pd.ExcelFile(PATH_YIELD+'/'+s+'/dt-area-prod-a.xls')
        table = xls.parse(0, index_col=None, na_values=['NA'])
        print table.columns
        ds = table['DIST'].unique().tolist()
        for didx in ds:
            ind = np.where(table['DIST'] == didx)[0][0]
            dname = table['DISTNAME'][ind]
            sidx = table['STCODE'][ind]
            sname = table['STNAME'][ind]
            print [didx, dname, sidx, sname]
            dists.append([didx, dname, sidx, sname])
    print str(len(dists)) + ' districts found.'
    return dists

def get_districts_from_weather():
    dists = list()
    for s in STATES:
        csv = pd.read_csv(PATH_WEATHER+'/'+s+'-stations-districts.csv', delimiter=';')
        print csv.columns
        ds = csv['DIST'].unique().tolist()
        for didx in ds:
            ind = np.where(csv['DIST'] == didx)[0][0]
            dname = csv['DISTNAME'][ind]
            sidx = csv['STCODE'][ind]
            sname = csv['STNAME'][ind]
            print [didx, dname, sidx, sname]
            dists.append([didx, dname, sidx, sname])
    print str(len(dists)) + ' districts found.'
    return dists

def get_weather(district_id):
    weather = {}
    station_ids = list()
    col_names = None
    for s in STATES:
        csv = pd.read_csv(PATH_WEATHER+'/'+s+'-stations-districts.csv', delimiter=';')
        if district_id in csv['DIST'].unique():
            inds = np.where(csv['DIST']==district_id)[0]
            station_ids.extend(csv['STATION'].iget(inds).unique())
            for i in station_ids:
                csv = pd.read_csv(PATH_WEATHER+'/'+s+'/weatherdata-'+str(i)+'.csv', delimiter=',', warn_bad_lines=True, parse_dates=False)
                col_names = csv.columns.values.tolist()[:-1]
                weather[i] = csv.as_matrix()[:,:-1]
            break
    print col_names
    return weather, station_ids, col_names

def get_yield(district_id):
    yields = None
    years = None
    col_names = None
    for s in STATES:
        xls = pd.ExcelFile(PATH_YIELD+'/'+s+'/dt-area-prod-a.xls')
        table = xls.parse(0, index_col=None, na_values=['NA'])
        col_names = table.columns.values.tolist()[5:]
        inds = np.where(district_id==table['DIST'])[0]
        if inds.size > 0:
            # print table['YEAR'][inds]
            # print table.loc[inds]
            years = table['YEAR'][inds].as_matrix()
            yields = table.loc[inds].as_matrix()[:,5:]
            break
    print col_names
    return yields, years, col_names

def generate_full_dataset():
    dists = get_districts_from_weather()
    cnt = len(dists)
    wdata = {}
    for d in dists:
        (wd, sid, wdata_col_name) = get_weather(d[0])
        wdata.update(wd)
        (ydata, years, yield_col_names) = get_yield(d[0])
        d.append(sid)
        d.append(years)
        d.append(ydata)
        print cnt
        cnt -= 1
    np.savez_compressed('data.npz', wdata=wdata, dists=dists, yield_col_names=yield_col_names, wdata_col_name=wdata_col_name)

def ols(vecX, vecy):
    # solve the ols regression with a single feature
    vecX = np.hstack((vecX[:,np.newaxis], np.ones((vecX.size, 1))))  #  datapoints x 2
    XXt = vecX.T.dot(vecX) 
    XtY = vecX.T.dot(vecy)
    w = np.linalg.inv(XXt).dot(XtY)
    y_pred = w.dot(vecX.T)
    return (vecy-y_pred)*(vecy-y_pred), y_pred > vecy

def calc_anom_threshold(se):
    # assume se is sorted (se_1 <= se_2 <= ... <= se_n)
    dse = se[1:] - se[:-1]
    ind = np.argmax(dse)
    cutoff = se[ind]
    return cutoff, dse

def generate_anomalies(plot=True):
    data = np.load('data.npz')
    dists = data['dists']
    yield_col_names = data['yield_col_names']
    print len(dists)
    print yield_col_names
    print yield_col_names[[0,1,12,13]]

    # weather data goes from 1979 - 2014
    # yield data starts from 1966 - 2009 (confirmed for all districts)

    # count of missing and total values
    cnt_missing = 0
    cnt_total = 0

    if plot:
        plt.figure(1)
        plt.title('Detrending: Sorted Squared Errors for each District (Time vs. Yield)')

    idx = 1
    total_data = 0
    total_anoms = 0
    time_anoms = np.zeros(len(dists[0][5]))
    dist_anoms = np.zeros(len(dists))

    lbl_dists = []
    for d in dists:
        # print d
        years = d[5]
        
        # cnt_missing += len(np.where(d[6][:, 1] <= 0.0)[0]) + len(np.where(d[6][:, 0] <= 0.0)[0]) + len(np.where(d[6][:, 13] <= 0.0)[0]) + len(np.where(d[6][:, 12] <= 0.0)[0])
        cnt_missing += len(np.where(d[6][:, 1] < 0.0)[0]) + len(np.where(d[6][:, 0] < 0.0)[0]) + len(np.where(d[6][:, 13] < 0.0)[0]) + len(np.where(d[6][:, 12] < 0.0)[0])
        cnt_total += 4.0*len(d[6][:, 1])

        yields = (d[6][:, 1]+d[6][:, 13]) / (d[6][:, 0]+d[6][:, 12])  # rice + maiz

        se, flag_anom = ols(years, yields.T)
        se_bak = np.array(se)        

        se = se[flag_anom]
        inds = np.argsort(se)
        cutoff, dse = calc_anom_threshold(se[inds])

        ainds = np.where((se_bak>=cutoff) & (flag_anom==True))[0]
        lbl = np.zeros(se_bak.size)
        lbl[ainds] = 1.0
        lbl_dists.append(lbl)
        
        time_anoms[np.array(years[ainds]-1966., dtype='i')] += 1
        dist_anoms[idx-1] = len(ainds)
        total_anoms += len(ainds)
        total_data += se_bak.size
        idx += 1

        if plot:
            plt.subplot(10, 10, idx)
            plt.plot(range(se.size), se[inds], '.-r')
            plt.plot([0, se.size], [cutoff, cutoff], '-b')
            plt.xticks([0, se.size+1], [])
            plt.yticks([0, max(1.1*se)], [])

    print '\nTotal num of datapoints, Number of anomalies, Fraction'
    print (total_data, total_anoms, float(total_anoms)/float(total_data))
    print '\nNum of neg. values encountered, Total number of values, Fraction of missing values'
    print (cnt_missing, cnt_total, 100.0*float(cnt_missing)/float(cnt_total))

    if plot:
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.bar(range(len(dists[0][5])), time_anoms)
        plt.xticks(range(time_anoms.size), years)
        plt.ylabel('Total Number of Anomalies')
        plt.xlabel('Year')

        plt.subplot(2, 1, 2)
        plt.bar(range(len(dists)), dist_anoms)
        plt.ylabel('Total Number of Anomalies')
        plt.xlabel('District')
        plt.show()

    return lbl_dists, years

def cut_weather_data():
    data = np.load('data.npz')
    dists = data['dists']
    wdata = data['wdata']

    used_stations = []
    for d in dists:
        used_stations.extend(d[4])
    used_stations = np.unique(used_stations)
    print '\nThere are {0} weather stations in total.'.format(len(used_stations))

    cnt = 0
    cnt_skip = 0

    stations_cnames = ['Id', 'Longitude', 'Latitude', 'Elevation']
    stations = list()
    data_cnames = ['Max Temperature', 'Min Temperature', 'Precipitation', 'Wind', 'Relative Humidity', 'Solar']
    data = list()
    dates_cnames = ['Day', 'Month', 'Year']
    dates = list()

    lens = list()
    flag = 0
    wdict = wdata[()]
    for wid in wdict.keys():
        entry = wdict[wid]
        stations.append([wid, entry[0, 1], entry[0, 2], entry[0, 3]])

        inds = []
        for i in range(entry.shape[0]):
            date = datetime.datetime.strptime(entry[i, 0], '%m/%d/%Y')
            if date.year>=1979 and date.year<=2009:
                inds.append(i)  # save index
                if not flag:
                    dates.append([date.day, date.month, date.year])
                cnt += 1
            else:
                cnt_skip += 1
        # skip non-used lines
        entry = entry[inds, 4:]
        print entry.shape
        if not entry.shape[0] in lens:
            lens.append(entry.shape[0])
        data.append(entry)
        flag = 1

    dates = np.array(dates, dtype='i')
    print dates.shape
    print lens
    print 'Total number of weather measurement: {0}'.format(cnt)
    print 'Total number of skipped weather measurement: {0}'.format(cnt_skip)

    return data, data_cnames, stations, stations_cnames, dates, dates_cnames


def cut_dists_data():
    data = np.load('data.npz')
    dists = data['dists']

    d = []
    for i in range(len(dists)):
        d.append(dists[i][:5])

    print d[:3]
    dists_cnames = ['DistCode', 'DistName', 'StateCode', 'StateName']
    return d, dists_cnames


# generate anomalies from yield data
(lbl, years) = generate_anomalies(plot=False)

# cut labels between years 1979 - 2009
cnt_all = 0
cnt_anom = 0
for i in range(len(lbl)):
    lbl[i] = lbl[i][13:]
    cnt_all += len(lbl[i])
    cnt_anom += len(np.where(lbl[i]==1.0)[0])

# cut years
years = years[13:]
print len(lbl[0])

# convert to array
lbl = np.array(lbl).shape
print cnt_all
print cnt_anom
print float(cnt_anom)/float(cnt_all)

(data, data_cnames, stations, stations_cnames, dates, dates_cnames) = cut_weather_data()
(dists, dists_cnames) = cut_dists_data()

np.savez_compressed('processed_data.npz', wdata=data, wdata_cols=data_cnames, dates=dates, dates_cols=dates_cnames, stations=stations, stations_cols=stations_cnames, dists=dists, dists_cols=dists_cnames, label=lbl, label_cols=years)
