import numpy as np
import sklearn.metrics as metrics
# Code and examples for Kernel Target Alignments (Christianini et al, NIPS 2001 and JMLR 2002).
# Author: Nico Goernitz, TU Berlin, 2016

def normalize_kernel(K):
    # A kernel K is normalized, iff K_ii = 1 \forall i
    N = K.shape[0]
    a = np.sqrt(np.diag(K)).reshape((N, 1))
    if any(np.isnan(a)) or any(np.isinf(a)) or any(np.abs(a) <= 1e-16):
        print 'Numerical instabilities.'
        C = np.eye(N)
    else:
        b = 1. / a
        C =  b.dot(b.T)
    return K * C


def center_kernel(K):
    # Mean free in feature space
    N = K.shape[0]
    a = np.ones((N, N)) / np.float(N)
    return K - a.dot(K) - K.dot(a) + a.dot(K.dot(a))


def kta_align_general(K1, K2):
    # Computes the (empirical) alignment of two kernels K1 and K2

    # Definition 1: (Empirical) Alignment
    #   a = <K1, K2>_Frob
    #   b = sqrt( <K1, K1> <K2, K2>)
    #   kta = a / b
    # with <A, B>_Frob = sum_ij A_ij B_ij = tr(AB')
    return K1.dot(K2.T).trace() / np.sqrt(K1.dot(K1.T).trace() * K2.dot(K2.T).trace())


def kta_align_binary(K, y):
    # Computes the (empirical) alignment of kernel K1 and
    # a corresponding binary label  vector y \in \{+1, -1\}^m

    m = np.float(y.size)
    YY = y.reshape((m, 1)).dot(y.reshape((1, m)))
    return K.dot(YY).trace() / (m * np.sqrt(K.dot(K.T).trace()))


def load_data(fname):
    import time
    t = time.time()
    data = np.load(fname)
    print '-- Version:', data['version']

    wdata = data['wdata']
    wdata_cols = data['wdata_cols']
    print '-- WDATA (stations x dates x measures):'
    print wdata.shape
    print wdata_cols

    dates = data['dates']
    dates_cols = data['dates_cols']
    print '-- DATES (day x month x year corresponding to wdata):'
    print dates.shape
    print dates_cols

    stations = data['stations']
    stations_cols = data['stations_cols']
    print '-- STATIONS (id, location for all stations corresponding to wdata):'
    print stations.shape
    print stations_cols

    dists = data['dists']
    dists_cols = data['dists_cols']
    print '-- DISTS (name and list of associated stations):'
    print dists.shape
    print dists_cols

    label = data['label']
    label_cols = data['label_cols']
    print '-- LABEL (for all districts and years):'
    print label.shape
    print label_cols
    print('{0} seconds for loading the dataset.'.format(time.time() - t))

    return wdata, dates, stations, dists, label


def get_task1_split_inds(dates, dists):
    import itertools
    # Returns indices of districts and years for train and test
    train_years = np.arange(1979, 2006+1)
    test_years = np.arange(2007, 2009+1)
    districts = np.arange(dists.shape[0])
    test = np.array(list(itertools.product(test_years, districts)))
    train = np.array(list(itertools.product(train_years, districts)))
    lens = np.round(0.2 * train.shape[0])
    inds = np.random.permutation(train.shape[0])[:lens]
    return train[inds, :], test


def get_task2_split_inds(dates, dists, perc=0.2):
    # Returns indices of districts and years for train and test
    import itertools
    # Returns indices of districts and years for train and test
    years = np.arange(1979, 2009+1)
    districts = np.arange(dists.shape[0])
    total = np.array(list(itertools.product(years, districts)))
    lens = np.round(perc * total.shape[0])
    inds = np.random.permutation(total.shape[0])
    return total[inds[:lens], :], total[inds[lens:], :]


def get_challenge_split_inds(year_dists_inds, perc=0.2):
    # Returns indices of districts and years for train and test
    import itertools
    # Returns indices of districts and years for train and test
    years = np.arange(1979, 2009+1)
    districts = np.unique(year_dists_inds[:, 1])
    total = np.array(list(itertools.product(years, districts)))
    lens = np.round(perc * total.shape[0])
    inds = np.random.permutation(total.shape[0])
    return total[inds[:lens], :], total[inds[lens:], :]


def split_train_test(X, year_dists_inds, y, perc=0.2, train_inds=None, test_inds=None):
    if train_inds is None or test_inds is None:
        # if not provided with training and test index lists, then assume
        # the standard task
        train_inds, test_inds = get_challenge_split_inds(year_dists_inds, perc)

    n_train = train_inds.shape[0]
    n_test = test_inds.shape[0]
    n_feats = X.shape[1]
    print 'Number of training year-district-pairs: ', n_train
    print 'Number of test year-district-pairs: ', n_test
    print 'Number of features: ', n_feats
    print 'Total number of data points: ', X.shape[0]

    inds1 = []
    for (i, j) in train_inds:
        inds1.extend(np.where((year_dists_inds[:, 0] == i) & (year_dists_inds[:, 1] == j))[0].tolist())
    inds2 = []
    for (i, j) in test_inds:
        inds2.extend(np.where((year_dists_inds[:, 0] == i) & (year_dists_inds[:, 1] == j))[0].tolist())

    print 'Number of train inds found: ', len(inds1)
    print 'Number of test inds found: ', len(inds2)
    print 'Total number of inds: ', len(inds1) + len(inds2)
    print 'Intersecting indices: ', np.intersect1d(inds1, inds2)
    assert len(inds1) + len(inds2) <= X.shape[0]

    X_train = X[inds1, :]
    year_dists_inds_train = year_dists_inds[inds1, :]
    y_train = y[inds1]

    X_test = X[inds2, :]
    year_dists_inds_test = year_dists_inds[inds2, :]
    y_test = y[inds2]
    return X_train, year_dists_inds_train, y_train, X_test, year_dists_inds_test, y_test


def get_basic_weather_station_feats(wdata, dates):
    # Return simple feats for the weather as in the challenge paper
    years = np.unique(dates[:, 2])
    X = np.zeros((wdata.shape[0], years.size, 7*5))
    for j in range(years.size):
        offset = 0
        for m in [6, 7, 8, 9, 10]:
            inds = np.where((dates[:, 2] == years[j]) & (dates[:, 1] == m))[0]
            X[:, j, offset:offset+6] = np.mean(wdata[:, inds, :], axis=1)
            X[:, j, offset+6] = np.sum(wdata[:, inds, 2] > 0.1, axis=1)
            offset += 7

    return X


def get_k_nearest_neighbors(k, X):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    for i in range(indices.shape[0]):
        assert i in indices[i, :]
    return indices


def get_k_nearest_weather_stations(k, stations):
    # Calls get_k_nearest_neighbors
    # Returns an array N \in I^{548 x k} of k nearest weather stations
    X = stations[:, 1:3]  # Longitude, Latitute, and Elevation
    return get_k_nearest_neighbors(k, X)


def evaluate(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=+1)
    return metrics.auc(fpr, tpr)


def get_exms_basic_feats(k, basic_weather_feats, stations, dates, dists, label,
                         add_year_bias=True, add_dist_bias=True, avg_neighbors=False):
    print('Generate examples based on basic weather features and k-nearest-neighbors (k={0}).'.format(k))
    X0 = basic_weather_feats  # stations x years x feats
    print 'Basic weather features size: ', X0.shape

    k_inds = get_k_nearest_weather_stations(k, stations)
    print 'Knn matrix size: ', k_inds.shape

    n_stations = X0.shape[0]
    n_feats = X0.shape[2]
    n_dists = dists.shape[0]
    years = np.unique(dates[:, 2])
    n_years = years.size
    print 'Number of weather stations: ', n_stations
    print 'Number of basic features: ', n_feats
    print 'Number of districts: ', n_dists
    print 'Number of years: ', n_years
    print 'Years: ', years

    # Outputs
    k_real = k
    if avg_neighbors:
        k_real = 1
    X = np.zeros((n_stations * n_years, n_feats * k_real + add_dist_bias*n_dists + add_year_bias*n_years))
    print 'Output feature matrix size: ', X.shape
    year_dist_inds = np.zeros((n_stations * n_years, 2), dtype=np.int)
    print 'Corresponding Year-District indices size: ', year_dist_inds.shape
    y = np.zeros((n_stations * n_years))
    print 'Corresponding label size: ', y.shape

    cnt = 0
    for i in range(n_dists):
        for sid in dists[i, 4]:
            sind = np.where(stations[:, 0] == sid)[0]
            assert sind.size == 1
            for j in range(n_years):
                y[cnt] = label[i, j]
                year_dist_inds[cnt, :] = years[j], i

                offset = 0
                if not avg_neighbors:  # stack neighbors atop each other
                    for nn in range(k):
                        X[cnt, offset:offset+n_feats] = X0[k_inds[sind, nn].flatten(), j, :]
                        offset += n_feats
                else:  # average neighbors
                    X[cnt, :n_feats] = np.mean(X0[k_inds[sind, :].flatten(), j, :], axis=0)
                    offset += n_feats

                if add_dist_bias:  # add bias for district
                    X[cnt, offset+i] = 1.
                    offset += n_dists
                if add_year_bias:  # add bias for year
                    X[cnt, offset+j] = 1.
                cnt += 1

    # convert y to -1,+1 := normal,anomaly:
    y[y==0] = -1
    print 'Convert label vector to -1/+1: ', np.unique(y)
    return X, year_dist_inds, y