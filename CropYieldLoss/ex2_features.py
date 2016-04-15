import numpy as np
import sklearn.svm as svm
import sklearn.metrics as metrics
import os

from feature_construction_utils import *
from data_conversion_utils import *


if __name__ == "__main__":
    FEATURE_FNAME = 'feats.npz'
    INPUT_DATA = 'cylad1.npz'

    if not os.path.isfile(FEATURE_FNAME):
        # Loading the dataset takes a long time. It is better to process
        # the features and save the intermediate result to file which
        # can be loaded much faster.
        wdata, dates, stations, dists, label = load_data(INPUT_DATA)
        F = get_basic_weather_station_feats(wdata, dates)
        np.savez(FEATURE_FNAME, F=F, dates=dates, stations=stations, dists=dists, label=label)

    # load the processed file (will be very fast)
    print '----- Load Data -----'
    foo = np.load(FEATURE_FNAME)
    F = foo['F']
    dates = foo['dates']
    stations = foo['stations']
    dists = foo['dists']
    label = foo['label']

    reps = 25
    knns = [1, 3, 5, 9, 13, 21]
    percs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

    # final results will be stored here
    all_aucs = np.zeros((len(knns), len(percs)))
    all_stds = np.zeros((len(knns), len(percs)))
    for k in range(len(knns)):
        X, inds, y = get_exms_basic_feats(knns[k], F, stations, dates, dists, label,
                                          add_dist_bias=True, add_year_bias=True, avg_neighbors=False)
        aucs = np.zeros((reps, len(percs)))
        for r in range(reps):
            for i in range(len(percs)):
                X_train, inds_train, y_train, X_test, inds_test, y_test = split_train_test(X, inds, y, perc=percs[i])
                mySvm = svm.LinearSVC(C=1., fit_intercept=False, class_weight='balanced')
                mySvm.fit(X_train, y_train)
                preds = mySvm.decision_function(X_test)
                fpr, tpr, thresholds = metrics.roc_curve(y_test, preds, pos_label=+1)
                aucs[r, i] = metrics.auc(fpr, tpr)
                print '\n> ', knns[k], r, i, ' AUC = ', aucs[r], '\n'

        all_aucs[k, :] = np.mean(aucs, axis=0)
        all_stds[k, :] = np.std(aucs, axis=0)

    print '--------- Done. ------------------'
    print all_aucs
    print all_stds
    np.savez('ex2_results.npz', aucs=all_aucs, stds=all_stds, reps=reps, knns=knns, percs=percs)
    print('\nThe End.')
