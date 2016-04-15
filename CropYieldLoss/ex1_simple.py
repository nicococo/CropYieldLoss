import sklearn.svm as svm
import sklearn.metrics as metrics
from feature_construction_utils import *

# 1. load the processed challenged data (download see website)
wdata, dates, stations, dists, label = load_data('cylad1.npz')
# 2. generate 7 basic features per weather station per year from wdata
F = get_basic_weather_station_feats(wdata, dates)
# 3. generate the final dataset based on the basic features. Stack the '3' nearest neighbors atop, add a bias
#    for each year and a bias for each district
X, inds, y = get_exms_basic_feats(3, F, stations, dates, dists, label,
                                  add_dist_bias=True, add_year_bias=True, avg_neighbors=False)
# 4. get a split for training/test
X_train, inds_train, y_train, X_test, inds_test, y_test = split_train_test(X, inds, y, perc=0.2)
# 5. train a linear support vector machine (without intercept, balance the classes)
mySvm = svm.LinearSVC(C=1., fit_intercept=False, class_weight='balanced')
mySvm.fit(X_train, y_train)
# 6. evaluate the test set and report the area under the ROC curve
fpr, tpr, thresholds = metrics.roc_curve(y_test, mySvm.decision_function(X_test), pos_label=+1)
print 'AUC: ', metrics.auc(fpr, tpr)

