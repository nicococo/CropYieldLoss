import numpy as np

from feature_construction_utils import *


if __name__ == "__main__":
    DATA_FILE = '../../Irene_ECML_data/raw/processed_data.npz'
    data = np.load(DATA_FILE)

    wdata=data['wdata']
    wdata_cols=data['wdata_cols']
    print '- WDATA:'
    print wdata.shape
    print wdata_cols

    dates=data['dates']
    dates_cols=data['dates_cols']
    print '- DATES:'
    print dates.shape
    print dates_cols

    stations=data['stations']
    stations_cols=data['stations_cols']
    print '- STATIONS:'
    print stations.shape
    print stations_cols

    dists=data['dists']
    dists_cols=data['dists_cols']
    print '- DISTS:'
    print dists.shape
    print dists_cols

    label=data['label']
    label_cols=data['label_cols']
    print '- LABEL:'
    print label.shape
    print label_cols
