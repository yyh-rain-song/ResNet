# initial trial
import os
import time
from copy import copy

import numpy as np
import h5py
from data_set import Data_set
import tensorflow as tf

DATAPATH='D:/1_My_File/LabWork/ST_ResNet/MyImplement/TaxiBJ'


# output: [None. 1]
def load_holiday(timeslots):
    timeslots = [bytes.decode(entry) for entry in timeslots]
    fname = os.path.join(DATAPATH, 'BJ_Holiday_13.txt')
    file = open(fname, 'r')
    holidays = file.readlines()
    holidays = [h[:8] for h in holidays]
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        entry = slot[:8]
        if entry in holidays:
            H[i] = 1
    return H[:, None]


# output: [None, 8]
def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = []
    for t in timestamps:
        t = bytes.decode(t)
        vec.append(time.strptime(t[:8], '%Y%m%d').tm_wday)
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    # ret: a list. For every timestamp, it's a 8-d vector. (the last bit =1 iff weekday)
    return np.asarray(ret)


def load_stdata(fname):
    f = h5py.File(fname, 'r')
    data = f['data'][:]
    timestamps = f['date'][:]
    f.close()
    return data, timestamps


def remove_incomplete_days(data, timestamps, T=48):
    days = []
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T < len(timestamps) and int(timestamps[i+T][8:]) == 1:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if(t[:8]) in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min()
        self.max = X.max()

    def transform(self, X):
        X = (1. * X) / (self.max - self.min)
        return X


def load_data(year):
    data_all = []
    fname = os.path.join(DATAPATH, 'BJ{}_M32x32_T30_InOut.h5'.format(year))
    print("file name: ", fname)
    data, timestamps = load_stdata(fname)
    # remove incomplete days (which doesn't have 48 timestamps)
    data, timestamps = remove_incomplete_days(data, timestamps)

    nb_flow = 2
    data = data[:, :nb_flow]
    data[data<0] = 0
    data_all.append(data)

    # min max scale (data --> [-1, 1])
    data_train = np.vstack(copy(data_all))
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]
    data_all_mmn = np.asarray(data_all_mmn[0])

    # load external feature
    week_feature = timestamp2vec(timestamps)
    holiday_feature = load_holiday(timestamps)
    extern_feature = np.hstack((week_feature, holiday_feature))
    return data_all_mmn, extern_feature, timestamps, mmn.min, mmn.max


def load_all_data():
    data, extern, timestamp, min_, max_ = load_data(13)
    data = data.tolist()
    extern = extern.tolist()
    for year in (14, 15, 16):
        da, ex, ti, mi, ma = load_data(year)
        data.extend(da)
        timestamp.extend(ti)
        extern.extend(ex)
        min_ = min(min_, mi)
        max_ = max(max_, ma)

    return np.asarray(data), np.asarray(extern), timestamp, min_, max_


data, extern, time, min_, max_ = load_all_data()
print(data.shape)