import numpy as np
from scipy.io import loadmat, savemat
from glob import glob
from sklearn.preprocessing import LabelEncoder, scale
from natsort import natsorted

# Load up the Mocap6 dataset
def load_mocap6_dataset():
    # Taken from https://github.com/michaelchughes/mocap6dataset
    path = 'datasets/mocap6/mocap6.mat'
    mat_dict = loadmat(path)
    time_series = []
    gt_labels = []
    for seq in mat_dict['DataBySeq']:
        this_time_series = seq[0][0] #np.array of (T, 12)
        this_gt_labels = seq[0][2] #np.array of (T, 1)
        time_series.append(this_time_series)
        gt_labels.append(this_gt_labels.flatten())

    return time_series, gt_labels

# Load up the bees dataset
def load_bees_dataset():
    # Taken from
    # Learning and Inferring Motion Patterns using Parametric Segmental Switching Linear Dynamic Systems
    # Sang Min Oh, James M. Rehg, Tucker Balch, Frank Dellaert
    # IJCV 2008
    path = 'datasets/bees/data/'
    folders = glob(path + 'seq*')
    time_series = []
    gt_labels = []
    le = LabelEncoder()
    for folder in natsorted(folders):
        folder += '/btf/'
        with open(folder + 'ximage.btf') as f:
            x = np.array([float(e.rstrip()) for e in f.readlines()])
        with open(folder + 'yimage.btf') as f:
            y = np.array([float(e.rstrip()) for e in f.readlines()])
        with open(folder + 'timage.btf') as f:
            theta = np.array([float(e.rstrip()) for e in f.readlines()])
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        with open(folder + 'label0.btf') as f:
            this_gt_labels = np.array([e.rstrip() for e in f.readlines()])
        this_time_series = np.concatenate((x[:,None],y[:,None],cos_theta[:,None],sin_theta[:,None]),axis=1)
        this_time_series = scale(this_time_series)
        time_series.append(this_time_series)
        gt_labels.append(this_gt_labels)

    le.fit(np.concatenate(gt_labels))
    for i in range(len(gt_labels)):
        gt_labels[i] = le.transform(gt_labels[i])

    return time_series, gt_labels