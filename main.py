from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import scipy.io as scio
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import random
from classifiers import resnet
from sklearn.metrics import accuracy_score
import time

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import savemat
import h5py
# import joblib
import pylab
import gc

time_start=time.time()

def label_smoothing(labels, factor=0.1):
    num_labels = labels.shape[1]
    labels = ((1-factor) * labels) + (factor/ num_labels)
    return labels

if True:
    # Modified in every py files  ##################################################
    classifier_name = 'inception_v9_v2'
    version = '_v001'
    data_date = '221028'
    loc_train_arr = np.array([0])
    loc_test_arr_vld = np.array([3])
    exp_seed = 1  # Choose different random seeds
    data_name1 = 'signal_arr_total605_2DMRS_wo_log_waverec_db6_6_norm_part1'
    data_name2 = 'signal_arr_total605_2DMRS_wo_log_waverec_db6_6_norm_part2'

    mat_v7_3 = False

    # For Training
    if len(loc_train_arr) > 1:
        mul_loc = True
    else:
        mul_loc = False
    dev_partial = False

    soft_label = True
    soft_factor = 0.1
    nb_epochs = 50
    batch_size = 64
    learning_rate = 0.003
    l2 = 0.1
    depth = 6
    verbose = 2

    # According to input data
    channel_num = 1
    ori_len = 1200 * channel_num
    complex_flag = False

    # For local vld ##################################################
    # local_vld = True # just validate the correctness of code on the laptop
    local_vld = False # Training on the server

    if mul_loc == True:
        loc_train_arr = np.array([0, 1, 2, 3])
        loc_test_arr_vld = np.array([1])

    if local_vld == True:
        nb_epochs = 1
        depth = 1

        train_ratio = 0.0008
        test_ratio = 0.0001
        vld_ratio = 0.0001
        train_ratio_vld = 0.0001
        test_ratio_vld  = 0.0001

        if mul_loc == False:
            data_train = np.zeros([ori_len, 3200], dtype=float)
            data_test = np.zeros([ori_len, 400], dtype=float)
            data_vld = np.zeros([ori_len, 400], dtype=float)
            data_train_vld = np.zeros([ori_len, 400], dtype=float)
            data_test_vld = np.zeros([ori_len, 400], dtype=float)
        else:
            data_train = np.zeros([ori_len, 4000], dtype=float)
            data_test = np.zeros([ori_len, 500], dtype=float)
            data_vld = np.zeros([ori_len, 500], dtype=float)
            data_train_vld = np.zeros([ori_len, 600], dtype=float)
            data_test_vld = np.zeros([ori_len, 600], dtype=float)

        if complex_flag == True:
            if mul_loc == False:
                data_train = np.zeros([ori_len, 3200], dtype=complex)
                data_test = np.zeros([ori_len, 400], dtype=complex)
                data_vld = np.zeros([ori_len, 400], dtype=complex)
                data_train_vld = np.zeros([ori_len, 400], dtype=complex)
                data_test_vld = np.zeros([ori_len, 400], dtype=complex)
            else:
                data_train = np.zeros([ori_len, 4000], dtype=complex)
                data_test = np.zeros([ori_len, 500], dtype=complex)
                data_vld = np.zeros([ori_len, 500], dtype=complex)
                data_train_vld = np.zeros([ori_len, 600], dtype=complex)
                data_test_vld = np.zeros([ori_len, 600], dtype=complex)

        root_dir = 'C:\\Code\\'  # Change according to your file location
        data_dir = 'C:\\DS1\\'  # Change according to your data location
        data_dir2 = 'C:\\DS2\\'  # Change according to your data location
    else:
        nb_epochs = 50
        depth = 6

        train_ratio = 0.8
        test_ratio = 0.1
        vld_ratio = 0.1
        train_ratio_vld = 0.001
        test_ratio_vld = 0.1

        if mul_loc == False:
            data_train = np.zeros([ori_len, 3200000], dtype=float)
            data_test = np.zeros([ori_len, 400000], dtype=float)
            data_vld = np.zeros([ori_len, 400000], dtype=float)
            data_train_vld = np.zeros([ori_len, 2000], dtype=float)
            data_test_vld = np.zeros([ori_len, 400000], dtype=float)
        else:
            data_train = np.zeros([ori_len, 4000000], dtype=float)
            data_test = np.zeros([ori_len, 500000], dtype=float)
            data_vld = np.zeros([ori_len, 500000], dtype=float)
            data_train_vld = np.zeros([ori_len, 600000], dtype=float)
            data_test_vld = np.zeros([ori_len, 600000], dtype=float)

        if complex_flag == True:
            if mul_loc == False:
                data_train = np.zeros([ori_len, 3200000], dtype=complex)
                data_test = np.zeros([ori_len, 400000], dtype=complex)
                data_vld = np.zeros([ori_len, 400000], dtype=complex)
                data_train_vld = np.zeros([ori_len, 2000], dtype=complex)
                data_test_vld = np.zeros([ori_len, 400000], dtype=complex)
            else:
                data_train = np.zeros([ori_len, 4000000], dtype=complex)
                data_test = np.zeros([ori_len, 500000], dtype=complex)
                data_vld = np.zeros([ori_len, 500000], dtype=complex)
                data_train_vld = np.zeros([ori_len, 600000], dtype=complex)
                data_test_vld = np.zeros([ori_len, 600000], dtype=complex)

        root_dir = 'C:\\Code\\'  # Change according to your file location
        data_dir = 'C:\\DS1\\'  # Change according to your data location
        data_dir2 = 'C:\\DS2\\'  # Change according to your data location

    dev_train_arr = np.array([0,1,2,3,4,5,6,7,8,9])
    dev_test_arr_vld = np.array([0,1,2,3,4,5,6,7,8,9])
    if dev_partial == True:
        dev_train_arr = np.array([0, 1, 2, 3, 5, 9])
        dev_test_arr_vld = np.array([0, 1, 2, 3, 5, 9])
    labels = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "dev8", "dev9", "dev10"]

    loc_arr = np.arange(0,4)

    len_train_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
    len_test_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
    len_vld_arr = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

    begin_flag = 1

    sample_train_num = 16000
    sample_vld_num   = 6000
    sample_train_p = 0
    sample_test_p  = 0
    sample_vld_p = 0
    for dev in dev_train_arr:
        for loc in loc_train_arr:
            if mat_v7_3 == False:
                data_dict = scio.loadmat(data_dir + '\\dev' + str(dev+1) + '\\loc' + str(loc+1) + \
                                     '\\' + data_name1 + '.mat')
                data_part1 = data_dict[data_name1]
                data_dict = scio.loadmat(data_dir + '\\dev' + str(dev + 1) + '\\loc' + str(loc + 1) + \
                                         '\\' + data_name2 + '.mat')
                data_part2 = data_dict[data_name2]
                data = np.hstack((data_part1, data_part2))
            else:
                a=1

            index = [i for i in range(data.shape[1])]
            random.seed(exp_seed)
            random.shuffle(index)
            data = data[:, index]
            data_train_par = data[:, 0:round(data.shape[1] * train_ratio)]
            data_vld_par = data[:, round(data.shape[1] * train_ratio):round(data.shape[1] * train_ratio) + round(
                data.shape[1] * vld_ratio)]
            data_test_par = data[:, round(data.shape[1] * train_ratio + data.shape[1] * vld_ratio): round(
                data.shape[1] * train_ratio + data.shape[1] * vld_ratio + data.shape[1] * test_ratio)]

            len_train_arr[loc][dev] = data_train_par.shape[1]
            len_test_arr[loc][dev] = data_test_par.shape[1]
            len_vld_arr[loc][dev] = data_vld_par.shape[1]

            data_train[:, sample_train_p:sample_train_p + data_train_par.shape[1]] = data_train_par
            data_test[:, sample_test_p :sample_test_p + data_test_par.shape[1]] = data_test_par
            data_vld[:, sample_vld_p : sample_vld_p + data_vld_par.shape[1]] = data_vld_par
            sample_train_p = sample_train_p + data_train_par.shape[1]
            sample_test_p  = sample_test_p + data_test_par.shape[1]
            sample_vld_p = sample_vld_p + data_vld_par.shape[1]

    len_train_arr = len_train_arr[:, :]
    len_test_arr  = len_test_arr[:, :]
    len_vld_arr = len_vld_arr[:, :]
    data_train = data_train[:, 0:sample_train_p]
    data_test  = data_test[:, 0:sample_test_p]
    data_vld = data_vld[:, 0:sample_vld_p]

    time_end = time.time()
    print('time cost', (time_end - time_start)/60, 'min')

    len_train_arr_vld = np.zeros([len(loc_arr), dev_train_arr[-1]+1])
    len_test_arr_vld  = np.zeros([len(loc_arr), dev_train_arr[-1]+1])

    sample_train_vld_p = 0
    sample_test_vld_p  = 0

    begin_flag = 1
    for dev in dev_test_arr_vld:
        for loc in loc_test_arr_vld:
            if mat_v7_3 == False:
                data_dict = scio.loadmat(data_dir2 + '\\dev' + str(dev + 1) + '\\loc' + str(loc + 1) + \
                                         '\\' + data_name1 + '.mat')
                data_part1 = data_dict[data_name1]
                data_dict = scio.loadmat(data_dir2 + '\\dev' + str(dev + 1) + '\\loc' + str(loc + 1) + \
                                         '\\' + data_name2 + '.mat')
                data_part2 = data_dict[data_name2]
                data = np.hstack((data_part1, data_part2))
            else:
                a = 1

            index = [i for i in range(data.shape[1])]
            random.seed(exp_seed)
            random.shuffle(index)
            data = data[:,index]
            data_train_par_vld = data[:, 0:round(data.shape[1] * train_ratio_vld)]
            data_test_par_vld  = data[:,round(data.shape[1] * train_ratio_vld):round(data.shape[1] * train_ratio_vld) + round(data.shape[1] * test_ratio_vld)]

            len_train_arr_vld[loc][dev] = data_train_par_vld.shape[1]
            len_test_arr_vld[loc][dev] = data_test_par_vld.shape[1]
            data_train_vld[:, sample_train_vld_p:sample_train_vld_p + data_train_par_vld.shape[1]] = data_train_par_vld
            data_test_vld[:, sample_test_vld_p:sample_test_vld_p + data_test_par_vld.shape[1]] = data_test_par_vld
            sample_train_vld_p = sample_train_vld_p + data_train_par_vld.shape[1]
            sample_test_vld_p  = sample_test_vld_p + data_test_par_vld.shape[1]

    len_train_arr_vld = len_train_arr_vld[:, :]
    len_test_arr_vld = len_test_arr_vld[:, :]
    data_train_vld = data_train_vld[:, 0:sample_train_vld_p]
    data_test_vld = data_test_vld[:, 0:sample_test_vld_p]

    data_train = data_train.T
    data_vld = data_vld.T
    data_test  = data_test.T
    data_train_vld = data_train_vld.T
    data_test_vld  = data_test_vld.T
    del data, data_train_par, data_vld_par, data_test_par, data_train_par_vld, data_test_par_vld
    gc.collect()

    if complex_flag == False:
        framelen = int(data_train.shape[1]/channel_num)
    else:
        framelen = int(data_train.shape[1])

    if channel_num == 2:
        data_train = np.squeeze(np.array([[np.real(data_train[:, 0:framelen])], [np.imag(data_train[:, 0:framelen])]]))
        data_train = np.transpose(data_train,(1,2,0))
        data_vld = np.squeeze(np.array([[np.real(data_vld[:, 0:framelen])], [np.imag(data_vld[:, 0:framelen])]]))
        data_vld = np.transpose(data_vld, (1, 2, 0))
        data_test = np.squeeze(np.array([[np.real(data_test[:, 0:framelen])], [np.imag(data_test[:, 0:framelen])]]))
        data_test = np.transpose(data_test,(1,2,0))
        data_train_vld = np.squeeze(np.array([[np.real(data_train_vld[:, 0:framelen])], [np.imag(data_train_vld[:, 0:framelen])]]))
        data_train_vld = np.transpose(data_train_vld,(1,2,0))
        data_test_vld = np.squeeze(np.array([[np.real(data_test_vld[:, 0:framelen])], [np.imag(data_test_vld[:, 0:framelen])]]))
        data_test_vld = np.transpose(data_test_vld,(1,2,0))
    elif channel_num == 3:
        data_train = np.squeeze(np.array([[data_train[:, 0:framelen]], [data_train[:, framelen:2 * framelen]],[data_train[:, 2 * framelen:3 * framelen]]]))
        data_train = np.transpose(data_train, (1, 2, 0))
        data_vld = np.squeeze(np.array([[data_vld[:, 0:framelen]], [data_vld[:, framelen:2 * framelen]],[data_vld[:, 2 * framelen:3 * framelen]]]))
        data_vld = np.transpose(data_vld, (1, 2, 0))
        data_test = np.squeeze(np.array([[data_test[:, 0:framelen]], [data_test[:, framelen:2 * framelen]],[data_test[:, 2 * framelen:3 * framelen]]]))
        data_test = np.transpose(data_test, (1, 2, 0))
        data_train_vld = np.squeeze(np.array([[data_train_vld[:, 0:framelen]], [data_train_vld[:, framelen:2 * framelen]],[data_train_vld[:, 2 * framelen:3 * framelen]]]))
        data_train_vld = np.transpose(data_train_vld, (1, 2, 0))
        data_test_vld = np.squeeze(np.array([[data_test_vld[:, 0:framelen]], [data_test_vld[:, framelen:2 * framelen]],[data_test_vld[:, 2 * framelen:3 * framelen]]]))
        data_test_vld = np.transpose(data_test_vld, (1, 2, 0))
    elif channel_num == 4:
        data_train = np.squeeze(np.array([[data_train[:, 0:framelen]], [data_train[:, framelen:2 * framelen]],[data_train[:, 2 * framelen:3 * framelen]],[data_train[:, 3 * framelen:4 * framelen]]]))
        data_train = np.transpose(data_train, (1, 2, 0))
        data_vld = np.squeeze(np.array([[data_vld[:, 0:framelen]], [data_vld[:, framelen:2 * framelen]], [data_vld[:, 2 * framelen:3 * framelen]],[data_vld[:, 3 * framelen:4 * framelen]]]))
        data_vld = np.transpose(data_vld, (1, 2, 0))
        data_test = np.squeeze(np.array([[data_test[:, 0:framelen]], [data_test[:, framelen:2 * framelen]],[data_test[:, 2 * framelen:3 * framelen]],[data_test[:, 3 * framelen:4 * framelen]]]))
        data_test = np.transpose(data_test, (1, 2, 0))
        data_train_vld = np.squeeze(np.array([[data_train_vld[:, 0:framelen]], [data_train_vld[:, framelen:2 * framelen]],[data_train_vld[:, 2 * framelen:3 * framelen]], [data_train_vld[:, 3 * framelen:4 * framelen]]]))
        data_train_vld = np.transpose(data_train_vld, (1, 2, 0))
        data_test_vld = np.squeeze(np.array([[data_test_vld[:, 0:framelen]], [data_test_vld[:, framelen:2 * framelen]],[data_test_vld[:, 2 * framelen:3 * framelen]],[data_test_vld[:, 3 * framelen:4 * framelen]]]))
        data_test_vld = np.transpose(data_test_vld, (1, 2, 0))

    len_train = np.sum(len_train_arr, axis=0)
    len_vld = np.sum(len_vld_arr, axis=0)
    len_test  = np.sum(len_test_arr, axis=0)
    len_train_vld = np.sum(len_train_arr_vld, axis=0)
    len_test_vld = np.sum(len_test_arr_vld, axis=0)

    len_train = len_train[len_train != 0]
    len_vld = len_vld[len_vld != 0]
    len_test = len_test[len_test != 0]
    len_train_vld = len_train_vld[len_train_vld != 0]
    len_test_vld = len_test_vld[len_test_vld != 0]

    label = np.arange(0, len(dev_train_arr))
    len_train = len_train.astype(int)
    len_vld = len_vld.astype(int)
    len_test  = len_test.astype(int)
    len_train_vld = len_train_vld.astype(int)
    len_test_vld  = len_test_vld.astype(int)

    y_train = np.repeat(label, len_train)
    y_train.shape = (len(y_train),1)
    y_vld = np.repeat(label, len_vld)
    y_vld.shape = (len(y_vld), 1)
    y_test = np.repeat(label, len_test)
    y_test.shape = (len(y_test), 1)
    y_train_vld = np.repeat(label, len_train_vld)
    y_train_vld.shape = (len(y_train_vld),1)
    y_test_vld = np.repeat(label, len_test_vld)
    y_test_vld.shape = (len(y_test_vld), 1)

    if len(data_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        data_train = data_train.reshape((data_train.shape[0], data_train.shape[1], 1))
        data_test = data_test.reshape((data_test.shape[0], data_test.shape[1], 1))
        data_train_vld = data_train_vld.reshape((data_train_vld.shape[0], data_train_vld.shape[1], 1))
        data_test_vld = data_test_vld.reshape((data_test_vld.shape[0], data_test_vld.shape[1], 1))

    output_directory = root_dir + '\\results\\' + classifier_name + version + '\\'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    if False:
    # if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        create_directory(output_directory)
        if classifier_name == 'inception_v9_v2':
            from classifiers import inception_v9_v2

        input_shape = data_train.shape[1:]
        nb_classes = len(dev_train_arr)
        # verbose = False
        # transform the labels from integers to one hot vectors
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
        y_train_arr = y_train
        y_vld_arr = y_vld
        y_test_arr = y_test
        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_vld = enc.transform(y_vld.reshape(-1, 1)).toarray()
        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        if soft_label == True:
            y_train = label_smoothing(y_train, soft_factor)

        enc.fit(np.concatenate((y_train_vld, y_test_vld), axis=0).reshape(-1, 1))
        y_train_vld_arr = y_train_vld
        y_test_vld_arr  = y_test_vld
        y_train_vld = enc.transform(y_train_vld.reshape(-1, 1)).toarray()
        y_test_vld = enc.transform(y_test_vld.reshape(-1, 1)).toarray()

        # save orignal y because later we will use binary
        y_vld_true = np.argmax(y_vld, axis=1)
        y_test_true = np.argmax(y_test, axis=1)
        y_train_vld_true = np.argmax(y_train_vld, axis=1)
        y_test_vld_true = np.argmax(y_test_vld, axis=1)

        model = inception_v9_v2.Classifier_INCEPTION_V9(output_directory, input_shape, nb_classes, verbose=verbose, nb_epochs=nb_epochs, batch_size = batch_size, depth=depth, lr = learning_rate, l2=l2)
        gc.collect()
        print('#############Start the training process#############')
        history = model.fit(data_train, y_train, data_vld, y_vld, y_vld_true)

        y_predict_test_acc = model.predict(data_test, y_test_true, data_train, y_train, y_test)
        # y_predict_train_vld_acc = model.predict(data_train_vld, y_train_vld_true, data_train, y_train, y_train_vld)
        y_predict_vld_acc = model.predict(data_vld, y_vld_true, data_train, y_train, y_vld)
        y_predict_test_vld_acc = model.predict(data_test_vld, y_test_vld_true, data_train, y_train, y_test_vld)
        print('y_predict_test:\n', y_predict_test_acc)
        print('y_predict_vld_acc:\n', y_predict_vld_acc)
        print('y_predict_test_vld:\n', y_predict_test_vld_acc)

        y_predict_test = model.predict(data_test, y_test_true, data_train, y_train, y_test, return_df_metrics=False)
        y_predict_train_vld = model.predict(data_train_vld, y_train_vld_true, data_train, y_train, y_train_vld,
                                            return_df_metrics=False)
        y_predict_test_vld = model.predict(data_test_vld, y_test_vld_true, data_train, y_train, y_test_vld,
                                           return_df_metrics=False)

        y_predict_test_arr = np.argmax(y_predict_test, axis=1)
        y_predict_train_vld_arr = np.argmax(y_predict_train_vld, axis=1)
        y_predict_test_vld_arr = np.argmax(y_predict_test_vld, axis=1)

        # plt.ion()
        cm = confusion_matrix(y_test_arr, y_predict_test_arr)
        # plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_test_arr')
        plt.savefig(output_directory + 'y_test_arr_cm.jpg')
        mdict = {'y_test_arr': cm}
        savemat(output_directory + 'y_test_arr.mat', mdict)

        cm = confusion_matrix(y_test_arr, y_predict_test_arr)
        cm = cm/cm.astype(np.float64).sum(axis=1,keepdims=True)
        # plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_test_arr_per')
        plt.savefig(output_directory + 'y_test_arr_cm_per.jpg')
        mdict = {'y_test_arr_per': cm}
        savemat(output_directory + 'y_test_arr_per.mat', mdict)

        cm = confusion_matrix(y_train_vld_arr, y_predict_train_vld_arr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_train_vld_arr')
        plt.savefig(output_directory + 'y_train_vld_arr_cm.jpg')
        mdict = {'y_train_vld_arr': cm}
        savemat(output_directory + 'y_train_vld_arr.mat', mdict)

        cm = confusion_matrix(y_train_vld_arr, y_predict_train_vld_arr)
        cm = cm / cm.astype(np.float64).sum(axis=1,keepdims=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_train_vld_arr_per')
        plt.savefig(output_directory + 'y_train_vld_arr_cm_per.jpg')
        mdict = {'y_train_vld_arr_per': cm}
        savemat(output_directory + 'y_train_vld_arr_per.mat', mdict)

        cm = confusion_matrix(y_test_vld_arr, y_predict_test_vld_arr)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_test_vld_arr')
        plt.savefig(output_directory + 'y_test_vld_arr_cm.jpg')
        mdict = {'y_test_vld_arr': cm}
        savemat(output_directory + 'y_test_vld_arr.mat', mdict)

        cm = confusion_matrix(y_test_vld_arr, y_predict_test_vld_arr)
        cm = cm / cm.astype(np.float64).sum(axis=1,keepdims=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('y_test_vld_arr_per')
        plt.savefig(output_directory + 'y_test_vld_arr_cm_per.jpg')
        mdict = {'y_test_vld_arr_per': cm}
        savemat(output_directory + 'y_test_vld_arr_per.mat', mdict)

        epochs = range(len(history.history['accuracy']))
        plt.figure()
        plt.plot(epochs, history.history['accuracy'], 'b', label='Training acc')
        plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation acc')
        plt.title('Traing and Validation accuracy')
        plt.legend()
        plt.savefig(output_directory + 'model_acc.jpg')
        mdict = {'accuracy': history.history['accuracy']}
        savemat(output_directory + 'accuracy.mat', mdict)
        mdict = {'val_accuracy': history.history['val_accuracy']}
        savemat(output_directory + 'val_accuracy.mat', mdict)

        plt.figure()
        plt.plot(epochs, history.history['loss'], 'b', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'r', label='Validation val_loss')
        plt.title('Traing and Validation loss')
        plt.legend()
        plt.savefig(output_directory + 'model_loss.jpg')
        plt.ioff()
        plt.show()
        mdict = {'loss': history.history['loss']}
        savemat(output_directory + 'loss.mat', mdict)
        mdict = {'val_loss': history.history['val_loss']}
        savemat(output_directory + 'val_loss.mat', mdict)

        create_directory(output_directory + '\\DONE')
