# -*-coding:utf-8*-

import sys
import os

#os.environ['KERAS_BACKEND'] = 'tensorflow'
#import keras.backend as K
#K.set_image_dim_ordering('tf')

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)


import pdb
from keras.models import Sequential, model_from_config
from keras.layers import Concatenate
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Dense, concatenate, merge, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.models import load_model
# from seya.layers.recurrent import Bidirectional
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import gzip
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
from scipy import sparse
import pdb
from math import sqrt
from sklearn.metrics import roc_curve, auc
import subprocess as sp
import scipy.stats as stats
from seq_motifs import *
import structure_motifs
from keras import backend as K
from rnashape_structure import run_rnashape
from keras.utils import plot_model
from theano import function, config, shared, tensor
import numpy
import time
import math

from  numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

def calculate_performance(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
#    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc,sensitivity, specificity, MCC


def transfer_label_from_prob(proba):
    label = [0 if val <= 0.5 else 1 for val in proba]
    return label


def merge_seperate_network(X_train1, X_train2, Y_train):
    left_hid = 128
    right_hid = 64
    left = get_rnn_fea(X_train1, sec_num_hidden=left_hid)
    right = get_rnn_fea(X_train2, sec_num_hidden=right_hid)

    conc = Concatenate([left.shape, right.shape])
    total_hid = left_hid + right_hid
    out = Dense(total_hid, 2)(conc)
    out = Dropout(0.3)(out)
    out = Activation('softmax')(out)

    model = Model([left.shape, right.shape], out)

    print 'model.summary()'
    plot_model(model, to_file='merge_seperate_network')

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)  # 'rmsprop')

    model.fit([left.output, right.output], Y_train, batch_size=100, nb_epoch=100, verbose=0)

    return model


def read_seq(seq_file):
    degree = 3
    encoder = buildseqmapper(degree)
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seqdata = GetSeqDegree(seq.upper(),degree)
                    seq_array = embed(seqdata,encoder)
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seqdata = GetSeqDegree(seq.upper(), degree)
            seq_array = embed(seqdata, encoder)
            seq_list.append(seq_array)

    return np.array(seq_list)


def load_label_seq(seq_file):
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
                label_list.append(int(label))
    return np.array(label_list)


def read_rnashape(structure_file):
    struct_dict = {}
    with gzip.open(structure_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[:-1]
            else:
                strucure = line[:-1]
                struct_dict[name] = strucure

    return struct_dict


def run_rnastrcutre(seq):
    # print 'running rnashapes'
    seq = seq.replace('T', 'U')
    struc_en = run_rnashape(seq)
    # fw.write(struc_en + '\n')
    return struc_en


def read_structure(seq_file, path):
    degree = 1
    encoder = buildstrucmapper(degree)
    structure_list = []
    struct_exist = False
    if not os.path.exists(path + '/structure.gz'):
        fw = gzip.open(path + '/structure.gz', 'w')
    else:
        fw = None
        struct_exist = True
        struct_dict = read_rnashape(path + '/structure.gz')
        # pdb.set_trace()
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    if struct_exist:
                        structure = struct_dict[old_name[:-1]]
                        strucdata = GetStrucDegree(seq,degree,fw,structure=structure)
                        struct = embed(strucdata,encoder)
                    else:
                        fw.write(old_name)
                        strucdata = GetStrucDegree(seq, degree, fw)
                        struct = embed(strucdata, encoder)
                    structure_list.append(struct)
                old_name = name
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            if struct_exist:
                structure = struct_dict[old_name[:-1]]
                strucdata = GetStrucDegree(seq, degree, fw, structure=structure)
                struct = embed(strucdata, encoder)
            else:
                fw.write(old_name)
                strucdata = GetStrucDegree(seq, degree, fw)
                struct = embed(strucdata, encoder)
            structure_list.append(struct)
    if fw:
        fw.close()
    return np.array(structure_list),np.array(strucdata)


def read_oli_feature(seq_file):
    trids4 = get_4_trids()
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_4_nucleotide_composition(trids4, seq)
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_4_nucleotide_composition(trids4, seq)
            seq_list.append(seq_array)

    return np.array(seq_list)


def get_4_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base = len(chars)
    end = len(chars) ** 4
    for i in range(0, end):
        n = i
        ch0 = chars[n % base]
        n = n / base
        ch1 = chars[n % base]
        n = n / base
        ch2 = chars[n % base]
        n = n / base
        ch3 = chars[n % base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3)
    return nucle_com


def get_4_nucleotide_composition(tris, seq, pythoncount=True):
    # pdb.set_trace()
    seq_len = len(seq)
    seq = seq.upper().replace('T', 'U')
    tri_feature = []

    if pythoncount:
        for val in tris:
            num = seq.count(val)
            tri_feature.append(float(num) / seq_len)
    else:
        k = len(tris[0])
        tmp_fea = [0] * len(tris)
        for x in range(len(seq) + 1 - k):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                tmp_fea[ind] = tmp_fea[ind] + 1
        tri_feature = [float(val) / seq_len for val in tmp_fea]
        # pdb.set_trace()
    return tri_feature


def load_data(path, seq=True, oli=False):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if seq:
        tmp = []
        tmp.append(read_seq(os.path.join(path, 'sequence.fa.gz')))
        seq_onehot, structure = read_structure(os.path.join(path, 'sequence.fa.gz'), path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        # data["structure"] = structure

    if oli: data["oli"] = read_oli_feature(os.path.join(path, 'sequence.fa.gz'))

    data["Y"] = load_label_seq(os.path.join(path, 'sequence.fa.gz'))
    # np.loadtxt(gzip.open(os.path.join(path,
    #                            "matrix_Response.tab.gz")),
    #                            skiprows=1)
    # data["Y"] = data["Y"].reshape((len(data["Y"]), 1))

    return data


def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq


def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))


def preprocess_data(X, scaler=None, stand=False):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def buildseqmapper(degree):
    length = degree
    alphabet = ['A', 'C', 'G', 'T']
    mapper = ['']
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        # delete the original conents
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1

    code = np.eye(len(mapper), dtype=int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i, :])

    number = int(math.pow(4, degree))
    encoder['N'] = [1.0 / number] * number
    return encoder


def buildstrucmapper(degree):
    length = degree
    alphabet = ['F', 'T', 'I', 'H','M','S']
    mapper = ['']
    while length > 0:
        mapper_len = len(mapper)
        temp = mapper
        for base in range(len(temp)):
            for letter in alphabet:
                mapper.append(temp[base] + letter)
        # delete the original conents
        while mapper_len > 0:
            mapper.pop(0)
            mapper_len -= 1

        length -= 1

    code = np.eye(len(mapper), dtype=int)
    encoder = {}
    for i in range(len(mapper)):
        encoder[mapper[i]] = list(code[i, :])

    number = int(math.pow(6, degree))
    encoder['N'] = [1.0 / number] * number
    return encoder


def embed(seq, mapper):
    mat = []
    for element in seq:
        if element in mapper:
            mat.append(mapper.get(element))
        elif "N" in element:
            mat.append(mapper.get("N"))
        else:
            print ("wrong")
    return np.asarray(mat)


def GetSeqDegree(seq, degree,motif_len = 10):
    half_len = motif_len/2
    length = len(seq)
    row = (length + motif_len - degree + 1)
    seqdata = []
    for i in range (half_len):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    for i in range(length - degree + 1):
        multinucleotide = seq[i:i + degree]
        seqdata.append(multinucleotide)

    for i in range (row-half_len,row):
        multinucleotide = 'N'
        seqdata.append(multinucleotide)

    return seqdata


def GetStrucDegree(seq, degree, fw, structure=None,motif_len = 10):
    if fw is None:
        struc_en = structure
    else:
        # print 'running rnashapes'
        seq = seq.replace('T', 'U')
        struc_en = run_rnashape(seq)
        fw.write(struc_en + '\n')

    length = len(struc_en)
    strucdata = []
    half_len = motif_len / 2
    row = (length + motif_len - degree + 1)

    for i in range (half_len):
        multinucleotide = 'N'
        strucdata.append(multinucleotide)


    for i in range(length - degree + 1):
        multinucleotide = struc_en[i:i + degree]
        strucdata.append(multinucleotide)

    for i in range (row-half_len,row):
        multinucleotide = 'N'
        strucdata.append(multinucleotide)

    return strucdata


def set_2d_cnn_network(input_length,input_dim):
    nb_conv = 4
    nb_pool = 2
    # model = Sequential()
    # model.add(Convolution2D(64, nb_conv, nb_conv,
    #                       border_mode='valid',
    #                       input_shape=(1, 107, 4)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, nb_conv, nb_conv))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())

    input = Input(shape=(1, input_length, input_dim))
    out = Convolution2D(64, nb_conv, nb_conv, border_mode='valid', )(input)
    out = Activation('relu')(out)
    out = Convolution2D(64, nb_conv, nb_conv)(out)
    out = Activation('relu')(out)
    out = MaxPooling2D(pool_size=(nb_pool, nb_pool))(out)
    out = Dropout(0.25)(out)
    out = Flatten()(out)
    model = Model(input, out)

    return out, model


def set_cnn_model(N,k1,k2,m):

    input_length = 111
    input_dim = 4

    model = Sequential()
    model.add(Convolution1D(input_dim=input_dim, input_length=input_length,
                            nb_filter=N,
                            filter_length=k1,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=m))

    model.add(Dropout(0.5))

    model = Model(input, out)
    print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # pdb.set_trace()
    print ('model training')

    return model
    # return out


def get_cnn_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    nbfilter = 16

    print('configure cnn network')
    seq_model, seq_model_out = set_cnn_model(111, 4) #degree = 1
    # seq_model, seq_model_out = set_cnn_model(110, 16) #degree = 2
    # seq_model, seq_model_out = set_cnn_model(109, 64) #degree = 3

    struct_model, struct_model_out = set_cnn_model(111, 6) #degree = 1
    # struct_model, struct_model_out = set_cnn_model(110, 36) #degree = 2
    # struct_model, struct_model_out = set_cnn_model(109, 216) #degree = 3
    # pdb.set_trace()
    # model = Sequential()
    conc = Concatenate(axis=1)([seq_model_out, struct_model_out])
    # lstmmodel = LSTM(2*nbfilter)
    # out = Bidirectional(lstmmodel)(conc)
    out = LSTM(2 * nbfilter)(conc)
    out = Dropout(0.10)(out)
    out = Dense(nbfilter * 2, activation='relu')(out)
    # model = Model(inputs = conc, outputs = out)
    #
    # aa = seq_model.input
    # bb = struct_model.input
    # cc = [seq_model.input, struct_model.input]
    # model = Model([aa,cc], output = out)

    model = Model([seq_model.input, struct_model.input], out)
    print (model.output_shape)
    return model,seq_model,struct_model


def set_sae_network():
    trainFile = '/home/disk1/wangya/ideeps/RBPdata1020/03_HITSCLIP_AGO2Karginov2013a_hg19/train/2/all_rna_feature_out.txt'
    datafile = load_sae_data_file(trainFile)
    x_train = datafile["seq_feature"][0]  # format is ndarray
    # data pre-processin
    print(x_train.shape)

    encoding_dim = 2
    input_fea = Input(shape=(25,))
    encoded = Dense(64, activation='relu')(input_fea)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(25, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_fea, output=decoded)

    # construct the encoder model
    encoder = Model(input=input_fea, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    print ('start sae fit')
    autoencoder.fit(x_train, x_train,epochs=20,batch_size=50,shuffle=True)
    print ('end sae fit')

    return encoder

def get_sae_feature(datafile):
    datafile = load_sae_data_file(datafile)
    x_train = datafile["seq_feature"][0]
    encoder = set_sae_network()
    sae_feature = encoder.predict(x_train)
    return sae_feature,encoder


def set_sae_multi_inputs_model():
    nbfilter = 16
    sae_input = Input(shape=(2,))
    cnn_lstm_network,seq_model,struct_model = get_cnn_network()

    conc = Concatenate(axis=1)([sae_input, cnn_lstm_network.output])
    # conc = keras.layers.concatenate([sae_input, cnn_lstm_network.output])

    out = Dropout(0.10)(conc)
    out = Dense(nbfilter * 2, activation='relu')(out)
    set_sae_multi_inputs_model = Model([sae_input,seq_model.input,struct_model.input], out)

    return set_sae_multi_inputs_model


def get_seq_network(N,k,m,l,initializer = 'glorot_uniform'):
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print ('configure cnn network')

    model = Sequential()

    model.add(Convolution1D(input_dim=64,input_length=109,
                            nb_filter=N,
                            filter_length=k,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1,kernel_initializer=initializer))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=m))
    model.add(Dropout(0.5))

    model.add(Convolution1D(nb_filter=N,
                            filter_length=k/2,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu', ))
    model.add(MaxPooling1D(pool_length=m))
    model.add(Dropout(0.25))

    model.add(Convolution1D(nb_filter=N,
                            filter_length=k/4,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu', ))
    model.add(MaxPooling1D(pool_length=m))
    model.add(Dropout(0.25))

    model.add(Convolution1D(nb_filter=N,
                            filter_length=1,
                            border_mode="valid",
                            # activation="relu",
                            subsample_length=1))
    model.add(Activation('relu', ))
    model.add(MaxPooling1D(pool_length=1))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(l, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    # pdb.set_trace()
    print 'model training'

    # input = Input(shape=(1, 111, 4)) # degree = 1
    # # input = Input(shape=(1, 110, 16)) # degree = 2
    # # input = Input(shape=(1, 109, 64)) # degree = 3
    # out = Convolution1D(nb_filter=nbfilter, filter_length=10, border_mode="valid", subsample_length=1)(input)
    # out = Activation('relu')(out)
    # out = MaxPooling1D(pool_length=3)(out)
    # out = Dropout(0.5)(out)
    # out = Flatten()(out)
    # out = Dense(nbfilter, activation='relu')(out)
    # out = Dropout(0.25)(out)
    #
    # model = Model(input, out)
    print(model.summary())
    return model


def get_struct_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    print ('configure cnn network')
    nbfilter = 16

    # model = Sequential()
    # model.add(Convolution1D(input_dim=6,input_length=111,
    #                         nb_filter=nbfilter,
    #                         filter_length=10,
    #                         border_mode="valid",
    #                         #activation="relu",
    #                         subsample_length=1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(pool_length=3))
    #
    # model.add(Dropout(0.5))
    #
    #
    # model.add(Flatten())
    #
    # model.add(Dense(nbfilter, activation='relu'))
    #
    # model.add(Dropout(0.25))

    input = Input(shape=(1, 111, 6)) # degree = 1
    # input = Input(shape=(1, 110, 36)) # degree = 2
    # input = Input(shape=(1, 109, 216)) # degree = 3
    out = Convolution1D(nb_filter=nbfilter, filter_length=10, border_mode="valid", subsample_length=1)(input)
    out = Activation('relu')(out)
    out = MaxPooling1D(pool_length=3)(out)
    out = Dropout(0.5)(out)
    out = Flatten()(out)
    out = Dense(nbfilter, activation='relu')(out)
    out = Dropout(0.25)(out)
    model = Model(input, out)
    print(model.summary())

    return model, out


def get_rnn_fea(train, sec_num_hidden=128, num_hidden=128):
    print ('configure network for', train.shape)
    # model = Sequential()

    # model.add(Dense(num_hidden, input_shape=(train.shape[1],), activation='relu'))
    # model.add(PReLU())
    # model.add(BatchNormalization(mode=2))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_hidden, input_dim=num_hidden, activation='relu'))
    # model.add(Dense(num_hidden, input_shape=(num_hidden,), activation='relu'))
    # model.add(PReLU())
    # model.add(BatchNormalization(mode=2))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    input = Input(shape=(train.shape[1],))
    x1 = Dense(num_hidden, activation='relu')(input)
    x1 = PReLU()(x1)
    x1 = BatchNormalization(mode=2)(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(num_hidden, input_dim=num_hidden, activation='relu')(x1)
    x1 = PReLU()(x1)
    x1 = BatchNormalization(mode=2)(x1)
    x1 = Dropout(0.5)(x1)
    model = Model(input, x1)
    return x1, model


def get_structure_motif_fig(filter_weights, filter_outs, out_dir, protein, seq_targets, sample_i=0, structure=None):
    print ('plot motif fig', out_dir)
    # seqs, seq_targets = get_seq_targets(protein)
    seqs = structure
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)

        seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]

    num_filters = filter_weights.shape[0]
    filter_size = 7  # filter_weights.shape[2]

    filters_ic = []
    meme_out = structure_motifs.meme_intro('%s/filters_meme.txt' % out_dir, seqs)

    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        structure_motifs.plot_filter_heat(filter_weights[f, :, :], '%s/filter%d_heat.pdf' % (out_dir, f))

        # write possum motif file
        structure_motifs.filter_possum(filter_weights[f, :, :], 'filter%d' % f, '%s/filter%d_possum.txt' % (out_dir, f),
                                       False)

        structure_motifs.plot_filter_logo(filter_outs[:, :, f], filter_size, seqs, '%s/filter%d_logo' % (out_dir, f),
                                          maxpct_t=0.5)

        filter_pwm, nsites = structure_motifs.make_filter_pwm('%s/filter%d_logo.fa' % (out_dir, f))
        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            structure_motifs.meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


def get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i=0):
    print ('plot motif fig', out_dir)
    seqs, seq_targets = get_seq_targets(protein)
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)

        # seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]

    num_filters = filter_weights.shape[0]
    filter_size = 7  # filter_weights.shape[2]

    # pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt' % out_dir, seqs)

    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f, :, :], '%s/filter%d_heat.pdf' % (out_dir, f))

        # write possum motif file
        filter_possum(filter_weights[f, :, :], 'filter%d' % f, '%s/filter%d_possum.txt' % (out_dir, f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:, :, f], filter_size, seqs, '%s/filter%d_logo' % (out_dir, f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa' % (out_dir, f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()

    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01
    subprocess.call('tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (
    out_dir, out_dir, 'Ray2013_rbp_RNA.meme'), shell=True)

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt' % out_dir, 'Ray2013_rbp_RNA.meme')

    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt' % out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print >> table_out, '%3s  %19s  %10s  %5s  %6s  %6s' % header_cols

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f, :, :])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:, :, f]), '%s/filter%d_dens.pdf' % (out_dir, f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print >> table_out, '%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols

    table_out.close()

    #################################################################
    # global filter plots
    #################################################################
    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print (filter_outs.shape)
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf' % out_dir)


def get_seq_targets(protein):
    path = "./RBPdata1020/%s/test/2" % protein
    data = load_data(path)
    seq_targets = np.array(data['Y'])

    seqs = []
    seq = ''
    fp = gzip.open(path + '/sequence.fa.gz')
    for line in fp:
        if line[0] == '>':
            name = line[1:-1]
            if len(seq):
                seqs.append(seq)
            seq = ''
        else:
            seq = seq + line[:-1].replace('T', 'U')
            seq = seq + line[:-1].replace('t', 'u')
    if len(seq):
        seqs.append(seq)
    fp.close()

    return seqs, seq_targets


# def get_features():
#     all_weights = []
#     for layer in model.layers:
#        w = layer.get_weights()
#        all_weights.append(w)
#
#     return all_weights

def _convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])


def get_feature(model, X_batch, index):
    inputs = [K.learning_phase()] + [model.inputs[index]]
    _convout1_f = K.function(inputs, model.layers[0].layers[index].layers[1].output)
    activations = _convout1_f([0] + [X_batch[index]])

    return activations


def get_motif(model, testing, protein, y, index=0, dir1='seq_cnn/', structure=None):
    sfilter = model.layers[0].layers[index].layers[0].get_weights()
    filter_weights_old = np.transpose(sfilter[0][:, 0, :, :], (2, 1, 0))  # sfilter[0][:,0,:,:]
    print (filter_weights_old.shape)
    # pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        # normalized, scale = preprocess_data(x)
        # normalized = normalized.T
        # normalized = normalized/normalized.sum(axis=1)[:,None]
        x = x - x.mean(axis=0)
        filter_weights.append(x)

    filter_weights = np.array(filter_weights)
    # pdb.set_trace()
    filter_outs = get_feature(model, testing, index)
    # pdb.set_trace()

    # sample_i = np.array(random.sample(xrange(testing.shape[0]), 500))
    sample_i = 0

    out_dir = dir1 + protein
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:
        get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i)
    else:
        get_structure_motif_fig(filter_weights, filter_outs, out_dir, protein, y, sample_i, structure)


def run_network(model, total_hid, training, testing, y, validation, val_y, protein=None, structure=None):
    input = Input(shape=(model.output_shape[1], ))
    x = Dense(2, input_shape=(total_hid,))(input)
    x = Activation('softmax')(x)
    model = Model(input,x)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # pdb.set_trace()
    print ('model training')
    # checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=50, nb_epoch=30, verbose=0, validation_data=(validation, val_y),
              callbacks=[earlystopper])

    # pdb.set_trace()
    # get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn1/')
    # get_motif(model, testing, protein, y, index = 1, dir1 = 'structure_cnn1/', structure = structure)

    predictions = model.predict_proba(testing)[:, 1]
    return predictions, model


def run_randomforest_classifier(data, labels, test):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(data, labels)
    # pdb.set_trace()
    pred_prob = clf.predict_proba(test)[:, 1]
    return pred_prob, clf


def run_svm_classifier(data, labels, test):
    # C_range = 10.0 ** np.arange(-1, 2)
    # param_grid = dict(C=C_range.tolist())
    clf = svm.SVC(probability=True, kernel='linear')
    # grid = GridSearchCV(svr, param_grid)
    clf.fit(data, labels)

    # clf = grid.best_estimator_
    pred_prob = clf.predict_proba(test)[:, 1]
    return pred_prob, clf


def calculate_auc(net, hid, train, test, true_y, train_y, rf=False, validation=None, val_y=None, protein=None, structure=None):
    # print 'running network'
    # net: cnn network or others
    if rf:
        print ('running oli')
        # pdb.set_trace()
        predict, model = run_svm_classifier(train, train_y, test)
    else:
        predict, model = run_network(net, hid, train, test, train_y, validation, val_y, protein=protein,
                                     structure=None)

    auc = roc_auc_score(true_y, predict)

    print ("Test AUC: ", auc)
    return auc, predict


def run_seq_struct_cnn_network(protein, seq=True, fw=None, oli=False, min_len=301):
    training_data = load_data("./RBPdata1020/%s/train/1" % protein, seq=seq, oli=oli)

    seq_hid = 16
    struct_hid = 16
    # pdb.set_trace()
    train_Y = training_data["Y"]
    print (len(train_Y))
    # pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    # pdb.set_trace()
    if seq:
        cnn_train = []
        cnn_validation = []
        seq_data = training_data["seq"][0]
        # pdb.set_trace()
        seq_train = seq_data[training_indice]
        seq_validation = seq_data[validation_indice]
        struct_data = training_data["seq"][1]
        struct_train = struct_data[training_indice]
        struct_validation = struct_data[validation_indice]
        cnn_train.append(seq_train)
        cnn_train.append(struct_train)
        cnn_validation.append(seq_validation)
        cnn_validation.append(struct_validation)
        seq_net = get_cnn_network()
        seq_data = []

    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder=encoder)

    training_data.clear()

    rf = False

    test_data = load_data("./RBPdata1020/%s/test/1" % protein, seq=seq, oli=oli)
    print (len(test_data))
    true_y = test_data["Y"].copy()

    print ('predicting')
    if seq:
        testing = test_data["seq"]
#        structure = test_data["structure"]
        seq_auc, seq_predict = calculate_auc(seq_net, seq_hid + struct_hid, cnn_train, testing, true_y, y,
                                             validation=cnn_validation, val_y=val_y, protein=protein, rf=rf,
                                             structure=None)
        seq_train = []
        seq_test = []

    print (str(seq_auc))
    fw.write(str(seq_auc) + '\n')

    mylabel = "\t".join(map(str, true_y))
    myprob = "\t".join(map(str, seq_predict))
    fw.write(mylabel + '\n')
    fw.write(myprob + '\n')


def split_training_validation(classes, validation_size=0.2, shuffle=False):
    """split samples based on balance classes"""
    num_samples = len(classes)
    classes = np.array(classes)
    classes_unique = np.unique(classes)
    num_classes = len(classes_unique)
    indices = np.arange(num_samples)
    # indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl = indices[classes == cl]
        num_samples_cl = len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual
        num_samples_each_split = int(num_samples_cl * validation_size)
        res = num_samples_cl - num_samples_each_split

        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res

        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl] * num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]

    return training_indice, training_label, validation_indice, validation_label


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)


def read_protein_name(filename='proteinnames'):
    protein_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key_name = values[0][1:-1]
            protein_dict[key_name] = values[1]
    return protein_dict


def read_result_file(filename='result_file_seq_whole_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        # protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            if index % 3 == 0:
                protein = values[0].split('_')[0]
            if index % 3 != 0:
                results.setdefault(protein, []).append(values)

            index = index + 1

    return results


def read_individual_auc(filename='result_file_all_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        # protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1:-1]

    return results


def read_ideep_auc(filename='result_mix_auc_new'):
    results = {}
    with open(filename, 'r') as fp:
        index = 0
        # protein = '28'
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            # pdb.set_trace()
            pro = values[0].split('_')[0]
            results[int(pro)] = values[1]

    return results


def plot_figure():
    protein_dict = read_protein_name()
    results = read_result_file()

    Figure = plt.figure(figsize=(12, 15))

    for key, values in results.iteritems():
        protein = protein_dict[key]
        # pdb.set_trace()
        labels = [int(float(val)) for val in values[0]]
        probability = [float(val) for val in values[1]]
        plot_roc_curve(labels, probability, protein)
    # plot_roc_curve(labels[1], probability[1], '')
    # plot_roc_curve(labels[2], probability[2], '')

    # title_type = 'stem cell circRNAs vs other circRNAs'
    title_type = 'ROC'
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title_type)
    plt.legend(loc="lower right")
    plt.savefig('roc1.eps', format='eps')
    plt.show()


def read_fasta_file(fasta_file):
    seq_dict = {}
    fp = gzip.open(fasta_file, 'r')
    name = ''
    name_list = []
    for line in fp:
        line = line.rstrip()
        # distinguish header from sequence
        if line[0] == '>':  # or line.startswith('>')
            # it is the header
            name = line[2:]  # discarding the initial >
            name_list.append(name)
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper().replace('U', 'T')
    fp.close()

    return seq_dict, name_list


def run_predict():
    data_dir = './RBPdata1020'
    fw = open('result_file_struct_auc', 'w')
    for protein in os.listdir(data_dir):
        print (protein)

        fw.write(protein + '\t')

        run_seq_struct_cnn_network(protein, seq=True, fw=fw)

    fw.close()


def load_data_file(inputfile, seq=True, onlytest=False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    if seq:
        tmp = []
        tmp.append(read_seq(inputfile))
        data["seq"] = tmp

        # seq_onehot, structure = read_structure(inputfile, path)
        # tmp.append(seq_onehot)
        # data["structure"] = structure

    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)

    return data


def read_sae_fea(seq_file):
    seq_namelist = []
    seq_feature = []
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                seq_namelist.append(name)
            else:
                seq = line[:-1].split(',')
                seq = [float(i) for i in seq]
                seq_feature.append(seq)

    return np.array(seq_namelist), np.array(seq_feature)


def load_sae_data_file(inputfile):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    tmp = []
    tmp2 = []
    tmp.append(read_sae_fea(inputfile)[0])
    tmp2.append(read_sae_fea(inputfile)[1])
    data["seq_name"] = tmp
    data["seq_feature"] = tmp2
    return data



def train_HOCNN_lncRBP(data_file,model_dir,batch_size, nb_epoch,N,k,m,l):
    training_data = load_data_file(data_file)

    # pdb.set_trace()
    train_Y = training_data["Y"]
    print (len(train_Y))
    y = preprocess_labels(train_Y)
    seq_data = training_data["seq"][0]
    # pdb.set_trace()
    my_classifier = get_seq_network(N,k,m,l)
    # 根据各个参数值的不同组合在(X_train, y_train)上训练模型
    my_classifier.fit(seq_data, y[0], batch_size=batch_size, epochs=nb_epoch)
    print(my_classifier.summary())
	
    # training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    # y = preprocess_labels(train_Y)
    # seq_data = training_data["seq"][0]
    # cnn_train = []
    # cnn_validation = []
    # seq_train = seq_data[training_indice]
    # seq_validation = seq_data[validation_indice]
    # cnn_train.append(seq_train)
    # cnn_validation.append(seq_validation)
    # y, encoder = preprocess_labels(training_label)
    # val_y, encoder = preprocess_labels(validation_label, encoder=encoder)
    #
    # # pdb.set_trace()
    # earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    # my_classifier = get_seq_network(N,k,m,l)
    # # 根据各个参数值的不同组合在(X_train, y_train)上训练模型
    #
    # my_classifier.fit(cnn_train, y, batch_size=batch_size, epochs=nb_epoch, verbose=0,
    #                   validation_data=(cnn_validation, val_y), callbacks=[earlystopper])
    #
    # print(my_classifier.summary())

    my_classifier.save(os.path.join(model_dir, 'seqcnn3_model.pkl'))


def test_HOCNN_lncRBP(data_file, model_dir, outfile='prediction.txt', fprfile='fpr_file.txt',tprfile='tpr_file.txt',
              metricsfile="metrics_file3.txt",onlytest=True):
    test_data = load_data_file(data_file)
    true_y = test_data["Y"]
    # truey = copy.copy(testlist)
    print ('predicting')

    testing = test_data["seq"][0]  # it includes one-hot encoding sequence and structure
    model = load_model(os.path.join(model_dir, 'seqcnn3_model.pkl'))

    predictions = model.predict(testing)
    # predictions = model.predict([testing,sae_feature])
    predictions_label = transfer_label_from_prob(predictions[:, 1])
    fw = open(outfile, 'w')
    myprob = "\n".join(map(str, predictions[:, 1]))
    # fw.write(mylabel + '\n')
    fw.write(myprob)
    fw.close()

    fpr,tpr,thresholds = roc_curve(true_y,predictions[:, 1])
    with open(fprfile, 'w') as f:
        writething = "\n".join(map(str, fpr))
        f.write(writething)
    with open(tprfile, 'w') as f:
        writething = "\n".join(map(str, tpr))
        f.write(writething)

    acc, sensitivity, specificity, MCC = calculate_performance(len(true_y), predictions_label, true_y)
    roc_auc = auc(fpr, tpr)

    out_rel = ['acc', acc, 'sn', sensitivity, 'sp', specificity, 'MCC', MCC, 'auc', roc_auc]
    with open(metricsfile, 'w') as f:
        writething = "\n".join(map(str, out_rel))
        f.write(writething)

    print 'acc,  sensitivity, specificity, MCC,auc \n', acc, sensitivity, specificity, MCC, roc_auc
#    print 'fpr',fpr
 #   print 'tpr',tpr



def get_structure_motif_fig_new(filter_weights, filter_outs, out_dir, structure, seq_targets=[], sample_i=0):
    print ('plot motif fig', out_dir)
    # seqs, seq_targets = get_seq_targets(protein)
    seqs = structure
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)

        # seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]

    num_filters = filter_weights.shape[0]
    filter_size = 7  # filter_weights.shape[2]

    filters_ic = []
    meme_out = structure_motifs.meme_intro('%s/filters_meme.txt' % out_dir, seqs)

    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        structure_motifs.plot_filter_heat(filter_weights[f, :, :], '%s/filter%d_heat.pdf' % (out_dir, f))

        # write possum motif file
        structure_motifs.filter_possum(filter_weights[f, :, :], 'filter%d' % f, '%s/filter%d_possum.txt' % (out_dir, f),
                                       False)

        structure_motifs.plot_filter_logo(filter_outs[:, :, f], filter_size, seqs, '%s/filter%d_logo' % (out_dir, f),
                                          maxpct_t=0.5)

        filter_pwm, nsites = structure_motifs.make_filter_pwm('%s/filter%d_logo.fa' % (out_dir, f))
        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            structure_motifs.meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()


def get_motif_fig_new(filter_weights, filter_outs, out_dir, seqs, sample_i=0):
    print ('plot motif fig', out_dir)
    # seqs, seq_targets = get_seq_targets(protein)
    if sample_i:
        print ('sampling')
        seqs = []
        for ind, val in enumerate(seqs):
            if ind in sample_i:
                seqs.append(val)

        # seq_targets = seq_targets[sample_i]
        filter_outs = filter_outs[sample_i]

    num_filters = filter_weights.shape[0]
    filter_size = 7  # filter_weights.shape[2]

    # pdb.set_trace()
    #################################################################
    # individual filter plots
    #################################################################
    # also save information contents
    filters_ic = []
    meme_out = meme_intro('%s/filters_meme.txt' % out_dir, seqs)

    for f in range(num_filters):
        print ('Filter %d' % f)

        # plot filter parameters as a heatmap
        plot_filter_heat(filter_weights[f, :, :], '%s/filter%d_heat.pdf' % (out_dir, f))

        # write possum motif file
        filter_possum(filter_weights[f, :, :], 'filter%d' % f, '%s/filter%d_possum.txt' % (out_dir, f), False)

        # plot weblogo of high scoring outputs
        plot_filter_logo(filter_outs[:, :, f], filter_size, seqs, '%s/filter%d_logo' % (out_dir, f), maxpct_t=0.5)

        # make a PWM for the filter
        filter_pwm, nsites = make_filter_pwm('%s/filter%d_logo.fa' % (out_dir, f))

        if nsites < 10:
            # no information
            filters_ic.append(0)
        else:
            # compute and save information content
            filters_ic.append(info_content(filter_pwm))

            # add to the meme motif file
            meme_add(meme_out, f, filter_pwm, nsites, False)

    meme_out.close()

    #################################################################
    # annotate filters
    #################################################################
    # run tomtom #-evalue 0.01
    subprocess.call('tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (
    out_dir, out_dir, 'Ray2013_rbp_RNA.meme'), shell=True)

    # read in annotations
    filter_names = name_filters(num_filters, '%s/tomtom/tomtom.txt' % out_dir, 'Ray2013_rbp_RNA.meme')

    #################################################################
    # print a table of information
    #################################################################
    table_out = open('%s/table.txt' % out_dir, 'w')

    # print header for later panda reading
    header_cols = ('', 'consensus', 'annotation', 'ic', 'mean', 'std')
    print >> table_out, '%3s  %19s  %10s  %5s  %6s  %6s' % header_cols

    for f in range(num_filters):
        # collapse to a consensus motif
        consensus = filter_motif(filter_weights[f, :, :])

        # grab annotation
        annotation = '.'
        name_pieces = filter_names[f].split('_')
        if len(name_pieces) > 1:
            annotation = name_pieces[1]

        # plot density of filter output scores
        fmean, fstd = plot_score_density(np.ravel(filter_outs[:, :, f]), '%s/filter%d_dens.pdf' % (out_dir, f))

        row_cols = (f, consensus, annotation, filters_ic[f], fmean, fstd)
        print >> table_out, '%-3d  %19s  %10s  %5.2f  %6.4f  %6.4f' % row_cols

    table_out.close()

    if True:
        new_outs = []
        for val in filter_outs:
            new_outs.append(val.T)
        filter_outs = np.array(new_outs)
        print (filter_outs.shape)
        # plot filter-sequence heatmap
        plot_filter_seq_heat(filter_outs, '%s/filter_seqs.pdf' % out_dir)


def read_seq_new(seq_file):
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    # seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq)
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            # seq_array = get_RNA_seq_concolutional_array(seq)
            seq_list.append(seq)

    return seq_list


def get_seq_structure_motif(model, testing, seqs, index=0, out_dir='motifs/seq_cnn/'):
    sf = model
    sfilter1 = model.layers[0]
    sfilter2 = sfilter1.layers[index]
    sfilter3 = sfilter2.layers[0]
    sfilter = sfilter3.get_weights()

    #sfilter = model.layers[0].layers[index].layers[0].get_weights()

    # filter_weights_old = np.transpose(sfilter[0][:,0,:,:], (2, 1, 0)) #sfilter[0][:,0,:,:]
    filter_weights_old = 0
    print (filter_weights_old.shape)
    # pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        x = x - x.mean(axis=0)
        filter_weights.append(x)

    filter_weights = np.array(filter_weights)
    # pdb.set_trace()
    filter_outs = get_feature(model, testing, index)

    sample_i = 0

    # out_dir = dir1 + protein
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:
        get_motif_fig_new(filter_weights, filter_outs, out_dir, seqs, sample_i)
    else:
        get_structure_motif_fig_new(filter_weights, filter_outs, out_dir, seqs, sample_i)


def identify_motif(data_file, model_dir='models/', motif_dir='motifs/', onlytest=True):
    test_data = load_data_file(data_file, onlytest=onlytest)
    seqs = read_seq_new(data_file)
    model = load_model(os.path.join(model_dir, 'model.pkl'))

    get_seq_structure_motif(model, test_data["seq"], seqs, index=0, out_dir=motif_dir + 'seq_cnn/')
    get_seq_structure_motif(model, test_data["seq"], test_data["structure"], index=1,
                            out_dir=motif_dir + 'structure_cnn/')

filename = '01_HITSCLIP_AGO2Karginov2013a_hg19'
number = '1'
bs=512
N=16
k=20
m=3
l=8

def run_train_HOCNN_lncRBP(

    data_file="./RBPdata1201/%s/train/%s/sequence.fa.gz"%(filename,number), train=True,batch_size=bs, n_epochs=300):
    model_dir = 'HOCNN_lncRBP_model/%s'%(filename)

    # data_file = "./datasets/clip/%s/30000/training_sample_0/sequences.fa.gz" % (filename), train = True, batch_size = bs , n_epochs = 300):
    # model_dir = '4seq_CNN_model/%s/seq_cnn3/300/' % (filename)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if train:
        print ('model training')
        train_HOCNN_lncRBP(data_file, model_dir, batch_size=batch_size, nb_epoch=n_epochs,N=N,k=k,m=m,l=l)

def run_test_HOCNN_lncRBP(
    data_file="./RBPdata1201/%s/test/%s/sequence.fa.gz"%(filename,number), predict=True):
    out_file = 'HOCNN_lncRBP_model/%s/prediction.txt'%(filename)
    model_dir = 'HOCNN_lncRBP_model/%s'%(filename)
    fpr_file = 'HOCNN_lncRBP_model/%s/fpr.txt'%(filename)
    tpr_file = 'HOCNN_lncRBP_model/%s/tpr.txt'%(filename)
    metricsfile = 'HOCNN_lncRBP_model/%s/metrics_file.txt' % (filename)

    # data_file = "./datasets/clip/%s/30000/test_sample_0/sequences.fa.gz" % (filename), predict = True):
    # out_file = '4seq_CNN_model/%s/seq_cnn3/300/prediction.txt' % (filename)
    # model_dir = '4seq_CNN_model/%s/seq_cnn3/300/' % (filename)
    # fpr_file = '4seq_CNN_model/%s/seq_cnn3/300/fpr.txt' % (filename)
    # tpr_file = '4seq_CNN_model/%s/seq_cnn3/300/tpr.txt' % (filename)
    # metricsfile = '4seq_CNN_model/%s/seq_cnn3/300/metrics_file3.txt' % (filename)


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if predict:
        print ('model prediction')
        test_HOCNN_lncRBP(data_file, model_dir, outfile=out_file, fprfile=fpr_file, tprfile=tpr_file, metricsfile=metricsfile,onlytest=True)


run_train_HOCNN_lncRBP()
run_test_HOCNN_lncRBP()