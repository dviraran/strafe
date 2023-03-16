import copy
import itertools
import random
import pickle
import sklearn.metrics
from sklearn.decomposition import PCA

import Utils.dbutils as dbutils
import Utils.data_utils as data_utils
import Utils.embedding_utils as embedding_utils
import Generators.CohortGenerator as CohortGenerator
import Generators.FeatureGenerator as FeatureGenerator
import Models.LogisticRegression.RegressionGen as lr_models
import Models.Transformer.visit_transformer as visit_transformer
import config
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F

import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import importlib
import csv
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.metrics import confusion_matrix
from concordance import concordance_index
from survival import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer, Encoder
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13

"""
Some the code was taken from omop-learn package jupyter notebooks: github.com/clinicalml/omop-learn
Some the code was taken from: github.com/shihux/sa_transformer


This script running the preprocess phase with the following steps:
1. Connects to database
2. Creates cohort
3. creates featureset
4. Create feature matrix for the baseline models (saved as feature_matrix_counts.pkl)
5. Saves relevant data to the deep models: SARD and STRAFE (saved as data_to_train.pkl)
6. Optinal: Embeddings training (saved with the name of the variable "embedding_filename")
"""


def plot_grad_flow(named_parameters):
    """
    :param named_parameters: clf.named_parameters() type
    :return: Prints table of parmaeters that their gradients are exploding/vanishing
    """
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.is_leaf and p.grad is not None:
            val = p.grad.abs().mean()
            layers.append(n)
            ave_grads.append(val)
    table_exploding = PrettyTable(["Layer name", "weight grad"])
    table_vanishing = PrettyTable(["Layer name", "weight grad"])
    for l, av in zip(layers, ave_grads):
        if av > 1:
            table_exploding.add_row([l, av])
        if av < 0.01:
            table_vanishing.add_row([l, av])
    print("table_exploding")
    print(table_exploding)
    print("table_vanishing")
    print(table_vanishing)


def error_analysis(pred_durations, true_durations, is_observed_all):
    """
    :param pred_durations: Predicted survival time
    :param true_durations: Real survival time
    :param is_observed_all: indices of patients that are observed
    :return: Confusion matrix of observed and censored patients
    """
    pred_obs = pred_durations[is_observed_all]
    pred_cens = pred_durations[~is_observed_all]
    true_obs = true_durations[is_observed_all]
    true_cens = true_durations[~is_observed_all]
    print("Obsereved confusion matrix")
    obs_matrix = confusion_matrix(true_obs, pred_obs.astype(int))
    for line in obs_matrix:
        print(*line)
    print("Censored confusion matrix")
    cens_matrix = confusion_matrix(true_cens, pred_cens.astype(int))
    for line in cens_matrix:
        print(*line)


def evaluate_surv_new(encoder, test_loader, train):
    """
    :param encoder: Encoder model
    :param test_loader: test set dataloader
    :param train: Not used
    :return: Dataset C-index and MAE
    """
    encoder.eval()
    pred_durations, true_durations, is_observed_all, is_observed_indices = [], [], [], []
    with torch.no_grad():
        g_i = 0
        print("len(test_loader) = ", len(test_loader))
        for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
            surv_pred, y_pred = encoder(X.cuda(), m.cuda(), train=train)
            surv_probs = torch.cumprod(surv_pred, dim=1)
            # surv_probs = surv_pred
            pred_durations += torch.sum(surv_probs, dim=1).tolist()
            true_durations_cur = np.asarray(duration)
            # true_durations += np.asarray([int(d * 12 / 365) for d in true_durations_cur]).tolist()
            true_durations += np.asarray(true_durations_cur).tolist()
            is_observed_all += is_observed.tolist()
            for i, idx in enumerate(is_observed):
                if is_observed[i] == 1:
                    is_observed_indices.append(g_i)
                g_i += 1

        print("is_observed_all", is_observed_all)
        pred_durations = np.array(pred_durations)
        true_durations = np.array(true_durations)
        pred_obs_durations = pred_durations[is_observed_indices]
        true_obs_durations = true_durations[is_observed_indices]

        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))
        is_observed_all = np.asarray(is_observed_all, dtype=bool)
        print(f"Number of observed patients is : {len(is_observed_indices)}")
        print(f"Number of censored patients is : {len(is_observed_all) - len(is_observed_indices)}")

        print('pred durations OBS', pred_durations[is_observed_all].round(),
              np.var(pred_durations[is_observed_all].round()))
        print('true durations OBS', true_durations[is_observed_all].round(),
              np.var(true_durations[is_observed_all].round()))
        print('pred durations CRS', pred_durations[~is_observed_all].round(),
              np.var(pred_durations[~is_observed_all].round()))
        print('true durations CRS', true_durations[~is_observed_all].round(),
              np.var(true_durations[~is_observed_all].round()))

        test_cindex = concordance_index(true_durations, pred_durations, is_observed_all)
        error_analysis(pred_durations, true_durations, is_observed_all)
    return test_cindex, mae_obs, None


def get_batches(arr, mbsz):
    """
    :param arr: list of indices
    :param mbsz: batch size
    :return: batches for indices list
    """
    curr, ret = 0, []
    while curr < len(arr) - 1:
        ret.append(arr[curr: curr + mbsz])
        curr += mbsz
    return ret


def get_dict(d, path, with_save=False):
    """
    :param d: dictionary
    :param path: path to save
    :param with_save: is to save the dictionary
    :return: save/save and load the dictionary
    """
    if with_save:
        a_file = open(path, "wb")
        pickle.dump(d, a_file)
        a_file.close()
    return pickle.load(open(path, "rb"))


if __name__ == '__main__':

    torch.cuda.empty_cache()
    # ********************* Loading libraries and device ***************************************************************
    visit_transformer = importlib.reload(visit_transformer)
    data_utils = importlib.reload(data_utils)
    embedding_utils = importlib.reload(embedding_utils)
    assert (torch.cuda.is_available())
    torch.cuda.set_device(0)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'device : {device}, OMOP_CDM_SCHEMA : {config.OMOP_CDM_SCHEMA}')

    # ********************* Creating Database **************************************************************************
    schema_name, cdm_schema_name, reset_schema = 'eol_deep_test', config.OMOP_CDM_SCHEMA, False
    connect_args = {"host": config.HOST, 'user': config.PG_USERNAME, 'password': config.PG_PASSWORD,
                    'database_name': config.DB_NAME, 'dialect': config.DIALECT, 'driver': config.DRIVER}
    config_path = '{dialect}+{driver}://{user}:{password}@{host}/{database_name}'  # .format(**connect_args)
    db = dbutils.Database(config_path, schema_name, connect_args, cdm_schema_name)  # set up database
    if reset_schema:
        db.execute('drop schema if exists {} cascade'.format(schema_name))
    db.execute('create schema if not exists {}'.format(schema_name))

    # ********************* Creating Cohort **************************************************************************
    # cohort_name, cohort_script_path = 'cohort_risk', config.SQL_PATH_COHORTS + '/cohort_risk.sql'
    cohort_name, cohort_script_path = 'cohort_survival', config.SQL_PATH_COHORTS + '/cohort_survival.sql'
    basic_time, days_before, days_after = 5, 90, 365

    params = {  # cohort parameters
        'cohort_table_name': cohort_name, 'schema_name': cdm_schema_name, 'aux_data_schema': config.CDM_AUX_SCHEMA,
        'training_start_date': '2015-01-01', 'training_end_date': '2021-01-01', 'Gap': basic_time,
        'CKD2_codes': (443601,443597),
        'CKD4_codes': (443612,443611),
        'Days_before': days_before, 'Days_after': days_after, 'outcome_window': '6 months'}
    cohort = CohortGenerator.Cohort(schema_name=cdm_schema_name, cohort_table_name=cohort_name,
                                    cohort_generation_script=cohort_script_path, cohort_generation_kwargs=params,
                                    outcome_col_name='y')
    cohort.build(db, replace=False)
    # ********************* Creating Feature Set **********************************************************************

    cache_data_path, nontemporal_cache_data_path = "temporal_file.csv", "nontemporal_file.csv"
    featureSet = FeatureGenerator.FeatureSet(db)
    featureSet.add_default_features(['drugs', 'conditions', 'procedures'], cdm_schema_name, cohort_name)
    # featureSet.add_default_features(['age', 'gender'], cdm_schema_name, cohort_name, temporal=False)
    featureSet.build(cohort, from_cached=True, cache_file=cache_data_path)
    # ********************* Data Pre-process **********************************************************************
    outcomes_filt, feature_matrix_3d_transpose, remap, good_feature_names, durations, observes, births, genders = FeatureGenerator.postprocess_feature_matrix(
        cohort, featureSet)

    person_ixs, time_ixs, code_ixs = feature_matrix_3d_transpose.coords

    all_codes_tensor = code_ixs
    people = sorted(np.unique(person_ixs))
    person_indices = np.searchsorted(person_ixs, people)
    person_indices = np.append(person_indices, len(person_ixs))
    person_chunks = [time_ixs[person_indices[i]: person_indices[i + 1]] for i in range(len(person_indices) - 1)]

    visit_chunks, visit_times_raw = [], []
    for i, chunk in enumerate(person_chunks):
        visits = sorted(np.unique(chunk))
        visit_indices_local = np.searchsorted(chunk, visits)
        visit_indices_local = np.append(visit_indices_local, len(chunk))
        visit_chunks.append(visit_indices_local)
        visit_times_raw.append(visits)

    n_visits = {i: len(j) for i, j in enumerate(visit_times_raw)}
    visit_days_rel = {i: (pd.to_datetime(params['training_end_date']) - pd.to_datetime(featureSet.time_map[i])).days for
                      i in featureSet.time_map}
    vdrel_func = np.vectorize(visit_days_rel.get)
    visit_time_rel = [vdrel_func(v) for v in visit_times_raw]

    maps = {'concept': featureSet.concept_map, 'id': featureSet.id_map, 'time': featureSet.time_map}
    dataset_dict = {
        'all_codes_tensor': all_codes_tensor,  # A tensor of all codes occurring in the dataset
        'person_indices': person_indices, 'visit_chunks': visit_chunks,
        'visit_time_rel': visit_time_rel,  # A list of times (as measured in days to the prediction date) for each visit
        'n_visits': n_visits,  # n_visits[i] is the number of visits made by the ith patient (dict)
        'outcomes_filt': outcomes_filt,  # outcomes_filt.iloc[i] is the outcome of the ith patient (pandas)
        'remap': remap, 'maps': maps, 'durations': durations,
        'is_observed': observes, 'births': births, 'genders': genders}
    # all_codes_tensor[person_indices[i]: person_indices[i+1]] are the codes assigned to the ith patient (list)
    # all_codes_tensor[person_indices[i]+visit_chunks[j]:person_indices[i]+visit_chunks[j+1]] are the codes assigned to the ith patient during their jth visit (list)

    good_feature_names = np.vectorize(dataset_dict['maps']['concept'].get)(dataset_dict['remap']['concept'])
    feature_matrix_counts, feature_names = data_utils.window_data_sorted(
        window_lengths=[1060], feature_matrix=feature_matrix_3d_transpose,
        all_feature_names=good_feature_names, cohort=cohort, featureSet=featureSet)
    # original window_lengths: = window_lengths=[30, 180, 365, 730, 10000]
    feature_matrix_counts = feature_matrix_counts.T
    print(feature_matrix_counts.shape)
    print(len(feature_names))
    c30,c90, c180, c365, c730, c10000 = 0, 0, 0, 0, 0, 0
    for name in feature_names:
        if '30' in name:
            c30 += 1
        elif '90' in name:
            c90 += 1
        elif '180' in name:
            c180 += 1
        elif '365' in name:
            c365 += 1
        elif '730' in name:
            c730 += 1
        elif '10000' in name:
            c10000 += 1
    print(c30,c90, c180, c365, c730, c10000)
    # ********************* Train-Test splitting **********************************************************************

    indices_all = range(len(dataset_dict['is_observed']))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        feature_matrix_counts, dataset_dict['is_observed'], indices_all, test_size=0.2, random_state=1,
        stratify=dataset_dict['is_observed'])
    val_size = int(len(y_test) / 2)

    print("len of indices train", len(indices_train), schema_name)
    print("len of indices test", len(indices_test), schema_name)


    mbsz = 256 # batch size
    save_for_baselines = True # Save baselines feature matrix
    if save_for_baselines:
        with open("feature_matrix_counts.pkl", "wb") as h:
            pickle.dump(feature_matrix_counts, h, protocol=pickle.HIGHEST_PROTOCOL)

    # ********************* SAVE OBJECTS FOT TRAIN FILE ******************************************************
    featureSet._db = None
    data_to_train = {"dataset_dict":dataset_dict, "featureSet":featureSet}
    with open("data_to_train.pkl", "wb") as h:
        pickle.dump(data_to_train, h, protocol=pickle.HIGHEST_PROTOCOL)


    # ********************* Embedding ****************************************************************
    to_train_enbedding = True # If true: traines embeddings, pay attention to the cohort being used
    all_concept_map_reverse = get_dict(featureSet.concept_map_rev, 'all_concept_map_reverse.pkl',
                                       with_save=to_train_enbedding)
    embedding_dim = 128  # size of embedding, must be multiple of number of heads --128
    window_days = 90  # number of days in window that defines a "Sentence" when learning the embedding

    embedding_filename = f"embedding_all_data_90d128"
    model_filename = f"{embedding_filename}_model"
    if to_train_enbedding:
        train_coords = np.nonzero(np.where(np.isin(person_ixs, range(len(observes))), 1, 0))
        embedding_filename = embedding_utils.train_embedding(featureSet, feature_matrix_3d_transpose, window_days,
                                                             person_ixs[train_coords], time_ixs[train_coords],
                                                             remap['time'],
                                                             embedding_dim, embedding_filename)

    exit(0)


    class MyDataset(Dataset):
        """ Class for Dataset creation"""

        def __init__(self, indices, base_model, masks, labels, durations, is_observed, train=True):
            """
            :param indices: indices of patients
            :param base_model: the class that creates the patient representation
            :param masks: masks the censored patients after the censoring time
            :param labels: label of target event for each time-step
            :param durations: real survival time for observed patients and censoring time for censored patients
            :param is_observed: list of observed/censored indicators
            :param train: train/test set

            """
            self.base_model = base_model
            self.X = []
            self.M = []
            self.masks = masks[indices]
            self.labels = labels[indices]
            self.durations = durations.iloc[indices].squeeze().tolist()
            self.is_observed = is_observed.iloc[indices].squeeze().tolist()
            chunk_size: int = 500  # Must be int
            chunk_size = int(min(chunk_size, len(indices)))
            num_chunks = int(len(indices) / chunk_size)
            print(num_chunks)
            for chunk in range(num_chunks):
                if chunk != num_chunks - 1:
                    x, m = self.base_model(indices[chunk * chunk_size:(chunk + 1) * chunk_size], train,
                                           return_mask=True)
                else:
                    x, m = self.base_model(indices[chunk * chunk_size:], train, return_mask=True)
                x = x.cpu()
                m = m.cpu()
                self.M += list(m)
                self.X += list(x)

            print(f"Mean of durations is : {np.mean(self.durations)}, variance: {np.std(self.durations)}")
            print(len(self.X))
            e1_indices = list(i for i, x in enumerate(self.is_observed) if x == 1)
            e0_indices = list(i for i, x in enumerate(self.is_observed) if x == 0)
            plt.hist(np.array(self.durations)[e1_indices], bins=100, color='b')
            plt.hist(np.array(self.durations)[e0_indices], bins=100, color='r')
            plt.show()

        def __getitem__(self, idx):
            return self.X[idx], self.M[idx], self.masks[idx], self.labels[idx], self.durations[idx], self.is_observed[
                idx]

        def __len__(self):
            return len(self.X)