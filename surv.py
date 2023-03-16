import copy
import sys
import itertools
import random
import pickle
from sklearn.decomposition import PCA
import Utils.data_utils as data_utils
import Utils.embedding_utils as embedding_utils
import Models.Transformer.visit_transformer as visit_transformer
import numpy as np
import torch
import torch.nn
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import importlib
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored as _cic, concordance_index_ipcw as _cii
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
from collections import Counter
from operator import itemgetter
from risk import plot_loss, count_parameters, get_dict
from concordance import concordance_index
from survival import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer, Encoder
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13


# ************************* Classes and Functions **************************************


class MyDataset(Dataset):
    """ Class for survival analysis dataset"""
    def __init__(self, indices, base_model, masks, labels, durations, is_observed, train=True):
        """
        :param indices:  indices of patients
        :param base_model: visit transformer preprocess class
        :param masks: auxiliary variables for loss function
        :param labels: auxiliary variables for loss function
        :param durations: survival times
        :param is_observed: survival labels
        :param train: train or test set
        """
        self.base_model = base_model
        self.X = []
        self.M = []
        self.masks = masks[indices]
        self.labels = labels[indices]
        self.durations = durations.iloc[indices].squeeze().tolist()
        if isinstance(is_observed, dict):
            self.is_observed = [is_observed[x] for x in indices]
        else:
            self.is_observed = is_observed.iloc[indices].squeeze().tolist()
        chunk_size: int = 500  # Must be int , worked with 500, for strafe worked with 20000
        chunk_size = int(min(chunk_size,len(indices)))
        num_chunks = int(len(indices) / chunk_size)
        for chunk in range(num_chunks):
            if chunk != num_chunks - 1:
                x, m = self.base_model(indices[chunk * chunk_size:(chunk + 1) * chunk_size], train, return_mask=True)

            else:
                x, m = self.base_model(indices[chunk * chunk_size:], train, return_mask=True)
            x = x.cpu()
            m = m.cpu()
            self.M += list(m)
            self.X += list(x)
        print(f"Mean of durations is : {np.mean(self.durations)}, variance: {np.std(self.durations)}")
        print(len(self.X))

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.masks[idx], self.labels[idx], self.durations[idx], self.is_observed[idx]

    def __len__(self):
        return len(self.X)


def evaluate_surv_new(encoder, test_loader, train, dists):
    """
    :param encoder: model
    :param test_loader: test dataset
    :param train: not used
    :param dists: not used
    :return: C-index and MAE
    """
    encoder.eval()
    pred_durations, true_durations, is_observed_all, is_observed_indices = [], [], [], []
    with torch.no_grad():
        g_i = 0
        print("\n")
        for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
            surv_pred, y_pred = clf(X.cuda(), m=None, train=True)
            surv_probs = torch.cumprod(surv_pred, dim=1)
            mean = True
            if mean: # mean survival time
                pred_durations += torch.sum(surv_probs, dim=1).tolist()
            else: # Median survival time
                for l in surv_probs:
                    pred_duration = 0
                    while True:
                        if l[pred_duration] < 0.5:
                            break
                        else:
                            pred_duration += 1
                            if pred_duration == l.shape[0]:
                                break
                    pred_durations.append(float(pred_duration))

            true_durations_cur = np.asarray(duration)
            true_durations += np.asarray(true_durations_cur).tolist()
            is_observed_all += is_observed.tolist()
            for i, idx in enumerate(is_observed):
                if is_observed[i] == 1:
                    is_observed_indices.append(g_i)
                g_i += 1

        pred_durations = np.array(pred_durations)
        true_durations = np.array(true_durations)
        pred_obs_durations = pred_durations[is_observed_indices]
        true_obs_durations = true_durations[is_observed_indices]

        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))
        is_observed_all = np.asarray(is_observed_all, dtype=bool)
        # print(f"Number of observed patients is : {len(is_observed_indices)}")
        # print(f"Number of censored patients is : {len(is_observed_all) - len(is_observed_indices)}")
        # print('pred durations OBS', pred_durations[is_observed_all].round(),
        #       "variance : ",np.var(pred_durations[is_observed_all].round()))
        # print('true durations OBS', true_durations[is_observed_all].round(),
        #       "variance : ",np.var(true_durations[is_observed_all].round()))
        # print('pred durations CRS', pred_durations[~is_observed_all].round(),
        #       "variance : ",np.var(pred_durations[~is_observed_all].round()))
        # print('true durations CRS', true_durations[~is_observed_all].round(),
        #       "variance : ",np.var(true_durations[~is_observed_all].round()))
        test_cindex = concordance_index(true_durations, pred_durations, is_observed_all)
        return test_cindex, mae_obs, None


def evaluate_risk_by_surv(encoder, test_loader, risk_time, train, ret_all=False):
    """
    :param encoder: model
    :param test_loader: test set
    :param risk_time: fixed risk prediction time
    :param train: not used
    :param ret_all: returns fpr and tpr (ret_all=epoch)
    :return: roc-auc score
    """
    np.set_printoptions(threshold=sys.maxsize)
    encoder.eval()
    preds_test, true_test = [], []
    with torch.no_grad():
        for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
            surv_pred, y_pred = encoder(X.cuda(), m.cuda(), train=train)
            surv_probs = torch.cumprod(surv_pred, dim=1)
            preds_test += surv_probs[:, int(12*risk_time/365)].tolist()
            true_test += is_observed.tolist()
    if ret_all:
        fpr, tpr, _ = roc_curve(np.array(true_test), 1 - np.array(preds_test), drop_intermediate=False)
        pickle.dump((fpr, tpr), open(f"STRAFE_ROC_{ret_all}", "wb"))
    return roc_auc_score(np.array(true_test), 1 - np.array(preds_test))


def survival_pre_process(ts, es, max_time):
    """
    :param ts: list of survival times
    :param es: list of observed/censored indicators
    :param max_time: maximum months for time-to-event
    :return: makss and labels which are used in Survival Dataset
    """
    masks, labels = [], []
    for duration, is_observed in zip(ts, es):
        if is_observed:
            mask = max_time * [1.]
            label = duration * [1.] + (max_time - duration) * [0.]
        else:
            # NOTE plus 1 to include day 0
            mask = (duration + 1) * [1.] + (max_time - (duration + 1)) * [0.]
            label = max_time * [1.]
        assert len(mask) == max_time
        assert len(label) == max_time
        masks.append(mask)
        labels.append(label)
    return np.array(masks), np.array(labels)


def quantiles_function(true_test, preds_test, qp: int = 100):
    """

    :param true_test: True survival times
    :param preds_test: predicted survival times
    :param qp: number of quantiles
    :return: number of positive in each quantile
    """
    assert len(true_test) == len(preds_test)
    n = len(true_test)
    dev = n % qp
    preds = np.sort(preds_test)
    label = true_test[np.argsort(preds_test)]
    label = label[dev:]
    preds = preds[dev:]

    assert len(label) / qp == int(len(label) / qp)
    n = len(label)
    q_ratio = []
    q_len = int(n / qp)
    for k in range(qp):
        st = q_len * k
        quantile = label[st:st + q_len]
        pos_num = list(quantile).count(1)
        q_ratio.append(pos_num)
    return q_ratio


def quantiles_function_survival(true_test, preds_test, qp:int = 10):
    """
        :param true_test: True survival times
        :param preds_test: predicted survival times
        :param qp: number of quantiles
        :return: number of positive in each quantile
        """
    assert len(true_test) == len(preds_test)
    n = len(true_test)
    dev = n % qp
    preds = np.sort(preds_test)
    label = true_test[np.argsort(preds_test)]
    label = label[dev:]
    assert len(label)/qp==int(len(label)/qp)
    n = len(label)
    q_ratio = []
    q_len = int(n/qp)
    for k in range(qp):
        st = q_len*k
        quantile = label[st:st+q_len]
        avg = np.mean(quantile)
        q_ratio.append(avg)

    return q_ratio


def survival_learner(dataset_dict, feature_matrix_counts, indices_train, indices_test, dists, risk_time, cox=True, RSF=True,
                     deepsurv=True, pca_dim=None):
    """

    :param dataset_dict: survival time information
    :param feature_matrix_counts: baselines input
    :param indices_train: patients in train set
    :param indices_test: patients in test set
    :param dists: not used
    :param risk_time: not used
    :param cox: run cox model - not used
    :param RSF: run RSF model
    :param deepsurv: run deephit
    :param pca_dim: pca dimension - not used
    :return: baselines evaluation
    """

    def risk_pred_from_survival(risk_time, preds, true_test, risk_indices_places):
        preds_test = preds[risk_indices_places, int(12*risk_time/365)].tolist()
        return roc_auc_score(np.array(true_test), 1 - np.array(preds_test))

    def get_risk_info(dataset_dict, indices_test, risk_time):
        risk_indices = []
        risk_labels = {}
        for i, (d, e) in enumerate(
                zip(dataset_dict["durations"][indices_test], dataset_dict["is_observed"][indices_test])):
            if e == 1:
                risk_indices.append(i)
                if d < risk_time:
                    risk_labels[i] = 1
                else:
                    risk_labels[i] = 0
            if e == 0 and d > risk_time:
                risk_indices.append(i)
                risk_labels[i] = 0
        print("Number of risk test patient is : ", len(risk_indices))
        risk_indices_places = risk_indices
        risk_labels = list(risk_labels.values())
        print(risk_labels.count(1), risk_labels.count(0))
        return risk_indices_places, risk_labels

    risk_indices_places180, risk_labels180 = get_risk_info(dataset_dict, indices_test, 180)
    risk_indices_places365, risk_labels365 = get_risk_info(dataset_dict, indices_test, 365)
    risk_indices_places730, risk_labels730 = get_risk_info(dataset_dict, indices_test, 730)

    t = time.time()
    feature_matrix_counts = feature_matrix_counts.toarray()
    print(feature_matrix_counts.shape)

    # Adding Non-temporal features
    genders = np.array(dataset_dict['genders'].tolist(), dtype=float)
    births = np.array(dataset_dict['births'].tolist(), dtype=float)
    births = (births-1920)/100


    feature_matrix_counts = hstack((feature_matrix_counts, csr_matrix(genders).T, csr_matrix(births).T))
    print(feature_matrix_counts.shape)
    X_train = feature_matrix_counts.tocsr()[indices_train].toarray()
    X_test = feature_matrix_counts.tocsr()[indices_test].toarray()
    T_train = dataset_dict['durations'][indices_train].tolist()
    T_test = dataset_dict['durations'][indices_test].tolist()

    T_train = np.floor(np.array(T_train) * 12 / 365).tolist()
    T_test = np.floor(np.array(T_test) * 12 / 365).tolist()
    T_train = np.clip(T_train,a_min=None, a_max=48)
    T_test = np.clip(T_test,a_min=None, a_max=48)


    E_train = dataset_dict['is_observed'][indices_train].tolist()
    E_test = dataset_dict['is_observed'][indices_test].tolist()
    E_train = list(map(bool, E_train))
    E_test = list(map(bool, E_test))
    Y_train = np.array(list(zip(E_train, T_train)))
    Y_test = np.array(list(zip(E_test, T_test)))

    obs_indices_train = [i for i in range(len(indices_train)) if E_train[i] == 1]
    obs_indices_test = [i for i in range(len(indices_test)) if E_test[i] == 1]

    print("end preprocess ", time.time() - t)
    t = time.time()

    if pca_dim > 0 and False: # remove False in order to use pca
        t_pca = time.time()
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)
        print("end PCA ", time.time() - t_pca)

    Y_train = np.core.records.fromarrays(Y_train.transpose(), names='E, T', formats='?, f8')
    Y_test = np.core.records.fromarrays(Y_test.transpose(), names='E, T', formats='?, f8')

    if RSF and True:
        print("Started RSF")
        rsf_time = time.time()
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                   min_samples_leaf=15, max_features="sqrt", n_jobs=4, random_state=1)
        rsf.fit(X_train, Y_train)
        print(f"RSF fit is completed, {time.time() - rsf_time} ")
        c_index = rsf.score(X_test, Y_test)
        preds = rsf.predict_survival_function(X_test, return_array=True)
        T_preds = []
        for p in preds:
            T_preds.append(sum(p))
        T_preds = np.array(T_preds)

        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds[obs_indices_test] - np.array(T_test)[obs_indices_test])))
        print('Auc risk 180: {:.4f}'.format(risk_pred_from_survival(180, preds, risk_labels180, risk_indices_places180)))
        print('Auc risk 365: {:.4f}'.format(risk_pred_from_survival(365, preds, risk_labels365, risk_indices_places365)))
        print('Auc risk 730: {:.4f}'.format(risk_pred_from_survival(730, preds, risk_labels730, risk_indices_places730)))

        print(f"RSF takes : {time.time() - t}")

    deephit = True
    if deephit:
        Y_train = np.array(list(zip(T_train, E_train)))
        Y_test = np.array(list(zip(T_test, E_test)))

        in_features = X_train.shape[1]
        out_features = 50
        num_nodes = [512, 128, 32]
        batch_norm = True
        dropout = 0.5
        batchsize = 256
        net = tt.practical.MLPVanilla(in_features, num_nodes,out_features,batch_norm,dropout)
        optimizer = tt.optim.Adam()
        model = DeepHitSingle(net, optimizer, alpha = 0.5, sigma =0.1)#, duration_index = labtrans.cuts)

        lr = 0.0005
        search_params = False  # find learning rate
        if search_params:
            lr_finder = model.lr_finder(input=X_train, target=Y_train, batch_size=batchsize, tolerance=3)#torch_ds_dl=True)
            lr = lr_finder.get_best_lr()
            print("lr : ", lr)

        model.optimizer.set_lr(lr)
        epochs = 30
        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(X_train, Y_train, batchsize, epochs, callbacks, val_data=(X_test, Y_test))
        preds = model.predict_surv_df(X_test)[1:]
        print(f"Deep Hit fit is completed, {time.time() - t} ")
        c_index = EvalSurv(preds, np.array(T_test), np.array(E_test), censor_surv = 'km').concordance_td('antolini')
        print("train cindex :  ",EvalSurv(model.predict_surv_df(X_train)[1:], np.array(T_train), np.array(E_train), censor_surv = 'km').concordance_td('antolini') )
        preds = preds.to_numpy().transpose()
        print(preds)
        T_preds = np.array([sum(p) for p in preds])
        print(T_preds)
        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds[obs_indices_test] - np.array(T_test)[obs_indices_test])))
        from sksurv.metrics import concordance_index_ipcw as _cii
        print(
            'Auc risk 180: {:.4f}'.format(risk_pred_from_survival(180, preds, risk_labels180, risk_indices_places180)))
        print(
            'Auc risk 365: {:.4f}'.format(risk_pred_from_survival(365, preds, risk_labels365, risk_indices_places365)))
        print(
            'Auc risk 730: {:.4f}'.format(risk_pred_from_survival(730, preds, risk_labels730, risk_indices_places730)))

        print(f"Deep Hit takes : {time.time() - t}")


def survival_linear_embedding(args_dict, data, indices_train, indices_test, dists, risk_time, cox=True, RSF=True,
                     deepsurv=True, pca_dim=None):
    """

       :param args_dict: survival time information
       :param data: baselines input
       :param indices_train: patients in train set
       :param indices_test: patients in test set
       :param dists: not used
       :param risk_time: not used
       :param cox: run cox model - not used
       :param RSF: run RSF model
       :param deepsurv: run deephit
       :param pca_dim: pca dimension - not used
       :return: baselines evaluation
       """

    def risk_pred_from_survival(risk_time, preds, true_test, risk_indices_places):
        preds_test = preds[risk_indices_places, int(12*risk_time/365)].tolist()
        return roc_auc_score(np.array(true_test), 1 - np.array(preds_test))

    def get_risk_info(args_dict, indices_test, risk_time):
        risk_indices = []
        risk_labels = {}
        for i, (d, e) in enumerate(
                zip(args_dict["durations"][indices_test], args_dict["is_observed"][indices_test])):
            if e == 1:
                risk_indices.append(i)
                if d < risk_time:
                    risk_labels[i] = 1
                else:
                    risk_labels[i] = 0
            if e == 0 and d > risk_time:
                risk_indices.append(i)
                risk_labels[i] = 0
        print("Number of risk test patient is : ", len(risk_indices))
        risk_indices_places = risk_indices
        risk_labels = list(risk_labels.values())
        print(risk_labels.count(1), risk_labels.count(0))

        return risk_indices_places, risk_labels


    quantiles_analysis_survival_gate = True

    risk_indices_places180, risk_labels180 = get_risk_info(args_dict, indices_test, int(180*12/365))
    risk_indices_places365, risk_labels365 = get_risk_info(args_dict, indices_test, int(365*12/365))
    risk_indices_places730, risk_labels730 = get_risk_info(args_dict, indices_test, int(730*12/365))

    t = time.time()
    print(data.shape)

    X_train = data[indices_train]
    X_test = data[indices_test]
    T_train = args_dict['durations'][indices_train].tolist()
    T_test = args_dict['durations'][indices_test].tolist()
    E_train = args_dict['is_observed'][indices_train].tolist()
    E_test = args_dict['is_observed'][indices_test].tolist()

    E_train = list(map(bool, E_train))
    E_test = list(map(bool, E_test))
    Y_train = np.array(list(zip(E_train, T_train)))
    Y_test = np.array(list(zip(E_test, T_test)))
    obs_indices_train = [i for i in range(len(indices_train)) if E_train[i] == 1]
    obs_indices_test = [i for i in range(len(indices_test)) if E_test[i] == 1]
    print("end preprocess ", time.time() - t)


    t = time.time()
    if pca_dim > 0 and False:  # remove False to use pca
        t_pca = time.time()
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)
        print("end PCA ", time.time() - t_pca)

    Y_train = np.core.records.fromarrays(Y_train.transpose(), names='E, T', formats='?, f8')
    Y_test = np.core.records.fromarrays(Y_test.transpose(), names='E, T', formats='?, f8')

    if RSF:
        print("Started RSF")
        rsf_time = time.time()
        from sksurv.ensemble import RandomSurvivalForest
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
                                   min_samples_leaf=15, max_features="sqrt",max_depth=12 , n_jobs=-1, random_state=1)
        rsf.fit(X_train, Y_train)
        print(f"RSF fit is completed, {time.time() - rsf_time} ")
        c_index = rsf.score(X_test, Y_test)

        preds = rsf.predict_survival_function(X_test, return_array=True)
        T_preds = []
        for p in preds:
            T_preds.append(sum(p))
        T_preds = np.array(T_preds)

        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds[obs_indices_test] - np.array(T_test)[obs_indices_test])))

        if quantiles_analysis_survival_gate:  # quantiles analysis
            preds_test = np.array(T_preds)[obs_indices_test]
            true_test = np.array(T_test)[obs_indices_test]
            q_ratios = []
            for f in range(100):  # Bootstrap
                indices = random.choices(list(range(len(true_test))), k=int(len(true_test) / 3))
                q_ratio = quantiles_function_survival(true_test[indices], preds_test[indices], qp=10)
                q_ratios.append(q_ratio)
            q_ratios = np.array(q_ratios)
            q10s = np.array(q_ratios)
            pickle.dump(q10s, open(f"RSF_quantiles", "wb"))
            return

        print(f"RSF takes : {time.time() - t}")

    deephit = True
    if deephit:

        Y_train = np.array(list(zip(T_train, E_train)))
        Y_test = np.array(list(zip(T_test, E_test)))
        in_features = X_train.shape[1]
        out_features = 50
        num_nodes = [512, 128, 32]
        batch_norm = True
        dropout = 0.5
        batchsize = 256
        net = tt.practical.MLPVanilla(in_features, num_nodes,out_features,batch_norm,dropout)
        optimizer = tt.optim.Adam()
        model = DeepHitSingle(net, optimizer, alpha = 0.5, sigma =0.1)#, duration_index = labtrans.cuts)
        lr = 0.0005
        search_params = False  # find learning rate
        if search_params:
            lr_finder = model.lr_finder(input=X_train, target=Y_train, batch_size=batchsize, tolerance=3)#torch_ds_dl=True)
            lr = lr_finder.get_best_lr()
            print("lr : ", lr)

        model.optimizer.set_lr(lr)
        epochs = 30
        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(X_train, Y_train, batchsize, epochs, callbacks, val_data=(X_test, Y_test))
        preds = model.predict_surv_df(X_test)[1:]
        print(f"Deep Hit fit is completed, {time.time() - t} ")
        c_index = EvalSurv(preds, np.array(T_test), np.array(E_test), censor_surv = 'km').concordance_td('antolini')
        print("train cindex :  ",EvalSurv(model.predict_surv_df(X_train)[1:], np.array(T_train), np.array(E_train), censor_surv = 'km').concordance_td('antolini') )
        preds = preds.to_numpy().transpose()
        print(preds)
        T_preds = np.array([sum(p) for p in preds])
        print(T_preds)
        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds[obs_indices_test] - np.array(T_test)[obs_indices_test])))
        from sksurv.metrics import concordance_index_ipcw as _cii
        print(
            'Auc risk 180: {:.4f}'.format(risk_pred_from_survival(180, preds, risk_labels180, risk_indices_places180)))
        print(
            'Auc risk 365: {:.4f}'.format(risk_pred_from_survival(365, preds, risk_labels365, risk_indices_places365)))
        print(
            'Auc risk 730: {:.4f}'.format(risk_pred_from_survival(730, preds, risk_labels730, risk_indices_places730)))



        if quantiles_analysis_survival_gate:  # deephit survival quantiles
            preds_test = np.array(T_preds)[obs_indices_test]
            true_test = np.array(T_test)[obs_indices_test]
            q_ratios = []
            for f in range(100):
                indices = random.choices(list(range(len(true_test))), k=int(len(true_test) / 3))
                q_ratio = quantiles_function_survival(true_test[indices], preds_test[indices], qp=10)
                q_ratios.append(q_ratio)
            q_ratios = np.array(q_ratios)
            q10s = np.array(q_ratios)
            pickle.dump(q10s, open(f"deephit_quantiles", "wb"))
            return

        print(f"Deep Hit takes : {time.time() - t}")


def get_risk_indices_and_labels(risk_time, dataset_dict, indices_all):
    """

    :param risk_time: Risk prediction time
    :param dataset_dict: survival data information
    :param indices_all: indices of patients
    :return: indices of patients eligible for risk (not censored in the risk prediction time)
    """
    risk_indices = []
    risk_labels = {}
    for i, (d, e) in enumerate(zip(dataset_dict["durations"][indices_all], dataset_dict["is_observed"][indices_all])):
        if i in indices_test:
            if e == 1:
                risk_indices.append(i)
                if d < risk_time:
                    risk_labels[i] = 1
                else:
                    risk_labels[i] = 0
            if e == 0 and d > risk_time:
                risk_indices.append(i)
                risk_labels[i] = 0
    print("Number of risk test patient is : ", len(risk_indices))
    vals = [x for x in risk_labels.values()]
    print(vals.count(1), vals.count(0))
    return risk_indices, risk_labels


if __name__ == '__main__':
    # ********************* Loading libraries, data and device ********************************************************
    visit_transformer = importlib.reload(visit_transformer)
    data_utils = importlib.reload(data_utils)
    embedding_utils = importlib.reload(embedding_utils)
    assert (torch.cuda.is_available())
    torch.cuda.set_device(0)

    with open("data_to_train.pkl", "rb") as h:
        data_to_train = pickle.load(h)
    dataset_dict = data_to_train["dataset_dict"]
    print(f"censored rate is : ", list(dataset_dict["is_observed"]).count(0)*100/len(dataset_dict["is_observed"]), "% ")
    featureSet = data_to_train["featureSet"]
    indices_all = list(range(len(dataset_dict['is_observed'])))
    stratify_by = dataset_dict['is_observed']

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        np.zeros(shape=(len(indices_all), 2)), stratify_by, indices_all, test_size=0.2, random_state=1,
        stratify=stratify_by)

    to_train_enbedding = False
    mbsz = 256
    print("Num train samples : ", len(indices_train))
    print("Num test samples : ", len(indices_test))

    risk_time = 365 * 2  # Fixed risk prediction time

    # **************************** survival baselines ******************************************************
    linear_baseline = False
    if linear_baseline:
        feature_matrix_counts = pickle.load(open("feature_matrix_counts.pkl", "rb"))

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            np.zeros(shape=(len(indices_all), 2)), stratify_by, indices_all, test_size=0.2, random_state=1,
            stratify=stratify_by)
        train_dist = np.core.records.fromarrays(np.array(list(zip(dataset_dict['is_observed'][indices_train].tolist(),
                                                                      dataset_dict['durations'][
                                                                          indices_train].tolist()))).transpose(),
                                                    names='E, T', formats='?, f8')
        test_dist = np.core.records.fromarrays(np.array(list(zip(dataset_dict['is_observed'][indices_test].tolist(),
                                                                     dataset_dict['durations'][
                                                                         indices_test].tolist()))).transpose(),
                                                   names='E, T', formats='?, f8')
        dists = [train_dist, test_dist]

        survival_learner(dataset_dict, feature_matrix_counts, indices_train, indices_test, dists, risk_time, cox=False, RSF=True,
                             deepsurv=False, pca_dim=500)

    # ********************* Embedding ****************************************************************
    all_concept_map_reverse = get_dict(featureSet.concept_map_rev, 'all_concept_map_reverse.pkl',
                                           with_save=to_train_enbedding)
    embedding_dim = 128  # size of embedding, must be multiple of number of heads --128
    window_days = 90  # number of days in window that defines a "Sentence" when learning the embedding
    embedding_filename = f"embedding_all_data_90d128"
    model_filename = f"{embedding_filename}_model"

    # ********************* Visit Transformer *******************************************************
    torch.cuda.empty_cache()
    mn_prefix = 'eol_experiment_prefix'  # Name for the model (mn_prefix) that will be used when saving checkpoints
    n_heads = 1
    assert embedding_dim % n_heads == 0
    model_params = {
        'embedding_dim': int(embedding_dim / n_heads), 'n_heads': n_heads, 'attn_depth': 4, 'dropout': 0.2,
        'use_mask': True, 'concept_embedding_path': embedding_filename, 'normalize_visits': False,
        'normalize_codes': False, 'use_RNN': False, 'time_emb_type': 'sin', 'use_mask_later': True,
        'all_concept_map_reverse': all_concept_map_reverse, 'pre_trained': True}
    base_model = visit_transformer.VisitTransformer(featureSet, **model_params, )
    base_model.set_data(
        torch.LongTensor(dataset_dict['all_codes_tensor']), dataset_dict['person_indices'],
        dataset_dict['visit_chunks'], dataset_dict['visit_time_rel'], dataset_dict['n_visits'],
        dataset_dict['births'], dataset_dict['genders'], )
    base_model.cuda()

    # ********************* Survival pre-process ****************************************************************
    max_time = 48
    dataset_dict['durations'] = dataset_dict['durations'] * 12 / 365
    dataset_dict['durations'] = dataset_dict['durations'].apply(np.floor).astype('int32')
    dataset_dict['durations'] = dataset_dict['durations'].clip(0, max_time - 1)
    masks, labels = survival_pre_process(dataset_dict['durations'], dataset_dict['is_observed'], max_time)
    dists = None

    # **************************** survival baselines Embedding ******************************************************
    linear_baseline_embedding = False # survival baselines with embeddings as input
    if linear_baseline_embedding:
        all_loader = DataLoader(
            MyDataset(indices_all, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                      train=True), batch_size=len(indices_all), shuffle=True)
        for idx in range(1):
            _, _, _, _, indices_train, indices_test = train_test_split(
                np.zeros(shape=(len(indices_all), 2)), stratify_by, indices_all, test_size=0.2,
                random_state=random.randint(0, 100),
                stratify=stratify_by)
            args_dict = {}
            data = None
            for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(all_loader):
                data = torch.sum(X, 1).numpy()
                args_dict["durations"] = duration.numpy()
                args_dict["is_observed"] = is_observed.numpy()

            dists = None
            survival_linear_embedding(args_dict, data, indices_train, indices_test, dists, risk_time,
                             cox=False, RSF=False, deepsurv=False, pca_dim=500)

        #exit(0)
    # ********************* Data Loaders ***********************************************************************

    train_loader = DataLoader(
        MyDataset(indices_train, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                  train=True), batch_size=mbsz, shuffle=True)
    test_loader = DataLoader(
        MyDataset(indices_test, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                  train=False), batch_size=mbsz, shuffle=False)

    risk_indices180, risk_labels180 = get_risk_indices_and_labels(int(180*12/365), dataset_dict, indices_all)
    risk_indices365, risk_labels365 = get_risk_indices_and_labels(int(365*12/365), dataset_dict, indices_all)
    risk_indices730, risk_labels730 = get_risk_indices_and_labels(int(730*12/365), dataset_dict, indices_all)
    risk_test_loader180 = DataLoader(
        MyDataset(risk_indices180, base_model, masks, labels, dataset_dict['durations'], risk_labels180,
                  train=False), batch_size=mbsz, shuffle=False)
    risk_test_loader365 = DataLoader(
        MyDataset(risk_indices365, base_model, masks, labels, dataset_dict['durations'], risk_labels365,
                  train=False), batch_size=mbsz, shuffle=False)
    risk_test_loader730 = DataLoader(
        MyDataset(risk_indices730, base_model, masks, labels, dataset_dict['durations'], risk_labels730,
                  train=False), batch_size=mbsz, shuffle=False)

    # ********************* Hyper parameters *******************************************************
    lr = 2e-3  # * 0.01
    n_epochs_pretrain = 35
    update_mod = 30  # update every update_mod batches

    # ********************* Survival Architecture *******************************************************
    c = copy.deepcopy
    # survival_architecture_parameters:
    #sap = {"num_heads": 1, "d_model": 128, "drop_prob": 0.3, "num_features": embedding_dim, "N": 1, "d_ff": 128} # original
    sap = {"num_heads": 1, "d_model": 256, "drop_prob": 0.3, "num_features": embedding_dim, "N": 2, "d_ff": 128}

    attn = MultiHeadedAttention(sap["num_heads"], sap["d_model"], sap["drop_prob"])
    ff = PositionwiseFeedForward(sap["d_model"], sap["d_ff"], sap["drop_prob"])
    encoder_layer = EncoderLayer(sap["d_model"], c(attn), c(ff), sap["drop_prob"])
    surv_encoder = Encoder(encoder_layer, sap["N"], sap["d_model"], sap["drop_prob"], sap["num_features"]).cuda()
    clf = visit_transformer.SA(base_model, surv_encoder, max_time, **model_params).cuda()

    # ********************* Training Section *******************************************************
    optimizer_clf = torch.optim.Adam(params=clf.parameters(), lr=lr  )# , weight_decay=20)
    torch.nn.utils.clip_grad_norm_(clf.parameters(), 50)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_clf, step_size=200, gamma=0.9)
    num_batches = int(len(train_loader) / mbsz) + 1
    task = 'eol'
    model_path = "model1"
    count_parameters(clf)
    train_losses, test_losses = [], []
    loaded_model = False
    for epoch in range(n_epochs_pretrain):
        if loaded_model: break
        t, batch_loss = time.time(), 0
        for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(train_loader):
            mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
            label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
            surv_pred, y_pred = clf(X.cuda(), m=None, train=True)
            surv_probs = torch.cumprod(surv_pred, dim=1)
            try:
                loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)
            except:
                print(surv_probs, " \n")
                print(mask, " \n")
                print(label, " \n")

            batch_loss += loss.item()
            loss.backward()
            if batch_num % update_mod == 0:
                optimizer_clf.step()
                optimizer_clf.zero_grad()

        batch_loss /= len(indices_train)
        t = time.time()
        clf.eval()
        # torch.save(clf.state_dict(), f"STRAFE_epoch{epoch+1}")
        with torch.no_grad():
            test_loss = 0
            for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
                mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
                label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
                surv_pred, y_pred = clf(X.cuda(), m=None, train=True)
                surv_probs = torch.cumprod(surv_pred, dim=1)
                loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)
                test_loss += loss.item()

            test_loss /= len(indices_test)
            # print(f'test loss: {test_loss}')
            print(f'Epochs: {epoch + 1} |  loss: {batch_loss} |  test loss: {test_loss}| epoch time: {time.time() - t}')
            train_cindex, train_mae, tarin_total_surv_probs = evaluate_surv_new(clf, train_loader, False, None)
            print('current train cindex', train_cindex, 'train mae', train_mae)
            val_cindex, val_mae, val_total_surv_probs = evaluate_surv_new(clf, test_loader, False, dists)
            print('current val cindex', val_cindex, 'val mae', val_mae)

            print('180 Risk Auc using survival function is : ', evaluate_risk_by_surv(clf, risk_test_loader180, risk_time=180, train=False))
            print('365 Risk Auc using survival function is : ' ,evaluate_risk_by_surv(clf, risk_test_loader365, risk_time=365, train=False))
            print('730 Risk Auc using survival function is : ',
                  evaluate_risk_by_surv(clf, risk_test_loader730, risk_time=730, train=False))
            print("\n")
        clf.train()
        scheduler.step()
        train_losses.append(batch_loss)
        test_losses.append(test_loss)
        if (epoch + 1) % 20 == 0 and False:
            plot_loss(train_losses, test_losses, epoch)

    exit(0)

    # ********************* Hyper Parameter Tuning *******************************************************
    hp_tuning = False
    if hp_tuning:
        def train_HP(sap, model_params, base_model, f):
            """
            :param sap: set of hyper parameters
            :param model_params: model paramaters
            :param base_model: model preprocess class
            :param f: file to write
            :return: model evaluations
            """
            f.write(
                f"num_heads = {hp[0]}, d_model = {hp[1]}, drop_prob = {hp[2]}, N = {hp[3]}, d_ff = {hp[4]}, lr = {hp[5]}, batch_size = {mbsz} \n")
            print(sap)
            update_mod = 30
            c = copy.deepcopy
            attn = MultiHeadedAttention(sap["num_heads"], sap["d_model"], sap["drop_prob"])
            ff = PositionwiseFeedForward(sap["d_model"], sap["d_ff"], sap["drop_prob"])
            encoder_layer = EncoderLayer(sap["d_model"], c(attn), c(ff), sap["drop_prob"])
            surv_encoder = Encoder(encoder_layer, sap["N"], sap["d_model"], sap["drop_prob"],
                                   sap["num_features"]).cuda()
            clf = visit_transformer.SA(base_model, surv_encoder, sap["max_time"], **model_params).cuda()
            optimizer_clf = torch.optim.Adam(params=clf.parameters(), lr=sap["lr"])  # , weight_decay=20)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_clf, step_size=20, gamma=0.9)
            train_losses, test_losses = [], []
            train_cindexs, test_cindexs = [], []
            test_cindex_ipws = []
            train_maes, test_maes = [], []
            auc_risks = []
            for epoch in range(sap["n_epochs_pretrain"]):
                t, batch_loss = time.time(), 0
                for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(train_loader):
                    mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
                    label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
                    surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=True)
                    surv_probs = torch.cumprod(surv_pred, dim=1)
                    try:
                        loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)
                    except:
                        print(surv_probs, " \n")
                        print(mask, " \n")
                        print(label, " \n")
                    batch_loss += loss.item()
                    loss.backward()
                    if batch_num % update_mod == 0:
                        optimizer_clf.step()
                        optimizer_clf.zero_grad()
                batch_loss /= len(indices_train)
                t = time.time()
                clf.eval()
                with torch.no_grad():
                    test_loss = 0
                    for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
                        mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
                        label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
                        surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=False)
                        surv_probs = torch.cumprod(surv_pred, dim=1)
                        loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)
                        test_loss += loss.item()
                    test_loss /= len(indices_test)
                    print(
                        f'Epochs: {epoch + 1} |  loss: {batch_loss} |  test loss: {test_loss}| epoch time: {time.time() - t}')
                    train_cindex, train_mae, tarin_total_surv_probs = evaluate_surv_new(clf, train_loader, False, None)
                    print('current train cindex', train_cindex, 'train mae', train_mae)
                    val_cindex, val_mae, val_cii = evaluate_surv_new(clf, test_loader, False, dists)
                    print('current val cindex', val_cindex, 'val mae', val_mae)

                    auc_risk_by_surv = evaluate_risk_by_surv(clf, risk_test_loader180, risk_time=180, train=False)
                    print('Risk Auc using survival function is : ', auc_risk_by_surv, "\n")
                clf.train()
                scheduler.step()
                train_losses.append(batch_loss)
                test_losses.append(test_loss)
                train_cindexs.append(train_cindex)
                test_cindexs.append(val_cindex)
                test_cindex_ipws.append(val_cii)
                train_maes.append(train_mae)
                test_maes.append(val_mae)
                auc_risks.append(auc_risk_by_surv)
                f.write(
                    f"Epoch : {epoch}, train loss: {format(batch_loss, '.2f')}, test loss: {format(test_loss, '.2f')}, train cindex: {format(train_cindex, '.2f')},"
                    f" test cindex: {format(val_cindex, '.2f')}, test cindex ipw : {format(val_cii, '.2f')}, train mae: {format(train_mae, '.2f')}, test mae : {format(val_mae, '.2f')}, auc_risk: {format(auc_risk_by_surv, '.2f')} \n ")

                if epoch > 0 and train_cindex - val_cindex > 0.05:
                    break

            return {"train_losses": train_losses, "test_losses": test_losses, "train_cindexs": train_cindexs,
                    "test_cindexs": test_cindexs, "test_cindex_ipws": test_cindex_ipws, "train_cindex_maes": train_maes,
                    "test_cindex_maes": test_maes, "auc_risks": auc_risks}


        _num_heads_ = [1, 2, 4, 8]
        _d_model_ = [128, 512, 256]
        _drop_prob_ = [0.3, 0.1]
        _N_ = [1, 2, 4, 8]
        _d_ff_ = [128, 64]
        _lr_ = [2e-3, 1e-4]
        res_dict = {}
        with open("results_log2.txt", 'a') as f:
            for hp in zip(itertools.product(*list([_num_heads_, _d_model_, _drop_prob_, _N_, _d_ff_, _lr_]))):
                s_time = time.time()
                print(hp)
                hp = hp[0]
                if hp[5] != 2e-3:
                    continue
                cur_sap = {"num_heads": hp[0], "d_model": hp[1], "drop_prob": hp[2], "num_features": embedding_dim,
                           "N": hp[3], "d_ff": hp[4],
                           "lr": hp[5], "n_epochs_pretrain": 10, "max_time": max_time}

                ret = train_HP(sap=cur_sap, model_params=model_params, base_model=base_model, f=f)
                res_dict[
                    f"num_heads = {hp[0]}, d_model = {hp[1]}, drop_prob = {hp[2]}, N = {hp[3]}," \
                    f" d_ff = {hp[4]}, lr = {hp[5]}, batch_size = {mbsz}"] = ret
                print(f"Iteration took : {time.time() - s_time}")

        exit(0)

    # ********************* Quantile analysis *******************************************************

    quantile_analysis = False
    if quantile_analysis:

        clf.load_state_dict(torch.load(f"STRAFE_epoch{25}"))
        clf.eval()
        preds_test, true_test, qp = [], [], 100
        with torch.no_grad():
            risk_time = 180
            for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(risk_test_loader180):
                surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=False)
                surv_probs = torch.cumprod(surv_pred, dim=1)
                preds_test += surv_probs[:, int(12 * risk_time / 365)].tolist()
                true_test += is_observed.tolist()

            true_test = np.array(true_test)
            preds_test = 1 - np.array(preds_test)

            q_ratios = []
            for f in range(100):  # Bootstrap
                indices = random.choices(list(range(len(true_test))), k=int(len(true_test)/3))
                q_ratio = quantiles_function(true_test[indices], preds_test[indices], qp=10)
                q_ratios.append(q_ratio)
            q_ratios = np.array(q_ratios)
            q10s = np.array(q_ratios) * 100 / 520
            q10_avg = np.mean(q10s, axis=0)
            q10_std = np.std(q10s, axis=0)
            q10_median = np.mean(np.median(q10s, axis=1))
            print("AVG : ", list(q10_avg))
            print("STD: ", list(q10_std))
            print("MEDIAN : ", q10_median)

    survival_time_ranking_analysis = False
    if survival_time_ranking_analysis:
        preds_test, true_test = [], []
        with torch.no_grad():
            for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
                surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=False)
                surv_probs = torch.cumprod(surv_pred, dim=1)
                observed = is_observed.tolist()
                positive = [i for i, x in enumerate(observed) if x == 1]
                true_test += duration[positive].tolist()
                preds_test += torch.sum(surv_probs[positive], dim=1).tolist()

            true_test = np.array(true_test)
            preds_test = np.array(preds_test)

        q_ratios = []
        for f in range(100):
            indices = random.choices(list(range(len(true_test))), k=int(len(true_test) / 3))
            q_ratio = quantiles_function_survival(true_test[indices], preds_test[indices], qp=10)
            q_ratios.append(q_ratio)
        q_ratios = np.array(q_ratios)
        q10s = np.array(q_ratios)
        pickle.dump(q10s, open(f"STRAFE_quantiles", "wb"))

    exit(0)
