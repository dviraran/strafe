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

from concordance import concordance_index
from survival import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer, Encoder
from prettytable import PrettyTable

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13

from torch.utils.data import Dataset, DataLoader


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []

    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.is_leaf and p.grad is not None:
            # print(p, n)
            # print(p.grad)
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
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.gcf().subplots_adjust(bottom=0.75)
    # plt.show()


def error_analysis(pred_durations, true_durations, is_observed_all):
    from sklearn.metrics import confusion_matrix

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


def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        #torch.nn.init.constant_(m.bias.data, 0)


class MyDataset(Dataset):
    def __init__(self, indices, base_model, masks, labels, durations, is_observed, train=True):
        self.base_model = base_model
        self.X = []
        self.M = []
        self.masks = masks[indices]
        self.labels = labels[indices]
        self.durations = durations.iloc[indices].squeeze().tolist()
        self.is_observed = is_observed.iloc[indices].squeeze().tolist()
        chunk_size: int = 500  # Must be int
        chunk_size = int(min(chunk_size,len(indices)))
        num_chunks = int(len(indices) / chunk_size)
        print(num_chunks)
        for chunk in range(num_chunks):
            if chunk != num_chunks - 1:
                x, m = self.base_model(indices[chunk * chunk_size:(chunk + 1) * chunk_size], train, return_mask=True)

            else:
                x, m = self.base_model(indices[chunk * chunk_size:], train, return_mask=True)
            # print(x.shape)
            x = x.cpu()
            m = m.cpu()

            self.M += list(m)
            self.X += list(x)
            # print(list(x))
            # print(len(self.X))

        # from sklearn.decomposition import PCA
        #
        # for i, x in enumerate(self.X):
        #    pca = PCA(n_components=32)
        #    x = pca.fit_transform(x.numpy())
        #    x = torch.from_numpy(x)
        #    self.X[i] = x

        print(f"Mean of durations is : {np.mean(self.durations)}, variance: {np.std(self.durations)}")
        print(len(self.X))
        e1_indices = list(i for i, x in enumerate(self.is_observed) if x == 1)
        e0_indices = list(i for i, x in enumerate(self.is_observed) if x == 0)
        plt.hist(np.array(self.durations)[e1_indices], bins=100, color='b')
        plt.hist(np.array(self.durations)[e0_indices], bins=100, color='r')
        plt.show()

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.masks[idx], self.labels[idx], self.durations[idx], self.is_observed[idx]

    def __len__(self):
        return len(self.X)


def plot_loss(train_loss, test_loss, epoch):
    epochs = range(1, epoch + 2)
    print(len(epochs))
    print(len(train_loss))
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, test_loss, 'b', label='validation loss')
    plt.title(f"Training_and_Validation_loss_epoch_{epoch}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"Training_and_Validation_loss_epoch_{epoch}")
    plt.show()


def evaluate_surv_new(encoder, test_loader, train):
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

        # print(is_observed_all, is_observed_indices)
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

        #print("train = ", train)
        #print(true_durations, pred_durations, is_observed_all)
        test_cindex = concordance_index(true_durations, pred_durations, is_observed_all)
        error_analysis(pred_durations, true_durations, is_observed_all)
    return test_cindex, mae_obs, None


def evaluate_surv(encoder, p_ranges_test, pred_method='mean'):
    # p_ranges = list(itertools.chain.from_iterable(p_ranges_test))
    # p_ranges = [[el] for el in p_ranges]
    encoder.eval()
    pred_durations, true_durations, is_observed_all = [], [], []
    with torch.no_grad():
        is_observed_indices = []
        pred_obs_durations, true_obs_durations = [], []
        # total_surv_probs = []
        # print(p_ranges)
        z = 0
        g_i = 0
        for p_ranges in p_ranges_test:
            z += 1
            sigmoid_preds = encoder(p_ranges)[0]
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            pred_durations += torch.sum(surv_probs, dim=1).tolist()
            true_durations_cur = np.asarray(dataset_dict['durations'].iloc[p_ranges].squeeze().tolist())
            true_durations += np.asarray([int(duration * 12 / 365) for duration in true_durations_cur]).tolist()
            is_observed = dataset_dict['is_observed'].iloc[p_ranges].squeeze().tolist()
            is_observed_all += is_observed
            for i, idx in enumerate(is_observed):
                if is_observed[i] == 1:
                    is_observed_indices.append(g_i)
                g_i += 1

        pred_durations = np.array(pred_durations)
        true_durations = np.array(true_durations)
        pred_obs_durations = pred_durations[is_observed_indices]
        true_obs_durations = true_durations[is_observed_indices]

        # print(pred_durations, len(pred_durations))
        # print(true_durations, len(true_durations))
        # print(is_observed_indices, len(is_observed_indices))
        # print(is_observed_all, len(is_observed_all))

        # total_surv_probs = torch.stack(pred_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))
        is_observed_all = np.asarray(is_observed_all, dtype=bool)
        print(f"Number of observed patients is : {len(is_observed_indices)}")
        print(f"Number of censored patients is : {len(is_observed_all) - len(is_observed_indices)}")

        print('pred durations OBS', pred_durations[is_observed_all].round())
        print('true durations OBS', true_durations[is_observed_all].round())
        print('pred durations CRS', pred_durations[~is_observed_all].round())
        print('true durations CRS', true_durations[~is_observed_all].round())
        test_cindex = concordance_index(true_durations, pred_durations, is_observed_all)

    return test_cindex, mae_obs, None


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def eval_curr_model_on(clf, dataset_dict, a, surv=False):
    with torch.no_grad():
        preds_test, true_test = [], []
        for batch_num, p_range in enumerate(a):
            if not surv:

                y_pred = clf(p_range)
                # y_pred = clf(p_range)
            else:
                y_pred = clf(p_range)[1]
            preds_test += y_pred.tolist()
            true_test += list(dataset_dict['outcomes_filt'].iloc[list(p_range)].values)
        return roc_auc_score(true_test, preds_test)


def plot_roc_curve(a, fig_name):
    with torch.no_grad():
        preds_test, true_test = [], []
        for batch_num, p_range in enumerate(a):
            y_pred = clf(p_range)
            preds_test += y_pred.tolist()
            true_test += list(dataset_dict['outcomes_filt'].iloc[list(p_range)].values)
        fpr, tpr, _ = roc_curve(true_test, preds_test)
        plt.plot(fpr, tpr)
        plt.savefig(fig_name)


def get_batches(arr, mbsz):
    curr, ret = 0, []
    while curr < len(arr) - 1:
        ret.append(arr[curr: curr + mbsz])
        curr += mbsz
    return ret


def survival_pre_process(ts, es, max_time):
    masks, labels = [], []
    for duration, is_observed in zip(ts, es):
        # duration = int(duration * 12 / 365)
        # if duration > max_time - 1:
        #    duration = max_time - 1
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


def train_linear(reg_lambdas, X_train, y_train, X_test, y_test, val_size, ):
    # train the regression model over several choices of regularization parameter
    lr_val_aucs = []
    for reg_lambda in reg_lambdas:
        clf_lr = lr_models.gen_lr_pipeline(reg_lambda)
        clf_lr.fit(X_train, y_train)
        pred_lr = clf_lr.predict_proba(X_test[:val_size])[:, 1]
        lr_val_aucs.append(roc_auc_score(y_test[:val_size], pred_lr))
        print('Validation AUC: {0:.3f}'.format(roc_auc_score(y_test[:val_size], pred_lr)))
        print('Train AUC: {0:.3f}'.format(roc_auc_score(y_train, clf_lr.predict_proba(X_train)[:, 1])))

    # pick the model with the best regularization
    clf_lr = lr_models.gen_lr_pipeline(reg_lambdas[np.argmax(lr_val_aucs)])
    clf_lr.fit(X_train, y_train)

    pred_lr_all = clf_lr.predict_proba(feature_matrix_counts)[:, 1]
    pred_lr = clf_lr.predict_proba(X_test[val_size:])[:, 1]
    print('Linear Model Test AUC: {0:.3f}'.format(roc_auc_score(y_test[val_size:], pred_lr)))
    coeffs = clf_lr['lr'].coef_[0]
    to_print = []
    for i, elem in enumerate(coeffs):
        if elem > 0:
            to_print.append(i)
    return pred_lr_all, to_print


def tensor_to_csv(t, path):
    t_np = t.numpy()  # convert to Numpy array
    df = pd.DataFrame(t_np)  # convert to a dataframe
    df.to_csv(path, index=False)  # save to file


def get_dict(d, path, with_save=False):
    if with_save:
        a_file = open(path, "wb")
        pickle.dump(d, a_file)
        a_file.close()
    return pickle.load(open(path, "rb"))


def save_for_inference(indices_for_surv, clf, db, cdm_schema_name, cohort_name, directory):
    """
    Creating 6 csv files:
    train_features : (id, embeddings \in R^(emb_size))
    val_features:   (id, embeddings \in R^(emb_size))
    test_features: (id, embeddings \in R^(emb_size))
    train_labels: (id, T, E)
    test_labels: (id, T, E)
    val_labels: (id, T, E)
    """
    clf.eval()

    for set_ in ["train", "val", "test"]:
        indices_ = indices_for_surv[f"p_ranges_{set_}"]
        with open(f"{directory}/{set_}_labels.csv", 'w') as file:
            output = csv.writer(file, delimiter=',')
            output.writerow(["t", "e"])
            for batch in indices_:
                sqlq = f"""select t, e from {cdm_schema_name}.{cohort_name} where example_id in {tuple(batch)}"""
                df = db.query(sqlq)
                output.writerows(df.values.tolist())
        with open(f"{directory}/{set_}_features.csv", 'w') as file:
            output = csv.writer(file, delimiter=',')
            first_attn = clf.get_attn_vec(indices_[0])
            print(first_attn.shape)
            num_features = first_attn.shape[1]
            output.writerow(list(range(num_features)))
            output.writerows(first_attn.tolist())
            for batch in indices_[1:]:
                attn = clf.get_attn_vec(batch)
                output.writerows(attn.tolist())


def survival_learner(dataset_dict, feature_matrix_counts, indices_train, indices_test, cox=True, RSF=True,
                     deepsurv=True, pca_dim=None):
    # from lifelines import CoxPHFitter
    # from cox_fit import CoxPHFitter
    # from pysurvival.models.survival_forest import RandomSurvivalForestModel
    # from pysurvival.utils.metrics import concordance_index as ci
    # from pysurvival.utils.metrics import mean_absolute_error as mae
    # from pysurvival.models.semi_parametric import CoxPHModel
    # from sklearn.decomposition import PCA
    t = time.time()
    feature_matrix_counts = feature_matrix_counts.toarray()
    # is_observed = dataset_dict['is_observed'].tolist()
    # is_observed_indices = [i for i in range(len(is_observed)) if is_observed[i] == 1]

    # Adding Non-temporal features
    genders = dataset_dict['genders'].tolist()
    births = dataset_dict['births'].tolist()
    # feature_matrix_counts = np.column_stack((feature_matrix_counts, genders, births))
    feature_matrix_counts = hstack((feature_matrix_counts, csr_matrix(genders).T, csr_matrix(births).T))
    X_train = feature_matrix_counts.tocsr()[indices_train].toarray()
    X_test = feature_matrix_counts.tocsr()[indices_test].toarray()
    T_train = dataset_dict['durations'][indices_train].tolist()
    T_test = dataset_dict['durations'][indices_test].tolist()

    T_train = np.floor(np.array(T_train) * 12 / 365).tolist()
    T_test = np.floor(np.array(T_test) * 12 / 365).tolist()

    E_train = dataset_dict['is_observed'][indices_train].tolist()
    E_test = dataset_dict['is_observed'][indices_test].tolist()
    E_train = list(map(bool, E_train))
    E_test = list(map(bool, E_test))
    Y_train = np.array(list(zip(E_train, T_train)))
    Y_test = np.array(list(zip(E_test, T_test)))
    print(X_train, X_train.shape, "\n")
    print(X_test, X_test.shape, "\n")
    print(T_train, len(T_train), "\n")
    print(T_test, len(T_test), "\n")
    print(E_train, len(E_train), "\n")
    print(E_test, len(E_test), "\n")
    print(Y_train, Y_train.shape, "\n")
    print(Y_test, Y_test.shape, "\n")
    obs_indices_train = [i for i in range(len(indices_train)) if E_train[i] == 1]
    obs_indices_test = [i for i in range(len(indices_test)) if E_test[i] == 1]

    print("end preprocess ", time.time() - t)
    t = time.time()
    t = time.time()

    if pca_dim > 0:
        t_pca = time.time()
        pca = PCA(n_components=pca_dim)
        X_train = pca.fit_transform(X_train)
        X_test = pca.fit_transform(X_test)
        print("end PCA ", time.time() - t_pca)
    print(X_train, X_train.shape, "\n")
    print(X_test, X_test.shape, "\n")

    Y_train = np.core.records.fromarrays(Y_train.transpose(), names='E, T', formats='?, f8')
    Y_test = np.core.records.fromarrays(Y_test.transpose(), names='E, T', formats='?, f8')

    if cox:
        t = time.time()
        print("Started COX")
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        cox = CoxPHSurvivalAnalysis(verbose=3).fit(X_train, Y_train)
        print(f"cox fit is completed, {time.time() - t}")
        c_index = cox.score(X_test, Y_test)

        preds = cox.predict_survival_function(X_test[obs_indices_test])
        T_preds = []
        for p in preds:
            T_preds.append(sum(p.y))
        T_preds = np.array(T_preds)

        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds - np.array(T_test)[obs_indices_test])))

    if RSF:
        print("Started RSF")
        from sksurv.ensemble import RandomSurvivalForest
        rsf = RandomSurvivalForest(n_estimators=1, min_samples_split=10,
                                   min_samples_leaf=15, max_features="sqrt", n_jobs=4, random_state=1)
        rsf.fit(X_train, Y_train)
        print(f"RSF fit is completed, {time.time() - t} ")
        c_index = rsf.score(X_test, Y_test)

        preds = rsf.predict_survival_function(X_test[obs_indices_test])
        T_preds = []
        for p in preds:
            T_preds.append(sum(p.y))
        T_preds = np.array(T_preds)

        print('C-index: {:.4f}'.format(c_index))
        print('MAE: {:.4f}'.format(np.mean(T_preds - np.array(T_test)[obs_indices_test])))

        print('C-index: {:.4f}'.format(c_index))

        # mae_score = sklearn.metrics.mean_absolute_error(T_test.loc[is_observed_indices].tolist(), rsf.predict(X_test.loc[is_observed_indices, :]))
        # print('mae score: {:.4f}'.format(mae_score))
        print(f"RSF takes : {time.time() - t}")

        """
        coxph = CoxPHModel()
        coxph.fit(X_train, T_train, E_train, lr=1, l2_reg=1e-2, init_method='uniform', dis)
        c_index = ci(coxph, X_test, T_test, E_test)
        print('C-index: {:.4f}'.format(c_index))
        mae_score = mae(T_test[is_observed_indices].tolist(), coxph.predict_risk(X_test[is_observed_indices]))
        print('mae score: {:.4f}'.format(mae_score))

        from lifelines import CoxPHFitter
        cph = CoxPHFitter()
        cph.fit(df_train, duration_col='durations', event_col='is_observed')
        cph.predict_expectation(df_test)
    print(f"cox takes : {time.time() - t}")
        """
    """
    if deepsurv:
        from pycox.models import CoxPH
        from pycox.evaluation import EvalSurv
        import torchtuples as tt
        in_features = X_train.shape[1]
        num_nodes = [32, 32]
        out_features = 1
        batch_norm = True
        dropout = 0.1
        output_bias = False
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)
        model = CoxPH(net, tt.optim.Adam)
        batch_size = 256
        lrfinder = model.lr_finder(X_train, y_train, batch_size, tolerance=10)
        model.optimizer.set_lr(0.1*lrfinder.get_best_lr())
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True
        model.fit(X_train, y_train, batch_size, epochs, callbacks, verbose,
                        val_data=val, val_batch_size=batch_size)
        surv = model.predict_surv_df(X_test)
        ev = EvalSurv(surv, T_test, E_test, censor_surv='km')
        print(ev.concordance_td())
    """


if __name__ == '__main__':

    torch.cuda.empty_cache()
    #print(torch.cuda.mem_get_info(0))
    #print(torch.cuda.mem_get_info(1))
    # ********************* Loading libraries and device ***************************************************************
    visit_transformer = importlib.reload(visit_transformer)
    data_utils = importlib.reload(data_utils)
    embedding_utils = importlib.reload(embedding_utils)
    assert (torch.cuda.is_available())
    torch.cuda.set_device(0)
    #print(torch.cuda.mem_get_info())

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
    # urinary infection (81902) | ckd2 codes (443601, 443597) | ckd4 codes (443612, 443611) |
    # diabetes 2 (201826, 40482801) | hypercholesterolemia (437827) | disorder due to diabetes 2 (443732, 443731)
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
    # exit(0)
    # ********************* Creating Feature Set **********************************************************************

    cache_data_path, nontemporal_cache_data_path = "temporal_file.csv", "nontemporal_file.csv"
    featureSet = FeatureGenerator.FeatureSet(db)
    featureSet.add_default_features(['drugs', 'conditions', 'procedures'], cdm_schema_name, cohort_name)

    # featureSet.add_default_features(['age', 'gender'], cdm_schema_name, cohort_name, temporal=False)
    featureSet.build(cohort, from_cached=True, cache_file=cache_data_path)
    # print(featureSet.concept_map)
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

    #plt.hist(list(n_visits.values()), bins=100)
    #plt.show()
    visit_days_rel = {i: (pd.to_datetime(params['training_end_date']) - pd.to_datetime(featureSet.time_map[i])).days for
                      i
                      in featureSet.time_map}
    vdrel_func = np.vectorize(visit_days_rel.get)
    visit_time_rel = [vdrel_func(v) for v in visit_times_raw]

    maps = {'concept': featureSet.concept_map, 'id': featureSet.id_map, 'time': featureSet.time_map}
    # print(n_visits)
    # plt.hist(list(n_visits[p] for p in range(len(n_visits))), bins=50)
    # plt.show()
    # exit(0)
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
    balance_surv_oversample = False
    balance_surv_downsample = False

    if balance_surv_oversample:
        # random.shuffle(indices_train*10)
        is_obs_train = dataset_dict['is_observed'].iloc[list(indices_train)].values
        p_add = []
        for i, idx in enumerate(indices_train):
            if is_obs_train[i] == 1:
                p_add.append(idx)
        a = len(p_add) / len(indices_train)
        print(f"number of observed is {len(p_add)}, a = {a}, number of patients is {len(indices_train)}")
        # p_add *= (int((1 - a) / (a)) - 1)
        p_add *= 1
        # TODO: balance by survival time distribution
        indices_train += p_add
        # indices_train *= 5
        print(f"number of patients is {len(indices_train)}")

    if balance_surv_downsample:
        # random.shuffle(indices_train*10)
        is_obs_train = dataset_dict['is_observed'].iloc[list(indices_train)].values
        obs, cens = [], []
        for i, idx in enumerate(indices_train):
            if is_obs_train[i] == 0:
                cens.append(idx)
            else:
                obs.append(idx)
        if len(obs) < len(cens):
            indices_train = obs + random.sample(cens, int(len(obs) * 0.66))
        # TODO: balance by survival time distribution
    print("len of indices train", len(indices_train), schema_name)
    print("len of indices test", len(indices_test), schema_name)


    mbsz = 256
    # p_ranges_train, p_ranges_test = [get_batches(arr, mbsz) for arr in (indices_train, indices_test)]
    # p_ranges_val = p_ranges_test[:val_size // mbsz]
    # p_ranges_test = p_ranges_test[val_size // mbsz:]
    save_for_baselines = True
    if save_for_baselines:
        with open("feature_matrix_counts.pkl", "wb") as h:
            pickle.dump(feature_matrix_counts, h, protocol=pickle.HIGHEST_PROTOCOL)

    exit(0)
    # ********************* SAVE OBJECTS FOT TRAIN FILE ******************************************************
    featureSet._db = None
    data_to_train = {"dataset_dict":dataset_dict, "featureSet":featureSet}
    with open("data_to_train.pkl", "wb") as h:
        pickle.dump(data_to_train, h, protocol=pickle.HIGHEST_PROTOCOL)




    #exit(0)
    # ********************* Linear model **********************************************************************
    # pred_lr_all, reg_lambdas = [], [2, 1, 0.5, 0.2, 0.02, 0.002]
    # pred_lr_all, feature_names_indices = train_linear(reg_lambdas, X_train, y_train, X_test, y_test, val_size)

    # ********************* Embedding ****************************************************************
    to_train_enbedding = True
    all_concept_map_reverse = get_dict(featureSet.concept_map_rev, 'all_concept_map_reverse.pkl',
                                       with_save=to_train_enbedding)
    # There are 26350 concepts
    embedding_dim = 128  # size of embedding, must be multiple of number of heads --128
    # embedding_dim = 32
    window_days = 90  # number of days in window that defines a "Sentence" when learning the embedding
    # embedding_filename = "wtv_data_window_{}d_20epochs_wv".format(window_days) # pretrained
    # embedding_filename = f"wtv_data_window_{window_days}d_20epochs_wv_big{embedding_dim}"
    # embedding_filename = f"wtv_data_window90_urinary2ckd"
    # embedding_filename = f"wtv_data_window90_diabetes2due_to_diabetes"
    # embedding_filename = f"wtv_data_window90_hypercholesterolemia2atrial_fibrillation_dim256"
    # embedding_filename = 'SNOMEDCT_isa.txt.emb_dims_200.nthreads_1.txt'

    embedding_filename = f"embedding_all_data_90d128"
    model_filename = f"{embedding_filename}_model"
    if to_train_enbedding:
        train_coords = np.nonzero(np.where(np.isin(person_ixs, range(len(observes))), 1, 0))
        embedding_filename = embedding_utils.train_embedding(featureSet, feature_matrix_3d_transpose, window_days,
                                                             person_ixs[train_coords], time_ixs[train_coords],
                                                             remap['time'],
                                                             embedding_dim, embedding_filename)
    # embedding_utils.plot_embedding(model_filename)
    # embedding_utils.closest_embedding(model_filename, all_concept_map_reverse)
    exit(0)
    # ********************* Visit Transformer *******************************************************
    torch.cuda.empty_cache()
    mn_prefix = 'eol_experiment_prefix'  # Name for the model (mn_prefix) that will be used when saving checkpoints
    n_heads = 1
    assert embedding_dim % n_heads == 0
    model_params = {
        'embedding_dim': int(embedding_dim / n_heads), 'n_heads': n_heads, 'attn_depth': 1, 'dropout': 0.3,
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
    max_time = 30
    dataset_dict['durations'] = dataset_dict['durations'] * 12 / 365
    # print(dataset_dict['durations'])
    dataset_dict['durations'] = dataset_dict['durations'].apply(np.floor).astype('int32')
    # print(dataset_dict['durations'])
    dataset_dict['durations'] = dataset_dict['durations'].clip(0, max_time - 1)
    # print(dataset_dict['durations'])
    # exit(0)
    masks, labels = survival_pre_process(dataset_dict['durations'], dataset_dict['is_observed'], max_time)

    # ********************* Data Loaders *******************************************************
    torch.cuda.empty_cache()
    #print(torch.cuda.mem_get_info(0))
    #print(torch.cuda.mem_get_info(1))

    train_loader = DataLoader(
        MyDataset(indices_train, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                  train=True), batch_size=mbsz, shuffle=True)
    test_loader = DataLoader(
        MyDataset(indices_test, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                  train=False), batch_size=mbsz, shuffle=False)

    """
    itw = indices_test[0]
    codes_of_patients = all_codes_tensor[person_indices[itw]: person_indices[itw+1]]
    print(f"codes_of_patients : {codes_of_patients}")
    print(list(featureSet.concept_map[x] for x in codes_of_patients))


    for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
        print(X, m, mask, label, duration, is_observed)
        break

    exit(0)
    """
    # ********************* Survival Baselines *******************************************************
    linear_baseline = False
    embedding_baseline = False

    if linear_baseline:
        survival_learner(dataset_dict, feature_matrix_counts, indices_train, indices_test, cox=True, RSF=True,
                         deepsurv=False, pca_dim=200)
    if embedding_baseline:
        data = MyDataset(indices_all, base_model, masks, labels, dataset_dict['durations'], dataset_dict['is_observed'],
                         train=False).X
        data = np.array([d.flatten().cpu().detach().numpy() for d in data])
        print(data.shape)

        survival_learner(dataset_dict, coo_matrix(data), indices_train, indices_test, cox=True, RSF=True,
                         deepsurv=False, pca_dim=200)
    if linear_baseline or embedding_baseline:
        exit(0)
    # ********************* Hyper parameters *******************************************************
    lr = 2e-3  # * 0.01
    n_epochs_pretrain = 1000
    update_mod = 30  # update every update_mod batches

    # ********************* Survival Architecture *******************************************************
    # clf = visit_transformer.VTClassifer(base_model, **model_params).cuda()
    c = copy.deepcopy
    # survival_architecture_parameters:
    sap = {"num_heads": 1, "d_model": 128, "drop_prob": 0.3, "num_features": embedding_dim, "N": 1, "d_ff": 128}
    attn = MultiHeadedAttention(sap["num_heads"], sap["d_model"], sap["drop_prob"])
    ff = PositionwiseFeedForward(sap["d_model"], sap["d_ff"], sap["drop_prob"])
    encoder_layer = EncoderLayer(sap["d_model"], c(attn), c(ff), sap["drop_prob"])
    surv_encoder = Encoder(encoder_layer, sap["N"], sap["d_model"], sap["drop_prob"], sap["num_features"]).cuda()
    clf = visit_transformer.SA(base_model, surv_encoder, max_time, **model_params).cuda()
    #clf.apply(initialize_weights)
    # clf = visit_transformer.Risk2Surv(base_model, surv_encoder, max_time, **model_params).cuda()

    optimizer_clf = torch.optim.Adam(params=clf.parameters(), lr=lr)# , weight_decay=20)
    # torch.nn.utils.clip_grad_norm_(clf.parameters(), 50)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_clf, step_size=200, gamma=0.9)
    num_batches = int(len(train_loader) / mbsz) + 1
    task = 'eol'
    model_path = "model1"
    # print("number of parameters: ", sum(p.numel() for p in clf.parameters()))
    count_parameters(clf)

    #print(torch.cuda.mem_get_info())
    print(torch.cuda.memory_allocated())
    train_losses, test_losses = [], []
    for epoch in range(n_epochs_pretrain):
        t, batch_loss = time.time(), 0
        for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(train_loader):

            # for idx in range(len(X)):
            #    print(X[idx], "\n", mask[idx], "\n", label[idx], "\n", duration[idx], "\n", is_observed[idx])
            #    print(X[idx].shape, "\n", mask[idx].shape, "\n", label[idx].shape, "\n", duration[idx].shape, "\n", is_observed[idx].shape)
            #
            #    print("\n\n\n")
            mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
            label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
            # print(X.shape)
            surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=True)
            surv_probs = torch.cumprod(surv_pred, dim=1)
            # surv_probs = surv_pred

            # surv_probs = torch.cumsum(surv_pred, dim=1)
            # label = torch.cumsum(label, dim=0)
            if batch_num == 0:
                # print(surv_probs)
                print(torch.var(surv_probs, dim=0))
                # print(label * mask)

            #weights = torch.tensor([1.0, 1.0, 10.0,20.0, 30.0, 40.0, 50.0, 40.0, 30.0, 20.0, 10.0, 1.0, 1.0, 1.0, 1.0]).cuda()
            loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)

            # + \
            #        0.0001*torch.nn.MSELoss(reduction='sum')(torch.sum(surv_probs, dim=1).cuda(), torch.tensor(duration).float().cuda())
            # loss = torch.nn.MSELoss(reduction='sum')(surv_probs * mask, label * mask)
            batch_loss += loss.item()
            loss.backward()
            # if batch_num == num_batches - 2:
            #   plot_grad_flow(clf.named_parameters())
            # plot_grad_flow(clf.named_parameters())
            if batch_num % update_mod == 0:
                optimizer_clf.step()
                optimizer_clf.zero_grad()
                # model.zero_grad(set_to_none=True)
            # if batch_num % 100 == 0: plot_grad_flow(clf.named_parameters())

        batch_loss /= len(indices_train)

        t = time.time()
        clf.eval()
        with torch.no_grad():
            test_loss = 0
            for batch_num, (X, m, mask, label, duration, is_observed) in enumerate(test_loader):
                mask = torch.tensor(np.vstack(mask).astype(np.float)).cuda()
                label = torch.tensor(np.vstack(label).astype(np.float)).cuda()
                surv_pred, y_pred = clf(X.cuda(), m.cuda(), train=False)
                #surv_probs = surv_pred
                surv_probs = torch.cumprod(surv_pred, dim=1)
                # surv_probs = torch.cumsum(surv_pred, dim=1)
                # label = torch.cumsum(label, dim=0)
                loss = torch.nn.BCELoss(reduction='sum')(surv_probs * mask, label * mask)

                # loss = torch.nn.MSELoss(reduction='sum')(surv_probs * mask, label * mask)
                test_loss += loss.item()
            test_loss /= len(indices_test)
            # print(f'test loss: {test_loss}')
            print(f'Epochs: {epoch + 1} |  loss: {batch_loss} |  test loss: {test_loss}| epoch time: {time.time() - t}')
            train_cindex, train_mae, tarin_total_surv_probs = evaluate_surv_new(clf, train_loader, train=False)
            print('current train cindex', train_cindex, 'train mae', train_mae)
            val_cindex, val_mae, val_total_surv_probs = evaluate_surv_new(clf, test_loader, train=False)
            print('current val cindex', val_cindex, 'val mae', val_mae)
            # test_cindex, test_mae, test_total_surv_probs = evaluate_surv(clf, p_ranges_test)
            # print('current test cindex', test_cindex, 'test mae', test_mae)
            # print(f"Evaluation time : {time.time() - t}")
            print("\n")
        clf.train()
        scheduler.step()
        train_losses.append(batch_loss)
        test_losses.append(test_loss)
        if (epoch + 1) % 20 == 0:
            plot_loss(train_losses, test_losses, epoch)

    # ********************* Save model *******************************************************
    # torch.save(clf.state_dict(), model_path)
    # indices_for_surv = {"p_ranges_train": p_ranges_train, "p_ranges_val": p_ranges_val, "p_ranges_test": p_ranges_test}
    # directory = "surv_data_big_100"
    exit(0)

    # TODO:
    # 1. output s(t) instead of q(t)
    # 2. loss on duration instead of every month
