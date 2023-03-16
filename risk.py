import copy
import random
import pickle
import Utils.data_utils as data_utils
import Utils.embedding_utils as embedding_utils
import Models.LogisticRegression.RegressionGen as lr_models
import Models.Transformer.visit_transformer as visit_transformer
import numpy as np
import torch
import torch.nn
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import importlib
from survival import MultiHeadedAttention, PositionwiseFeedForward, EncoderLayer, Encoder
from prettytable import PrettyTable
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 13


# ************************* Classes and Functions **************************************

class RiskDataset(Dataset):
    """ Class for risk dataset """
    def __init__(self, indices, base_model, labels, train=True):
        """
            :param indices: indices of patients
            :param base_model: the class that creates the patient representation
            :param labels: label of target event
            :param train: train/test set
        """

        self.base_model = base_model
        self.X = []
        self.M = []
        self.labels = [labels[x] for x in indices]

        chunk_size: int = 5000  # Must be int, worked with 500
        chunk_size = int(min(chunk_size,len(indices)))
        num_chunks = int(len(indices) / chunk_size)
        print(num_chunks)
        for chunk in range(num_chunks):
            if chunk != num_chunks - 1:
                x, m = self.base_model(indices[chunk * chunk_size:(chunk + 1) * chunk_size], train, return_mask=True)
            else:
                x, m = self.base_model(indices[chunk * chunk_size:], train, return_mask=True)
            x = x.cpu()
            m = m.cpu()

            self.M += list(m)
            self.X += list(x)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx],  self.labels[idx]

    def __len__(self):
        return len(self.X)


def plot_loss(train_loss, test_loss, epoch):
    """
    :param train_loss: train set loss values
    :param test_loss: test set loss values
    :param epoch: The last epoch to include in the plot
    :return: Plot loss values for each epoch
    """
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


def count_parameters(model):
    """
    Counts model parameters for each layer
    """
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


def train_linear(reg_lambdas, X_train, y_train, X_test, y_test, val_size, ret_all=False):
    # train the regression model over several choices of regularization parameter
    lr_val_aucs = []
    for reg_lambda in reg_lambdas:
        clf_lr = lr_models.gen_lr_pipeline(reg_lambda)
        clf_lr.fit(X_train, y_train)
        pred_lr = clf_lr.predict_proba(X_test[:val_size])[:, 1]
        lr_val_aucs.append(roc_auc_score(y_test[:val_size], pred_lr))
        print('Validation AUC: {0:.5f}'.format(roc_auc_score(y_test[:val_size], pred_lr)))
        print('Train AUC: {0:.5f}'.format(roc_auc_score(y_train, clf_lr.predict_proba(X_train)[:, 1])))

    # pick the model with the best regularization
    idx = np.argmax(lr_val_aucs) # highest
    idx = np.argsort(lr_val_aucs)[0] # The second highest
    clf_lr = lr_models.gen_lr_pipeline(reg_lambdas[idx])
    clf_lr.fit(X_train, y_train)

    pred_lr_all = clf_lr.predict_proba(X_test)[:, 1]
    pred_lr = clf_lr.predict_proba(X_test[val_size:])[:, 1]

    print('Linear Model Test AUC: {0:.5f}'.format(roc_auc_score(y_test[val_size:], pred_lr)))
    to_print = []
    if ret_all:
        fpr, tpr, thr = roc_curve(np.array(y_test[val_size:]), np.array(pred_lr), drop_intermediate=False)
        print(len(thr))
        pickle.dump((fpr, tpr), open(f"LR_ROC_{ret_all}", "wb"))
    return pred_lr_all, to_print


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


def eval_curr_model_on(clf, test_loader, ret_all=False):
    """

    :param clf: model
    :param test_loader: test dataset
    :param ret_all: if to return fpr and tpr values (ret all = epoch)
    :return: roc_auc score
    """
    clf.eval()
    preds_test, true_test = [], []
    with torch.no_grad():
        for batch_num, (X, m, label) in enumerate(test_loader):
            y_pred = clf(X.cuda(), m.cuda())
            preds_test += y_pred.tolist()
            true_test += label.tolist()
    print(len(true_test), len(preds_test))
    if ret_all:
        fpr, tpr, thr = roc_curve(np.array(true_test), np.array(preds_test), drop_intermediate=False)
        print(len(thr))
        pickle.dump((fpr, tpr), open(f"SARD_ROC_{ret_all}", "wb"))

    return roc_auc_score(np.array(true_test), np.array(preds_test))


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
    featureSet = data_to_train["featureSet"]

    # ********************* Risk eligible patients selection (exclude censored) *********************************

    risk_time = 365*2   # Fixed risk prediction time
    risk_indices = []
    risk_labels = {}

    for i, (d, e) in enumerate(zip(dataset_dict["durations"], dataset_dict["is_observed"])):
        if e == 1:
            risk_indices.append(i)
            if d<risk_time:
                risk_labels[i] = 1
            else:
                risk_labels[i] = 0
        if e == 0 and d>risk_time:
            risk_indices.append(i)
            risk_labels[i] = 0

    vals = [x for x in risk_labels.values()]
    print(vals.count(1), vals.count(0))  # prints number of positive and negative samples
    indices_all = risk_indices
    print(len(risk_indices))

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        np.zeros(shape=(len(indices_all), 2)), range(len(indices_all)), indices_all, test_size=0.2, random_state=random.randint(0,100),
        stratify=list(risk_labels.values()))

    assert set(indices_train).union(set(indices_test)) == set(risk_indices)
    assert len(set(indices_train).intersection(set(indices_test))) == 0
    to_train_enbedding = False
    mbsz = 256
    print("Num train samples : ", len(indices_train))
    print("Num test samples : ", len(indices_test))
    # ********************* Linear model **********************************************************************
    risk_baseline = False
    if risk_baseline:
        val_size = int(len(indices_test) / 2)
        pred_lr_all, reg_lambdas = [], [2, 1, 0.5, 0.2, 0.02, 0.002]
        feature_matrix_counts = pickle.load(open("feature_matrix_counts.pkl", "rb"))
        pred_lr_all, feature_names_indices = \
        train_linear(reg_lambdas, feature_matrix_counts[indices_train], dataset_dict['is_observed'].iloc[indices_train],
                     feature_matrix_counts[indices_test], dataset_dict['is_observed'].iloc[indices_test], val_size)

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

    #********************** Risk baseline with embeddings ********************************************************
    risk_baseline_embedding = False
    if risk_baseline_embedding:
        indices_train_in_risk = [i for i,x in enumerate(risk_indices) if x in indices_train]
        indices_test_in_risk = [i for i,x in enumerate(risk_indices) if x in indices_test]
        indices_train = indices_train_in_risk
        indices_test = indices_test_in_risk
        all_loader = DataLoader(
                RiskDataset(indices_all, base_model, risk_labels, train=True), batch_size=len(indices_all), shuffle=True)


        for batch_num, (X, m, label) in enumerate(all_loader): # only one batch
            X_train = X[indices_train]
            X_test = X[indices_test]
            y_train = label.numpy()[indices_train]
            y_test = label.numpy()[indices_test]
            X_train = torch.sum(X_train,1).numpy()
            X_test = torch.sum(X_test, 1).numpy()
        val_size = int(len(indices_test) / 2)
        pred_lr_all, reg_lambdas = [], [2, 1, 0.5, 0.2, 0.02, 0.002]
        pred_lr_all, feature_names_indices = train_linear(reg_lambdas, X_train,y_train,X_test, y_test, val_size, ret_all=-1)
        exit(0)
    # ********************* Risk pre-process for deep model *********************************************************
    train_loader = DataLoader(
        RiskDataset(indices_train, base_model, risk_labels, train=True), batch_size=mbsz, shuffle=True)
    test_loader = DataLoader(
        RiskDataset(indices_test, base_model, risk_labels, train=False), batch_size=mbsz, shuffle=False)

    # ********************* Hyper parameters *******************************************************
    lr = 2e-3  # * 0.01
    n_epochs_pretrain = 30
    update_mod = 35  # update every update_mod batches

    # ********************* Risk Architecture *******************************************************
    c = copy.deepcopy
    # survival_architecture_parameters:
    sap = {"num_heads": 1, "d_model": 128, "drop_prob": 0.3, "num_features": embedding_dim, "N": 1, "d_ff": 128}
    attn = MultiHeadedAttention(sap["num_heads"], sap["d_model"], sap["drop_prob"])
    ff = PositionwiseFeedForward(sap["d_model"], sap["d_ff"], sap["drop_prob"])
    encoder_layer = EncoderLayer(sap["d_model"], c(attn), c(ff), sap["drop_prob"])
    surv_encoder = Encoder(encoder_layer, sap["N"], sap["d_model"], sap["drop_prob"], sap["num_features"]).cuda()
    clf = visit_transformer.Risk(base_model, **model_params).cuda()

    optimizer_clf = torch.optim.Adam(params=clf.parameters(), lr=lr  )# , weight_decay=20)
    # torch.nn.utils.clip_grad_norm_(clf.parameters(), 50)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_clf, step_size=200, gamma=0.9)
    count_parameters(clf)

    train_losses, test_losses = [], []
    loaded_model = True # skip training if the model is loaded
    for epoch in range(n_epochs_pretrain):
        if loaded_model: break
        t, batch_loss = time.time(), 0
        for batch_num, (X, m, label) in enumerate(train_loader):
            y_pred = clf(X.cuda(), m.cuda())
            try:
                loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(y_pred, torch.tensor(label, dtype=torch.float).cuda().clone().detach())
            except Exception as e:
                print(e)
                print(y_pred,y_pred.shape, " \n")
                print(label,label.shape, " \n")
            batch_loss += loss.item()
            loss.backward()
            if batch_num % update_mod == 0:
                optimizer_clf.step()
                optimizer_clf.zero_grad()
        batch_loss /= len(indices_train)
        print("loss")

        t = time.time()
        clf.eval()
        # torch.save(clf.state_dict(), f"STRAFE_epoch{epoch + 1}")
        with torch.no_grad():
            test_loss = 0
            for batch_num, (X, m, label) in enumerate(test_loader):
                y_pred = clf(X.cuda(), m.cuda())
                loss = torch.nn.BCEWithLogitsLoss(reduction='sum')(y_pred, torch.tensor(label, dtype=torch.float).cuda().clone().detach())
                test_loss += loss.item()
            test_loss /= len(indices_test)
            print(f'Epochs: {epoch + 1} |  loss: {batch_loss} |  test loss: {test_loss}| epoch time: {time.time() - t}')
            print('AUC TEST of SARD : ', eval_curr_model_on(clf, test_loader))
            print('AUC TRAIN of SARD : ', eval_curr_model_on(clf, train_loader))
            print("\n")
        clf.train()
        scheduler.step()
        train_losses.append(batch_loss)
        test_losses.append(test_loss)
        if (epoch + 1) % 20 == 0:
            plot_loss(train_losses, test_losses, epoch)

    # ********************* Evaluation *******************************************************
    epochs_list = []
    for epoch in epochs_list:
        clf.load_state_dict(torch.load(f"SARD_epoch{epoch+1}"))
        clf.eval()
        print('Risk AUC is : ', eval_curr_model_on(clf, test_loader, ret_all=epoch))
    exit(0)

