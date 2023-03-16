'''
    The original Transformer code is from: http://nlp.seas.harvard.edu/2018/04/03/attention.html

    To run this code, you also need the concordance.py file (https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/utils/concordance.py) in the same directory.

    Please note that this code is for research purposes only.
    Author: Shi Hu
    Email: s.hu@uva.nl
    Last Modified: Oct 24, 2021
'''


# This file contains the Transformer-based architecture defined in the paper:
"""
@inproceedings{hu2021transformer,
title = {Transformer-based deep survival analysis},
author = {Hu, Shi and Fridgerisson, Egill and van Wingen, Guido and Welling, Max},
booktitle = {Survival Prediction-Algorithms, Challenges and Applications},
pages={132--148},
year = {2021},
organization = {PLMR}
}

Author github: github.com/shihux/sa_transformer
"""


# This include some exceptions.
# Every function which is differ from the original is marked with the comment: "# Changed" before function definition .

import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# import seaborn
from sklearn.decomposition import PCA
# seaborn.set_context(context="talk")
import matplotlib
# from lifelines import *
import pandas as pd
import argparse
from operator import itemgetter
from concordance import concordance_index



parser = argparse.ArgumentParser(description='Survival analysis')
parser.add_argument('--max_time', type=int, default=12, help='max number of months')
parser.add_argument('--num_epochs', type=int, default=600)
parser.add_argument('--N', type=int, default=1, help='number of modules')  # 6
parser.add_argument('--num_heads', type=int, default=1)  # 8
parser.add_argument('--d_model', type=int, default=64)  # 64
parser.add_argument('--d_ff', type=int, default=16)  # 256
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--drop_prob', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--coeff', type=float, default=1.0)
parser.add_argument('--coeff2', type=float, default=1.0)
parser.add_argument('--data_dir', type=str, default='surv_original_data')  # surv_data_big_100, surv_original_data, gaussian
parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints')
parser.add_argument('--report_interval', type=int, default=5)
parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
opt = parser.parse_args()
print(opt)



# Changed
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Changed
class TranDataset(Dataset):
    def __init__(self, features, labels, is_train=True):
        self.is_train = is_train
        self.data = []
        temp = []
        for feature, label in zip(features, labels):
            feature = torch.from_numpy(feature).float()
            duration, is_observed = label[0], label[1]
            duration = int(duration * 12 / 365)
            if duration+1 > opt.max_time:
                continue
            temp.append([duration, is_observed, feature])

        sorted_temp = temp
        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, is_observed, feature in new_temp:
            if is_observed:
                mask = opt.max_time * [1.]
                label = duration * [1.] + (opt.max_time - duration) * [0.]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(),
                     torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.] + (opt.max_time - (duration + 1)) * [0.]
                label = opt.max_time * [1.]
                feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(), torch.tensor(duration).float().cuda(), torch.tensor(mask).float().cuda(),
                     torch.tensor(label).cuda(), torch.tensor(is_observed).byte().cuda()])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a + 1, len(self.data))
            return [[self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a]))]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -torch.tensor(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# initial embedding for raw input
class SrcEmbed(nn.Module):
    def __init__(self, input_dim, d_model):
        super(SrcEmbed, self).__init__()
        self.w = nn.Linear(input_dim, d_model)
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.w(x))


# Changed
# final layer for the transformer
class TranFinalLayer(nn.Module):
    def __init__(self, d_model):
        super(TranFinalLayer, self).__init__()
        self.w_one_step = nn.Linear(d_model, 1)
        self.w_1 = nn.Linear(d_model, d_model // 2, )
        self.norm = LayerNorm(d_model // 2)
        self.w_2 = nn.Linear(d_model // 2, d_model //4)
        self.norm2 = LayerNorm(d_model //4)
        self.w_3  = nn.Linear(d_model //4, 1)


    def forward(self, x):
        x = F.leaky_relu(self.w_1(x))
        x = self.norm(x)
        x = F.leaky_relu(self.w_2(x))
        x = self.norm2(x)
        x = self.w_3(x)

        return torch.sigmoid(x.squeeze(-1))


        # print(torch.sigmoid(x.squeeze(-1)))
        # print(torch.tanh(x.squeeze(-1)))
        # print(torch.softmax(x.squeeze(-1),dim=-1))
        # print(x.squeeze(-1))

        # exit(0)
        # return torch.tanh(x.squeeze(-1))
        # return torch.relu(x.squeeze(-1))+0.00000000000000001

# Changed
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N, d_model, dropout, num_features):
        super(Encoder, self).__init__()
        self.src_embed = SrcEmbed(num_features, d_model)
        self.position_encode = PositionalEncoding(d_model, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.final_layer = TranFinalLayer(d_model)
        self.lstm = nn.RNN(input_size=d_model, hidden_size=d_model, batch_first=True)#, proj_size=1)


    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        x = self.position_encode(self.src_embed(x))
        # Transformer prediction head (Un/Contextualize STRAFE)
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_layer(x)
        # LSTM prediction head (Un/Contextualize LSTM)
        #x, _ = self.lstm(x)
        #x = torch.sigmoid(x)

        # x is (batch size, max_time, d_model)
        #return self.final_layer(x)



class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def evaluate(encoder, test_loader):
    encoder.eval()
    with torch.no_grad():
        pred_durations, true_durations, is_observed = [], [], []
        pred_obs_durations, true_obs_durations = [], []

        total_surv_probs = []

        # NOTE batch size is 1
        for features, durations, mask, label, is_observed_single in test_loader:
            is_observed.append(is_observed_single)
            sigmoid_preds = encoder.forward(features)
            surv_probs = torch.cumprod(sigmoid_preds, dim=1).squeeze()
            total_surv_probs.append(surv_probs)

            if opt.pred_method == 'mean':
                pred_duration = torch.sum(surv_probs).item()
            elif opt.pred_method == 'median':
                pred_duration = 0
                while True:
                    if surv_probs[pred_duration] < 0.5:
                        break
                    else:
                        pred_duration += 1
                        if pred_duration == len(surv_probs):
                            break

            true_duration = durations.squeeze().item()
            pred_durations.append(pred_duration)
            true_durations.append(true_duration)

            if is_observed_single:
                pred_obs_durations.append(pred_duration)
                true_obs_durations.append(true_duration)

        total_surv_probs = torch.stack(total_surv_probs)

        pred_obs_durations = np.asarray(pred_obs_durations)
        true_obs_durations = np.asarray(true_obs_durations)
        mae_obs = np.mean(np.abs(pred_obs_durations - true_obs_durations))

        pred_durations = np.asarray(pred_durations)
        true_durations = np.asarray(true_durations)
        is_observed = np.asarray(is_observed, dtype=bool)
        print(len(pred_durations))
        print(len(true_durations))
        print('pred durations OBS', pred_durations[is_observed].round())
        print('true durations OBS', true_durations[is_observed].round())

        print('pred durations CRS', pred_durations[~is_observed].round())
        print('true durations CRS', true_durations[~is_observed].round())

        test_cindex = concordance_index(true_durations, pred_durations, is_observed)
        # print('c index', test_cindex, 'mean abs error (OBS)', mae_obs)

    return test_cindex, mae_obs, total_surv_probs


def checkpoint(model, total_surv_probs):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}/best_model_trainset_{}.pth".format(opt.save_ckpt_dir, datadir)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # save npy array
    surv_probs_out_path = "{}/best_surv_probs_test_{}.pth".format(opt.save_ckpt_dir, datadir)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())

# Changed
def train(features, labels, encoder):
    train_loader = DataLoader(TranDataset(features[0], labels[0], is_train=True), batch_size=opt.train_batch_size,
                              shuffle=True)

    train_infer_loader = DataLoader(TranDataset(features[0], labels[0], is_train=False), batch_size=1,
                                    shuffle=True)
    # NOTE VAL batch size is 1
    val_loader = DataLoader(TranDataset(features[1], labels[1], is_train=False), batch_size=1, shuffle=False)
    # NOTE TEST batch size is 1
    test_loader = DataLoader(TranDataset(features[2], labels[2], is_train=False), batch_size=1, shuffle=False)
    # NOTE using C index as early stopping criterion
    best_val_cindex, best_test_cindex, best_test_mae, best_epoch = -1, -1, 9999999, 0

    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    print("params num", sum(param.numel() for param in encoder.parameters()))

    for t in range(opt.num_epochs):
        print('epoch', t)
        encoder.train()

        tot_loss = 0.
        for features, true_durations, mask, label, is_observed in train_loader:
            #print(features[0].size(), "\n" ,true_durations[0].size(), "\n", mask[0].size(), "\n",label[0].size(), "\n",is_observed[0].size(),"\n")
            optimizer.zero_grad()

            is_observed_a = is_observed[0]
            mask_a = mask[0]
            mask_b = mask[1]
            label_a = label[0]
            label_b = label[1]
            true_durations_a = true_durations[0]
            true_durations_b = true_durations[1]

            sigmoid_a = encoder.forward(features[0])
            surv_probs_a = torch.cumprod(sigmoid_a, dim=1)
            mask_a = (0 + 1) * [1.] + (opt.max_time - (0 + 1)) * [0.]
            mask_a = torch.tensor(mask_a).float().cuda()
            batch_weighting = False
            if batch_weighting:
                batch_size = len(label_a)
                batch_weights = torch.zeros(batch_size).double().cuda()
                batch_weights = torch.where(is_observed_a == 0, batch_weights, 5.0)
                batch_weights = torch.where(is_observed_a == 1, batch_weights, 100.0)
                batch_weights = batch_weights.repeat(opt.max_time).reshape(batch_size, opt.max_time)
                # loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)
                loss = nn.BCELoss(weight=batch_weights)(surv_probs_a * mask_a, label_a * mask_a)
            else:
                loss = nn.BCELoss()(surv_probs_a * mask_a, label_a * mask_a)

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('train total loss', tot_loss)
        if t > 0 and t % opt.report_interval == 0 and False:
            print('TRAIN')
            train_cindex, train_mae, tarin_total_surv_probs = evaluate(encoder, train_infer_loader)
            print('current train cindex', train_cindex, 'train mae', train_mae)

        # evaluate VAL and TEST
        if t > 0 and t % opt.report_interval == 0 and True:
            print('TRAIN')
            train_cindex, train_mae, tarin_total_surv_probs = evaluate(encoder, train_infer_loader)
            print('current train cindex', train_cindex, 'train mae', train_mae)

            print('VAL')
            val_cindex, val_mae, val_total_surv_probs = evaluate(encoder, val_loader)
            print('TEST')
            test_cindex, test_mae, test_total_surv_probs = evaluate(encoder, test_loader)

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex
                best_val_mae = val_mae
                best_test_cindex = test_cindex
                best_test_mae = test_mae
                best_epoch = t
                checkpoint(encoder, test_total_surv_probs)

            print('current val cindex', val_cindex, 'val mae', val_mae)
            print('BEST val cindex', best_val_cindex, 'mae', best_val_mae, 'at epoch', best_epoch)
            print('its test cindex', best_test_cindex, 'mae', best_test_mae)


def main(features, labels, num_features):
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = Encoder(encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features).cuda()
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda()
    train(features, labels, encoder)







if __name__ == '__main__':
    assert (torch.cuda.is_available())
    torch.cuda.set_device(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device : {device}')

    train_features = pd.read_csv(opt.data_dir + '/train_features.csv', header=0, index_col=False).to_numpy()
    val_features = pd.read_csv(opt.data_dir + '/val_features.csv', header=0, index_col=False).to_numpy()
    test_features = pd.read_csv(opt.data_dir + '/test_features.csv', header=0, index_col=False).to_numpy()
    features = [train_features, val_features, test_features]

    train_labels = pd.read_csv(opt.data_dir + '/train_labels.csv', header=0, index_col=False).to_numpy()
    val_labels = pd.read_csv(opt.data_dir + '/val_labels.csv', header=0, index_col=False).to_numpy()
    test_labels = pd.read_csv(opt.data_dir + '/test_labels.csv', header=0, index_col=False).to_numpy()
    labels = [train_labels, val_labels, test_labels]

    num_features = train_features.shape[1]
    # num_features = 100
    print('train features shape', train_features.shape)
    print('train labels shape', train_labels.shape)
    print('val features shape', val_features.shape)
    print('val labels shape', val_labels.shape)
    print('test features shape', test_features.shape)
    print('test labels shape', test_labels.shape)
    print()

    print('num features', num_features)
    total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
    print('total data', total_data)
    print()

    print('train max label', train_labels[:, 0].max())
    print('train min label', train_labels[:, 0].min())
    print('val max label', val_labels[:, 0].max())
    print('val min label', val_labels[:, 0].min())
    print('test max label', test_labels[:, 0].max())
    print('test min label', test_labels[:, 0].min())


    train_files = [opt.data_dir + "/train_features.csv", opt.data_dir + "/train_labels.csv"]
    test_files =  [opt.data_dir + "/test_features.csv", opt.data_dir + "/test_labels.csv"]
    main(features, labels, num_features)




