import random

from matplotlib import pyplot as plt
import matplotlib
import Models.Transformer.transformers as transformers
from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn
import math
import importlib
import pandas as pd


importlib.reload(transformers)

"""
All the code was taken from omop-learn package: github.com/clinicalml/omop-learn
This include some exceptions. 
Every function which is differ from the original is marked with the comment: "# Changed" before function definition .
"""

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class VTClassifer(torch.nn.Module):
    def __init__(
            self, bert_model,
            n_targets=1,
            attn_classifier=True,  # False
            pred_dropout=True,
            rnn_classifier=False,
            visit_dropout=False,
            convolutional_classifier=False,
            two_lin_layer_conv=False,
            multi_visit_kernel_conv=False,
            kernel_size_visits=2,
            averaging_classifier=False,
            **kwargs
    ):
        super(VTClassifer, self).__init__()
        self.n_targets = n_targets
        self.emb_size = bert_model.embedding_dim
        self.bert = bert_model
        self.attn_classifier = attn_classifier
        self.n_parallel_pools = 1  # 10
        self.pred_dropout = pred_dropout
        self.rnn_classifier = rnn_classifier
        self.visit_dropout = visit_dropout
        self.convolutional_classifier = convolutional_classifier
        self.averaging_classifier = averaging_classifier
        self.two_lin_layer_conv = two_lin_layer_conv
        self.multi_visit_kernel_conv = multi_visit_kernel_conv
        self.kernel_size_visits = kernel_size_visits

        if self.visit_dropout:
            self.visit_dropout_layer = torch.nn.Dropout2d(bert_model.dropout)

        if self.attn_classifier:
            self.collector = torch.nn.Parameter(torch.randn(self.n_parallel_pools, self.emb_size))
            self.keys = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.values = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.dropout = torch.torch.nn.Dropout(bert_model.dropout)
            self.linear = torch.nn.Linear(self.emb_size * self.n_parallel_pools, n_targets)

        elif self.convolutional_classifier:
            if self.multi_visit_kernel_conv:
                self.convs = torch.nn.Conv1d(
                    self.emb_size, self.emb_size * self.n_parallel_pools, self.kernel_size_visits)
                self.maxpool = torch.nn.MaxPool1d(self.bert.max_visits - self.kernel_size_visits)
                self.linear = torch.nn.Linear(self.emb_size * self.n_parallel_pools, n_targets)
            else:
                self.convs = torch.nn.Conv1d(self.emb_size, self.emb_size * self.n_parallel_pools, 1)
                self.maxpool = torch.nn.MaxPool1d(self.bert.max_visits)
                if self.two_lin_layer_conv:
                    self.linear1 = torch.nn.Linear(self.emb_size * self.n_parallel_pools, self.emb_size)
                    self.linear2 = torch.nn.Linear(self.emb_size, n_targets)
                else:
                    self.linear = torch.nn.Linear(self.emb_size * self.n_parallel_pools, n_targets)

        elif self.averaging_classifier:
            self.linear = torch.nn.Linear(self.emb_size, n_targets)
            self.dropout = torch.nn.Dropout(bert_model.dropout)
        else:
            self.pooler = torch.nn.Linear(self.bert.max_visits, self.n_parallel_pools)
            self.linear = torch.nn.Linear(self.emb_size * self.n_parallel_pools, n_targets)
            self.dropout = torch.nn.Dropout(bert_model.dropout)

    def forward(self, x, train=False, interpret_debug=False):

        if self.attn_classifier:
            x, m = self.bert(x, train=train, return_mask=True)
            if self.visit_dropout:
                x = self.visit_dropout_layer(x)
            k = self.keys(x)
            v = self.values(x)
            attn_scores = torch.matmul(self.collector.expand(x.shape[0], -1, -1), k.transpose(-2, -1)) / math.sqrt(
                self.emb_size)
            attn_scores = attn_scores.masked_fill(m.unsqueeze(1).expand(-1, self.n_parallel_pools, -1) == 0, -1e9)
            p_attn = torch.nn.functional.softmax(attn_scores, dim=-1)
            if self.pred_dropout:
                p_attn = self.dropout(p_attn)

            pooled = torch.matmul(p_attn, v)
            pooled = pooled.view(pooled.shape[0], -1)
            y_pred = self.linear(torch.nn.ReLU()(pooled))

            if self.n_targets == 1:
                return y_pred.flatten(0, -1)
            return y_pred

        elif self.rnn_classifier:
            x = self.bert(x, train=False)
            pass

        elif self.convolutional_classifier:
            if self.multi_visit_kernel_conv:
                x, m = self.bert(x, train=train, return_mask=True)
                cv = self.convs(x.transpose(1, 2))
                cv_masked = cv.masked_fill(
                    m[:, :-self.kernel_size_visits + 1].unsqueeze(1).expand(-1, self.emb_size * self.n_parallel_pools,
                                                                            -1) == 0, -1e9)
                pooled = torch.nn.Sigmoid()(self.maxpool(cv_masked))
                y_pred = self.linear(pooled.squeeze())
                if self.n_targets == 1:
                    return y_pred.flatten(0, -1)
                return y_pred
            else:
                x, m = self.bert(x, train=train, return_mask=True)  # (batch_size, max_visits, embedding_dim)
                cv = self.convs(x.transpose(1, 2))
                cv_masked = cv.masked_fill(m.unsqueeze(1).expand(-1, self.emb_size * self.n_parallel_pools, -1) == 0,
                                           -1e9)
                pooled = torch.nn.Sigmoid()(self.maxpool(cv_masked))
                if self.two_lin_layer_conv:
                    y_pred = torch.nn.ReLU()(self.linear1(pooled.squeeze()))
                    y_pred = self.linear2(y_pred)
                    if self.n_targets == 1:
                        return y_pred.flatten(0, -1)
                    return y_pred
                else:
                    y_pred = self.linear(pooled.squeeze())
                    if self.n_targets == 1:
                        if interpret_debug:
                            return y_pred.flatten(0, -1), pooled, cv_masked
                        return y_pred.flatten(0, -1)
                    return y_pred

        else:
            x = self.bert(x, train=False)
            if self.visit_dropout:
                x = self.visit_dropout_layer(x)
            if self.averaging_classifier:
                y_pred = self.linear(torch.nn.ReLU()(x.mean(1)))
                if self.n_targets == 1:
                    return y_pred.flatten(0, -1)
                return y_pred
            else:
                pooled = self.pooler(x.transpose(1, 2)).view(-1, 10 * self.emb_size)
                pooled = self.dropout(pooled)
                y_pred = self.linear(torch.nn.ReLU()(pooled))
                if self.n_targets == 1:
                    return y_pred.flatten(0, -1)
                return y_pred


# Changed: the goal of this class is to create the patient representation as a sequence of visits by
# using the embedding representation


class VisitTransformer(torch.nn.Module):
    def __init__(
            self, featureSet,
            embedding_dim=300,
            n_heads=2, attn_depth=2,
            dropout=0.3,
            concept_embedding_path=None,
            time_emb_type='sin',
            use_RNN=False,
            use_mask=False,  # False
            max_days=365*4,  # 365
            normalize_visits=False,
            normalize_codes=False,
            backwards_attn=False,
            index_embedding=False,
            use_mask_later=True,
            all_concept_map_reverse=None,
            pre_trained=True,
            **kwargs
    ):
        super(VisitTransformer, self).__init__()

        self.time_emb_type = time_emb_type
        self.data_set = False
        self.concept_embedding_path = concept_embedding_path
        self.featureSet = featureSet

        self.n_heads = n_heads
        self.embedding_dim = embedding_dim * n_heads
        self.dropout = dropout

        self.max_days = max_days
        self.max_visits = 100  # 512 - 2
        self.attn_depth = attn_depth

        self.mask_prob = 0.15
        self.rand_prob = 0.1
        self.keep_prob = 0.1

        self.use_mask = use_mask
        self.backwards_attn = backwards_attn
        self.normalize_visits = normalize_visits
        self.normalize_codes = normalize_codes
        self.index_embedding = index_embedding

        self._initialize_concept_embeddings(all_concept_map_reverse, pre_trained)

        if self.time_emb_type == 'sin':
            self.timescales = (((1 / 10000) ** (1 / self.embedding_dim)) ** torch.FloatTensor(
                range(self.embedding_dim // 2))).cuda()

        elif self.time_emb_type == 'learnfreq':
            self.ampl = torch.nn.Parameter(torch.randn(self.embedding_dim // 2))
            self.freq = torch.nn.Parameter(
                ((1 / 10000) ** (1 / self.embedding_dim)) ** torch.FloatTensor(range(self.embedding_dim // 2)))
            self.phase = torch.nn.Parameter(torch.randn(self.embedding_dim // 2))

        else:
            self.time_embedder = torch.nn.Embedding(self.max_days, self.embedding_dim)

        if not self.use_mask:
            self.start_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
            self.pad_embedding = torch.nn.Parameter(torch.zeros(self.embedding_dim))
            self.mask_embedding = torch.nn.Parameter(torch.randn(self.embedding_dim))
        else:
            self.pad_embedding = torch.zeros(self.embedding_dim).cuda()

        if use_RNN:
            self.tfs = torch.nn.ModuleList([transformers.RNNBlock(self.embedding_dim, self.dropout)])
        else:
            self.tfs = torch.nn.ModuleList([
                transformers.TransformerBlock(self.embedding_dim, self.n_heads, self.dropout,
                                              backwards_only=self.backwards_attn) for _ in range(self.attn_depth)])

    def _initialize_concept_embeddings(self, all_concept_map_reverse, pre_trained):

        count_key_error = 0
        if not pre_trained: # Train embedding as part of the model workflow
            self.concept_embedder = torch.nn.Embedding(len(self.featureSet.concept_map), self.embedding_dim)
            if self.concept_embedding_path is not None:
                wv = KeyedVectors.load(self.concept_embedding_path, mmap='r')
                for i in range(len(self.featureSet.concept_map)):
                    try:
                        self.concept_embedder.weight.data[i, :] = torch.FloatTensor(
                            wv[str(all_concept_map_reverse[self.featureSet.concept_map[i]])])
                    except KeyError:
                        pass


        if pre_trained: # Use pretrained embeddings file
            concept_embedding_matrix = torch.zeros(len(self.featureSet.concept_map), self.embedding_dim)
            if self.concept_embedding_path is not None:
                wv = KeyedVectors.load(self.concept_embedding_path, mmap='r')
                # wv = KeyedVectors.load_word2vec_format(self.concept_embedding_path, binary=False)
                for i in range(len(self.featureSet.concept_map)):
                    try:
                        concept_embedding_matrix[i, :] = torch.FloatTensor(
                            wv[str(all_concept_map_reverse[self.featureSet.concept_map[i]])])
                    except KeyError:
                        count_key_error += 1
                        pass
            print(concept_embedding_matrix)
            print("Number of embedding key errors : ", count_key_error)
            self.concept_embedder = torch.nn.Embedding(len(self.featureSet.concept_map),
                                                       self.embedding_dim).from_pretrained(concept_embedding_matrix)

    def set_data(self, all_codes_tensor, person_indices, visit_chunks, visit_time_rel, n_visits, births, genders):

        self.all_codes_tensor = all_codes_tensor
        self.person_indices = person_indices
        self.visit_chunks = visit_chunks
        self.visit_time_rel = visit_time_rel
        self.n_visits = n_visits
        self.births = births
        self.genders = genders
        self.data_set = True

    def forward(self, person_range, train=True, return_mask=False):
        with torch.no_grad():
            use_mask = self.use_mask

            assert (self.data_set)
            z = self.all_codes_tensor[np.concatenate([np.array(range(self.person_indices[p] + self.visit_chunks[p][0],
                                                                     self.person_indices[p] + self.visit_chunks[p][-1]
                                                                     )) for p in person_range])]
            embedded_raw = self.concept_embedder(torch.tensor(z).cuda())
            curr = 0
            person_ix = []
            sum_indices = []
            for p in person_range:
                for v in range(len(self.visit_chunks[p]) - 1):
                    sum_indices += [curr for _ in range(self.visit_chunks[p][v + 1] - self.visit_chunks[p][v])]
                    if self.visit_chunks[p][v + 1] > self.visit_chunks[p][v]:
                        person_ix.append(p)
                    curr += 1
            summed = torch.zeros(len(person_ix), self.embedding_dim).cuda()
            summed = summed.index_add_(0, torch.tensor(sum_indices).cuda(), embedded_raw)

            if self.normalize_codes:
                emb_code_norm = torch.ones(embedded_raw.shape[0], 1).cuda()
                sum_norm = torch.zeros(len(person_ix), 1).cuda()
                sum_norm = sum_norm.index_add_(0, torch.tensor(sum_indices).cuda(), emb_code_norm)
                summed /= sum_norm

            reshaped = self.pad_embedding.unsqueeze(0).repeat(len(person_range), self.max_visits, 1)

            if use_mask:
                mask = torch.zeros(reshaped.shape[:-1]).cuda()

            curr = 0
            for i, p in enumerate(person_range):
                curr += self.n_visits[p]
                seq_len = min(self.max_visits, self.n_visits[p])
                reshaped[i, -seq_len:, :] = summed[curr - seq_len: curr, :]
            if self.normalize_visits:
                norm = torch.FloatTensor(np.vectorize(self.n_visits.get)(person_range)).cuda()
                norm = torch.clamp(norm, 0, self.max_visits).unsqueeze(-1).unsqueeze(-1)
                reshaped /= norm
            times = torch.cat([torch.FloatTensor(self.visit_time_rel[p]).cuda() for p in person_range]).clamp(0,
                                                                                                              self.max_days - 1)

            if self.index_embedding:
                indices_raw = torch.cat(
                    [torch.FloatTensor(range(len(self.visit_time_rel[p]))).cuda() for p in person_range]).clamp(0,
                                                                                                                self.max_days - 1)

            if self.time_emb_type == 'sin':

                time_embedding_unshaped = torch.cat([
                    torch.sin(torch.ger(times, self.timescales)),
                    torch.cos(torch.ger(times, self.timescales))
                ], 1)
                if self.index_embedding:
                    index_embedding_unshaped = torch.cat([
                        torch.sin(torch.ger(indices_raw, self.timescales)),
                        torch.cos(torch.ger(indices_raw, self.timescales))
                    ], 1)

            elif self.time_emb_type == 'learnfreq':
                time_embedding_unshaped = torch.cat([
                    self.ampl * torch.sin(self.phase + torch.ger(times, self.freq)),
                    self.ampl * torch.cos(self.phase + torch.ger(times, self.freq))
                ], 1)

            else:
                time_embedding_unshaped = self.time_embedder(
                    torch.cat([torch.tensor(self.visit_time_rel[p]).cuda() for p in person_range]).clamp(0,
                                                                                                         self.max_days - 1))
            if self.time_emb_type == 'none':
                time_embedding_unshaped = 0 * time_embedding_unshaped

            time_embedding = torch.zeros(reshaped.shape).cuda()
            if self.index_embedding:
                indices_embedding = torch.zeros(reshaped.shape).cuda()

            curr = 0
            for i, p in enumerate(person_range):
                curr += self.n_visits[p]
                seq_len = min(self.max_visits, self.n_visits[p])
                time_embedding[i, -seq_len:, :] = time_embedding_unshaped[curr - seq_len: curr, :]
                if self.index_embedding:
                    indices_embedding[i, -seq_len:, :] = index_embedding_unshaped[curr - seq_len: curr, :]
                if use_mask:
                    mask[i, -seq_len:] = 1
            output_emb = reshaped  # + time_embedding
            if self.index_embedding:
                output_emb += indices_embedding

            # NonTemporal embedding

            births = self.births.iloc[person_range].tolist()
            genders = self.genders.iloc[person_range].tolist()
            birth_emb = torch.zeros(reshaped.shape).cuda()
            gender_emb = torch.zeros(reshaped.shape).cuda()
            for i, p in enumerate(person_range):
                age = (int(births[i]) - 1900) / 120
                gender = 2 * genders[i] - 1
                birth_emb[i, :, :] = torch.ones(birth_emb.shape[1], birth_emb.shape[2]) * age * 0.1
                gender_emb[i, :, :] = torch.ones(birth_emb.shape[1], birth_emb.shape[2]) * gender * 0.1

            output_emb += birth_emb
            output_emb += gender_emb
            if return_mask:
                return output_emb, mask
            else:
                return output_emb, None


# Changed
# This is the SARD model classifier
class Risk(torch.nn.Module):
    def __init__(
            self, bert_model,
            n_targets=1,
            attn_classifier=True,  # False
            pred_dropout=True,
            rnn_classifier=False,
            visit_dropout=False,
            convolutional_classifier=False,
            two_lin_layer_conv=False,
            multi_visit_kernel_conv=False,
            kernel_size_visits=1,
            averaging_classifier=False,
            **kwargs
    ):
        super(Risk, self).__init__()
        self.n_targets = n_targets
        self.emb_size = bert_model.embedding_dim
        self.bert = bert_model
        self.attn_classifier = attn_classifier
        self.n_parallel_pools = 1  # 10
        self.pred_dropout = pred_dropout
        self.rnn_classifier = rnn_classifier
        self.visit_dropout = visit_dropout
        self.convolutional_classifier = convolutional_classifier
        self.averaging_classifier = averaging_classifier
        self.two_lin_layer_conv = two_lin_layer_conv
        self.multi_visit_kernel_conv = multi_visit_kernel_conv
        self.kernel_size_visits = kernel_size_visits

        if self.visit_dropout:
            self.visit_dropout_layer = torch.nn.Dropout2d(bert_model.dropout)

        if self.attn_classifier:
            self.collector = torch.nn.Parameter(torch.randn(self.n_parallel_pools, self.emb_size))
            self.keys = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.values = torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
            self.dropout = torch.torch.nn.Dropout(bert_model.dropout)
            self.linear = torch.nn.Linear(self.emb_size * self.n_parallel_pools, n_targets)

    def forward(self, x, m):
        if self.attn_classifier:

            if self.visit_dropout:
                x = self.visit_dropout_layer(x)
            k = self.keys(x)
            v = self.values(x)
            attn_scores = torch.matmul(self.collector.expand(x.shape[0], -1, -1), k.transpose(-2, -1)) / math.sqrt(
                self.emb_size)
            attn_scores = attn_scores.masked_fill(m.unsqueeze(1).expand(-1, self.n_parallel_pools, -1) == 0, -1e9)
            p_attn = torch.nn.functional.softmax(attn_scores, dim=-1)
            if self.pred_dropout:
                p_attn = self.dropout(p_attn)

            pooled = torch.matmul(p_attn, v)
            pooled = pooled.view(pooled.shape[0], -1)
            y_pred = self.linear(torch.nn.ReLU()(pooled))

            if self.n_targets == 1:
                return y_pred.flatten(0, -1)
            return y_pred


# This is the STARFE architecture
class SA(torch.nn.Module):
    def __init__(
            self, bert_model, surv_encoder, max_time=24,
            n_targets=1,
            supervision=False,
            attn_classifier=True,  # False
            pred_dropout=True,
            rnn_classifier=False,
            visit_dropout=False,
            convolutional_classifier=False,
            two_lin_layer_conv=False,
            multi_visit_kernel_conv=False,
            kernel_size_visits=1,
            averaging_classifier=False,
            use_mask_later=True,

            **kwargs
    ):
        super(SA, self).__init__()
        self.n_targets = n_targets
        self.emb_size = bert_model.embedding_dim
        self.bert = bert_model
        self.attn_classifier = attn_classifier
        self.n_parallel_pools = 1  # 10
        self.pred_dropout = pred_dropout
        self.rnn_classifier = rnn_classifier
        self.visit_dropout = visit_dropout
        self.convolutional_classifier = convolutional_classifier
        self.averaging_classifier = averaging_classifier
        self.two_lin_layer_conv = two_lin_layer_conv
        self.multi_visit_kernel_conv = multi_visit_kernel_conv
        self.kernel_size_visits = kernel_size_visits
        self.surv_encoder = surv_encoder
        self.supervision = supervision
        self.max_time = max_time
        self.use_mask_later = use_mask_later

        if self.visit_dropout:
            self.visit_dropout_layer = torch.nn.Dropout2d(bert_model.dropout)

        if self.attn_classifier:

            self.conv = torch.nn.Conv1d(self.bert.max_visits, self.max_time, kernel_size=kernel_size_visits)
            self.conv_to_1 = torch.nn.Conv1d(self.bert.max_visits, 1, kernel_size=kernel_size_visits)
            self.linear_to_1 = torch.nn.Linear(self.emb_size, self.max_time)
            self.linear_reduction = torch.nn.Linear(self.bert.max_visits * self.emb_size, self.max_time * self.emb_size)
            self.tfs = torch.nn.ModuleList([
                transformers.TransformerBlock(self.emb_size, self.bert.n_heads, self.bert.dropout,
                                              backwards_only=self.bert.backwards_attn) for _ in
                range(self.bert.attn_depth)])

            self.sublayers = torch.nn.ModuleList(
                [transformers.SublayerConnection(self.emb_size, self.bert.dropout) for _ in
                 range(self.bert.attn_depth)])

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.emb_size * self.bert.max_visits, 1600),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(1600, 400),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(400, 100),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(100, self.max_time),
                torch.nn.Sigmoid(),
            )

    def forward(self, x, m, train=True):
        if not self.use_mask_later:
            m = None
        batch_size = x.shape[0]

        # Visits contextualized representation is shifted to months of time-to-event by dimension reduction. This can be
        # done by using convolution layer or fully connected layer
        conv = True
        linear_reduction = False


        # if embedding_version=True then we ablate the self-attention from the representation phase
        embedding_version = False
        if embedding_version:
            pooled = x
        else:
            for tf, sl in zip(self.tfs, self.sublayers):
                pooled = sl(x, lambda x: tf(x, m))

        if conv:  # pooled: (batch_size,max_visits, embedding_dim) -----> (batch_size,max_time, embedding_dim)
            pooled_surv = self.conv(pooled)

        if linear_reduction:
            cur = self.linear_reduction(pooled.reshape(batch_size, self.bert.max_visits * self.emb_size))
            cur = torch.nn.ReLU()(cur)
            pooled_surv = cur.reshape(batch_size, self.max_time, self.emb_size)

        # pooled_surv size : (batch_size, max_time, embedding_size)


        surv_pred = self.surv_encoder(pooled_surv)  # surv_pred shape: (batch_size, max_time)
        # surv_encoder is self-attentio or lstm and need to be determined in the survival.py file



        # This part is used when attention weights are printed
        if train == "sard_weights":
            torch.set_printoptions(threshold=100000, precision=4)
            for i, block in enumerate(self.tfs):
                if i!=3: continue
                print(f"Number head is :{i}, ", "\n")
                attn = block.attention.attn
                #print(attn.shape)
                attn = attn.detach().cpu()#.numpy()
                attn = attn[:,0,:,:]
                print(attn.shape)
                highest = []
                for patient in range(len(attn)):
                    print(f"Number head is :{i}, ", "\n")
                    pat=attn[patient]
                    print(pat.shape)
                    k = 5
                    v_pat, i_pat = torch.topk(pat.flatten(), k=k)
                    pat_top_ind = np.array(np.unravel_index(i_pat.numpy(), pat.shape)).T
                    for l in range(k):
                        print(f"Score in place {l+1} in fast is : {pat[pat_top_ind[l][0]][pat_top_ind[l][1]]} got by : {pat_top_ind[l]}")
                    highest.append([pat_top_ind[0]])
                    num_visits = 100
                    pat_gate = True
                    for v1 in range(len(pat)):
                        if pat[v1][0]==0 and pat_gate:
                            for v2 in range(len(pat)):
                                if pat[v1][v2]!=0 and pat_gate:
                                    num_visits = 100-v2
                                    print(f"fast number of visits : {num_visits}")
                                    pat_gate = False
                    reduced_pat = pat[100-num_visits:, 100-num_visits:]
                    print(reduced_pat, reduced_pat.shape)
            return surv_pred, highest

        return surv_pred, None
