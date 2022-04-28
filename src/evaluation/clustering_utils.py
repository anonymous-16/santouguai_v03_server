import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from src.utils.utilities import read_lst
from src.utils.torch_utilities import predict


def device(x):
    return torch.from_numpy(x).cuda()


def kmeans(rep, n_clusters):
    y = rep.cpu().numpy()
    n_clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(y)
    center = n_clustering.cluster_centers_
    label = n_clustering.labels_
    return device(label), device(center)


def frame_2_sample(target, rep):
    N, T = target.shape
    nind = (torch.arange(N)[:, None].to(target.device) + torch.zeros_like(target)).flatten()
    tind = (torch.arange(T)[None, :].to(target.device) + torch.zeros_like(target)).flatten()
    target = target.flatten()
    rep = rep.view(N, -1, T).transpose(1, 2).flatten(0, 1)
    x = nind[target > 0]
    y = tind[target > 0]
    z = rep[target > 0]
    return z, torch.stack([x, y], 1)


def parse_index(ind, target):
    _, T = target.shape
    ind = ((ind[:, 0] * T) + ind[:, 1]).long()
    return ind


def sample_2_frame(target, index, cluster, n_clusters):
    target_clusters = []
    index = parse_index(index, target)
    N, T = target.shape
    print(cluster.shape, index.shape)
    for i in range(n_clusters):
        ind = cluster == i
        temp = torch.zeros_like(target).flatten()
        temp[index[ind]] = 1
        target_clusters.append(temp.view(N, T))
    return target_clusters


def estimate_target(note_prob, onset_prob):
    TOL = 5
    NOL = 63

    onset_target = predict(onset_prob, 0.2)
    note_target = predict(note_prob, 0.5)
    onset_target = note_target * onset_target

    res = torch.zeros_like(onset_target)
    for i in range(note_target.shape[0]):
        in_seg = False
        out_seg = 0
        note_seg = 0
        for j in range(note_target.shape[-1]):
            if onset_target[i, j] == 1:
                in_seg = True
                out_seg = TOL

            if in_seg:
                note_seg = 0
                if note_target[i, j] == 1:
                    res[i, j - TOL + out_seg: j + 1] = 1
                    out_seg = TOL
                else:
                    out_seg -= 1
                    if out_seg == 0:
                        in_seg = False
            elif note_target[i, j] == 1:
                note_seg += 1
            else:
                note_seg = 0
            if note_seg > NOL:
                res[i, j + 1 - note_seg: j + 1] = 1

    note_target = res
    res = torch.zeros_like(onset_target)
    for i in range(note_target.shape[0]):
        seg_len = 0
        for j in range(note_target.shape[-1]):
            if onset_target[i, j] == 1:
                seg_len += 1
            else:
                if seg_len > 0:
                    res[i, j - seg_len] = 1
                seg_len = 0
    onset_target = res
    return note_target


def smooth(sample_rep, sample_index):
    slen = sample_index.shape[0]
    index = torch.arange(slen).to(sample_index.device)
    TB = 3
    TU = 11
    pre = 0
    i = 1
    while i <= slen:
        if i == slen or not sample_index[pre, 0] == sample_index[i, 0] \
                or sample_index[i, 1] - sample_index[i - 1, 1] > TB \
                or sample_index[i, 1] - sample_index[pre, 1] >= TU:
            if i - pre > 1:
                emb = sample_rep[pre: i]
                dis = ((emb[:, None] - emb[None, :]) ** 2).mean(-1)
                for j in range(i - pre):
                    dis[j, j] = 233333
                ind = torch.argmin(dis, 1)
                flag = torch.zeros_like(ind)

                def connect(a, v):
                    if flag[a] > 0:
                        return v
                    flag[a] = 1
                    b = ind[a].item()
                    v = index[b + pre] if v > index[b + pre] else v
                    v = connect(b, v)
                    index[b + pre] = v
                    return v

                for j in range(i - pre):
                    index[j + pre] = connect(j, index[j + pre])

            if i < slen and (
                    sample_index[i, 1] - sample_index[i - 1, 1] > TB or not sample_index[pre, 0] == sample_index[i, 0]):
                pre = i
            else:
                pre += 1
        i += 1
    for i in range(slen):
        ind = index == i
        if len(index[ind]) > 1:
            sample_rep[ind] = sample_rep[ind].mean(0)
        elif len(index[ind]) == 1:
            index[ind] = -1
    return sample_rep, sample_index


def load_note(path):
    notes = read_lst(path, "\t\t")


class EmbeddingCenter(nn.Module):
    def __init__(self, sample_reps, batch_size=16, loss_weights=None):
        super().__init__()
        n_clusters = len(sample_reps)
        param_list = []
        for i in range(n_clusters):
            param_list.append(nn.Parameter(torch.randn(sample_reps[i].shape[0])))
        self.param_list = nn.ParameterList(param_list)
        labels = []
        for i in range(n_clusters):
            labels.append(torch.zeros_like(sample_reps[i][:, 0]) + i)

        self.data = torch.cat(sample_reps, 0)
        self.label = torch.cat(labels, 0).long()
        if loss_weights is None:
            self.loss_weight = torch.ones_like(self.label)
        self.batch_size = batch_size
        self.sample_reps = sample_reps
        self.index = torch.arange(self.data.shape[0]).to(self.data.device)

    def compute_center(self):
        centers = []
        for i in range(len(self.sample_reps)):
            centers.append((F.softmax(self.param_list[i], 0)[:, None] * self.sample_reps[i]).sum(0))
        return torch.stack(centers, -1)

    def forward(self):
        random.shuffle(self.index)
        samples_num = self.data.shape[0]
        for i in range(0, samples_num, self.batch_size):
            center = self.compute_center()
            st = i
            ed = i + self.batch_size if i + self.batch_size < samples_num else samples_num
            ind = self.index[st: ed]
            dis = - ((self.data[ind][:, :, None] - center[None, :]) ** 2).sum(1)
            loss = nn.CrossEntropyLoss(reduction='none')(dis, self.label[ind])
            loss = (loss * self.loss_weight[ind]).sum()
            yield loss


def clustering_from_preds(target, rep, n_clusters):
    sample_rep, sample_index = frame_2_sample(target, rep)
    sample_rep, sample_index = smooth(sample_rep, sample_index)
    cluster, center = kmeans(sample_rep, n_clusters)
    target_clusters = sample_2_frame(target, sample_index, cluster, n_clusters)
    sample_rep, sample_index, cluster, center = fit_center(target_clusters, rep)
    target_clusters = sample_2_frame(target, sample_index, cluster, n_clusters)
    return target_clusters, torch.cat([sample_index, cluster[:, None] + 1, sample_rep], -1), center


def fit_center(targets, rep):
    sample_reps, sample_indexs, sample_labels = [], [], []
    for i, target in enumerate(targets):
        sample_rep, sample_index = frame_2_sample(target, rep)
        sample_reps.append(sample_rep)
        sample_indexs.append(sample_index)
        sample_labels.append(torch.zeros_like(sample_rep[:, 0]) + i)
    model = EmbeddingCenter(sample_reps)

    sample_rep = torch.cat(sample_reps, 0)
    sample_index = torch.cat(sample_indexs, 0)
    sample_label = torch.cat(sample_labels, 0)

    model.cuda()
    loss_total = 23333
    loss_tol = 0.001
    lr = 1e-3
    MAX_STEP = 100
    opt = torch.optim.Adam(model.param_list, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    step = 0.
    while loss_total > loss_tol:
        loss_total = 0.
        step += 1
        for loss in model():
            loss_total += loss.item()
            loss = loss / model.batch_size
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_total += loss
        loss_total = loss_total / sample_rep.shape[0]
        print(f"step {step}", loss_total.item())
        if step > MAX_STEP:
            break
    model.eval()
    with torch.no_grad():
        center = model.compute_center()
        label = torch.argmin(((sample_rep[:, :, None] - center[None, :]) ** 2).sum(1), -1)
        print("acc", len(label[sample_label == label]), "/", label.shape[0])
    return sample_rep, sample_index, label, center.transpose(-1, -2)


class ClusteringUtil(object):
    def __init__(self, target, rep, n_clusters):
        self.target = target
        self.rep = rep
        self.n_clusters = n_clusters

    def preprocess_data(self):
        target = self.target
        mask = target.sum(0).flatten()
        mask[mask > 1] = 0
        sources_num = target.shape[0]
        sample_reps = []
        for i in range(sources_num):
            sample_rep, sample_index = frame_2_sample(target[i], self.rep)
            index = parse_index(sample_index, target[i])
            sample_mask = mask[index]
            sample_rep = sample_rep[sample_mask > 0]
            sample_reps.append(sample_rep)
        return sample_reps
