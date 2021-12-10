# coding: utf-8
"""
dembedder is a software aiming at inducing taxonomies from embedding spaces.
Copyright (C) 2021 Solocal-SA and CNRS

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

dembedder was developped by: François Torregrossa, Robin Allesiardo, Vincent Claveau and Guillaume Gravier.
"""

from nltk.corpus import wordnet as wn
from tqdm import tqdm
import codecs
import numpy as np

def add_descendants(entity_set, entity_name):
    entity_set.add(entity_name)
    for x in wn.synset(entity_name).hyponyms():
        entity_set.add(x.name())
        add_descendants(entity_set, x.name())

def wordnet_ground_truth_edges(entity_name):

    if entity_name == "all_nouns":
        __entities = set(map(lambda x: x.name(), wn.all_synsets(pos='n')))
    elif entity_name == "all_verbs":
        __entities = set(map(lambda x: x.name(), wn.all_synsets(pos='v')))
    else:
        entities = set()
        add_descendants(entities, entity_name)
        __entities = set(entities)

    gt_edges = set([(wn.synset(sn).name(), v.name()) for sn in tqdm(__entities) for v in wn.synset(sn).hypernyms() if v.name() in __entities])

    return gt_edges

def serialize_edges(out, edges):
    with codecs.open(out, "w", encoding='utf-8') as outstream:
        for u, v in edges:
            outstream.write("%s\t%s\n" % (u, v))

def ncommon(a, b):
    if len(a) >= len(b):
        return sum(1 for w in b if w in a)
    else:
        return ncommon(b, a)

def ncommon_undirected(a, b):
    _a = set([tuple(sorted(x)) for x in a])
    _b = set([tuple(sorted(x)) for x in b])
    if len(a) >= len(b):
        return ncommon(_a, _b)
    else:
        return ncommon(_b, _a)

def graph_precision(edges, gt_edges):
    return ncommon(edges, gt_edges) / len(edges) if len(edges) > 0 else 0

def graph_precision_undirected(edges, gt_edges):
    return ncommon_undirected(edges, gt_edges) / len(edges) if len(edges) > 0 else 0

def graph_recall(edges, gt_edges):
    return ncommon(edges, gt_edges) / len(gt_edges)

def graph_recall_undirected(edges, gt_edges):
    return ncommon_undirected(edges, gt_edges) / len(gt_edges)

def graph_fscore(edges, gt_edges):
    precision = graph_precision(edges, gt_edges)
    recall = graph_recall(edges, gt_edges)
    return {
        'fscore': (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0,
        'precision': precision,
        'recall': recall}

def graph_fscore_undirected(edges, gt_edges):
    precision = graph_precision_undirected(edges, gt_edges)
    recall = graph_recall_undirected(edges, gt_edges)
    return {
        'fscore': (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0,
        'precision': precision,
        'recall': recall}

def haussdorff_ged(g1, g2):

    from aproximated_ged.VanillaHED import VanillaHED
    hed = VanillaHED(del_node = 1, ins_node = 1, del_edge = 1, ins_edge = 1)
    return hed.ged(g1, g2)[0]

def mean_rank(map_neighbors, gt_edges):

    cum_rank = 0
    count = 0

    for k, neighbors in map_neighbors.items():
        keep = 0
        for i, x in enumerate(neighbors, 1):
            if (k, x[0]) in gt_edges:
                cum_rank += i
                keep = 1
                break
        if keep:
            count += 1

    return cum_rank / count

def mean_average_precision(map_neighbors, gt_edges):

    from sklearn.metrics import average_precision_score

    cum_map = 0
    count = 0

    for k, neighbors in map_neighbors.items():
        labels = list(map(lambda x: 1 if (k, x[0]) in gt_edges else 0, neighbors))
        if np.sum(labels) > 0:
            dist = list(map(lambda x: -x[1], neighbors))
            cum_map += average_precision_score(labels, dist)
            count += 1

    return cum_map / count

def compute_distorsion(emb, tree, dfun):

    dist = lambda x, y :dfun(emb.project(x).reshape(1, -1), emb.project(y).reshape(1, -1))
    pl = tree.path_lengths(dfun=dist)

    avg = 0
    c = 0
    max_dist = float("-inf")

    pbar = tqdm(enumerate(emb.voc), total=len(emb.voc))
    for i, w1 in pbar:
        for w2 in emb.voc[i+1:]:
            local_dist = pl[w1][w2] / dist(w1, w2)
            avg = (avg * c + local_dist) / (c + 1)
            c += 1
            max_dist = max(local_dist, max_dist)
            pbar.set_description("max distortion: %.2f -- avg distortion: %.2f" % (max_dist, avg))

    return max_dist, avg