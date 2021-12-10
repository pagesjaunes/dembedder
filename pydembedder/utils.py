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

import codecs
import numpy as np
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants
from networkx.algorithms.tree.recognition import is_arborescence, is_tree

class NetworkX111Wrapper:

    def __init__(
        self,
        g):

        self.n = len(g)
        self.edge = {}
        self.nodes = g.nodes
        for u in g.nodes():
            for v in g.edges(u):
                if u not in self.edge.keys():
                    self.edge[u] = {}
                self.edge[u][v] = {'weight': 1}

    def __len__(
        self):

        return self.n

class Embedding:

    def __init__(
        self,
        word2idx,
        vecs):

        self.w2i = word2idx
        self.data = vecs
        self.voc = list(self.w2i.keys())

    def project(
        self,
        w):
        return self.data[self.w2i[w], :]


class Tree:

    @staticmethod
    def fill_graph(
        g,
        edges):

        for u, v in edges:
            g.add_edge(u, v, weight=1)

    @staticmethod
    def recursive_directed_edges(g, start, directed_g, excluded, edges):
        excluded.add(start)
        for _, n in g.edges(start):
            if n not in excluded:
                directed_g.add_edge(start, n)
                edges.add((start, n))
                Tree.recursive_directed_edges(g, n, directed_g, excluded, edges)

    def __init__(
        self,
        edges,
        voc_restrict = None):

        self.edges = edges if voc_restrict is None else set((u, v) for u, v in edges if u in voc_restrict and v in voc_restrict)
        self.g = nx.DiGraph()
        Tree.fill_graph(self.g, self.edges)

    @property
    def nodes(self):
        return self.g.nodes()

    def is_valid(
        self,
        pair):
        u, v = pair
        return (u, v) in self.edges

    def add_edge(
        self,
        u,
        v):

        self.edges.add((u, v))
        self.g.add_edge(u, v)

    def is_valid_undirected(
        self,
        pair):
        u, v = pair
        return (u, v) in self.edges or (v, u) in self.edges

    def ancestor_edges(
        self):

        voc = set(map(lambda x: x[0], self.edges))
        voc.update(set(map(lambda x: x[1], self.edges)))
        return set([(w, hypernym) for w in voc for hypernym in ancestors(self.g, w)])

    def descendants(
        self,
        x):

        desc = set(descendants(self.g, x))
        return desc

    def descending_edges(
        self,
        x):

        edges = set()
        for y in self.successors(x):
            edges.add((x, y))
            edges.update(self.descending_edges(y))
        return edges


    def reorganize(
        self,
        emb):

        new_edges = set()
        for u, v in self.edges:
            vec_u, vec_v = emb.project(u), emb.project(v)
            if np.linalg.norm(vec_u) > np.linalg.norm(vec_v):
                new_edges.add((u, v))
            else:
                new_edges.add((v, u))
        self.g = nx.DiGraph()
        self.edges = new_edges
        Tree.fill_graph(self.g, self.edges)

    def successors(self, x):
        return self.g.successors(x)

    def predecessors(self, x):
        return self.g.predecessors(x)

    def rooten(
        self,
        w):

        g = self.g.to_undirected()
        self.g = nx.DiGraph()
        self.edges = set()
        Tree.recursive_directed_edges(g, w, self.g, set(), self.edges)

    def path_lengths(self, dfun=lambda x, y: 1):
        g = nx.Graph()
        for u, v in self.g.edges:
            g.add_edge(u, v, weight=dfun(u, v))
        return dict(nx.shortest_path_length(g, weight="weight"))

    def save(self, filepath, centroid):
        with codecs.open(filepath, "w", encoding="utf-8") as outstream:
            if centroid is not None:
                outstream.write("centroid\t%s\n" % centroid)
            for u, v in self.edges:
                outstream.write("%s\t%s\n" % (u, v))

    def get_components(
        self):

        from networkx.algorithms.components import connected_components
        return list(connected_components(self.g.to_undirected()))

    def clear_pearls(
        self,
        centroid):

        g = nx.DiGraph()
        new_centroid = []
        self.clear_pearls_aux(g, centroid, None, new_centroid)
        assert len(new_centroid) == 1
        self.g = g
        self.edges = set()
        for u, v in self.g.edges:
            self.edges.add((u, v))
        return new_centroid[0]

    def clear_pearls_aux(
        self,
        g,
        start,
        origin,
        centroid):

        if self.g.out_degree(start) != 1:
            if origin is not None:
                g.add_edge(origin, start)
            else:
                centroid.append(start)
            for x in self.g.successors(start):
                self.clear_pearls_aux(g, x, start, centroid)
        else:
            self.clear_pearls_aux(g, next(self.g.successors(start)), origin, centroid)

def load_vec(infile):

    with codecs.open(infile, 'r', encoding='utf-8') as instream:

        nwords, ndim = map(int, instream.readline().replace("\n", "").split())
        w2i = {}
        array = np.zeros((nwords, ndim))

        for i, l in enumerate(map(lambda x: x.replace('\n', ""), instream)):
            elements = l.split(" ")
            for k in range(ndim):
                array[i, k] = float(elements[k + 1])

            w2i[elements[0]] = i

    return Embedding(w2i, array)


def load_tree(infile, use_edges=False, with_centroid=False, reverse=False):

    with codecs.open(infile, 'r', encoding='utf-8') as instream:
        if with_centroid:
            attr, centroid = instream.readline().replace("\n", '').split('\t')
            assert attr == "centroid"
        else:
            centroid = None
        edges = set(map(lambda x: (x[1], x[0]) if reverse else x, map(lambda x: tuple(x.replace('\n', '').split('\t')), instream)))

    if not use_edges:
        return centroid, Tree(edges)
    else:
        return centroid, edges

def load_edges(infile):
    with codecs.open(infile, 'r', encoding='utf-8') as instream:
        attr, centroid = instream.readline().replace("\n", '').split('\t')
        assert attr == "centroid"
        edges = list(map(lambda x: tuple(x.replace('\n', '').split('\t')), instream))
    voc = set(list(map(lambda x: x[0], edges)) +  list(map(lambda x: x[1], edges)))
    return edges, voc

def extract_centroid(infile):

    with codecs.open(infile, 'r', encoding='utf-8') as instream:
        attr, centroid = instream.readline().replace("\n", '').split('\t')
        assert attr == "centroid"
    return centroid


def set_minus(s1, s2):
    return set(x for x in s1 if x not in s2)

def is_a(u, v, alpha, distance):
    return (1 + alpha * (np.linalg.norm(u) - np.linalg.norm(v))) * distance(u, v)

def euclidean_distance(u, v):
    return np.linalg.norm(u - v)

def to_hyperboloid(vecs, beta):

    N, d = vecs.shape
    proj = np.zeros((N, d + 1))

    proj[:, 1:] = 2 * vecs / (1 - np.sum(vecs * vecs, axis=1).reshape(-1, 1))
    proj[:, 0] = np.sqrt(np.sum(proj[:, 1:] * proj[:, 1:], axis=1) + beta)

    return proj

def lorentzian_distance(u, v, beta=1):

    U = to_hyperboloid(u, beta).reshape(-1)
    V = to_hyperboloid(v, beta).reshape(-1)
    return - 2 * beta - 2 * (-U[0] * V[0] + np.sum(U[1:] * V[1:]))

def hyperbolic_distance(u, v, c=1):

    mu = -u
    musquare = np.sum(mu * mu)
    vsquare = np.sum(v * v)
    dotmuv = np.sum(mu * v)

    norm = np.linalg.norm(((1 + 2 * c * dotmuv + c * vsquare) * mu + (1 - c * musquare) * v) / (1 + 2 * c * dotmuv + c * c * musquare * vsquare))
    return 2 / np.sqrt(c) * np.arctanh(norm * np.sqrt(c))

def save_edges(edges, file):

    with codecs.open(file, "w", encoding='utf-8') as outstream:
        for u, v in edges:
            outstream.write("%s\t%s\n" % (u, v))

def extract_hp_from_fn(fn):
    els = fn.split("/")
    dir = els[-2].split(".")
    if len(dir) > 3:
        # wordnet
        dataset = ".".join(dir[:3])
        dim = dir[3]
        negs = dir[-1][1:]
    else:
        # semeval
        dataset = "%s_%s" % (els[-4], els[-3])

        dim = els[-2].split('.')[1]
        negs = els[-1].split('.')[-2].split("_")[-2]

    tree = els[-1]

    if "mst" in tree or 'MST' in tree:
        algorithm = "mst"
    elif "npoint" in tree or 'NPOINT' in tree:
        algorithm = "npoint"
    elif "naive" in tree or 'NAIVE' in tree:
        algorithm = "naive"
    elif "TAXI" in tree:
        algorithm = "TAXI"
        if "refined" in tree:
            algorithm += "-refined"
    elif "USAAR" in tree:
        algorithm = "USAAR"
        if "refined" in tree:
            algorithm += "-refined"
    elif "JUNLP" in tree:
        algorithm = "JUNLP"
        if "refined" in tree:
            algorithm += "-refined"
    else:
        algorithm = "none"

    recursive = "no" if "recursive" not in tree else "yes"
    extended = "no" if "extended" not in tree else "yes"
    if "poincarenips" in tree or "poincare_nips" in tree:
        embedding = "poincare_nips"
    elif "hyperboliccones" in tree or "hyperbolic_cones" in tree:
        embedding = "hyperbolic_cones"
    elif "euclideannips" in tree or "euclidean_nips" in tree:
        embedding = "euclidean_nips"
    elif "euclideancones" in tree or "euclidean_cones" in tree:
        embedding = "euclidean_cones"
    elif "lorentzian" in tree or "law19a" in tree:
        embedding = "lorentzian"
    else:
        embedding = "none"

    return [dataset, dim, negs, algorithm, recursive, extended, embedding, tree]

def concat_csv_cleaner(csvfile, filteredfile):

    wordnet_metadata = {
        'attribute.n.02': (7261, 7380, 9),
        'communication.n.02': (4450, 4498, 5),
        'group.n.01': (8182, 8223, 3),
        'measure.n.02': (2308, 2333, 3),
        'event.n.01': (7879, 7958, 8),
        'cognition.n.01': (3912, 3945, 3),
        'relation.n.01': (5145, 5286, 9),
        'causal_agent.n.01': (8142, 8392, 16),
        'matter.n.03': (6576, 6903, 21),
        'location.n.01': (1262, 1265, 2),
        'natural_object.n.01': (1113, 1131, 3),
        'person.n.01': (6979, 7163, 16),
        'animal.n.01': (3999, 4033, 5),
        'plant.n.02': (4487, 4493, 2),
        'process.n.06': (1653, 1669, 3),
        'thing.n.12': (2355, 2381, 4)}

    add_headers = ["dataset", "dim", "negatives", "algorithm", "recursive", "extended", "embedding", "tree"]
    data = []
    with codecs.open(csvfile, "r", encoding="utf-8") as instream:
        headerline = instream.readline()
        ncols = len(headerline.split(','))
        for content in map(lambda x: x.replace('\n', '').split(','), filter(lambda x: x != headerline, instream)):
            if len(content) > ncols:
                start = 1
                end = 1 + len(content) - ncols + 1
                data.append(extract_hp_from_fn(content[0]) + [";".join(content[start:end])] + content[end:])
            else:
                data.append(extract_hp_from_fn(content[0]) + content[1:])

    print('Writing file at location %s' % filteredfile)
    with codecs.open(filteredfile, "w", encoding="utf-8") as outstream:
        outstream.write(",".join(add_headers) + "," + ",".join(headerline.split(",")[1:]))
        for l in data:
            outstream.write("%s\n" % ",".join(l))

def extract_subvoc_edges(edge_file, output, subvoc, csv_hierarx):

    check_voc = set()

    if csv_hierarx:
        with codecs.open(edge_file, "r", encoding="utf-8") as instream:
            with codecs.open(output, "w", encoding="utf-8") as outstream:
                for u, v, sim in map(lambda x: x.replace("\n", "").split(","), instream):
                    if u in subvoc and v in subvoc:
                        outstream.write("%s,%s,%s\n" % (u, v, sim))
                        check_voc.add(u)
                        check_voc.add(v)
    else:
        with codecs.open(edge_file, "r", encoding="utf-8") as instream:
            with codecs.open(output, "w", encoding="utf-8") as outstream:
                for u, v in map(lambda x: x.replace("\n", "").split("\t"), instream):
                    if u in subvoc and v in subvoc:
                        outstream.write("%s\t%s\n" % (u, v))
                        check_voc.add(u)
                        check_voc.add(v)

