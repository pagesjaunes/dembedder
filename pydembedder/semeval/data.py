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

import pandas as pd
import numpy as np
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from googletrans import Translator
import time, json, codecs, os

SPARQL_WEBISA = SPARQLWrapper("http://webisa.webdatacommons.org/sparql")
MAXVOCSIZE = 500

def translate_voc(
        source_voc,
        sourcelang,
        targetlang):

    translator = lambda x: Translator().translate(x, src=sourcelang, dest=targetlang).text.split("\n")

    if len(source_voc) < MAXVOCSIZE:
        return dict(zip(source_voc, translator('\n'.join(source_voc))))
    else:
        diclist = []
        for i in tqdm(range(len(source_voc) // MAXVOCSIZE + 1)):
            subvoc = source_voc[i * MAXVOCSIZE:(i + 1) * MAXVOCSIZE]
            diclist.append(dict(zip(subvoc, translator('\n'.join(subvoc)))))
            time.sleep(2)
        return dict(sum(map(lambda x: list(x.items()), diclist), []))

def get_map_for_lang(
        source_voc,
        sourcelang,
        langs):

    vocs = dict()
    for lg in langs:
        vocs[lg] = translate_voc(source_voc, sourcelang, lg)
        time.sleep(5)

    return vocs

def query_sparql(sparql, w):

    sparql.setQuery("""
        PREFIX isa: <http://webisa.webdatacommons.org/concept/>
        PREFIX isaont: <http://webisa.webdatacommons.org/ontology#> 
        SELECT ?concept ?hypernymLabel ?hyponymLabel ?confidence ?freq
        WHERE{
            GRAPH ?g {
                ?concept skos:broader ?hyponym.
            }
            ?concept rdfs:label "%s".
            ?concept rdfs:label ?hypernymLabel.
            ?hyponym rdfs:label ?hyponymLabel.
            ?g isaont:hasConfidence ?confidence.
            ?g isaont:hasFrequency ?freq.
        }
        ORDER BY DESC(?confidence)
    """ % (w))
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results

def query_voc(voc, lang="en", cache=None, cache_df=None):

    if cache_df is None or not os.path.exists(cache_df):

        if lang != "en":
            print("Translating...")
            if cache is not None:
                if os.path.exists(cache):
                    maps = json.load(codecs.open(cache, 'r', encoding='utf-8'))
                else:
                    maps = get_map_for_lang(voc, lang, ["en"])
                    json.dump(maps, codecs.open(cache, "w", encoding='utf-8'))
            else:
                maps = get_map_for_lang(voc, lang, ["en"])
            _voc = list(map(lambda x: maps["en"][x], voc))
            rev = dict(map(lambda x: (x[1], x[0]), maps["en"].items()))
            print("Done.")
        else:
            _voc = voc

        vocset = set(_voc)
        edges = dict()

        for w in tqdm(_voc):
            res = query_sparql(SPARQL_WEBISA, w)
            for r in res["results"]["bindings"]:
                if r['hyponymLabel']['value'] in vocset:
                    edges[(r['hyponymLabel']['value'], r['hypernymLabel']['value'])] = {
                        "confidence": r['confidence']['value'],
                        "freq": r['freq']['value']
                    }

        df = {
            'hyponym': [],
            'hypernym': [],
            'freq': [],
            'confidence': []
        }

        for hypo, hyper in edges.keys():
            if lang == "en":
                df['hyponym'].append(hypo)
                df['hypernym'].append(hyper)
            else:
                df['hyponym'].append(rev[hypo])
                df['hypernym'].append(rev[hyper])

            vals = edges[(hypo, hyper)]
            df['freq'].append(int(vals['freq']))
            df['confidence'].append(float(vals['confidence']))

        df = pd.DataFrame(df)
        if cache_df is not None:
            df.to_csv(cache_df)
    else:
        df = pd.read_csv(cache_df)
    return df

def import_file(
        path):
    return pd.read_csv(path, delimiter='\t', error_bad_lines=False, encoding='utf-8', skiprows=[39791])

def import_gold_file(
        path):
    return pd.read_csv(path, delimiter='\t',  header=None)

def get_voc_gold(
        df):

    voc = set(df.iloc[:, 1])
    voc.update(df.iloc[:, 2])
    return set(map(str, voc))

def filter_voc(
        df,
        voc):

    hypodf = df[df['hyponym'].isin(voc)]
    return hypodf[hypodf['hypernym'].isin(voc)]

def score(a, b):
    return  len(b) / len(a) if match(a, b) else 0

def match(a, b):
    return b in a

def combine_pmi(
        dflist,
        svdreg = 0):

    voc = set()
    ppmis = dict()
    part = dict()

    for i, df in enumerate(dflist):
        Wmax = df.sum()['freq']
        edges = dict()
        hypo_occur = dict()
        hyper_occur = dict()
        for _, minidf in df.iterrows():
            hypo, hyper = minidf['hyponym'], minidf['hypernym']
            if hypo != hyper:
                edges[(hypo, hyper)] = minidf['freq']

                if (hypo, hyper) not in edges.keys():
                    edges[(hypo, hyper)] = 0
                if hypo not in hypo_occur.keys():
                    hypo_occur[hypo] = 0
                if hyper not in hyper_occur.keys():
                    hyper_occur[hyper] = 0

                edges[(hypo, hyper)] += minidf['freq'] / Wmax
                hypo_occur[hypo] += minidf['freq'] / Wmax
                hyper_occur[hyper] += minidf['freq'] / Wmax
                voc.add(hypo)
                voc.add(hyper)

        for (hypo, hyper) in edges.keys():
            if (hypo, hyper) not in ppmis.keys():
                ppmis[(hypo, hyper)] = np.zeros((len(dflist)))
            ppmis[(hypo, hyper)][i] = max(0, np.log(edges[hypo, hyper] / (hyper_occur[hyper] * hypo_occur[hypo])))

    vmax = np.max(np.array([v for v in ppmis.values()]), axis=0)

    keepdf = {
        'hyponym': [],
        'hypernym': [],
        'freq': []
    }
    for hypo, hyper in ppmis.keys():
        keepdf['hyponym'].append(hypo)
        keepdf['hypernym'].append(hyper)
        keepdf['freq'].append(np.mean(ppmis[(hypo, hyper)][:-1]) + ppmis[(hypo, hyper)][-1])
        #np.sum(ppmis[(hypo, hyper)]) / np.sum((ppmis[(hypo, hyper)] > 0)) * (1 + score(hypo, hyper)))

    keepdf = pd.DataFrame(keepdf)


    if svdreg > 0:
        voclist = list(voc)
        arr = np.zeros((len(voc), len(voc)))
        for i, w1 in tqdm(enumerate(voclist), total=len(voc)):
            for j, w2 in enumerate(voclist):
                if (w1, w2) in ppmis.keys():
                    df = keepdf[keepdf['hyponym'] == w1]
                    df = df[df['hypernym'] == w2]
                    if df.shape[0] > 0:
                        arr[i, j] = df['freq']
        u, s, v = np.linalg.svd(arr)
        s = np.diag(s[:svdreg])
        u = u[:, :svdreg]
        v = v[:svdreg, :]

        spmi = lambda x, y: np.dot(np.dot(u[voclist.index(x), :].reshape(1, -1), s), v[:, voclist.index(y)].reshape(-1, 1))[0][0]

        keepdf = {
            'hyponym': [],
            'hypernym': [],
            'freq': []
        }
        for hypo, hyper in ppmis.keys():

            keepdf['hyponym'].append(hypo)
            keepdf['hypernym'].append(hyper)
            keepdf['freq'].append(spmi(hypo, hyper))

        keepdf = pd.DataFrame(keepdf)

    #keepdf['freq'] = (keepdf['freq'] - keepdf['freq'].mean()) / keepdf['freq'].std()
    return keepdf

def substring_df(
        voc):

    ss_df = {
        'hyponym': [],
        'hypernym': [],
        'freq': []
    }

    for hypo in voc:
        for hyper in voc:
            sim = score(hypo, hyper)
            if sim > 0 and hypo != hyper :
                ss_df['hyponym'].append(hypo)
                ss_df['hypernym'].append(hyper)
                ss_df['freq'].append(sim)

    return pd.DataFrame(ss_df)

def extract_edges(
        df):

    edges = {}
    for _, minidf in tqdm(df.iterrows(), total=df.shape[0]):
        edges[(minidf['hyponym'], minidf['hypernym'])] = minidf['freq']
    return edges

def filter_edges(
        edges,
        threshold = 5):

    new_edges = {}
    for hypo, hyper in edges.keys():
        if edges[(hypo, hyper)] > threshold and hypo != hyper:
            if (hyper, hypo) in edges.keys():
                if edges[(hypo, hyper)] > edges[(hyper, hypo)]:
                    new_edges[(hypo, hyper)] = edges[(hypo, hyper)]
                else:
                    continue
            else:
                new_edges[(hypo, hyper)] = edges[(hypo, hyper)]

    return new_edges

def keep_neighbors(
        edges,
        topk=3):

    hypo_edges = {}
    for hypo, hyper in edges.keys():
        if hypo not in hypo_edges.keys():
            hypo_edges[hypo] = []
        hypo_edges[hypo].append((hyper, edges[(hypo, hyper)]))

    new_edges = {}
    for hypo, neighbors in hypo_edges.items():
        neighbors = sorted(neighbors, key=lambda x: -x[1])
        for hyper, val in neighbors[:topk]:
            new_edges[(hypo, hyper)] = val
    return new_edges

def extract_spmi_mat(gold_file, global_file, domain_file, outfile_edges, lang="en", cache=None, cache_webisa=None):

    print('Extracting vocabulary and relations')
    terms = get_voc_gold(import_gold_file(gold_file))
    df = import_file(domain_file)
    global_df = import_file(global_file)
    webisa = query_voc(list(terms), lang=lang, cache=cache, cache_df=cache_webisa)


    print('Parsing data')
    filtered_A = filter_voc(df, terms)
    filtered_B = filter_voc(global_df, terms)
    filtered_C = filter_voc(webisa, terms)
    filtered_D = substring_df(terms)

    print('Extracting & writing relations')
    d = combine_pmi([filtered_A, filtered_B, filtered_C, filtered_D], svdreg=50)
    d.to_csv(outfile_edges, header=True, index=False, sep='\t')

def extract_edges_from_spmi(weighted_edges, outfile_edges, threshold=0, topk=5):
    df = pd.read_csv(weighted_edges, sep="\t")
    edges = keep_neighbors(filter_edges(extract_edges(df), threshold=threshold if threshold >= 0 else df['freq'].mean()), topk=topk)
    with codecs.open(outfile_edges, "w", encoding="utf-8") as out:
        for u, v in edges.keys():
            out.write("%s\t%s\n" % ("_".join(str(u).split()), "_".join(str(v).split())))


def extend_tree(
    centroid,
    tree,
    terms,
    edge_weight_df,
    lang,
    percentile=0.95):

    ## supposed rooted tree
    from networkx.algorithms.tree.recognition import is_arborescence
    tree.rooten(centroid)
    assert is_arborescence(tree.g)

    spacy_model = {
        'fr': 'fr_core_news_lg',
        'en': 'en_core_web_lg',
        'it': 'it_core_news_lg',
        'nl': 'nl_core_news_lg'}

    ### ATTACH using relations
    print("Attach using relations")
    dict_of_weight = dict()
    for _, minidf in edge_weight_df[edge_weight_df['freq'] > edge_weight_df['freq'].quantile(percentile)].iterrows():
        dict_of_weight[minidf["hypernym"], minidf["hyponym"]] = minidf["freq"]

    N = 0
    while len(tree.edges) != N:
        N = len(tree.edges)
        disconnected = set(x for x in terms if x not in tree.nodes)
        ## cycle are impossible
        for _, minidf in edge_weight_df[edge_weight_df['freq'] > edge_weight_df['freq'].quantile(percentile)].iterrows():
            hypernym, hyponym = minidf["hypernym"], minidf['hyponym']
            if (hyponym, hypernym) not in dict_of_weight or dict_of_weight[(hyponym, hypernym)] <= minidf["freq"]:
                if hyponym in disconnected or hypernym in disconnected:
                    tree.add_edge(hypernym, hyponym)
                    print("Adding %s -> %s" % (hypernym, hyponym))

    ### ATTACH uing substring
    print("\nAttach using substring")
    N = 0
    while len(tree.edges) != N:
        N = len(tree.edges)
        disconnected = set(x for x in terms if x not in tree.nodes)
        for c in disconnected:
            p = max(filter(lambda x: x != c, tree.nodes), key=lambda p: score(c, p))
            if score(c, p) > 0.25:
                tree.add_edge(p, c)
                print("Adding %s -> %s" % (p, c))


    ### ATTACH using vectors

    print("\nAttach using word vectors")
    import spacy
    m = spacy.load(spacy_model[lang])
    proj = lambda x: m(x.replace("_", " ")).vector
    node_word = list(filter(lambda w: np.linalg.norm(proj(w)) > 0, tree.nodes))
    node_vecs = [proj(w) for w in node_word if np.linalg.norm(proj(w)) > 0]

    # compute threshold for cosine similarity
    cos = []
    for x in tree.nodes:
        for p in tree.predecessors(x):
            if np.any(proj(x) != 0) and np.any(proj(p) != 0):
                cos.append(np.dot(proj(x), proj(p)) / np.linalg.norm(proj(x)) / np.linalg.norm(proj(p)))
    thres = min(0.95, np.mean(cos) + np.std(cos))

    N = 0
    while len(tree.edges) != N:
        N = len(tree.edges)
        disconnected = set(x for x in terms if x not in tree.nodes)
        for c in disconnected:
            vecs = np.array(node_vecs)
            sims = (np.dot(proj(c).reshape(1, -1), vecs.T) / np.linalg.norm(proj(c)).reshape(-1) / np.linalg.norm(vecs, axis=1).reshape(1, -1)).reshape(-1)
            brother = node_word[np.argmax(sims)]
            potential_parent = list(tree.predecessors(brother))

            if len(potential_parent) > 0 and np.max(sims) > thres:
                tree.add_edge(potential_parent[0], c)
                print("Adding %s -> %s" % (potential_parent[0], c))
                if np.linalg.norm(proj(c)) > 0:
                    node_word.append(c)
                    node_vecs.append(proj(c))
            else:
                pass


    ### Reconnect disconnected nodes and components to centroid
    print("\nRecompose components")

    component_list = tree.get_components()
    print("\nAttach component roots")
    for component in component_list:
        if centroid in component:
            pass
        else:
            for n in filter(lambda n: tree.g.in_degree(n) == 0, component):
                tree.add_edge(centroid, n)
                print("Adding %s -> %s" % (centroid, n))

    assert is_arborescence(tree.g)
    return centroid, tree