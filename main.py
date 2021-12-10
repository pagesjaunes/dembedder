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

import os
import codecs
import numpy as np
import math
from tqdm import tqdm
import networkx as nx


FAMILIES = [
    'attribute.n.02',
    'communication.n.02',
    'group.n.01',
    'measure.n.02',
    'event.n.01',
    'cognition.n.01',
    'relation.n.01',
    'causal_agent.n.01',
    'matter.n.03',
    'location.n.01',
    'natural_object.n.01',
    'person.n.01',
    'animal.n.01',
    'plant.n.02',
    'process.n.06',
    'thing.n.12']

DIM = [2, 5, 10, 20, 50, 100]

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Specify command name', dest='command')

    write_wordnet_taxs = subparsers.add_parser('wordnet-tax', help="Extract file")
    write_wordnet_taxs.add_argument('expdir', type=str, help='path to experiment directory')
    write_wordnet_taxs.add_argument('entity_name', type=str, help='root entity')

    write_wordnet_main_taxs = subparsers.add_parser('wordnet-main-tax', help="Extract file")
    write_wordnet_main_taxs.add_argument('expdir', type=str, help='path to experiment directory')

    clean_csv = subparsers.add_parser('clean-csv', help="Extract file")
    clean_csv.add_argument('input', type=str, help='path to experiment directory')
    clean_csv.add_argument('output', type=str, help='path to experiment directory')

    gen_params = subparsers.add_parser('gen-param', help="Extract file")
    gen_params.add_argument('out', type=str, help='path to experiment directory')

    convert_taxi = subparsers.add_parser('convert-taxi', help="")
    convert_taxi.add_argument("input", type=str)
    convert_taxi.add_argument("output", type=str)
    convert_taxi.add_argument("centroid", type=str)

    convert_tree = subparsers.add_parser('convert-tree', help="")
    convert_tree.add_argument("input", type=str)
    convert_tree.add_argument("output", type=str)
    convert_tree.add_argument("--no_digit", action="store_true")
    convert_tree.add_argument("--keep_underscore", action="store_true")

    extend_tree = subparsers.add_parser('extend-tree', help="")
    extend_tree.add_argument("expdir", type=str)
    extend_tree.add_argument("input", type=str)
    extend_tree.add_argument("output", type=str)
    extend_tree.add_argument("lang", type=str)
    extend_tree.add_argument("gold_tree", type=str)

    semeval_extract = subparsers.add_parser('semeval-extract', help="Extract semeval data")
    semeval_extract.add_argument('expdir', type=str, help="expdir")
    semeval_extract.add_argument('lang', type=str, help="lang")
    semeval_extract.add_argument('domain', type=str, help="domain")
    semeval_extract.add_argument('-topk', type=int, default=5, help="domain")
    semeval_extract.add_argument('-threshold', type=float, default=-1, help="domain")
    semeval_extract.add_argument('-semeval_dir', type=str, default=os.path.join("third-party", "Taxonomy_Refinement_Embeddings"), help="semeval dir")

    train_emb = subparsers.add_parser('train-emb', help="train embedding")
    train_emb.add_argument('expdir', type=str, help='experiment directory')
    train_emb.add_argument('-dim', type=int, help='experiment directory')
    train_emb.add_argument('-negative', type=int, default=2, help='experiment directory')
    train_emb.add_argument('-beta', type=float, default=0.01, help='experiment directory')
    train_emb.add_argument('-nepoch', type=int, default=200, help='experiment directory')
    train_emb.add_argument('-train_class', type=str, default="Law19a", help='experiment directory')
    train_emb.add_argument('-custom_fn', type=str, default="", help='experiment directory')
    train_emb.add_argument('-lr', type=float, default=None, help='experiment directory')
    train_emb.add_argument('-momentum', type=float, default=None, help='experiment directory')
    train_emb.add_argument('--recursive', action="store_true", help="storing true")
    train_emb.add_argument('-tree_extractor', type=str, default="naive", help="storing true")
    train_emb.add_argument('-minvoc', type=str, default=100, help="storing true")
    train_emb.add_argument('--lorentzian', action="store_true")
    train_emb.add_argument('-on_edges', type=str, default="edges.tsv")
    train_emb.add_argument('--rerun', action="store_true")

    compute_tree = subparsers.add_parser('compute-tree', help="Apply dembedder cpp")
    compute_tree.add_argument("expdir", type=str, help='experiment directory')
    compute_tree.add_argument("algorithm", type=str, help='algorithm to use')
    compute_tree.add_argument("-delimiter", type=str, default=None, help='delimiter if csv file')
    compute_tree.add_argument("-root", type=str, default=None, help="root to seed the algorithm")
    compute_tree.add_argument("-tol", type=float, default=None, help="tolerance")
    compute_tree.add_argument("-distance", type=str, default=None, help="distance")
    compute_tree.add_argument("-dparam", type=float, default=None, help="distance param")
    compute_tree.add_argument("-verbose", type=int, default=None, help="verbosity")
    compute_tree.add_argument("-tree_name", type=str, default=None, help="name for tree path")
    compute_tree.add_argument("-use_tree_root", type=str, default=None, help="use same centroid")
    compute_tree.add_argument("-use_emb", type=str, default="emb.vec", help="use same centroid")
    compute_tree.add_argument("-tree_input", type=str, default=None, help="use same centroid")
    compute_tree.add_argument("--random_tree", action="store_true", help="use same centroid")
    compute_tree.add_argument('--rerun', action="store_true")

    eval_tree = subparsers.add_parser('eval-tree', help="Tree evaluation")
    eval_tree.add_argument("expdir", type=str, help="experiment directory")
    eval_tree.add_argument("-tree_name", type=str, default=None, help="name for tree")
    eval_tree.add_argument("-out", type=str, default="results.csv", help="output")
    eval_tree.add_argument("--headers", action="store_true", help="output")
    eval_tree.add_argument("--no_reverse", action="store_true", help="output")
    eval_tree.add_argument("--unrooted", action="store_true", help="output")
    eval_tree.add_argument("-distortion", type=str, default=None, help="output")
    eval_tree.add_argument("-dparam", type=float, default=1, help="output")
    eval_tree.add_argument("-use_emb", type=str, default='emb.vec', help="output")
    eval_tree.add_argument("-on_test", type=str, default=None, help="output")
    eval_tree.add_argument("-on_edges", type=str, default="gt_edges.tsv", help="edge file")

    generate_random_tree = subparsers.add_parser('generate-random-tree', help="Tree evaluation")
    generate_random_tree.add_argument("tree_path", type=str, help="experiment directory")
    generate_random_tree.add_argument("-number_nodes", type=int, help="experiment directory")
    generate_random_tree.add_argument("--random_node_number", action="store_true")
    generate_random_tree.add_argument("-min", type=int, default=50, help="experiment directory")
    generate_random_tree.add_argument("-max", type=int, default=1000, help="experiment directory")
    generate_random_tree.add_argument("--no_pearls", type=int, default=1000, help="experiment directory")

    args = parser.parse_args()

    if args.command == 'wordnet-tax':
        from nltk.corpus import wordnet as wn
        from pydembedder.eval import serialize_edges, wordnet_ground_truth_edges
        if not os.path.exists(os.path.join(args.expdir, "%s_edges.tsv" % args.entity_name)):
            edges = wordnet_ground_truth_edges(args.entity_name)
            serialize_edges(os.path.join(args.expdir, "%s_edges.tsv" % args.entity_name), edges)

            ## folllowing adapted from https://github.com/pagesjaunes/HierarX/blob/master/pyhierarx/__main__.py
            ## under GNU affero 3.0 license

            funcsims = {
                'lch': lambda x, y: x.lch_similarity(y),
                'path': lambda x, y: x.path_similarity(y),
                'wup': lambda x, y: x.wup_similarity(y)
            }


            def add_to(hierarchie, entity_name):
                    hierarchie.add(entity_name)
                    for hypo in wn.synset(entity_name).closure(lambda s: s.hyponyms()):
                        hierarchie.add(hypo.name())

            entities = set()
            add_to(entities, args.entity_name)


            with open(os.path.join(args.expdir, '%s_similarities.hierarx' % args.entity_name), 'w') as ofstream:
                    __entities = list(entities)
                    for i, x in tqdm(enumerate(__entities), total=len(__entities), unit='entity'):
                        for j in range(i + 1, len(__entities)):
                            ofstream.write('%s,%s,%.6f\n' % (x, __entities[j], funcsims['lch'](wn.synset(x), wn.synset(__entities[j]))))

    elif args.command == 'wordnet-main-tax':
        from pydembedder.eval import serialize_edges, wordnet_ground_truth_edges

        desc = {}
        for entity_name in FAMILIES:

            if os.path.exists(os.path.join(args.expdir, "%s_edges.tsv" % entity_name)):
                continue
            print('Processing %s' % entity_name)
            edges = wordnet_ground_truth_edges(entity_name)
            serialize_edges(os.path.join(args.expdir, "%s_edges.tsv" % entity_name), edges)
            voc = set(list(map(lambda x: x[0], edges)) + list(map(lambda x: x[1], edges)))

            g = nx.Graph()
            for u, v in edges:
                g.add_edge(u, v)

            desc[entity_name] = (len(voc), len(edges))

    elif args.command == 'gen-param':

        with codecs.open(args.out, 'w', encoding='utf-8') as out:
            for f in FAMILIES:
                for d in DIM:
                    out.write("%s %d\n" % (f, d))

    elif args.command == "convert-taxi":

        edges = set()
        with codecs.open(args.input, 'r' , encoding='utf-8') as instream:
            for u, v in map(lambda x: x.replace("\n", "").split('\t')[1:], instream):
                edges.add(("_".join(v.split()), "_".join(u.split())))

        from pydembedder.utils import Tree
        tr = Tree(edges)
        tr.save(args.output, args.centroid)

    elif args.command == "convert-tree":
        from pydembedder.utils import load_tree
        r, tr = load_tree(args.input, with_centroid=True)
        tr.rooten(r)
        with codecs.open(args.output, 'w', encoding='utf-8') as outstream:
            for i, x in enumerate(tr.edges):
                if args.no_digit:
                    outstream.write("%s\t%s\n" % (x[1].replace("_", " ") if not args.keep_underscore else x[1], x[0].replace("_", " ") if not args.keep_underscore else x[0]))
                else:
                    outstream.write("%d\t%s\t%s\n" % (i, x[1].replace("_", " ") if not args.keep_underscore else x[1], x[0].replace("_", " ") if not args.keep_underscore else x[0]))

    elif args.command == "extend-tree":

        from pydembedder.utils import load_tree
        from pydembedder.semeval.data import extend_tree
        import pandas as pd

        tree_path = os.path.join(args.expdir, args.input)
        gt_path = os.path.join(args.expdir, args.gold_tree)
        weight_path = os.path.join(args.expdir, "edges_weighted.tsv")
        assert os.path.exists(tree_path)
        assert os.path.exists(gt_path)
        assert os.path.exists(weight_path)

        centroid, tree = load_tree(tree_path, with_centroid=True)
        _, gt_edges = load_tree(gt_path, use_edges=True)
        terms = set(list(map(lambda x: x[0], gt_edges)) + list(map(lambda x: x[1], gt_edges)))

        weight_df = pd.read_csv(weight_path, sep="\t")
        _, tree = extend_tree(centroid, tree, terms, weight_df, args.lang.lower())

        tree.save(os.path.join(args.expdir, args.output), centroid)

    elif args.command == "semeval-extract":

        from pydembedder.semeval.data import extract_spmi_mat, extract_edges_from_spmi
        import subprocess

        semeval_data_path = os.path.join(args.semeval_dir, "data")
        semeval_eval_path = os.path.join(args.semeval_dir, "eval")

        gold_file = os.path.join(semeval_eval_path, args.lang.upper(), "gold_%s.taxo" % args.domain.lower())
        global_file = os.path.join(semeval_data_path, args.lang.upper(), "%s.csv" % args.lang.lower())
        domain_file = os.path.join(semeval_data_path, args.lang.upper(), "%s_%s.csv" % (args.lang.lower(), args.domain.lower()))

        assert os.path.exists(gold_file)
        assert os.path.exists(global_file)
        assert os.path.exists(domain_file)

        weighted_edges = os.path.join(args.expdir, "edges_weighted.tsv")
        extract_spmi_mat(
            gold_file,
            global_file,
            domain_file,
            weighted_edges,
            lang=args.lang.lower(),
            cache=os.path.join(args.expdir, "translate.json"),
            cache_webisa=os.path.join(args.expdir, "webisa.csv"))
        assert os.path.exists(weighted_edges)

        print("Extract edges from pmi")
        extract_edges_from_spmi(weighted_edges, os.path.join(args.expdir, "edges.tsv"), threshold=args.threshold, topk=args.topk)

        with codecs.open(gold_file, "r", encoding='utf-8') as instream:
            with codecs.open(os.path.join(args.expdir, "gt_edges.tsv"), "w", encoding="utf-8") as outstream:
                for l in map(lambda x : ("_".join(x[0].split()), "_".join(x[1].split())), map(lambda x: x.replace("\n", "").split("\t")[1:], instream)):
                    outstream.write("\t".join(l) + "\n")

    elif args.command == "clean-csv":

        from pydembedder.utils import concat_csv_cleaner
        concat_csv_cleaner(args.input, args.output)

    elif args.command == 'train-emb':

        embedding_name = {
            'EuclideanNIPS': "emb_euclideannips.vec",
            'PoincareNIPS': "emb_poincarenips.vec",
        }

        train_params = {
            'dim': args.dim,
            'negative': args.negative,
            'beta': args.beta,
            'nepoch': args.nepoch,
            'lorentzian_gensim': args.lorentzian,
            'niter': args.nepoch,
            'nthread': os.environ["NTHREAD_HIERARX"] if "NTHREAD_HIERARX" in os.environ else 1,
            'lr': args.lr,
            'momentum': args.momentum,
            'bs': 1 + args.negative,
            'lorentzian': args.lorentzian,
            'plateau': 0.1,
            'early_stop': True,
            'outfile': ""
        }

        train_func = None
        file_path = os.path.join(args.expdir, embedding_name[args.train_class]) if not args.recursive else os.path.join(args.expdir, "recursive_%s_%s" % (args.train_class.lower(), args.tree_extractor))

        if args.rerun or not os.path.exists(file_path):

            from pydembedder.train import train_cones_edges
            edge_file = os.path.join(args.expdir, args.on_edges)
            assert os.path.exists(edge_file)

            def train_cones(expdir, edge, out, train_class, **kwargs):
                kwargs["outfile"] = out
                train_cones_edges(
                    expdir,
                    edge,
                    train_class,
                    **kwargs)

            train_func = lambda x, y, z: train_cones(x, y, z, args.train_class, **train_params)

            train_func(args.expdir, edge_file, args.custom_fn)

    elif args.command == 'compute-tree':
        from pydembedder.algo import run_cpp_dembedder
        from pydembedder.utils import extract_centroid

        input_file = os.path.join(args.expdir, args.use_emb)
        assert os.path.exists(input_file)

        if args.use_tree_root is not None:
            tree_file = os.path.join(args.expdir, args.use_tree_root)
            assert tree_file
            assert args.root is not None
            root = extract_centroid(tree_file)
        else:
            root = args.root
            tree_file = None

        if args.tree_input is not None:
            tree_file = os.path.join(args.expdir, args.tree_input)
            assert(os.path.exists(tree_file) or args.random_tree)
            if args.random_tree:
                from networkx.generators.trees import random_tree
                from pydembedder.utils import load_vec
                emb = load_vec(input_file)
                N = len(emb.voc)
                t = random_tree(N)
                with codecs.open(tree_file, "w", encoding='utf-8') as outstream:
                    outstream.write("centroid\t%s\n" % root)
                    for u, v in t.edges:
                        outstream.write("%s\t%s\n" % (emb.voc[u], emb.voc[v]))

        output_file = os.path.join(args.expdir, "tree.tsv" if args.tree_name is None else args.tree_name)

        if args.rerun or not os.path.exists(output_file):
            run_cpp_dembedder(
                input_file,
                output_file,
                args.algorithm,
                delimiter=args.delimiter,
                root=root if root != 'NULL' and root is not None else "",
                tol=args.tol,
                distance=args.distance,
                dparam=args.dparam,
                verbose=args.verbose,
                tree_input=tree_file)

    elif args.command == 'eval-tree':

        from pydembedder.utils import load_vec, load_tree, set_minus, Tree
        from pydembedder.eval import graph_fscore, graph_fscore_undirected
        import json

        pred_tree_file = os.path.join(args.expdir, args.tree_name if args.tree_name is not None else "tree.tsv")
        gt_tree_file = os.path.join(args.expdir, args.on_edges)
        emb_file = os.path.join(args.expdir, args.use_emb)

        assert os.path.exists(emb_file) or args.distortion is None
        assert os.path.exists(pred_tree_file)
        assert os.path.exists(gt_tree_file)

        centroid, pred_tree = load_tree(pred_tree_file, with_centroid=True)
        _, gt_tree = load_tree(gt_tree_file, reverse=not args.no_reverse)
        pred_tree = Tree(pred_tree.edges, voc_restrict=set(gt_tree.nodes))

        print("Evaluation of %s" % pred_tree_file)
        print("Score without organisation")
        print("-- exact undirected match")
        undir_match = graph_fscore_undirected(pred_tree.edges, gt_tree.edges)
        print(json.dumps(undir_match, indent=4))

        print("Eval GED")
        from pydembedder.utils import NetworkX111Wrapper
        from aproximated_ged import VanillaHED
        pred_n111 = NetworkX111Wrapper(pred_tree.g.to_undirected())
        gt_n111 = NetworkX111Wrapper(gt_tree.g.to_undirected())
        #print(pred_tree.edges, gt_tree.edges)
        hed = VanillaHED(del_node = 0, ins_node = 0, del_edge = 1, ins_edge = 1)
        hed_ged = hed.ged(pred_n111, gt_n111)[0]
        print("--- haussdorff edit distance: %.2f" % hed_ged)
        #hed_ged = -1

        import gmatch4py as gm
        hed = gm.kernels.weisfeiler_lehman.WeisfeleirLehmanKernel(h=5)
        swlk = hed.compare([pred_tree.g.to_undirected(), gt_tree.g.to_undirected()], None)[0, 1]
        print("--- Weisfeiler Lehman kernel: %.4f" % swlk)
        hed = gm.kernels.shortest_path_kernel.ShortestPathGraphKernel()
        sspk = hed.compare([pred_tree.g.to_undirected(), gt_tree.g.to_undirected()], None)[0, 1]
        print("--- shortest path kernel: %.4f" % sspk)
        #sspk = -1

        if not args.unrooted:
            pred_tree.rooten(centroid)
        print("")
        print("Score with rooted tree at %s" % centroid)
        print("-- exact match")
        basic_match = graph_fscore(pred_tree.edges, gt_tree.edges)
        print(json.dumps(basic_match, indent=4))
        print("-- hypernym match")
        transitive_match = graph_fscore(pred_tree.ancestor_edges(), gt_tree.ancestor_edges())

        print(json.dumps(transitive_match, indent=4))
        print("-- non basic-edge match")
        non_basic_edges = set_minus(gt_tree.ancestor_edges(), gt_tree.edges)
        non_basic_match = graph_fscore(set_minus(pred_tree.ancestor_edges(), gt_tree.edges), non_basic_edges)
        print(json.dumps(non_basic_match, indent=4))
        if args.on_test is not None:
            test_file = os.path.join(args.expdir, args.on_test)
            assert os.path.exists(test_file)
            non_basic_test_edges = load_tree(test_file, use_edges=True)[1]
            print(("-- non basic-edge match (on test %s" % test_file) + ")")
            non_basic_match_test = graph_fscore(set_minus(pred_tree.ancestor_edges(), set_minus(non_basic_edges, non_basic_test_edges)), non_basic_test_edges)
            print(json.dumps(non_basic_match_test, indent=4))
        else:
            non_basic_match_test = {'fscore': -1, 'precision': -1, 'recall': -1}




        if args.distortion is not None:
            print("Compute distortion")
            from pydembedder.algo import run_cpp_dembedder_distortion
            tmpfile = os.path.join(args.expdir, "tmp.txt")
            run_cpp_dembedder_distortion(emb_file, pred_tree_file, tmpfile, distance= args.distortion, dparam=args.dparam)
            assert os.path.exists(tmpfile)
            with codecs.open(tmpfile, 'r', encoding='utf-8') as infile:
                def parse_row(l):
                    return float(l.replace("\n", "").split()[1])
                max_expand = parse_row(infile.readline())
                avg_expand = parse_row(infile.readline())
                max_contrac = parse_row(infile.readline())
                avg_contrac = parse_row(infile.readline())

                maxd = max_expand * max_contrac
                avgd = avg_expand * avg_contrac

        else:
            maxd, avgd = -1, -1
            max_expand = -1
            avg_expand = -1
            max_contrac = -1
            avg_contrac = -1
            map = -1
            mr = -1

        with codecs.open(os.path.join(args.expdir, args.out), "w", encoding='utf-8') as outstream:
            headers = [
                'file',
                'root',
                '#nodes',
                '#edges',
                '#extracted_nodes',
                '#extracted_edges',
                'undirected_precision',
                'undirected_recall',
                'undirected_fscore',
                'basic_precision',
                'basic_recall',
                'basic_fscore',
                'transitive_precision',
                'transitive_recall',
                'transitive_fscore',
                'non_basic_precision',
                'non_basic_recall',
                'non_basic_fscore',
                'non_basic_precision_test',
                'non_basic_recall_test',
                'non_basic_fscore_test',
                'maximum_distortion',
                'average_distortion',
                'max_expansion',
                'average_expansion',
                'max_contaction',
                'average_contraction',
                'similarity_weisfeleir_lehman_kernel',
                'similarity_shortest_paths_kernel',
                'hed_ged']

            content = [pred_tree_file, centroid, str(len(gt_tree.nodes)), str(len(gt_tree.edges)), str(len(pred_tree.nodes)), str(len(pred_tree.edges))]
            for d in [undir_match, basic_match, transitive_match, non_basic_match, non_basic_match_test]:
                for k in ['precision', 'recall', 'fscore']:
                    content.append("%.4f" % d[k])

            content.append("%.6f" % maxd)
            content.append("%.6f" % avgd)
            content.append("%.6f" % max_expand)
            content.append("%.6f" % avg_expand)
            content.append("%.6f" % max_contrac)
            content.append("%.6f" % avg_contrac)
            content.append("%.4f" % swlk)
            content.append("%.4f" % sspk)
            content.append("%.2f" % hed_ged)

            assert len(headers) == len(content)
            if args.headers:
                outstream.write(','.join(headers) + "\n")
            outstream.write(','.join(content) + "\n")

        print(json.dumps(dict(zip(headers, content)), indent=4))

    elif args.command == "generate-random-tree":

        from networkx.generators.trees import random_tree
        from pydembedder.utils import Tree
        from networkx.algorithms.tree.recognition import is_arborescence

        if args.random_node_number:
            import random
            N = random.randint(args.min, args.max)
        else:
            N = args.number_nodes

        t = random_tree(N)
        edges = set()
        for u, v in t.edges:
            edges.add((str(u), str(v)))

        tr = Tree(edges)
        centroid = str(random.randint(0, N-1))
        tr.rooten(centroid)
        if args.no_pearls:
            centroid = tr.clear_pearls(centroid)
            tr.rooten(centroid)

        assert is_arborescence(tr.g)
        if len(tr.nodes) > (args.min / 2):
            tr.save(args.tree_path, centroid)

    else:
        raise ValueError("Invalid command %s" % args.command)