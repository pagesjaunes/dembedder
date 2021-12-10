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

from pydembedder.utils import load_tree
from gensim.models import KeyedVectors
import os
import numpy as np

def train_cones_edges(
    expdir,
    edge_file,
    train_class,
    dim=2,
    negative=2,
    nepoch=200,
    outfile="",
    **kwargs):

    from pydembedder.deps.hyperbolic_cones.relations import Relations

    print("Reading")
    all_edges = Relations(edge_file, True)

    from pydembedder.deps.hyperbolic_cones.params import default_params, non_default_params
    from pydembedder.deps.hyperbolic_cones.utils import setup_logger

    logger = setup_logger("HYPERBOLIC:CONES", also_stdout=False)
    params = default_params.copy()
    params['num_negative'] = negative
    params['dim'] = dim
    params['epochs'] = nepoch
    params['epochs_init'] = 150

    model = None
    if train_class == "PoincareNIPS":

        from pydembedder.deps.hyperbolic_cones.poincare_model import PoincareModel
        model = PoincareModel(
            train_data=all_edges,
            dim=params['dim'],
            logger=logger,
            init_range=(-0.0001, 0.0001),
            lr=params['lr_init'],
            opt=params['opt'],  # rsgd or exp_map
            burn_in=params['epochs_init_burn_in'],
            epsilon=params['epsilon'],
            seed=params['seed'],
            num_negative=params['num_negative'],
            neg_sampl_strategy=params['neg_sampl_strategy_init'],
            where_not_to_sample=params['where_not_to_sample'],
            neg_edges_attach=params['neg_edges_attach'],
            always_v_in_neg=True,
            neg_sampling_power=params['neg_sampling_power_init'],
            loss_type='nll')

    elif train_class == "EuclideanNIPS":

        from pydembedder.deps.hyperbolic_cones.eucl_simple_model import EuclSimpleModel
        model = EuclSimpleModel(
            train_data=all_edges,
            dim=params['dim'],
            logger=logger,
            init_range=(-0.0001, 0.0001),
            lr=params['lr_init'],
            burn_in=params['epochs_init_burn_in'],
            seed=params['seed'],
            num_negative=params['num_negative'],
            neg_sampl_strategy=params['neg_sampl_strategy_init'],
            where_not_to_sample=params['where_not_to_sample'],
            neg_edges_attach=params['neg_edges_attach'],
            always_v_in_neg=True,
            neg_sampling_power=params['neg_sampling_power_init'])

    assert(model is not None)

    print("Training %s model (cones)" % train_class)
    model.train(epochs=params['epochs'], batch_size=params['batch_size'], print_every=params['print_every'])

    if outfile == "":
        outfile = ('emb_%s' % train_class.lower()) + ".vec"

    filename = os.path.join(expdir,  outfile)
    print('Converting model at %s' % filename)
    model.kv.save_word2vec_format(filename)
