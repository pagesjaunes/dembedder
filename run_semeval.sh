# dembedder is a software aiming at inducing taxonomies from embedding spaces.
# Copyright (C) 2021 Solocal-SA and CNRS
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# dembedder was developped by: François Torregrossa, Robin Allesiardo, Vincent Claveau and Guillaume Gravier.


LANG=EN
DOMAIN=science
ROOT=science # root of the taxonomy may differ in other languages

# Create experiment directory
mkdir -p semeval
mkdir -p semeval/$LANG
mkdir -p semeval/$LANG/$DOMAIN

export EXPDIR=$PWD/semeval/$LANG/$DOMAIN

# Extract semeval features and build the main graph
python main.py semeval-extract $EXPDIR $LANG $DOMAIN

# train embedding on the main graph
PYTHONPATH=$PWD/pydembedder/deps/hyperbolic_cones:$PYTHONPATH python main.py train-emb $EXPDIR -dim 100 -nepoch 1500 -negative 10 -train_class PoincareNIPS
python main.py compute-tree $EXPDIR NAIVE -tree_name tree_naive_poincare_nips.tsv -use_emb emb_poincarenips.vec -distance poincare -dparam 1 -root $ROOT

# refine taxonomy
python main.py extend-tree $EXPDIR tree_naive_poincare_nips.tsv tree_naive_poincare_nips_extended.tsv $LANG gt_edges.tsv

# eval taxonomy
python main.py eval-tree $EXPDIR -tree_name tree_naive_poincare_nips.tsv -out eval_naive_poincare_nips.csv -use_emb emb_poincarenips.vec -distortion poincare -dparam 1
python main.py eval-tree $EXPDIR -tree_name tree_naive_poincare_nips_extended.tsv -out eval_naive_poincare_nips_extended.csv -use_emb emb_poincarenips.vec -distortion poincare -dparam 1