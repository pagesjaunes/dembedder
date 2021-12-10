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

ROOTDIR=.
WORDNETDIR=${ROOTDIR}/wordnet
ENTITY=$1
DIM=100
NEPOCH=1500
NEGATIVE=10

mkdir -p $WORDNETDIR

python main.py wordnet-tax $WORDNETDIR $ENTITY

ROOT_EXPDIR=$WORDNETDIR/$ENTITY
EXPDIR=$ROOT_EXPDIR/tree_extract.$DIM.n$NEGATIVE
mkdir -p $EXPDIR
echo "Copying $WORDNETDIR/${ENTITY}_edges.tsv"
cp $WORDNETDIR/${ENTITY}_edges.tsv $ROOT_EXPDIR/edges.tsv
cp $WORDNETDIR/${ENTITY}_edges.tsv $ROOT_EXPDIR/gt_edges.tsv
cp $WORDNETDIR/${ENTITY}_edges.tsv $EXPDIR/edges.tsv
cp $WORDNETDIR/${ENTITY}_edges.tsv $EXPDIR/gt_edges.tsv
cp $WORDNETDIR/${ENTITY}_similarities.hierarx $EXPDIR/sims.hierarx

PYTHONPATH=$PWD/pydembedder/deps/hyperbolic_cones:$PYTHONPATH python main.py train-emb $EXPDIR -dim $DIM -nepoch $NEPOCH -negative $NEGATIVE -train_class PoincareNIPS
python main.py compute-tree $EXPDIR MST -tree_name tree_naive_poincare_nips.tsv -use_emb emb_poincarenips.vec -distance poincare -dparam 1
python main.py eval-tree $EXPDIR -tree_name tree_naive_poincare_nips.tsv -out eval_naive_poincare_nips.csv -use_emb emb_poincarenips.vec -distortion poincare -dparam 1