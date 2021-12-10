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

## Introduction

This project is the official implementation for the paper: *Unsupervised Tree Extraction in Embedding Spaces for Taxonomy Induction* by François Torregrossa, Robin Allesiardo, Vincent Claveau and Guillaume Gravier, presented at WI-IAT 2021 international conference.

## Installation

### Requirements
- python 3.6.12 (with tcl-tk 8.6)
- boost 1.72
- gcc 10.2

### Procedure
```sh
git clone {link_to_project}
cd ./dembedder
git submodule init
git submodule update
mkdir pydembedder/deps/hyperbolic_cones/logs
pip install -r requirements.txt
cd third-party/aproximated_ged 
pip install .
cd ../GMath4py
pip install -r requirements.txt
pip install .
cd ../..
cd build
export ARGH_HOME=$PWD/../third-party/argh
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX
make dembedder 
cd ..
export DEMBEDDER_BIN=$PWD/build/dembedder

python -m spacy download en_core_web_lg # you may want to download {lang_code}_core_news_lg for fr, it, nl.
```

Do not forget to download data from the project: https://github.com/uhh-lt/Taxonomy_Refinement_Embeddings.git (follow readme instruction). Unzip the data.zip at the root of this project.

## Citation

If you use this code, please cite our paper:
```
@inproceedings{
    title = {Unsupervised Tree Extraction in Embedding Spaces for Taxonomy Induction},
    author = {
        François Torregrossa and
        Robin Allesiardo and
        Vincent Claveau and
        Guillaume Gravier
    },
    year = {2021},
    booktitle = {WI-IAT 2021}
}
``` 