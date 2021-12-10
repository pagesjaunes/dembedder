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

import subprocess
import os
import codecs

def py_command_builder(
        binary,
        *args,
        **kwargs):

    command = "%s" % binary
    for a in args:
        command += " %s" % a
    for k, v in kwargs.items():
        if v is not None:
            if type(v) != bool:
                if type(v) == str:
                    command += ' -%s "%s"' % (k, v)
                else:
                    command += " -%s %s" % (k, str(v))
            else:
                if v:
                    command += " --%s" % k
    return command

def command_builder(
    binary,
    **kwargs):

    command = "%s" % binary
    for k, v in kwargs.items():
        if v is not None:
            if type(v) != bool:
                if type(v) == str:
                    command += ' -%s="%s"' % (k, v)
                else:
                    command += " -%s=%s" % (k, str(v))
            else:
                if v:
                    command += " --%s" % k
    return command

def run_cpp_dembedder_neighbors(
    embeddingfile,
    output,
    delimiter=None,
    distance=None,
    dparam=None):

    assert "DEMBEDDER_BIN" in os.environ

    subprocess.run(command_builder(
        os.environ["DEMBEDDER_BIN"],
        neighbors=True,
        input=embeddingfile,
        output=output,
        distance=distance,
        dparam=dparam,
        delimiter=delimiter
    ), shell=True, check=True)

def run_cpp_dembedder_distortion(
    embeddingfile,
    treefile,
    output,
    delimiter=None,
    distance=None,
    dparam=None):

    assert "DEMBEDDER_BIN" in os.environ

    subprocess.run(command_builder(
        os.environ["DEMBEDDER_BIN"],
        distortion=True,
        input=embeddingfile,
        tree_input=treefile,
        output=output,
        distance=distance,
        dparam=dparam,
        delimiter=delimiter
    ), shell=True, check=True)

def run_cpp_tree_distortion(
        goldtree,
        predtree,
        output):

    assert "DEMBEDDER_BIN" in os.environ

    subprocess.run(command_builder(
        os.environ["DEMBEDDER_BIN"],
        tree_distortion=True,
        tree_input_1=predtree,
        tree_input_2=goldtree,
        output=output
    ), shell=True, check=True)

def run_cpp_dembedder(
    inputfile,
    outputfile,
    method,
    delimiter=None,
    root=None,
    tol=None,
    distance=None,
    dparam=None,
    verbose=None,
    tree_input=None):

    assert "DEMBEDDER_BIN" in os.environ

    subprocess.run(command_builder(
        os.environ["DEMBEDDER_BIN"],
        input=inputfile,
        output=outputfile,
        algorithm=method,
        delimiter=delimiter,
        root=root,
        tol=tol,
        distance=distance,
        dparam=dparam,
        verbose=verbose,
        tree_input=tree_input
    ), shell=True, check=True)


def write_vec(out, emb):
    with codecs.open(out, 'w', encoding='utf-8') as outstream:
        outstream.write("%d %d\n" % (len(emb.keys()), emb.dim))
        for w in emb.keys():
            outstream.write("%s " % w)
            outstream.write(" ".join(map(str, emb.project(w))))
            outstream.write("\n")

def dembed(
    emb,
    expdir,
    method,
    delimiter=None,
    root=None,
    tol=None,
    distance=None,
    dparam=None,
    verbose=None):

    intput_file = os.path.join(expdir, "emb.vec")
    output_file = os.path.join(expdir, "tree_%s.tsv" % method)
    write_vec(intput_file, emb)

    run_cpp_dembedder(
        intput_file,
        output_file,
        method,
        delimiter=delimiter,
        root=root,
        tol=tol,
        distance=distance,
        dparam=dparam,
        verbose=verbose)