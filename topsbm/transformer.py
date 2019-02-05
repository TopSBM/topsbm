"""
This contains the hSBM topic modelling transformer, TopSBM
"""
# This file is part of TopSBM
# Copyright 2017-8, Martin Gerlach and the University of Sydney
#
# TopSBM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TopSBM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TopSBM.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import graph_tool
from graph_tool import Graph
from graph_tool.inference import minimize_nested_blockmodel_dl
import scipy.sparse
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state


class TopSBM(BaseEstimator):
    """A Scikit-learn compatible transformer for hSBM topic models

    Parameters
    ----------
    n_init : int, default=1
        Number of random initialisations to perform in order to avoid a local
        minimum of MDL. The minimum MDL solution is chosen.
    min_groups : int, default=None
        The minimum number of word and docuent groups to infer. This is also a
        lower bound on the number of topics.
    max_groups : int, default=None
        The maximum number of word and docuent groups to infer. This also an
        upper bound on the number of topics.
    weighted_edges : bool, default=True
        When True, edges are weighted instead of adding duplicate edges.
    random_state : None, int or np.random.RandomState
        Controls randomization. See Scikit-learn's glossary.

        Note that if this is set, the global random state of libcore will be
        affected, and the global random state of numpy will be temporarily
        affected.

    Attributes
    ----------
    graph_ : graph_tool.Graph
        Bipartite graph between samples (the first `n_samples_` vertices) and
        features (the remaining vertices)
    state_
        Inference state from graphtool
    n_levels_ : int
        The number of levels in the inferred hierarchy of groups.
    groups_ : dict
        Results of group membership from inference.
        Key is an integer, indicating the level of grouping (starting from 0).
        Value is a dict of information about the grouping which contains:

        B_d : int
            number of doc-groups
        B_w : int
            number of word-groups
        p_tw_d : array of shape (B_w, d)
            doc-topic mixtures:
            prob of word-group tw in doc d P(tw | d)
        p_td_d : array of shape (B_d, n_samples)
            doc-group membership:
            prob that doc-node d belongs to doc-group td: P(td | d)
        p_tw_w : array of shape (B_w, n_features)
            word-group-membership:
            prob that word-node w belongs to word-group tw: P(tw | w)
        p_w_tw : array of shape (n_features, B_w)
            topic distribution:
            prob of word w given topic tw P(w | tw)

        Here "d"/document refers to samples; "w"/word refers to features.
    mdl_
        minimum description length of inferred state
    n_features_ : int
    n_samples_ : int

    References
    ----------
    Martin Gerlach, Tiago P. Peixoto, and Eduardo G. Altmann,
    `“A network approach to topic models,”
    <http://advances.sciencemag.org/content/4/7/eaaq1360>`_.
    Science Advances (2018)
    """

    def __init__(self, n_init=1, min_groups=None, max_groups=None,
                 weighted_edges=True, random_state=None):

        self.n_init = n_init
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.weighted_edges = weighted_edges
        self.n_init = n_init
        self.random_state = random_state

    def __make_graph(self, X):
        # make a graph
        g = Graph(directed=False)
        # define node properties
        # kind: docs - 0, words - 1
        kind = g.vp["kind"] = g.new_vp("int")
        if self.weighted_edges:
            ecount = g.ep["count"] = g.new_ep("int")

        # add all documents first
        doc_vertices = [g.add_vertex() for _ in range(X.shape[0])]
        word_vertices = [g.add_vertex() for _ in range(X.shape[1])]

        # add all documents and words as nodes
        # add all tokens as links
        X = scipy.sparse.coo_matrix(X)

        if not self.weighted_edges and X.dtype != int:
            X_int = X.astype(int)
            if not np.allclose(X.data, X_int.data):
                raise ValueError('Data must be integer if '
                                 'weighted_edges=False')
            X = X_int

        for row, col, count in zip(X.row, X.col, X.data):
            doc_vert = doc_vertices[row]
            kind[doc_vert] = 0
            word_vert = word_vertices[col]
            kind[word_vert] = 1

            if self.weighted_edges:
                e = g.add_edge(doc_vert, word_vert)
                ecount[e] = count
            else:
                for n in range(count):
                    g.add_edge(doc_vert, word_vert)
        return g

    def __fit_hsbm(self):
        clabel = self.graph_.vp['kind']

        state_args = {'clabel': clabel, 'pclabel': clabel}
        if "count" in self.graph_.ep:
            state_args["eweight"] = self.graph_.ep.count

        self.mdl_ = np.inf
        for _ in range(self.n_init):
            # the inference
            state = minimize_nested_blockmodel_dl(self.graph_, deg_corr=True,
                                                  # overlap=overlap,  # TODO:
                                                  # implement overlap
                                                  state_args=state_args,
                                                  B_min=self.min_groups,
                                                  B_max=self.max_groups)

            mdl = state.entropy()
            if mdl < self.mdl_:
                self.state_ = state.copy()
                self.mdl_ = mdl
                del state

        # collect group membership for each level in the hierarchy
        n_levels = len(self.state_.levels)

        if n_levels == 2:
            # only trivial bipartite structure
            self.groups_ = {0: self.__get_groups(level=0)}
        else:
            # omit trivial levels:
            # - level=n_levels-1 (single group),
            # - level=n_levels-2 (bipartite)
            self.groups_ = {level: self.__get_groups(level=level)
                            for level in range(n_levels - 2)}

        self.n_levels_ = len(self.groups_)

    def fit(self, X, y=None):
        """Fit the hSBM topic model

        Constructs a graph representation of X and infers clustering.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Word frequencies for each document, represented as  non-negative
            integers.
        y : ignored

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the hSBM topic model

        Constructs a graph representation of X, infers clustering, and reports
        the cluster probability for each sample in X.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            Word frequencies for each document, represented as  non-negative
            integers.
        y : ignored

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_components)
            The cluster probability for each sample in X
        """
        X = check_array(X, accept_sparse=True)
        np_random_state = np.random.get_state()  # we restore this later
        random_state = check_random_state(self.random_state)
        if self.random_state is not None:
            graph_tool.seed_rng(
                random_state.randint(0, np.iinfo(np.int32).max))
        np.random.set_state(random_state.get_state())

        try:
            self.graph_ = self.__make_graph(X)
            self.n_features_ = X.shape[1]
            self.n_samples_ = X.shape[0]

            self.__fit_hsbm()
        finally:
            np.random.set_state(np_random_state)

        level = 0
        Xt = self.groups_[level]['p_tw_d'].T

        self.num_components_ = Xt
        return Xt

    def __get_groups(self, level=0):
        '''extract group membership statistics from the inferred state.

        return dict
        '''
        level_state = self.state_.project_level(level).copy(overlap=True)
        level_state_edges = level_state.get_edge_blocks()

        # count labeled half-edges, group-memberships
        n_groups = level_state.B
        n_wb = np.zeros((self.n_features_, n_groups))
        n_db = np.zeros((self.n_samples_, n_groups))
        n_dbw = np.zeros((self.n_samples_, n_groups))

        for e in self.graph_.edges():
            z1, z2 = level_state_edges[e]
            v1 = int(e.source())
            v2 = int(e.target())
            n_db[v1, z1] += 1
            n_dbw[v1, z2] += 1
            n_wb[v2 - self.n_samples_, z2] += 1

        # p_w = np.sum(n_wb,axis=1) / float(np.sum(n_wb))

        n_db = n_db[:, np.any(n_db, axis=0)]
        Bd = n_db.shape[1]

        n_wb = n_wb[:, np.any(n_wb, axis=0)]
        Bw = n_wb.shape[1]

        n_dbw = n_dbw[:, np.any(n_dbw, axis=0)]

        # group-membership distributions
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # group membership of each doc-node P(t_d | d)
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        # topic-distribution for words P(w | t_w)
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

        # Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T

        result = {}
        result['Bd'] = Bd
        result['Bw'] = Bw
        result['p_tw_w'] = p_tw_w
        result['p_td_d'] = p_td_d
        result['p_w_tw'] = p_w_tw
        result['p_tw_d'] = p_tw_d

        return result

    def plot_graph(self, filename=None, n_edges=1000):
        """Plots arcs from documents to words coloured by inferred group

        Parameters
        ----------
        filename : str, optional
            Path to write to (e.g. 'something.png').
            Otherwise returns a displayable object.
        n_edges : int
            Size of subsample to plot (reducing memory requirements)
        """
        self.state_.draw(layout='bipartite', output=filename,
                         subsample_edges=n_edges, hshortcuts=1, hide=0)
