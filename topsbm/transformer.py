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

from collections import defaultdict

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
        Bipartite graph between samples (kind=0) and features (kind=1)
    num_features_ : int
    num_samples_ : int
    state_
         Inference state from graphtool
    groups_
        results of group membership from inference

        # TODO: document structure and semantics of this
    mdl_
        minimum description length of inferred state

    References
    ----------
    Martin Gerlach, Tiago P. Peixoto, and Eduardo G. Altmann,
    `“A network approach to topic models,”
    <http://advances.sciencemag.org/content/4/7/eaaq1360>`_.
    Science Advances (2018)
    """

    def __init__(self, weighted_edges=True, n_init=1, random_state=None):
        self.weighted_edges = weighted_edges
        self.n_init = n_init
        self.random_state = random_state

    def __make_graph(self, X):
        num_samples = X.shape[0]

        list_titles = ['Doc#%d' % h for h in range(num_samples)]

        # make a graph
        # create a graph
        g = Graph(directed=False)
        # define node properties
        # name: docs - title, words - 'word'
        # kind: docs - 0, words - 1
        idx = g.vp["idx"] = g.new_vp("int")
        kind = g.vp["kind"] = g.new_vp("int")
        if self.weighted_edges:
            ecount = g.ep["count"] = g.new_ep("int")

        docs_add = defaultdict(lambda: g.add_vertex())
        words_add = defaultdict(lambda: g.add_vertex())

        # add all documents first
        for i_d in range(num_samples):
            title = list_titles[i_d]
            docs_add[title]

        # add all documents and words as nodes
        # add all tokens as links
        X = scipy.sparse.coo_matrix(X)
        for row, col, count in zip(X.row, X.col, X.data):
            doc_vert = docs_add[row]
            idx[doc_vert] = row
            kind[doc_vert] = 0

            word_vert = words_add[col]

            idx[word_vert] = col
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
                                                  state_args=state_args)

            mdl = state.entropy()
            if mdl < self.mdl_:
                self.state_ = state.copy()
                self.mdl_ = mdl
                del state

        # collect group membership for each level in the hierarchy
        L = len(self.state_.levels)
        dict_groups_L = {}

        # only trivial bipartite structure
        if L == 2:
            self.L_ = 1
            for l in range(L - 1):
                dict_groups_l = self.__get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        # omit trivial levels: l=L-1 (single group), l=L-2 (bipartite)
        else:
            self.L_ = L - 2
            for l in range(L - 2):
                dict_groups_l = self.__get_groups(l=l)
                dict_groups_L[l] = dict_groups_l
        self.groups_ = dict_groups_L

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
            self.num_features_ = X.shape[1]
            self.num_samples_ = X.shape[0]

            self.__fit_hsbm()
        finally:
            np.random.set_state(np_random_state)

        l = 0
        Xt = self.groups_[l]['p_tw_d'].T

        '''
        np.zeros(self.num_samples_, self.groups_[l]['Bw'])
        dict_groups =  self.groups_[l]
        for doc_index in range(self.num_samples_):
            p_tw_d = dict_groups['p_tw_d']
            list_topics_tw = []
            for tw, p_tw in enumerate(p_tw_d[:,doc_index]):
                    list_topics_tw += [(tw, p_tw)]
        '''
        self.num_components_ = Xt
        return Xt

    def __get_groups(self, l=0):
        '''extract group membership statistics from the inferred state.

        return dictionary
        - B_d, int, number of doc-groups
        - B_w, int, number of word-groups
        - p_tw_w, array B_w x V; word-group-membership:
         prob that word-node w belongs to word-group tw: P(tw | w)
        - p_td_d, array B_d x D; doc-group membership:
         prob that doc-node d belongs to doc-group td: P(td | d)
        - p_w_tw, array V x B_w; topic distribution:
         prob of word w given topic tw P(w | tw)
        - p_tw_d, array B_w x d; doc-topic mixtures:
         prob of word-group tw in doc d P(tw | d)
        '''
        V = self.num_features_
        D = self.num_samples_

        g = self.graph_
        state = self.state_
        idx = g.vp["idx"]
        state_l = state.project_level(l).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()

        # count labeled half-edges, group-memberships
        B = state_l.B
        n_wb = np.zeros((V, B))
        n_db = np.zeros((D, B))
        n_dbw = np.zeros((D, B))

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            v1 = e.source()
            v2 = e.target()
            n_db[idx[v1], z1] += 1
            n_dbw[idx[v1], z2] += 1
            n_wb[idx[v2], z2] += 1

        # p_w = np.sum(n_wb,axis=1) / float(np.sum(n_wb))

        ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
        Bd = len(ind_d)
        n_db = n_db[:, ind_d]

        ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
        Bw = len(ind_w)
        n_wb = n_wb[:, ind_w]

        ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
        n_dbw = n_dbw[:, ind_w2]

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
        result['p_tw_d'] = p_tw_d  # XXX: this appears to be 1-p_td_d

        return result

    def plot_graph(self, filename, n_edges=1000):
        """Plots arcs from documents to words coloured according to inferred groups

        Parameters
        ----------
        n_edges : int
            ???

        Returns
        -------
        ax : matplotlib.axes.Axes
            An axes with the plot on it
        """
        self.state_.draw(layout='bipartite', output=filename,
                         subsample_edges=n_edges, hshortcuts=1, hide=0)
