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
import pytest
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import fetch_20newsgroups, make_multilabel_classification
from sklearn.feature_extraction.text import CountVectorizer
from topsbm import TopSBM


X_mmlc, _ = make_multilabel_classification(n_samples=20, n_features=100)
bunch_20n = fetch_20newsgroups(categories=['alt.atheism',
                                           'soc.religion.christian'],
                               shuffle=True, random_state=0,
                               remove=('headers', 'footers'))
X_20n = CountVectorizer().fit_transform(bunch_20n.data)


def test_common():
    return check_estimator(TopSBM)


def check_graph_structure(X, model):
    n_vertices = model.n_samples_ + model.n_features_
    for e in model.graph_.edges():
        assert model.n_samples_ > e.source() >= 0
        assert n_vertices > e.target() >= model.n_samples_

    if model.weighted_edges:
        expected_feature_degree = (X > 0).sum(axis=0)
        expected_sample_degree = (X > 0).sum(axis=1)
    else:
        expected_feature_degree = X.sum(axis=0)
        expected_sample_degree = X.sum(axis=1)
    # in case of sparse:
    expected_sample_degree = np.asarray(expected_sample_degree).ravel()
    expected_feature_degree = np.asarray(expected_feature_degree).ravel()

    for i, v in enumerate(model.graph_.vertices()):
        # Check vertices are aligned with features
        if i < model.n_samples_:
            assert v.out_degree() == expected_sample_degree[i]
        else:
            assert (v.out_degree() ==
                    expected_feature_degree[i - model.n_samples_])


@pytest.mark.parametrize('weighted_edges', [True, False])
def test_trivial(weighted_edges):
    X = np.zeros((20, 10))
    # two populations of samples with non-overlapping feature spaces
    X[:10, :8] = 1
    X[:10, 0] = 2
    X[10:, 8:] = 1
    model = TopSBM(random_state=0, weighted_edges=weighted_edges)
    Xt = model.fit_transform(X)

    check_graph_structure(X, model)
    assert model.state_ is not None
    assert model.mdl_ > 0
    assert model.n_features_ == X.shape[1]
    assert model.n_samples_ == X.shape[0]

    # rows sum to 1
    assert np.allclose(Xt.sum(axis=1), 1)
    # rows consist of 0 and 1
    assert np.allclose(np.ptp(Xt, axis=1), 1)
    # There should be no topical overlap between the two populations
    tuples = [tuple(row) for row in Xt]
    assert not set(tuples[:10]) & set(tuples[10:])

    # more specifically:
    assert Xt.shape == (20, 2)

    # TODO: also test other outputs in groups_

    # TODO: explore the effect of increasing topic overlap


@pytest.mark.xfail
def test_n_init(n_samples=20, n_features=1000):
    feat = np.random.RandomState(0).choice(X_20n.shape[1], n_features)
    X = X_20n[:n_samples, feat]
    model1 = TopSBM(random_state=0).fit(X)
    model10 = TopSBM(random_state=0, n_init=10).fit(X)
    assert model10.mdl_ < model1.mdl_
    assert np.isclose(model1.state_.entropy(), model1.mdl_,
                      atol=0, rtol=1e-8)
    pytest.skip('Failure due to '
                'https://git.skewed.de/count0/graph-tool/issues/546')
    assert np.isclose(model10.state_.entropy(), model10.mdl_,
                      atol=0, rtol=1e-8)


def test_random_state(n_samples=20, n_features=100):
    feat = np.random.RandomState(0).choice(X_20n.shape[1], n_features)
    X = X_20n[:n_samples, feat]
    Xt0a = TopSBM(random_state=0).fit_transform(X)
    Xt0b = TopSBM(random_state=0).fit_transform(X)
    Xt2 = TopSBM(random_state=2).fit_transform(X)
    np.testing.assert_allclose(Xt0a, Xt0b)
    assert Xt0a.shape != Xt2.shape or not np.allclose(Xt0a, Xt2)


def test_min_max_groups(n_samples=300, n_features=1000):
    feat = np.random.RandomState(0).choice(X_20n.shape[1], n_features)
    X = X_20n[:n_samples, feat]
    model1 = TopSBM(random_state=0).fit(X)
    check_graph_structure(X, model1)
    model2 = TopSBM(random_state=0, min_groups=10).fit(X)
    model3 = TopSBM(random_state=0, max_groups=2).fit(X)
    # TODO: more explicitly test the effect on the number of groups
    assert not np.isclose(model1.mdl_, model2.mdl_)
    assert not np.isclose(model1.mdl_, model3.mdl_)
