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


def test_transformer():
    return check_estimator(TopSBM)


def test_random_state(n_samples=20, n_features=100):
    feat = np.random.RandomState(0).choice(X_20n.shape[1], n_features)
    X = X_20n[:n_samples, feat]
    Xt0a = TopSBM(random_state=0).fit_transform(X)
    Xt0b = TopSBM(random_state=0).fit_transform(X)
    Xt2 = TopSBM(random_state=2).fit_transform(X)
    np.testing.assert_allclose(Xt0a, Xt0b)
    assert Xt0a.shape != Xt2.shape or not np.allclose(Xt0a, Xt2)
