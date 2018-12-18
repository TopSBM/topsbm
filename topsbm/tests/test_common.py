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


def test_n_init(n_samples=20, n_features=1000):
    feat = np.random.RandomState(0).choice(X_20n.shape[1], n_features)
    X = X_20n[:n_samples, feat]
    model1 = TopSBM(random_state=0).fit(X)
    model10 = TopSBM(random_state=0, n_init=10).fit(X)
    assert model10.mdl_ < model1.mdl_
    assert model1.state_.entropy() == model1.mdl_
    assert model10.state_.entropy() == model10.mdl_
