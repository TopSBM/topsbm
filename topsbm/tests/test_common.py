import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from topsbm import TopSBM


def test_transformer():
    return check_estimator(TopSBM)


def test_random_state():
    X = np.random.randint(0, 2, size=(10, 5))
    Xt0a = TopSBM(random_state=0).fit_transform(X)
    Xt0b = TopSBM(random_state=0).fit_transform(X)
    Xt1 = TopSBM(random_state=1).fit_transform(X)
    assert np.allclose(Xt0a, Xt0b)
    # XXX: confirm whether we should expect different results
    # assert not np.allclose(Xt0a, Xt1)
