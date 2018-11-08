from sklearn.utils.estimator_checks import check_estimator
from topsbm import TopSBM


def test_transformer():
    return check_estimator(TopSBM)
