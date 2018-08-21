from sklearn.utils.estimator_checks import check_estimator
from hSBM import (hSBMTransformer)


def test_transformer():
    return check_estimator(hSBMTransformer)
