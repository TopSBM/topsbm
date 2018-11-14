"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
import numpy as np
from topsbm import TopSBM
from matplotlib import pyplot as plt

if (0):
    X = np.arange(100).reshape(100, 1)
    y = np.zeros((100, ))
    estimator = TopSBM()
    estimator.fit(X, y)
    plt.plot(estimator.predict(X))
    plt.show()
