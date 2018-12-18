from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy  # noqa
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy  # noqa
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


def setup_package():
    if sys.version_info[0] < 3:
        import __builtin__ as builtins
    else:
        import builtins
    builtins.__IN_SETUP__ = True
    import topsbm  # does not import estimator due to builtins hack
    setup(name='topsbm',
          version=topsbm.__version__,
          description='A scikit-learn extension for Topic Modelling with '
                      'Hierarchical Stochastic Block Models',
          author=', '.join(['Martin Gerlach',
                            'Eduardo Altmann',
                            'Vijay Raghunath',
                            'Joel Nothman']),
          packages=find_packages(),
          install_requires=INSTALL_REQUIRES,
          author_email='martin.gerlach@northwestern.edu',
          )
    delattr(builtins, '__IN_SETUP__')


if __name__ == '__main__':
    setup_package()
