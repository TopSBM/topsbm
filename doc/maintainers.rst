Maintaining the Package
=======================

This document contains information for the software developers and maintainers.
Issues can be posted at `https://github.com/TopSBM/topsbm/issues`_.

Travis CI
---------

When a commit is made to any branch of the repository, or a pull request is
made, Travis CI pulls in the changes and runs the tests. It will give a green
tick if the tests run successfully.

Anyone listed in GitHub as a repository owner can administrate Travis too.

Building the Documentation
--------------------------

You can build the documentation on your own machine by installing sphinx and
nbsphinx. Then, in the ``doc/`` directory, run ``make html``.

Recompiling the documentation will re-run examples in Jupyter notebooks *only
if* all cells' output has been cleared.  Otherwise, the documentation will show
the output already in the notebook.

Note that the ReadTheDocs service currently refuses to re-run the example
notebook, as it takes longer than that service allows.

ReadTheDocs
-----------

ReadTheDocs recompiles the documentation when any commit is made to the
``master`` branch, and publishes it to `https://topsbm.readthedocs.io`_.

?Anyone listed in GitHub as a repository owner can administrate ReadTheDocs too.

Releasing to PyPI
-----------------

When you are ready to release a new version of the software, you should first
make sure that you are authorised to maintain the `PyPI package
<https://pypi.org/project/topsbm/>`__ (it lists maintainers on that page).

Then follow these steps:

1. Make sure the version is correct in the ``__version__`` variable in
   [``topsbm/__init__.py``](https://github.com/TopSBM/topsbm/blob/master/topsbm/__init__.py).
   For releases, remove suffixes like ``dev0``. Commit that change.
2. Tag the commit with the version number, with a command such as ``git tag v0.2``.
3. Push the tags to github. ``git push --tags``
4. Make sure setuptools and twine are installed. ``pip install setuptools twine``
4. Remove any files from previous releases in the ``dist`` directory: ``rm dist/*.tar.gz``
5. Run ``python setup.py sdist`` to create new entries in ``dist/``
6. Ensure your PyPI credentials are stored in ``~/.pypirc``.
7. Run ``twine upload dist/*.tar.gz``.
8. If you want, create a corresponding `GitHub release <https://github.com/TopSBM/topsbm/releases>`__
