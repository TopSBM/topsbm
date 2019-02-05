# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

set -ex

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget http://repo.continuum.io/miniconda/Miniconda-3.6.0-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b
export PATH=/home/travis/miniconda/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
cat environment.yml
conda env create
source activate topsbm
conda install --yes python=$PYTHON_VERSION pip pytest \
      numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
      scikit-learn=$SKLEARN_VERSION

if [[ "$COVERAGE" == "true" ]]; then
    pip install pytest-cov coverage coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python setup.py develop
