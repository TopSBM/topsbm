set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cd $TEST_DIR

pytest --showlocals --pyargs topsbm

pip install flake8
flake8 topsbm examples
