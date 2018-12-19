set -ex

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cd $TEST_DIR

cp "$TRAVIS_BUILD_DIR"/setup.cfg $TEST_DIR
pytest --rootdir="$TRAVIS_BUILD_DIR" --showlocals --pyargs topsbm

cd -

pip install flake8
flake8 topsbm examples

