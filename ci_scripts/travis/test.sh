set -e

# Get into a temp directory to run test from the installed scikit learn and
# check if we do not leave artifacts
mkdir -p $TEST_DIR

cd $TEST_DIR

if [[ "$COVERAGE" == "true" ]]; then
    nosetests -s --with-coverage --cover-package=$MODULE $MODULE
else
    nosetests -s $MODULE
fi

pip install flake8
flake8 topsbm examples
