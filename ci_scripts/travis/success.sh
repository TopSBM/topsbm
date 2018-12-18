set -e

if [[ "$COVERAGE" == "true" ]]; then
    coveralls
fi
