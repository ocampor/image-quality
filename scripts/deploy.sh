#!/bin/bash
if [ "$TRAVIS_BRANCH" == "master" ]; then
  pip install --user --upgrade twine
  python setup.py sdist
  python -m twine upload -u ${PYPI_USER} -p ${PYPI_PASSWORD} dist/*
fi
