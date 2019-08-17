#!/bin/bash
if [ "$TRAVIS_BRANCH" == "master" ]; then
  pip install --upgrade twine
  python setup.py sdist bdist_wheel
  python -m twine upload -u ${PYPI_USER} -p ${PYPI_PASSWORD} dist/*
fi
