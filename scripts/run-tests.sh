unit_tests() {
  pytest
}

package_validation() {
  pip install --upgrade twine
  python setup.py sdist
  validation_output=$(python -m twine check dist/*)

  if [ "${validation_output: -6}" != "Passed" ]; then
    echo "The python package is invalid"
    echo "${validation_output}"
    exit 1
  fi
}

unit_tests && package_validation
