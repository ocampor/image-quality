unit_tests() {
  pytest
}

lower() {
  word=${1}
  echo "${word}" | tr '[:upper:]' '[:lower:]'
}

package_validation() {
  pip install --upgrade twine
  python setup.py sdist
  validation_output=$(python -m twine check dist/*)
  validation_output=$(lower "${validation_output}")

  if [ "${validation_output: -6}" != "passed" ]; then
    echo "The python package is invalid"
    echo "${validation_output}"
    exit 1
  fi
}

unit_tests && package_validation
