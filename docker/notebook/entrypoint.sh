#!/bin/bash bash
NOTEBOOKS_PATH="${HOME}/notebooks"

generate_password() {
  python -c "from notebook.auth import passwd; print(passwd('${JUPYTER_PASS}'))"
}

create_working_directory() {
  if [ ! -d "${NOTEBOOKS_PATH}" ]; then
    echo "Creating directory ${NOTEBOOKS_PATH}"
    mkdir -p "${NOTEBOOKS_PATH}"
  fi
}

run_jupyter_notebook() {
  if [ -z "${JUPYTER_PASS}" ]; then
    jupyter notebook \
      --no-browser \
      --ip 0.0.0.0 \
      --port 8000 \
      --notebook-dir "${NOTEBOOKS_PATH}"
  else
    echo "WARNING: Running jupyter notebook with password in env JUPYTER_PASS"
    jupyter notebook \
      --no-browser \
      --ip 0.0.0.0 \
      --port 8000 \
      --notebook-dir "${NOTEBOOKS_PATH}" \
      --NotebookApp.password="$(generate_password)"
  fi
}

create_working_directory && run_jupyter_notebook
