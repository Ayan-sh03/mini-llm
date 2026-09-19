#!/usr/bin/env bash
# Run INSIDE a rented Linux machine; never creates/terminates a cloud instance.
set -euo pipefail
mini_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$mini_project_root"
mini_python="${MINI_PYTHON:-python3}"
"$mini_python" -c 'import sys; assert sys.version_info[:2] in [(3,11),(3,12)], "Use Python 3.11 or 3.12 (set MINI_PYTHON if needed)"'
command -v nvidia-smi >/dev/null || { echo 'No NVIDIA driver visible. Check the rental image.'; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [[ ! -d .venv-hyperbolic ]]; then
    "$mini_python" -m venv .venv-hyperbolic
fi
mini_venv_python="$mini_project_root/.venv-hyperbolic/bin/python"
"$mini_venv_python" -m pip install 'torch==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
"$mini_venv_python" -m pip install -r requirements.txt
"$mini_venv_python" -m pip check
unset MINI_MODAL_VOLUME
"$mini_venv_python" -m mini.preflight
echo 'Setup complete. Run: source .venv-hyperbolic/bin/activate'
echo 'IMPORTANT: exiting training does not terminate your rented instance.'
