#!/usr/bin/env bash
# Install Python packages needed for llm_mpc_changed.py into an EXISTING virtual environment.
#
# Usage (pick one):
#   1) Activate your venv, then run from project root:
#        source /path/to/your/venv/bin/activate
#        ./install_llm_mpc_requirements.sh
#
#   2) Pass the venv directory explicitly:
#        ./install_llm_mpc_requirements.sh /path/to/your/venv
#
#   3) If you use the repo's .venv and it is not activated, this script will
#      auto-activate ./.venv when present.
#
# Requires: git, network (pip + optional Hugging Face model download on first run).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

activate_or_use_venv() {
  local venv_path="${1:-}"

  if [[ -n "$venv_path" ]]; then
    if [[ ! -f "$venv_path/bin/activate" ]]; then
      echo "error: not a virtualenv (missing $venv_path/bin/activate)" >&2
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$venv_path/bin/activate"
    echo "[info] Activated venv: $venv_path"
    return
  fi

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[info] Using already-active venv: $VIRTUAL_ENV"
    return
  fi

  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$ROOT/.venv/bin/activate"
    echo "[info] Activated repo venv: $ROOT/.venv"
    return
  fi

  echo "error: no virtual environment active and no ./.venv found." >&2
  echo "  Activate your venv first, or run:" >&2
  echo "    $0 /path/to/your/venv" >&2
  exit 1
}

activate_or_use_venv "${1:-}"

python -m pip install --upgrade pip setuptools wheel

pip install \
  "python-dotenv>=1.0.0" \
  "pandas" \
  "scipy" \
  "gymnasium" \
  "gym==0.26.2" \
  "Pillow" \
  "numba>=0.55.2" \
  "pyglet>=1.4.10,<1.5" \
  "PyOpenGL" \
  "langchain==0.3.22" \
  "langchain-experimental==0.3.4" \
  "langchain-community" \
  "langchain-text-splitters" \
  "langchain-huggingface" \
  "langchain-core" \
  "sentence-transformers" \
  "chromadb==0.6.3" \
  "langchain-openai==0.3.11"

pip install "numpy>=2.0"

F1TENTH_DIR="$ROOT/external/f1tenth_gym"
if [[ ! -d "$F1TENTH_DIR/.git" ]]; then
  mkdir -p "$ROOT/external"
  git clone --depth 1 https://github.com/f1tenth/f1tenth_gym.git "$F1TENTH_DIR"
fi

pip uninstall -y f110_gym 2>/dev/null || true
pip install --no-deps -e "$F1TENTH_DIR"

echo ""
echo "Done. Python in use: $(command -v python)"
echo "Try: python3 llm_mpc_changed.py --help"
