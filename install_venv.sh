#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

pip install -r requirements_from_pip.txt

if [[ -d "external/f1tenth_gym" ]]; then
  pip install --no-deps -e external/f1tenth_gym
else
  pip install --no-deps -e git+https://github.com/f1tenth/f1tenth_gym.git
fi

echo "\nDone. Activate the venv with: source .venv/bin/activate"
echo "Then run your project, for example: python llm_mpc.py --help"