#!/bin/bash
# Build the pinned Evalchemy client environment (CPU only; the model is served by vLLM in another env). Run through sbatch from a login node:
#   ROOT=<workspace> sbatch -p debug -t 01:00:00 build_evalchemy_env.sbatch
# Mirrors the Marin eval policy's runtime (marin.evaluation.evalchemy.runtime): Evalchemy at the pinned commit, Python 3.12, the serve-eval extra plus the extras of
# the tasks we run, s3fs/gcsfs, and CPU-only torch. The environment goes to $ROOT/envs/evalchemy_pin and is never overwritten.
#SBATCH -N 1
set -uo pipefail
: "${ROOT:?workspace root}"
ENVDIR=$ROOT/envs/evalchemy_pin
[[ ! -e "$ENVDIR" ]] || { echo "$ENVDIR exists: not overwriting"; exit 1; }
EVALCHEMY_SHA=c1d5c62cdc3939b42f64411af44fd50716b53454   # marin-community/evalchemy commit pinned by Marin 75a5690b0f (lib/marin/src/marin/external_dependencies.py)
TORCH_WHL=https://download.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl   # the policy's aarch64 CPU torch wheel
command -v uv >/dev/null || python3 -m pip install --user -q uv || { echo "no uv"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
EXTRAS=serve-eval,humanevalplus,mbppplus,ifbench,cruxeval,math500,aime24,gpqadiamond
uv venv --python 3.12 "$ENVDIR" && uv pip install --python "$ENVDIR/bin/python" \
    "evalchemy[$EXTRAS] @ git+https://github.com/marin-community/evalchemy.git@$EVALCHEMY_SHA" s3fs gcsfs "$TORCH_WHL" || { echo "install failed"; exit 1; }
"$ENVDIR/bin/python" -I -c "import eval, finestore, lm_eval; print('evalchemy env ok')"
