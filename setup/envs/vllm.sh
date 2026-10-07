# Recipe for the `vllm` env: serving (generation, hidden-state server).
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME, env_build, env_check.
# Add a dependency: put the install step in env_build (every machine it applies to),
# make env_check verify it, then run `setup/setup.sh --freeze`.

ENV_NAME=vllm

# CMake and nvcc want a classic CUDA_HOME (lib64/, unversioned lib*.so, a libcuda stub), which the
# pip cuda-toolkit does not provide: it has bin/ include/ nvvm/ and only versioned libs. Build one
# from symlinks, outside the env. The libcuda.so stub is link-time only, taken from a Vista module
# (gg nodes have no driver); at run time the real driver's libcuda.so.1 is used.
# Usage: vllm_make_cuda_home <pip toolkit dir, nvidia/cu13> <dest>
vllm_make_cuda_home() {
    local pip=$1 dest=$2 stub
    stub=$( module load gcc/14.2.0 cuda/13.1 >/dev/null 2>&1
            r=$(tr ':' '\n' <<<"$PATH" | grep -m1 'cuda/13\.1/bin$') && echo "${r%/bin}"/targets/sbsa-linux/lib/stubs/libcuda.so )
    [[ -f "$stub" ]] || { echo "libcuda stub not found (module cuda/13.1: $stub)" >&2; return 1; }
    rm -rf "$dest"; mkdir -p "$dest/lib64/stubs" || return 1
    local d f
    for d in bin include nvvm; do ln -s "$pip/$d" "$dest/$d"; done
    for f in "$pip"/lib/*; do ln -sf "$f" "$dest/lib64/$(basename "$f")"; done
    for f in "$dest"/lib64/lib*.so.*; do   # libX.so.N -> libX.so (skip if a plain libX.so exists)
        local base=${f%%.so.*}.so
        [[ -e "$base" ]] || ln -s "$(basename "$f")" "$base"
    done
    ln -s "$dest/lib64" "$dest/lib"
    ln -s "$stub" "$dest/lib64/stubs/libcuda.so"
}

env_build() {
    case "$MACHINE" in
    genai)
        # Reconstructed from the working env; setup/locks/genai/vllm.txt is authoritative.
        # The Marin vLLM fork (commit 39e62869693c) is a local wheel, not on PyPI.
        : "${VLLM_WHEEL:=$PROJECT_ROOT/tmp/vllm-0.0.0.dev20260929+marin.39e62869693c.cu132-cp38-abi3-manylinux_2_28_x86_64.whl}"
        [[ -f "$VLLM_WHEEL" ]] || { echo "Marin vLLM wheel not found: $VLLM_WHEEL (set VLLM_WHEEL)" >&2; return 1; }
        conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
        conda activate "$ENV_NAME" || return 1
        pip install "$VLLM_WHEEL" --extra-index-url https://download.pytorch.org/whl/cu132 || return 1
        pip uninstall -y torchaudio                      # no cu132 build; vLLM does not need it
        pip install "mooncake-transfer-engine-cuda13==0.3.13.post1" || return 1
        pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
        ;;
    vista)
        # aarch64: no prebuilt wheel covers both gh (sm_90) and gb (sm_100), so the Marin fork
        # (pinned in repos.txt) is built from source, mirroring its release recipe
        # (vllm/infra/release/config.json + gpu-constraints.txt, docker/Dockerfile).
        # Run via scripts/vista/build_vllm_env.sbatch: a full gg node, hours. The wheel is
        # kept in $VLLM_WHEEL_DIR (on $WORK, not purged) and reused, so a $SCRATCH purge only
        # costs an install. Delete the wheel to force a rebuild.
        local src=$PROJECT_ROOT/vllm sha=39e62869693c
        local cons=$src/infra/release/gpu-constraints.txt
        local cu=https://download.pytorch.org/whl/cu132
        : "${VLLM_WHEEL_DIR:=$WORK/wheels}"
        [[ -f "$cons" ]] || { echo "vllm source missing at $src (run setup/setup.sh)" >&2; return 1; }
        git -C "$src" rev-parse HEAD | grep -q "^$sha" || { echo "$src is not at $sha" >&2; return 1; }
        conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
        conda activate "$ENV_NAME" || return 1
        module load gcc/14.2.0 || return 1              # host compiler for nvcc
        pip install -c "$cons" --extra-index-url $cu \
            "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" \
            "cuda-toolkit[nvcc,cccl,cuobjdump]==13.2.1" || return 1   # nvcc 13.2.1 = Marin's
        # The toolkit comes from pip, not a Vista module (its 13.1/13.3 differ from Marin's 13.2.1).
        local cuda_home
        cuda_home=$(dirname "$(dirname "$(find "$CONDA_PREFIX/lib" -path '*/nvidia/*/bin/nvcc' | head -1)")")
        [[ -x "$cuda_home/bin/nvcc" ]] || { echo "nvcc from cuda-toolkit not found under $CONDA_PREFIX" >&2; return 1; }
        vllm_make_cuda_home "$cuda_home" "$CONDA_PREFIX/cuda-home" || return 1
        cuda_home=$CONDA_PREFIX/cuda-home
        # Run time too: FlashInfer JITs kernels with the `nvcc` on PATH, which would otherwise be the
        # nvidia module's 12.5 (fails: nvcc fatal: Unknown option '--compress-mode=size').
        mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
        cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_home.sh" <<'HOOK'
export _VLLM_OLD_PATH=$PATH CUDA_HOME=$CONDA_PREFIX/cuda-home
export PATH=$CUDA_HOME/bin:$PATH
HOOK
        cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda_home.sh" <<'HOOK'
export PATH=$_VLLM_OLD_PATH; unset _VLLM_OLD_PATH CUDA_HOME
HOOK
        mkdir -p "$VLLM_WHEEL_DIR"
        local wheel
        wheel=$(ls -t "$VLLM_WHEEL_DIR"/vllm-*marin."$sha"*.whl 2>/dev/null | head -1)
        if [[ -z "$wheel" ]]; then
            pip install -c "$cons" --extra-index-url $cu -r "$src/requirements/build/cuda.txt" || return 1
            command -v uv >/dev/null || { echo "uv needed for DeepGEMM interpreters" >&2; return 1; }
            (
                cd "$src" || exit 1
                export CUDA_HOME=$cuda_home PATH=$cuda_home/bin:$PATH
                export TORCH_CUDA_ARCH_LIST="9.0;10.0"        # explicit: never auto-detect (no GPU here)
                export VLLM_TARGET_DEVICE=cuda CMAKE_BUILD_TYPE=Release
                export MAX_JOBS=${MAX_JOBS:-24} NVCC_THREADS=${NVCC_THREADS:-2}
                export SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev$(git show -s --format=%cd --date=format:%Y%m%d HEAD)+marin.$sha"
                export DEEPGEMM_VENV_PREFIX=$TMPDIR/dgenv
                export DEEPGEMM_PYTHON_INTERPRETERS=$(tools/setup_deepgemm_pythons.sh) || exit 1
                rm -rf .deps && mkdir -p .deps                # stale CMake fetch state (as in Marin's Dockerfile)
                python setup.py bdist_wheel --dist-dir="$VLLM_WHEEL_DIR" --py-limited-api=cp38
            ) || return 1
            wheel=$(ls -t "$VLLM_WHEEL_DIR"/vllm-*marin."$sha"*.whl 2>/dev/null | head -1)
            [[ -n "$wheel" ]] || { echo "build produced no wheel in $VLLM_WHEEL_DIR" >&2; return 1; }
        fi
        echo "installing $wheel"
        pip install -c "$cons" --extra-index-url $cu --extra-index-url https://flashinfer.ai/whl/ "$wheel" || return 1
        pip install "mooncake-transfer-engine-cuda13==0.3.13.post1" || return 1
        pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
        ;;
    *) echo "no vllm recipe for MACHINE=$MACHINE" >&2; return 1 ;;
    esac
}

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import os, sys, shutil
import torch, vllm, hs_connectors
print(f"vllm {vllm.__version__} | torch {torch.__version__} (cuda {torch.version.cuda})")
problems = []
if "+marin" not in vllm.__version__:
    problems.append("vllm is not the Marin fork build")
if shutil.which("mooncake_master") is None:
    problems.append("mooncake_master not on PATH (needed by the mooncake hidden-states backend)")
import torch
need = os.environ.get("REQUIRED_CUDA_ARCHS", "").split()
have = torch.cuda.get_arch_list()   # compiled-in archs; works without a GPU
missing = [a for a in need if a not in have and a.replace("sm_", "compute_") not in have]
if missing:
    problems.append(f"torch lacks CUDA archs {missing} (has {have})")
# vLLM's compiled kernels (Vista builds from source for several archs): list the SASS targets.
if os.environ.get("MACHINE") == "vista":
    import glob, re, subprocess
    tool = shutil.which("cuobjdump") or next(iter(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/bin/cuobjdump"))), None)
    sos = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_C*.so"))
    if not tool or not sos:
        problems.append("cannot inspect vLLM kernels (cuobjdump or vllm/_C*.so missing)")
    else:
        out = subprocess.run([tool, "--list-elf", sos[0]], capture_output=True, text=True).stdout
        built = sorted(set("sm_" + m for m in re.findall(r"sm_(\d+)", out)))
        print("vllm kernel SASS:", built)
        problems += [f"vllm kernels lack {a}" for a in need if a not in built]
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
}
