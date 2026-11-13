# vllm env on horizon: build steps, adapted from vista/envs/vllm.sh (sm_100 only, no gcc/cuda Vista modules). Sourced by setup/setup.sh after setup/envs/vllm.sh (which has env_check).

# CMake and nvcc want a classic CUDA_HOME (lib64/, unversioned lib*.so, a libcuda stub), which the
# pip cuda-toolkit does not provide: it has bin/ include/ nvvm/ and only versioned libs. Build one
# from symlinks, outside the env. The libcuda.so stub is link-time only, taken from a Vista module
# (gg nodes have no driver); at run time the real driver's libcuda.so.1 is used.
# Usage: vllm_make_cuda_home <pip toolkit dir, nvidia/cu13> <dest>
vllm_make_cuda_home() {
    local pip=$1 dest=$2 stub
    # Horizon: the `cuda` module (default 13.3) has the libcuda stub; its root is $CUDA_HOME while the module is loaded (the build script saves it as CUDA_MODULE_HOME before conda activate changes CUDA_HOME) (we are about to replace CUDA_HOME).
    stub=$(ls "${CUDA_MODULE_HOME:-${CUDA_HOME:-/nonexistent}}"/targets/sbsa-linux/lib/stubs/libcuda.so 2>/dev/null | head -1)
    [[ -f "$stub" ]] || { echo "libcuda stub not found (need the cuda module loaded: $stub)" >&2; return 1; }
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
    # aarch64: no prebuilt wheel covers both gh (sm_90) and gb (sm_100), so the Marin fork
    # (pinned in repos.txt) is built from source, mirroring its release recipe
    # (vllm/infra/release/config.json + gpu-constraints.txt, docker/Dockerfile).
    # Run via setup/machines/vista/slurm/build_vllm_env.sbatch: a full gg node, hours. The wheel is
    # kept in $VLLM_WHEEL_DIR (on $WORK, not purged) and reused, so a $SCRATCH purge only
    # costs an install. Delete the wheel to force a rebuild.
    # VLLM_SRC and VLLM_SHA build another commit (a fork branch) into its own env and wheel; the defaults are the pinned Marin release.
    local src=${VLLM_SRC:-$PROJECT_ROOT/vllm} sha=${VLLM_SHA:-39e62869693c}
    local cons=$src/infra/release/gpu-constraints.txt
    local cu=https://download.pytorch.org/whl/cu132
    : "${VLLM_WHEEL_DIR:=$PROJECT_ROOT/wheels}"
    [[ -f "$cons" ]] || { echo "vllm source missing at $src (run setup/setup.sh)" >&2; return 1; }
    git -C "$src" rev-parse HEAD | grep -q "^$sha" || { echo "$src is not at $sha" >&2; return 1; }
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install uv || return 1                      # needed by tools/setup_deepgemm_pythons.sh (Horizon has no uv)
    gcc --version | head -1                         # host compiler for nvcc: Horizon's system gcc 14.x (no gcc/14.2.0 module here)
    # torch, torchvision and the CUDA toolkit are taken from the checkout's own constraints file (Marin moves them between releases: the pinned release has torch 2.13.0 / toolkit 13.2.1,
    # main later 2.14.1 / 13.2.2); the literals are the pinned release's, used only if a line is missing.
    local torch_pin tv_pin ctk_ver
    torch_pin=$(sed -nE 's/^(torch==[^ ;#\\]*).*/\1/p' "$cons" | head -1);        torch_pin=${torch_pin:-torch==2.13.0+cu132}
    tv_pin=$(sed -nE 's/^(torchvision==[^ ;#\\]*).*/\1/p' "$cons" | head -1);     tv_pin=${tv_pin:-torchvision==0.28.0+cu132}
    ctk_ver=$(sed -nE 's/^cuda-toolkit==([^ ;#\\]*).*/\1/p' "$cons" | head -1);  ctk_ver=${ctk_ver:-13.2.1}
    echo "toolchain from $cons: $torch_pin $tv_pin cuda-toolkit==$ctk_ver"
    pip install -c "$cons" --extra-index-url $cu \
        "$torch_pin" "$tv_pin" \
        "cuda-toolkit[nvcc,cccl,cuobjdump]==$ctk_ver" || return 1   # nvcc = Marin's (13.2.1 for the pinned release)
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
export _VLLM_OLD_PATH=$PATH _VLLM_OLD_CPATH=${CPATH:-} CUDA_HOME=$CONDA_PREFIX/cuda-home
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH CUDA_PATH CUDA_ROOT   # Horizon's cuda module (13.3) sets these; its headers clash with the pip 13.2.x nvcc
export PATH=$CUDA_HOME/bin:$PATH
HOOK
    cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda_home.sh" <<'HOOK'
export PATH=$_VLLM_OLD_PATH; [[ -n $_VLLM_OLD_CPATH ]] && export CPATH=$_VLLM_OLD_CPATH; unset _VLLM_OLD_PATH _VLLM_OLD_CPATH CUDA_HOME
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
            unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH CUDA_PATH CUDA_ROOT   # module cuda/13.3 headers vs pip nvcc 13.2: 'compiler and toolkit headers are incompatible'
            export TORCH_CUDA_ARCH_LIST="10.0"           # sm_100 only (GB200); explicit, never auto-detect
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
}
