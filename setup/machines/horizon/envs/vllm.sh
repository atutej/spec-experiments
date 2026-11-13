# vllm env on horizon: same aarch64 GB200 stack as Vista, so the Vista build steps are reused. Module names (gcc/14.2.0, cuda/13.1 for the libcuda stub) are Vista's
# and untested here: check `module spider gcc cuda` and edit setup/machines/vista/envs/vllm.sh or copy it here if they differ.
source "$(dirname "${BASH_SOURCE[0]}")/../../vista/envs/vllm.sh"
