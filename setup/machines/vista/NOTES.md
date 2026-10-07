# TACC Vista

Facts checked on 2026-10-07. The porting history is in `docs/vista_setup.md`.

- **CPU:** every node type is aarch64 (Grace). No x86 wheel or env carries over.
- **Node types / partitions:** `gg` (CPU only, 144 cores, ~237 GB), `gh` (1x H100, sm_90), `gb` (4x
  GB200, sm_100; driver 590.48.01 on the node I tested), plus `gh-dev`. Nodes are exclusive.
- **Account:** `CCR24067`. Write it in upper case in `#SBATCH -A`, or Slurm errors out. QOS limits a
  job to 2 days.
- **Submitting:** `sbatch` is refused on compute nodes, including idev sessions. Submit from a login
  node. Job output goes to `$PROJECT_ROOT/logs/slurm/` (set by a literal path in the `#SBATCH -o` line). idev sessions on `gg` and `gb` last up to 2 h.
- **Filesystems:** `$HOME` (small), `$WORK` (not purged; holds conda and the vLLM wheel in `wheels/`),
  `$SCRATCH` (large, purged when unused, ~89% full). The workspace is `$SCRATCH/marin_speculator`.
- **Internet from compute nodes:** works in idev sessions (Hugging Face, GitHub, PyPI, PyTorch index)
  and in batch jobs. One batch job (`i617-051`) failed a DNS lookup once; the build script now
  checks DNS first and aborts clearly.
- **Modules:** the default `nvidia/24.7` module sets `CC=nvc`, `CXX=nvc++` and puts nvcc 12.5 on
  `PATH`. Both break torch inductor and FlashInfer. `env.sh` sets `CC=gcc CXX=g++`, and the `vllm`
  env brings its own nvcc 13.2.1 (pip) on `conda activate vllm`. `module spider cuda` also lists 12.8
  to 13.3, but nothing in the workspace depends on a CUDA module.
- **Not yet tested here:** a `gh` node (see the "Vista status" section of `docs/vista_setup.md`).
