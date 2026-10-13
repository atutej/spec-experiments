# TACC Vista

Facts checked on 2026-10-07. The porting history is in `docs/vista_setup.md`.

- **CPU:** every node type is aarch64 (Grace). No x86 wheel or env carries over.
- **Node types / partitions:** `gg` (CPU only, 144 cores, ~237 GB), `gh` (1x H100, sm_90), `gb` (4x
  GB200, sm_100; driver 590.48.01 on the node I tested), plus `gh-dev`. Nodes are exclusive.
- **Account:** `CCR24067`. Write it in upper case in `#SBATCH -A`, or Slurm errors out. QOS limits (`sacctmgr show qos`): `qgb` 12:00:00 per job and only **3 submitted jobs per user**;
  `qgg`, `qgh` 2-00:00:00 and 40 submits. Rule of thumb: sbatch `-t` = the QOS maximum.
- **Reservation `CGAI-gb`** (2 gb nodes, `c672-[006-007]`, user atutej, active 2026-10-13 to 2026-10-27; check with `scontrol show reservation`): gb jobs are submitted with
  `--reservation=CGAI-gb`. It is added by the submit scripts (`common/slurm.sh`, `gb_reservation_args`) on the command line for gb jobs ONLY: the shared `run.sbatch` files also run the gg
  stages, and a gb reservation cannot hold a gg job, so it is not an `#SBATCH` header. If the reservation does not exist (expired) the scripts warn and submit without it; `RESERVATION=` (empty)
  turns it off, `RESERVATION=<name>` uses another. A job in a reservation runs only on its nodes (two, and jobs are exclusive), and its `-t` must end before the reservation does.
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
- **Temp space:** `TMPDIR=/tmp` (node-local, set via `SPEC_TMPDIR` in `env.sh`). It doesn't survive the job and
  isn't shared between nodes. `/tmp` is 1.1 TB on a gb node; gg and gh sizes are unchecked.
- **Not yet tested here:** a `gh` node (see the "Vista status" section of `docs/vista_setup.md`).
