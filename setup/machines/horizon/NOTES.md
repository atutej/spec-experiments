# TACC Horizon (draft)

From docs.tacc.utexas.edu/hpc/horizon on 2026-11-13, a draft page; access was limited to internal users. Nothing here was run on Horizon.

- **Nodes:** GB nodes have 2 Grace CPUs (144 cores) and 4 GB200 GPUs (185 GiB HBM3 each); VV nodes have 176 Vera cores and no GPU. The page names Grace and Vera CPUs and does not state the instruction set.
- **Queues:** `gpu-dev` (GB, 8 nodes, 2 h), `gpu` (GB, 64 nodes, 48 h), `gpu-large` (GB, 256 nodes, 48 h), `gpu-shared` (up to 3 GPUs, 48 h); `cpu-dev`, `cpu`, `cpu-large` (VV). `gpu` costs 2 SU per node-hour, 15 minute minimum.
- **Submitting:** `sbatch -p <queue> -N <nodes>`; `-A <project>` only with several projects. `--gres` and `--gpus-per-task` are rejected, and the page says to avoid `--export`.
- **Filesystems:** `$SCRATCH` has no quota and is purged when untouched for 10 days; `$WORK` is not available to early users; `/corral` is on login nodes only.
- **Login nodes:** run AI-assisted work on compute nodes; do not run `ssh-keygen`.

Not verified: the hostname regex in `env.sh`; where conda lives; the module names the Vista vLLM recipe loads; whether compute nodes have outbound internet (the page is silent; `tools/agentic_eval` checks it and stops with a message); whether `idev` exists.
