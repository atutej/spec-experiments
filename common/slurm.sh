# Sourced by the submit_*.sh scripts (experiments' vista/ folders). Slurm helpers shared by all of them.
#
# RESERVATION: the Slurm reservation used for gb jobs. Default CGAI-gb (2 gb nodes, active 2026-10-13 to 2026-10-27, user atutej). RESERVATION= (empty)
# submits without one; RESERVATION=<other name> uses another. It is added on the sbatch command line, for gb jobs ONLY: the shared run.sbatch files
# also run the gg stages (export, prepare, ...) and a gb reservation cannot hold a gg job, so it cannot be an #SBATCH header line there.
# A job in a reservation can only run on the reservation's nodes (here two, and the jobs are --exclusive: two at a time at most).
RESERVATION=${RESERVATION-CGAI-gb}

# CPU_PARTITION: where the CPU-only stages (export, prepare, convert, upload) run. Default gb, so that every job of a chain is in the reservation (these
# stages simply leave the GPUs idle). CPU_PARTITION=gg puts them back on the CPU partition (2-day limit, no reservation). On gb the QOS allows only 3
# submitted jobs per user and 12 h per job, so the chains put short CPU stages in the same job as a GPU stage (comma-joined: `prepare,train`).
CPU_PARTITION=${CPU_PARTITION:-gb}

# Prints the sbatch option for a gb job, or nothing: when no reservation is wanted, or when it does not exist (expired, wrong name; a warning
# says so and the job is submitted without it instead of being rejected).
gb_reservation_args() {
    [[ -n "$RESERVATION" ]] || return 0
    if scontrol show reservation "$RESERVATION" >/dev/null 2>&1; then
        echo "--reservation=$RESERVATION"
    else
        echo "WARNING: Slurm reservation '$RESERVATION' not found (expired?); submitting gb jobs without it" >&2
    fi
}
