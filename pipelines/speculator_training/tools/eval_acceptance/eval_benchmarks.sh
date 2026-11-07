# Benchmarks for eval_matrix: name -> required files and the runner's settings. Sourced by eval_matrix.sh (preflight) and eval_matrix.sbatch (runs).
# Settings are those of the earlier figure evals: SPEED-Bench qualitative (AIPerf, fixed dispatch, concurrency 32, 32,768 window) and the Datakit-SFT / big-mix held-out sets
# (first + later turns, concurrency 16, 102,400 window, 4,096 new tokens). To add a benchmark: one line in each of the three functions below.
ROOT=${ROOT:-/scratch/09749/atutej/marin_speculator}; RUNS=$ROOT/runs
EVAL_DIR=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance
SB_DATA=${SB_DATA:-/scratch/09749/atutej/marin_speed_bench/aiperf_data/qualitative.jsonl}
BENCHES_KNOWN="speedbench datakit bigmix"

bench_files() {  # bench_files <bench>: files that must exist (one per line)
    case $1 in
        speedbench) echo "$SB_DATA" ;;
        datakit)    echo "$RUNS/val_prompts_multiturn_1024/first_turn.jsonl" ;;
        bigmix)     echo "$RUNS/bigpool_val_prompts_1024/first_turn.jsonl" ;;
        *) return 1 ;;
    esac
}

bench_env() {  # bench_env <bench>: set RUNNER and export the runner's settings (call inside a subshell)
    case $1 in
        speedbench)
            RUNNER=$EVAL_DIR/serve_speedbench.sh
            export DISPATCH=sessions CONC=32 OSL=4096 MAX_MODEL_LEN=32768 ;;
        datakit|bigmix)
            local d; if [[ $1 == datakit ]]; then d=$RUNS/val_prompts_multiturn_1024; else d=$RUNS/bigpool_val_prompts_1024; fi
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$d PROMPT_SETS="$1=$d:4096:first_turn,later_turn" CONCURRENCIES=16 MAX_MODEL_LEN=102400 MAX_TOKENS=4096 GPU_LOG=1
            unset PLAIN SERVER_EXTRA ;;
        *) return 1 ;;
    esac
}

bench_done() {  # bench_done <bench> <result dir of one drafter>: 0 when the run left its result files
    case $1 in
        speedbench) [[ -f "$2/speed_bench_report.csv" ]] ;;
        datakit|bigmix) compgen -G "$2/$1/c*/acceptance.csv" >/dev/null || compgen -G "$2/acceptance.csv" >/dev/null ;;
        *) return 1 ;;
    esac
}
