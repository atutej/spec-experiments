# Benchmarks for eval_matrix: name -> required files and the runner's settings. Sourced by eval_matrix.sh (preflight) and eval_matrix.sbatch (runs).
# Settings are those of the earlier figure evals: SPEED-Bench qualitative (AIPerf, fixed dispatch, concurrency 32, 32,768 window) and the Datakit-SFT / big-mix held-out sets
# (first + later turns, concurrency 16, 102,400 window, 4,096 new tokens). To add a benchmark: one line in each of the three functions below.
ROOT=${ROOT:-/scratch/09749/atutej/marin_speculator}; RUNS=$ROOT/runs
EVAL_DIR=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance
SB_DATA=${SB_DATA:-/scratch/09749/atutej/marin_speed_bench/aiperf_data/qualitative.jsonl}
# paper = GSM8K, MATH-500, HumanEval, MBPP, MT-Bench (turn 1) as in the DFlash/DSpark papers; prompts from build_paper_prompts.py; all five sets are served in ONE server session (T=1, 2,048 new tokens, concurrency 16)
PAPER_DIR=${PAPER_PROMPTS:-$RUNS/paper_prompts_20261107}; PAPER_SETS="gsm8k math500 humaneval mbpp mtbench"
FINANCE_DIR=${FINANCE_PROMPTS:-$RUNS/finance_val_prompts_20261111}   # finance_v1 task rows from the Datakit validation split (build_finance_val_prompts.py): 256 prompts of about 30k tokens each
DATAKIT_FULL_DIR=${DATAKIT_FULL_PROMPTS:-$RUNS/val_prompts_uncapped_512_20261110}   # 512 first-turn prompts sampled uniformly from the Datakit validation rows, prompts up to 31,000 tokens (the datakit set caps them at 28,000 and under-represents long rows)
SNOWBALL_SWE_DIR=${SNOWBALL_SWE_PROMPTS:-$RUNS/snowball_swe_replay_prompts_20261112}   # agent contexts replayed from published Snowball v0.1 SWE-bench Verified trajectories (build_snowball_swe_replay_prompts.py): 1,366 turns in 5 context-length buckets
SNOWBALL_SWE_SUBSETS="ctx_lt4k ctx_4k_16k ctx_16k_32k ctx_32k_64k ctx_64k_128k"
AIME_DIR=${AIME_PROMPTS:-$RUNS/aime_prompts_20261112}   # AIME 2024 + 2025 (60 problems) x 4 repeats = 240 prompts, MATH-500 template (build_aime_prompts.py); long reasoning replies: 32,000 new tokens
BENCHES_KNOWN="speedbench datakit bigmix paper finance datakit_full snowball_swe aime"

bench_files() {  # bench_files <bench>: files that must exist (one per line)
    case $1 in
        speedbench) echo "$SB_DATA" ;;
        datakit)    echo "$RUNS/val_prompts_multiturn_1024/first_turn.jsonl" ;;
        bigmix)     echo "$RUNS/bigpool_val_prompts_1024/first_turn.jsonl" ;;
        finance)    echo "$FINANCE_DIR/first_turn.jsonl" ;;
        datakit_full) echo "$DATAKIT_FULL_DIR/first_turn.jsonl" ;;
        snowball_swe) echo "$SNOWBALL_SWE_DIR/manifest.json"; for s in $SNOWBALL_SWE_SUBSETS; do echo "$SNOWBALL_SWE_DIR/$s.jsonl"; done ;;
        paper)      for s in $PAPER_SETS; do echo "$PAPER_DIR/$s/first_turn.jsonl"; done ;;
        aime)       echo "$AIME_DIR/aime/first_turn.jsonl" ;;
        *) return 1 ;;
    esac
}

bench_env() {  # bench_env <bench>: set RUNNER and export the runner's settings (call inside a subshell)
    case $1 in
        speedbench)
            RUNNER=$EVAL_DIR/serve_speedbench.sh
            export DISPATCH=sessions CONC=32 OSL=4096 MAX_MODEL_LEN=32768 SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        datakit|bigmix)
            local d; if [[ $1 == datakit ]]; then d=$RUNS/val_prompts_multiturn_1024; else d=$RUNS/bigpool_val_prompts_1024; fi
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$d PROMPT_SETS="$1=$d:4096:first_turn,later_turn" CONCURRENCIES=16 MAX_MODEL_LEN=102400 MAX_TOKENS=4096 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        snowball_swe)
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$SNOWBALL_SWE_DIR PROMPT_SETS="snowball_swe=$SNOWBALL_SWE_DIR:2048:${SNOWBALL_SWE_SUBSETS// /,}" CONCURRENCIES=16 MAX_MODEL_LEN=140000 MAX_TOKENS=2048 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        datakit_full)
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$DATAKIT_FULL_DIR PROMPT_SETS="datakit_full=$DATAKIT_FULL_DIR:4096:first_turn" CONCURRENCIES=16 MAX_MODEL_LEN=102400 MAX_TOKENS=4096 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        finance)
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$FINANCE_DIR PROMPT_SETS="finance=$FINANCE_DIR:4096:first_turn" CONCURRENCIES=16 MAX_MODEL_LEN=102400 MAX_TOKENS=4096 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        aime)
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$AIME_DIR/aime PROMPT_SETS="aime=$AIME_DIR/aime:32000:first_turn" CONCURRENCIES=16 MAX_MODEL_LEN=32768 MAX_TOKENS=32000 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        paper)
            local ps=""; for s in $PAPER_SETS; do ps+="${ps:+ }$s=$PAPER_DIR/$s:2048:first_turn"; done
            RUNNER=$EVAL_DIR/serve_eval.sh
            export PROMPTS_DIR=$PAPER_DIR/gsm8k PROMPT_SETS="$ps" CONCURRENCIES=16 MAX_MODEL_LEN=32768 GPU_LOG=1
            unset PLAIN; export SERVER_EXTRA="${EVAL_SERVER_EXTRA:-}" ;;
        *) return 1 ;;
    esac
}

bench_done() {  # bench_done <bench> <result dir of one drafter>: 0 when the run left its result files
    case $1 in
        speedbench)
            if [[ "$(basename "$2")" == plain-* ]]; then compgen -G "$2/speed_bench_*/profile_export_aiperf.json" >/dev/null   # plain arm: throughput only
            else [[ -f "$2/speed_bench_report.csv" ]]; fi ;;
        datakit|bigmix|finance|datakit_full|snowball_swe|aime)
            if [[ "$(basename "$2")" == plain-* ]]; then compgen -G "$2/$1/c*/results.json" >/dev/null
            else compgen -G "$2/$1/c*/acceptance.csv" >/dev/null || compgen -G "$2/acceptance.csv" >/dev/null; fi ;;
        paper)
            if [[ "$(basename "$2")" == plain-* ]]; then compgen -G "$2/*/c*/results.json" >/dev/null
            else local s; for s in $PAPER_SETS; do compgen -G "$2/$s/c*/acceptance.csv" >/dev/null || return 1; done; fi ;;
        *) return 1 ;;
    esac
}
